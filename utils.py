# Imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import scipy.stats as stats
from collections import namedtuple
from scipy.ndimage import gaussian_filter1d
from brian2 import *

# Own imports
from visualizations import *

# Define Izhkevich Neurons
def IzhikevichNeuron(
    voltages,
    izhikevich_us,
    injected_current,
    izhikevich_a,
    izhikevich_b,
    izhikevich_c,
    izhikevich_d,
    spike_threshold = 30,
    t_step_size = 1,
    diagnostic_printing = False
):
    '''
    A function to model an Izhikevich Neuron.
    Implemented with from "Simple Model of Spiking Neurons", Izhikevich (2003). 

    Parameters:
    ----------
        voltages: nx1 Numpy Array. The voltages of each neuron in the network. Each row is the voltage of an individual neuron.
        izhikevich_us: nx1 Numpy Array. The u recovery parameter for the Izhikevich neuron. Each row is the value of u for an individual neuron.
        injected_current: nx1 Numpy array. The amount of external injected current to each individual neuron. Each row is the value of injected current being applied to an individual neuron.
        izhikevich_a:
        izhikevich_b:
        izhikevich_c:
        izhikevich_d:

    Returns:
    -------
        voltages: nx1 Numpy Array. The voltages of each neuron in the network. Each row is the voltage of an individual neuron.
        izhikevich_us: nx1 Numpy Array. The u recovery parameter for the Izhikevich neuron. Each row is the value of u for an individual neuron.
    '''

    # First check fo  firing 
    spike_idx = voltages >= spike_threshold
    if (np.sum(spike_idx) != 0) and diagnostic_printing:
        print('Voltages:', voltages)
        print('Voltages shape:', voltages.shape)
        print('Spike Index:', spike_idx)
        print('Spike Index Shape:', spike_idx.shape, '\n')
    voltages[spike_idx] = izhikevich_c[spike_idx]
    izhikevich_us[spike_idx] += izhikevich_d[spike_idx]

    # Calculate the change in the voltages and the u parameters
    dv = (
        (0.04 * voltages * voltages) + (5 * voltages) + 140 - izhikevich_us + injected_current
    ) * t_step_size
        
    du = (
        izhikevich_a * (izhikevich_b * voltages - izhikevich_us)
    ) * t_step_size

    # Update the voltages and u parametesr
    # Have to reshape from row vectors to shapeless vectors
    voltages += dv # .reshape(-1)
    izhikevich_us += du # .reshape(-1)

    # Reset peaks to 30 spike threshold just so that they're equalized, following Izhikevich (2003) who does this in Fig. 3.
    voltage_cap_idx = voltages >= spike_threshold
    voltages[voltages >= spike_threshold] = spike_threshold

    return voltages, izhikevich_us, np.array(spike_idx, dtype = np.int64)

def GeneratePoissoinPointProcess(
    n,
    q,
    num_time_steps,
    t_step_size,
    diagnostic_printing = False,
):
    '''
    A function to generate random noise in the network as poissoin point processes.

    Paramters:
    ---------
        n: Int. Number of neurons in the network.
        q: Float. The average firing rate in hertz (1/s)
        num_time_steps: Int. Number of time steps, unitless
        t_step_size: Float. The step size in time (ms)
        

    Returns:
    -------
        external_noise_array: Numpy Array. The first column is the node index that should get injected current and the second column is the time step when it should get that injected current.
    
    '''
    total_time_ms = num_time_steps * t_step_size # Total time in milliseconds (ms)

    if diagnostic_printing:
        print('n:', n)
        print('q:', q)
    
    # Convert total time to seconds
    total_time_s = total_time_ms / 1000  # 1000 ms in a second
    lam = q * total_time_s # Create lambda from the number of seconds that it's being run and using the expected firing rate for each second

    if diagnostic_printing:
        print('lam:', lam)
    
    # Find the external noise
    external_noise = np.random.poisson(lam, size = n)
    
    # An array to keep the values for the external noise to the neuron
    # Beacuse it'd be a sparse matrix to have all neurons on rows and all time steps on columns
    # I'm going to create an array with the first column being the neuron and the second being a time step that it should fire
    external_noise_array = []
    
    # Determine when those firings should happen
    for i in range(n):
        # Find where the neuron should fire
        if diagnostic_printing:
            print('Num time steps:', num_time_steps)
            print('External noise:', external_noise[i])
        firing_locations = np.random.choice(np.arange(0, num_time_steps), size = external_noise[i], replace = False)
        external_noise_array += [np.column_stack([np.repeat(i, external_noise[i]), firing_locations])]
    
    # Concatenate the list of numpy arrays into a single numpy array 
    external_noise_array = np.concat(external_noise_array)

    return external_noise_array

def PosArray_to_SignalTiming(
    pos_array,
    edges,
    config_dict,
    constant_diameter = None, # A way to set the diameter of axons constant and change conduction delay ony by axon length,
    myelinated_index = None
):
    '''
    
    
    Inputs:
        
    
    '''

    ## Put the parameters into the function name space instead of constantly indexing the dictinoary
    # Timing
    num_time_steps = config_dict['timing']['num_time_steps']
    t_step_size = float(config_dict['timing']['t_step_size'])
    noise_injected_current_amount = config_dict['timing']['noise_injected_current_amount']

    # Electical parameters
    rushton_constant = config_dict['electrical-parameters']['rushton_constant']  
    allometric_scaling_exponent = config_dict['electrical-parameters']['allometric_scaling_exponent'] 

    # G-Ratio
    unmyelinated_gratio = config_dict['g-ratio']['unmyelinated_gratio']
    gratio_sd = config_dict['g-ratio']['gratio_sd']
    gratio_lower_bound = config_dict['g-ratio']['gratio_lower_bound']
    theta_scalar_myelin = config_dict['g-ratio']['theta_scalar_myelin']
    theta_scalar_non_myelin = config_dict['g-ratio']['theta_scalar_non_myelin']
    gratio_threshold = config_dict['g-ratio']['gratio_threshold']
    unmyelinated_constant_diameter = config_dict['g-ratio']['unmyelinated_constant_diameter'] 
    myelinated_constant_diamteter = config_dict['g-ratio']['myelinated_constant_diamteter']
    
    
    # Constants defined in the function
    gratio_mean = np.exp(-1/2) # ~0.6 from Rushton
    num_edges = edges.shape[0]

    # Create a distribution for G-Ratios
    gratio_distribution = stats.truncnorm(
        (gratio_lower_bound - gratio_mean) / gratio_sd,
        (unmyelinated_gratio - gratio_mean) / gratio_sd,
        loc = gratio_mean, 
        scale = gratio_sd
    )

    gratio_distribution = stats.truncnorm(
        (gratio_lower_bound - gratio_mean) / gratio_sd,
        (unmyelinated_gratio - gratio_mean) / gratio_sd,
        loc = gratio_mean, 
        scale = gratio_sd
    )

    # Randomly draw g-ratios based on the 
    g_ratios = np.array(gratio_distribution.rvs(num_edges), dtype = np.float64)
    
    # Calculate the length of each axon
    distances = np.linalg.norm(
        pos_array[edges[:, 0], :] - pos_array[edges[:, 1], :], 
        axis = 1
    )
    
    # Defining the inner diameter of the neuron (d)
    if not constant_diameter:
        d = distances**allometric_scaling_exponent # Scale internal diameter via allometric scaling

        # Find which edges are myelinated or not
        myelinated_index = (d >= gratio_threshold)
        unmyelinated_index = (1 - myelinated_index).astype(np.bool_)
    
    elif constant_diameter:
        # We have a  set myelinated index, find unmyelinated
        unmyelinated_index = (1 - myelinated_index).astype(np.bool_)
        
        # Get array of internal diameters based on set myelination values
        d = (myelinated_index) * np.array([myelinated_constant_diamteter] * num_edges) + (unmyelinated_index) * np.array([unmyelinated_constant_diameter] * num_edges)

    # Calculating how far the g-ratios for unmyelinated axons are from the desired g-ratio for unmyelinated neurons
    g_ratios[unmyelinated_index] = unmyelinated_gratio
    
    # Defining the outer diameter
    # If it's myelinated, it should be the same as d. 
    D = (d / g_ratios)

    ### Calculate the value for theta, scale by the time step size so it's an actual speed value
    thetas = (
        myelinated_index * (d * np.sqrt(-np.log(g_ratios)) ) * (theta_scalar_myelin / t_step_size) # Myelinated calculation for theta from Rushton
        + unmyelinated_index * (np.sqrt(d) * (theta_scalar_non_myelin / t_step_size)) # Unmyelinated calculation for theta
    )
    
    # Values for time to complete action potentials
    time_to_complete_ap = distances / thetas

    return time_to_complete_ap

def LoadYAMLFile(config_filepath):
    '''
    A function to load a yaml file. The function returns loaded dictionary object.
    
    Parameters:
    ----------
    config_filepath: String 
        The file path to the desired configuation file. 
    
    Returns:
    -------
    config_dict: Dictionary
        Python dictionary containing the information from the provided yaml file.
    '''
    
    with open(config_filepath) as config:
        config_dict = yaml.safe_load(config)
        
    return config_dict

# Create a function for generating the node positions
def GeneratePosArray(
    implement_nueron_modules_positions,
    n_dim = 2,
    radius = 4000,
    n = 100, # The number of neurons when in a randomized space without modules
    num_groups = 3,
    n_per_group = 100,
    seed = 2302,
    std = 0.05,
    x1_mean = 0.9, 
    x2_mean = 0.6,
    x3_mean = 0.9,
    y1_mean = 0.3,
    y2_mean = 0.6,
    y3_mean = 0.9,
):
    '''
    A general function for creating the positions array.

    Parameters:
    ----------
        implement_nueron_modules_positions: A boolean to decide if the neurons should be clustered in modules or randomly dispered in the physical space.
        
    '''

    # Create positions for the neurons in a physical space
    if not implement_nueron_modules_positions:
        if seed is not None: # Allow for seed to be none in case the notebook needs to be ran once through
            np.random.seed(seed)
        pos_array = np.random.rand(n, n_dim) * radius
    
    elif implement_nueron_modules_positions:
        # Make 3 seperate groups
        n = n_per_group * num_groups
        if seed is not None:
            np.random.seed(seed) # Allow for seed to be none in case the notebook needs to be ran once through

        # Make it so that the first neuron is placed exactly at the location of the mean, this will be the "rich-club" neuron for stimulation
        x1 = np.random.normal(loc = x1_mean, scale = std, size = n_per_group-1)
        x2 = np.random.normal(loc = x2_mean, scale = std, size = n_per_group-1)
        x3 = np.random.normal(loc = x3_mean, scale = std, size = n_per_group-1) 
        
        y1 = np.random.normal(loc = y1_mean, scale = std, size = n_per_group-1)
        y2 = np.random.normal(loc = y2_mean, scale = std, size = n_per_group-1)
        y3 = np.random.normal(loc = y3_mean, scale = std, size = n_per_group-1)

        # Concatenate the set positions now with the randomized positions
        x1 = np.concat([np.array([x1_mean]), x1])
        x2 = np.concat([np.array([x2_mean]), x2])
        x3 = np.concat([np.array([x3_mean]), x3])

        y1 = np.concat([np.array([y1_mean]), y1])
        y2 = np.concat([np.array([y2_mean]), y2])
        y3 = np.concat([np.array([y3_mean]), y3])
        
        # Pair them together 
        xs = np.concat([x1, x2, x3])
        ys = np.concat([y1, y2, y3])
        pos_array = np.column_stack([xs, ys]) * radius

    return pos_array

def GetDistanceMatrix(
    pos_array,
    n_dim = 2,
    check_symmetry = False
):
    '''
    A function to calculate the distance matrix given a specific positions array.
    '''

    n = pos_array.shape[0]
    distance_matrix = np.zeros((n, n), dtype = np.float64) # Create a distance matrix to capture the eucilidean distances between each nodes

    # Loop over each node, and calculate it's distance from the other neurons
    for i in range(n):
    
        # Repeat the positions array n-1 times to subtract from the other positions arrays
        # Repeating it (n-1) - 1 so that I don't re-calculate previously calculated distances, this will be a symmetric matrix
        ith_pos_array = np.repeat(
            pos_array[i, :].reshape(1, n_dim), (n-1) - i, axis = 0
        ).reshape((n-1) - i, 2)
    
        # Calculate the euclidean distance between the nodes 
        distance_vec = np.linalg.norm(
            ith_pos_array - pos_array[(i + 1):, :], axis = 1
        )
    
        # Fill the distance matrix using this distance vector
        distance_matrix[i, (i + 1):] = distance_vec
    
    # Make it symmetrix by filling in the lower-traingle 
    distance_matrix += distance_matrix.T

    # Make sure it's symmetric
    if check_symmetry:
        print(
            'Distance matrix symmetric?:', 
            np.sum(distance_matrix.T == distance_matrix) == n*n
        )

    return distance_matrix

def GetAdjacencyMatrix(
    implement_neuron_modules_connections,
    distance_matrix,
    pos_array,
    radius,
    n_dim,
    adjacency_matrix_path,
    network_graph_path,
    n_per_group = 100,
    num_groups = 3,
    distance_connection_probability_exponent_constant = None,
    num_rich_club_neurons_per_module = 1,
    visualize_adjacency_matrix = True,
    visualize_connected_network = True,
    constant_connection_probability = None
):
    '''
    A function to create the adjacency matrix for the network.

    Parameters:
    ----------
        implement_neuron_modules_connections: A boolean. If True, networks will be connected as modules. If false, they will have their connection probability 

    Returns:
    -------
        A: The adjacency matrix.
        edges: A mx(n_dim) numpy array of the edges. The first column is the outgoing neuron and the second is the incoming neuron.
        rich_club_neurons:
    '''
    
    if not implement_neuron_modules_connections:
        # Define the number of neurons in the network
        n = pos_array.shape[0]
        
        # Now make it so that connection probabilties exponentially decay with distance
        if distance_connection_probability_exponent_constant is None:
            distance_connection_probability_exponent_constant = 10

        if constant_connection_probability is None:
            # Scaling distances by the maximum distance in the space so that it is a value value between 0 and 1 for probability. 
            max_distance = np.sqrt((radius)**2 * n_dim)
            connection_probabilty_exponent = ( distance_matrix / max_distance) * distance_connection_probability_exponent_constant # Add a constant to change the strength of this value
            connection_probability = np.exp(-connection_probabilty_exponent)
        else:
            connection_probability = np.ones((n,n)) * constant_connection_probability
        
        # Calculate the adjacency matrix as a binomial process based on connection probabilties
        A = np.random.binomial(n = 1, p = connection_probability)
        A -= np.eye(n, dtype = np.int64) # Subtract out self-connections
        
        # Check if any neurons do not have a single connection
        no_connection_neurons = np.argwhere((np.sum(A, axis = 1) + np.sum(A, axis = 0)) == 0).flatten() # find neurons without a connection
        closest_neurons = np.argsort(distance_matrix[no_connection_neurons, :], axis = 1)[:, 1] # Find which neurons they're closest to, skip the result that is a self connection
        new_edges = np.column_stack([no_connection_neurons, closest_neurons]) # Combine into a vector
        A[new_edges[:, 0], new_edges[:, 1]] = A[new_edges[:, 0], new_edges[:, 1]] = 1 # Create double sided connection between the two neurons
    
        # Remove self connections
        diag_indices = np.arange(n)
        A[diag_indices, diag_indices] = np.zeros(n)

        # In this case there are no rich club neurons
        rich_club_neurons = None

        edges = np.argwhere(A == 1)
    
    elif implement_neuron_modules_connections:
        if distance_connection_probability_exponent_constant is None:
            distance_connection_probability_exponent_constant = 100

        n = n_per_group * num_groups
        
        # Go through the different groups, connect each by their closest neurons 
        max_distance = np.sqrt((radius)**2 * n_dim)
        connection_probabilty_exponent = ( distance_matrix / max_distance) * distance_connection_probability_exponent_constant # Add a constant to change the strength of this value
        connection_probability = np.exp(-connection_probabilty_exponent)
    
        # Loop over the groups to connect them
        A = np.zeros((n, n))

        # Start with ordered connections between the neurons in the set position, will have characteristic flow then
        A[0, n_per_group] = 1 # Module 1 to 2
        A[n_per_group, n_per_group*2]  = 1 # Module 2 to 3
        A[n_per_group*2, 0]  = 1 # Module 3 to 1
        
        for group_idx in range(num_groups):
            start_index = n_per_group * group_idx
            end_index = start_index + n_per_group
    
            # Calculate the connections within the module
            A_sub = np.random.binomial(n = 1, p = connection_probability[start_index:end_index, start_index:end_index])
    
            # Fill in the full connection matrix using the sub module
            A[start_index:end_index, start_index:end_index] = A_sub

        # Remove self connections
        diag_indices = np.arange(n)
        A[diag_indices, diag_indices] = np.zeros(n)
    
        # Check if any neurons do not have a single connection
        no_connection_neurons = np.argwhere((np.sum(A, axis = 1) + np.sum(A, axis = 0)) == 0).flatten() # find neurons without a connection
        closest_neurons = np.argsort(distance_matrix[no_connection_neurons, :], axis = 1)[:, 1] # Find which neurons they're closest to, skip the result that is a self connection
        new_edges = np.column_stack([no_connection_neurons, closest_neurons]) # Combine into a vector
        A[new_edges[:, 0], new_edges[:, 1]] = A[new_edges[:, 0], new_edges[:, 1]] = 1 # Create double sided connection between the two neurons
    
        # Remove self connections
        diag_indices = np.arange(n)
        A[diag_indices, diag_indices] = np.zeros(n)
    
        # Find the highest degree nodes in each module, and connect them with the others
        rich_club_neurons = []
        total_degrees = np.sum(A, axis = 0) + np.sum(A, axis = 1)
        for group_idx in range(num_groups):
            start_index = n_per_group * group_idx
            end_index = start_index + n_per_group
            
            rich_club_neurons_for_module = np.argsort(-total_degrees[start_index:end_index])[:num_rich_club_neurons_per_module]
            rich_club_neurons += [rich_club_neurons_for_module + start_index] # Add them to a list, add starting index to make index the correct basis
        
        # Turn list of arrays into an array
        rich_club_neurons = np.concat(rich_club_neurons)
        rich_club_connections = np.array(np.meshgrid(rich_club_neurons, rich_club_neurons)).T.reshape(-1, 2) # Get all combinations of rich club neurons
    
        # Create double sided connections between all rich club neurons
        A[rich_club_connections[:, 0], rich_club_connections[:, 1]] = 1
    
        # Remove self connections again
        A[diag_indices, diag_indices] = np.zeros(n)
            
        # Generate the edges
        edges = np.argwhere(A == 1)
    
    
    print(f'Network Density: {(np.sum(A) / (A.shape[0] * A.shape[1]) * 100):.2f}%')
    
    if visualize_adjacency_matrix: 
        sns.heatmap(A, cmap = 'bwr', alpha = 0.5, cbar_kws={'label': 'Adjacency Matrix Value'})
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('Neuron Index')
        plt.ylabel('Neuron Index')
        plt.title('Neuron Adjacency Matrix')
        plt.savefig(adjacency_matrix_path)
        print('Adjacency Matrix:')
        plt.show()
        print('\n')

    if visualize_connected_network:
        fig = ShowNetwork(
            pos_array = pos_array,
            edges = edges,
            A = A,
            axis_limit = radius,
            showticklabels = True,
            gridwidth = 2,
            gridcolor = 'black',
            node_size = 15,
            title = 'Neuron Graph Network',
            title_x = 0.5, 
            node_size_floor=3,
            degree_based_size_scalar=0.5
        )
        fig.write_image(network_graph_path)
        print('Network Configuration:')
        fig.show(renderer="png")
        print('\n')

    return A, edges, rich_club_neurons

# Function for Simulation
def SimulateNeurons(
    pos_array,
    edges,
    weights,
    num_time_steps,
    time_to_complete_ap = None,
    q = 1, # Random process added in
    constant_injected_current_index = None,
    vary_neuron_type = False,
    resting_voltage = -70,
    default_a = 0.02,
    default_b = 0.2,
    default_c = -65,
    default_d = 8,
    superthreshold_injected_current = 15,
    highest_injected_current = 1,
    t_step_size = 1,
    injected_current_decay_parameter = 0,
    diagnostic_printing = False,
    num_time_steps_with_injection = None
):
    '''

    A function to simulate the neurons for a given period of time.

    Parameters:
    ----------

    Returns:
    -------

    '''
    # Create a named tuple object to save results for later
    SimulationResults = namedtuple(
        'SimulationResults', 
        [
            'voltages', 'us', 'injected_current',
            'spikes', 'q', 'num_time_steps', 't_step_size',
            't_index'
        ]
    )

    # Get the number of neurons and edges in the simulation
    n = pos_array.shape[0]
    m = edges.shape[0]
    
    # Create arrays to save the voltages, injected_current, and u's at each time step
    voltages_over_time = np.zeros((num_time_steps, n), dtype = np.float64)
    izhikevich_us_over_time = np.zeros((num_time_steps, n), dtype = np.float64)
    injected_current_over_time = np.zeros((num_time_steps, n), dtype = np.float64)
    spikes_over_time = np.zeros((num_time_steps, n), dtype = np.float64)

    if vary_neuron_type == True:
        # Fill this in more 
        izhikevich_a = np.array([default_a] * n, dtype = np.float64)
        izhikevich_b = np.array([default_b] * n, dtype = np.float64)
        izhikevich_c = np.array([default_c] * n, dtype = np.float64)
        izhikevich_d = np.array([default_d] * n, dtype = np.float64)
        
    else:
        # Have all neurons be regular spiking Izhikevich Neurons
        izhikevich_a = np.array([default_a] * n, dtype = np.float64)
        izhikevich_b = np.array([default_b] * n, dtype = np.float64)
        izhikevich_c = np.array([default_c] * n, dtype = np.float64)
        izhikevich_d = np.array([default_d] * n, dtype = np.float64)     

    if num_time_steps_with_injection is None:
        num_time_steps_with_injection = num_time_steps

    # Create an array for the injected current
    highest_injected_current_vec = np.array([highest_injected_current]*n, dtype = np.float64).reshape(1, n)

    # Create arrays for the starting voltages and izhikeivch_us
    voltages = np.array([resting_voltage] * n, dtype = np.float64)
    izhikevich_us = default_b * voltages # np.array([], dtype = np.float64)
    injected_current = np.zeros(n, dtype = np.float64)
    spike_idx = np.zeros(n, dtype = np.float64)

    # Generate random input to the system
    external_noise_array = GeneratePoissoinPointProcess(
        n = n, q = q, 
        num_time_steps = num_time_steps, 
        t_step_size = t_step_size * 1000, # Convert the t_step_size that is in seconds to ms
        diagnostic_printing = diagnostic_printing
    ) 

    # Incorporating timing dynamics
    if time_to_complete_ap is not None:
        # Create a counter for the signal along an axon
        # t_index = np.zeros(m, dtype = np.int64)
        # t_index_over_time = np.zeros((num_time_steps, m), dtype = np.float64)

        # Create an array of when spikes need to occur to account for delay
        spike_delays = np.empty((0, 2))
        
        
    for t in range(num_time_steps):
        # Create a new array for random noise at this time step
        random_noise_at_time_step = np.zeros(n, dtype = np.float64)
        
        # Create a diagonal matrix such that that diagonal entry says if this neuron spiked or not
        spike_idx_diag = np.diag(spike_idx)

        # Incorporate timing if passed as a parameter
        if time_to_complete_ap is not None:

            # Create a matrix to hold the indices of which nodes are getting injected current
            nodes_getting_injected_current_mat = np.zeros((n,n), dtype = np.float64)
            
            # Check if any axons are actively carrying signal
            # update_idx = t_index > 0
            # t_index[update_idx] += 1 # If axon is carrying signal, update the timing for how many time steps its been since initial firing

            # Look for axons whose firing ended now
            # ended_firing_index = t_index >= time_to_complete_ap
            # t_index[ended_firing_index] = 0 # Reset their t_index now to 0
            ended_firing_index = spike_delays[:, 1] <= t
            
            if ended_firing_index.shape[0] != 0:
                ended_firing_edge_idx = spike_delays[ended_firing_index, 0].astype(np.int64)
                end_of_firing_edges = edges[ended_firing_edge_idx, :]
                
                ## Calculate the injected current at time step taking into account timing
                # Find the edges that finished firing, including pre and post-synaptic neurons
                # end_of_firing_edges = edges[ended_firing_index, :]
            
                nodes_getting_injected_current_mat[end_of_firing_edges[:, 0], end_of_firing_edges[:, 1]] = 1 # Fill in with a 1 as an indicator
            
                # Add injected current to the post-synaptic neurons
                spike_adjusted_weights_mat = nodes_getting_injected_current_mat * weights # Filter weights based on if the axon finished firing
                injected_current_at_step = highest_injected_current_vec @ spike_adjusted_weights_mat 
            
                # Update the total injected current for the neuron, to take into account any previous
                injected_current += injected_current_at_step.reshape(-1)

                # Remove the delays from the spike delay values if finished with delay
                spike_delays = spike_delays[~ended_firing_index]
                
        else:
            ## Calculate the injected current at time step without taking into account timing
            # Multiply spike_idx_diag by weights to select the rows (neurons) that have spiked.
            spike_adjusted_weights_mat = spike_idx_diag @ weights 

            # Then multiply by the injected current vec to sum down the column
            # getting the total injected current to the neuron at this time
            injected_current_at_step = highest_injected_current_vec @ spike_adjusted_weights_mat

            # Update the total injected current for the neuron, to take into account any previous
            injected_current += injected_current_at_step.reshape(-1)

        # Check if any neurons are getting random injected current at this time step
        if q != 0:
            external_noise_at_this_time_step_idx = external_noise_array[:, 1] == t
            neurons_getting_external_noise_at_this_time_step = external_noise_array[external_noise_at_this_time_step_idx, 0]
            random_noise_at_time_step[neurons_getting_external_noise_at_this_time_step] = superthreshold_injected_current
    
            if (np.sum(external_noise_at_this_time_step_idx) != 0) and diagnostic_printing:
                print(f'{t}')
                print('external_noise_at_this_time_step_idx:', external_noise_at_this_time_step_idx)
                print('random_noise_at_time_step:', random_noise_at_time_step, '\n')

            # Add the noise to the injected current at this time step
            injected_current_at_step = random_noise_at_time_step
            
            # Update the total injected current for the neuron, to take into account any previous
            injected_current += injected_current_at_step.reshape(-1)

        if (constant_injected_current_index is not None) and (num_time_steps_with_injection > t):
            # If there are neurons being constantly injected with current, add that current
            constant_injected_current_needed_idx = injected_current[constant_injected_current_index] < superthreshold_injected_current
            constant_injected_current_needed_idx = constant_injected_current_index[constant_injected_current_needed_idx]
            injected_current[constant_injected_current_needed_idx] = superthreshold_injected_current
        
        # Update voltages, us, and capture nodes that spiked
        voltages, izhikevich_us, spike_idx = IzhikevichNeuron(
            voltages = voltages,
            izhikevich_us = izhikevich_us,
            injected_current = injected_current,
            izhikevich_a = izhikevich_a,
            izhikevich_b = izhikevich_b,
            izhikevich_c = izhikevich_c,
            izhikevich_d = izhikevich_d,
            spike_threshold = 30,
            t_step_size = t_step_size * 1000, # Convert the t_step_size that is in seconds to ms
            diagnostic_printing = diagnostic_printing
        )

        if time_to_complete_ap is not None:
            ## Update the t_index values for the edges that have the pre-synaptic neuron firing            
            # Get pre-synaptic neurons based on if they fired
            presynaptic_neurons = np.argwhere(spike_idx == 1).flatten()
            
            # Find edges with this presynaptic axon
            # start_ap_counter_idx = np.isin(edges[:, 0], presynaptic_neurons)
            # t_index[start_ap_counter_idx] += 1 # Start their t_index counter

            if presynaptic_neurons.shape[0] != 0:
                start_ap_count_edge_idx = np.argwhere(np.isin(edges[:, 0], presynaptic_neurons)).flatten()
                delivery_time = time_to_complete_ap[start_ap_count_edge_idx] + t
                # post_synaptic_neurons = edges[start_ap_count_edge_idx, 1]
                new_spike_delays = np.column_stack([start_ap_count_edge_idx, delivery_time])
    
                spike_delays = np.concat([spike_delays, new_spike_delays])

        # Reset injected current for neurons who spiked to 0
        injected_current[spike_idx] = 0
        
        # Save results to larger matrices
        voltages_over_time[t, :] = voltages
        izhikevich_us_over_time[t, :] = izhikevich_us
        injected_current_over_time[t, :] = injected_current
        spikes_over_time[t, :] = spike_idx
        if time_to_complete_ap is not None:
            # t_index_over_time[t, :] = t_index
            t_index_over_time = None

        # Exponential decay to allow the injected current to leak
        injected_current *= np.exp(-injected_current_decay_parameter)

        if diagnostic_printing and (time_to_complete_ap is not None):
            print(f'Spike Delays as of ({t}):', spike_delays)

    if time_to_complete_ap is None:
        t_index_over_time = None
        
    # Compile results in a named tuple for ease of use
    out = SimulationResults(
        voltages = voltages_over_time,
        us = izhikevich_us_over_time,
        injected_current = injected_current_over_time,
        spikes = spikes_over_time,
        q = q,
        num_time_steps = num_time_steps,
        t_step_size = t_step_size * 1000, # Convert the t_step_size from seconds to ms
        t_index = t_index_over_time
    )

    return out

def Brian2SimulateNeurons(
    pos_array,
    edges,
    config_dict,
    weights, 
    implement_timing,
    time_to_complete_ap,
    constant_injected_current_index,
):
    '''
    A function to use the Brian2 Simulation method for 
    '''

    start_scope()

    # Changing the default clock so that the dt is 0.01 instead of 0.1
    num_time_steps = 100000
    dt = 0.01
    defaultclock.dt = dt*ms
    n = pos_array.shape[0]
    
    # Excitatory Parameters
    spike_threshold = 30*mV
    a = 0.02/ms
    b = 0.2/ms
    c = -65*mV
    d = 8*mV/ms
    
    # Conductance changes
    # Values for these conductances taken from https://brian2.readthedocs.io/en/stable/examples/frompapers.Stimberg_et_al_2018.example_1_COBA.html
    E_e = 0*mV 
    E_i = -80*mV 
    tau_e = 5*ms
    tau_i = 10*ms
    tau_r = 5*ms # A refractory period
    C_m = 198*pF           # Membrane capacitance
    w_e = 23*nS # 0.05*nS          # Excitatory synaptic conductance
    w_i = 20*nS # 1.0*nS           # Inhibitory synaptic conductance
        
    eqs = '''
    dv/dt = (0.04/ms/mV)*v**2+(5/ms)*v+140*mV/ms-u+ (I_syn+I_ext) / C_m: volt
    du/dt = a*(b*v-u) : volt/second 
    I_syn = g_e * (E_e - v) + g_i * (E_i - v) : amp
    dg_e/dt = -g_e / tau_e: siemens
    dg_i/dt = -g_i / tau_i: siemens
    I_ext : amp
    '''
    
    # The reset of the dynamics after a spike
    reset = '''
    v = c
    u = u + d
    '''
    
    # Creating the network
    G = NeuronGroup(n, eqs, threshold=f'v >= spike_threshold', reset=reset, method='euler')
    G.v = c # Setting the resting potentials
    G.u = b * G.v # Setting the resting value of u
    G.g_e = 0*nS # Setting the deaful
    G.g_i = 0*nS 
    
    # Set up dynamic clipping, don't want the conductance values to reach unrealisitc values
    G.run_regularly('g_e = clip(g_e, 0*nS, 30*nS)', dt=defaultclock.dt)
    G.run_regularly('g_i = clip(g_i, 0*nS, 50*nS)', dt=defaultclock.dt)

    # Apply injected current to neurons
    # np.random.seed(constant_injected_current_index_seed)
    # constant_injected_current_index = np.random.choice(
    #     np.arange(n), num_neurons_to_stimluate, 
    #     replace = False
    # )

    # Set external activity, equivalent to 7.5 mV
    G.I_ext = 0*pA
    G.I_ext[constant_injected_current_index] = 1485*pA
    
    # Creating synapses
    if not implement_timing:
        S = Synapses(G, G, '''w : 1''', on_pre='g_e_post += w*w_e')
        S.connect(i = edges[:, 0], j = edges[:, 1])

    # When adding conductance delays
    if implement_timing:

        if time_to_complete_ap is None:
            time_to_complete_ap = PosArray_to_SignalTiming(
                pos_array, edges, config_dict, 
                constant_diameter = False, myelinated_index = None
            )
    
        # Have to change the results so that the timings are on correct scale
        time_to_complete_ap = (time_to_complete_ap) * ms
    
        # Create the synapses
        S = Synapses(G, G, '''w : 1''', on_pre='g_e_post += w*w_e')
        S.connect(i = edges[:, 0], j = edges[:, 1])
    
        # Implement conduction delays
        S.delay = time_to_complete_ap

    # Implement the weights
    S.w = weights[edges[:, 0], edges[:, 1]]
    
    # group.v = 'rand()'
    state = StateMonitor(G, 'v', record = True)
    
    # Run a network
    run(num_time_steps*dt*ms)
    
    # Way to get the voltages out and make them 
    voltages = np.array(state.v.T) * 1000 # Sending to mV instead of Volts
    
    # Create the out named tuple
    out = CreateNamedTuple(
        voltages,
        t_step_size = dt,
        num_time_steps = num_time_steps
    )
    
    return out

def FullSimulation(
    config_dict,
    array_id,
    seed,
    n,
    n_per_group = 100,
    n_dim = 2,
    radius = 4000,
    implement_nueron_modules_positions = True,
    implement_neuron_modules_connections = True,
    module_std = 0.05,
    distance_connection_probability_exponent_constant = None,
    num_rich_club_neurons_per_module = 2,
    q = 0,
    injected_current_decay_parameter = 0.01,
    include_constant_injected_current_to_rich_club = False,
    x1_mean = 0.9, 
    x2_mean = 0.6,
    x3_mean = 0.9,
    y1_mean = 0.3,
    y2_mean = 0.6,
    y3_mean = 0.9,
    check_distance_matrix_symmetry = False,
    visualize_adjacency_matrix= True,
    adjacency_matrix_path = None,
    visualize_connected_network= True,
    network_graph_path = None,
    num_time_steps = False,
    t_step_size = False,
    save_outputs = True,
    implement_timing = True,
    implement_constant_weights = False,
    vary_neuron_type = False,
    diagnostic_printing = False,
    resting_voltage = -70,
    default_a = 0.02,
    default_b = 0.2,
    default_c =  -65,
    default_d = 8,
    highest_injected_current = 1,
    num_groups = 3,
    number_of_rich_club_neurons_to_stimulate = 1,
    constant_weight_value = 0.10,
    visualize_spike_trace = False,
    spike_trace_path = None,
    visualize_voltage_trace = False,
    voltage_trace_path = None,
    constant_injected_current_index = None,
    mean_weight_val = 0.3,
    std_weight_val = 0.1,
    constant_connection_probability = None,
    num_time_steps_with_injection = None,
    constant_diameter = False,
    superthreshold_injected_current = 15,
    weight_rich_club_connections_more = False,
    Brian2Implementation = False
):
    '''
    A function to generate positions array, caculate distance matrix, and simluate the neurons.
    '''

    # Define the positions array
    pos_array = GeneratePosArray(
            implement_nueron_modules_positions,
            n_dim = n_dim,
            radius = radius,
            n = n,
            num_groups = num_groups,
            n_per_group = n_per_group,
            seed = seed,
            std = module_std,
            x1_mean = x1_mean, 
            x2_mean = x2_mean,
            x3_mean = x3_mean,
            y1_mean = y1_mean,
            y2_mean = y2_mean,
            y3_mean = y3_mean, 
    )

    # Re-introduce n into the scope in case modules have been created
    n = pos_array.shape[0]

    # Calculate the euclidean distance between all nodes
    distance_matrix = GetDistanceMatrix(
        pos_array = pos_array,
        n_dim = n_dim,
        check_symmetry = check_distance_matrix_symmetry
    )

    # Define the adjacency matrix and edges of the network
    A, edges, rich_club_neurons = GetAdjacencyMatrix(
        implement_neuron_modules_connections = implement_neuron_modules_connections,
        distance_matrix = distance_matrix,
        pos_array = pos_array,
        radius = radius,
        n_dim = n_dim,
        adjacency_matrix_path = adjacency_matrix_path,
        network_graph_path = network_graph_path,
        n_per_group = n_per_group,
        num_groups = num_groups,
        distance_connection_probability_exponent_constant = distance_connection_probability_exponent_constant,
        num_rich_club_neurons_per_module = num_rich_club_neurons_per_module,
        visualize_adjacency_matrix = visualize_adjacency_matrix,
        visualize_connected_network = visualize_connected_network,
        constant_connection_probability = constant_connection_probability
    )

    # Some parameter adjustements before simulation
    if include_constant_injected_current_to_rich_club:
        constant_injected_current_index = np.random.choice(rich_club_neurons, number_of_rich_club_neurons_to_stimulate, replace = False)
    
    if num_time_steps is None:
        num_time_steps = config_dict['timing']['num_time_steps']

    if t_step_size is None:
        t_step_size = float(config_dict['timing']['t_step_size'])

    if implement_timing:
        # Find axons that should be myelinated if constant diameter
        myelinated_index = np.zeros(edges.shape[0])

        # Myelinated the signals between modules
        if implement_neuron_modules_connections:
            mod_1_to_2 = np.argwhere(np.logical_and(edges[:,0] == 0, edges[:, 1] == n_per_group)).flatten()
            mod_2_to_3 = np.argwhere(np.logical_and(edges[:,0] == n_per_group, edges[:, 1] == n_per_group*2)).flatten()
            mode_3_to_2 = np.argwhere(np.logical_and(edges[:,0] == n_per_group*3, edges[:, 1] == 0)).flatten()
    
            myelinated_index[mod_1_to_2] = 1
            myelinated_index[mod_2_to_3] = 1
            myelinated_index[mode_3_to_2] = 1
        
        time_to_complete_ap = PosArray_to_SignalTiming(
            pos_array, edges, config_dict, 
            constant_diameter, myelinated_index
        )
    else:
        time_to_complete_ap = None
    
    if implement_constant_weights:
        weight_adjustement = np.ones((n,n)) * constant_weight_value
    else:
        weight_adjustement = np.random.normal(loc = mean_weight_val, scale = std_weight_val, size = (n,n))

        if weight_rich_club_connections_more:
            module_linking_neurons = np.array([0, n_per_group, n_per_group*2])
            weight_adjustement[module_linking_neurons, :] /= weight_adjustement[module_linking_neurons, :] # Set these all to 1 so that they have the maximum weight

    # Simluate the neurons
    if not Brian2Implementation:
        out = SimulateNeurons(
            pos_array = pos_array,
            edges = edges,
            weights = A * weight_adjustement,
            num_time_steps = num_time_steps,
            time_to_complete_ap = time_to_complete_ap,
            q = q, # Random process added in
            constant_injected_current_index = constant_injected_current_index,
            vary_neuron_type = vary_neuron_type,
            resting_voltage = resting_voltage,
            default_a = default_a,
            default_b = default_b,
            default_c = default_c,
            default_d = default_d,
            superthreshold_injected_current = superthreshold_injected_current,
            highest_injected_current = highest_injected_current,
            t_step_size = t_step_size,
            injected_current_decay_parameter = injected_current_decay_parameter,
            diagnostic_printing = diagnostic_printing,
            num_time_steps_with_injection = num_time_steps_with_injection
        )
        
    else:
        out = Brian2SimulateNeurons(
            pos_array = pos_array,
            edges = edges,
            config_dict = config_dict,
            weights = A * weight_adjustement, 
            implement_timing = implement_timing,
            time_to_complete_ap = time_to_complete_ap,
            constant_injected_current_index = constant_injected_current_index,
        )
        
    # Save the files associated with it
    if save_outputs:
        if not Brian2Implementation:
            np.savetxt(f'outputs/voltages/voltages-array-id-{array_id}.csv', out.voltages, delimiter = ',') # Voltages
            np.savetxt(f'outputs/spikes/spikes-array-id-{array_id}.csv', out.spikes, delimiter = ',') # Spikes
            np.savetxt(f'outputs/injected-current/injected-current-array-id-{array_id}.csv', out.injected_current, delimiter = ',') # Injected current
        else:
            np.savetxt(f'updated-outputs/voltages/voltages-array-id-{array_id}.csv', out.voltages, delimiter = ',')

    if visualize_spike_trace:
        PlotSpikeTrace(
            out, flip_axes=True,
            include_node_indices=False,
            cmap = 'grey_r',
            save_graphic_path = spike_trace_path
        )

    if visualize_voltage_trace:
        PlotVoltageTrace(
            out, flip_axes=True, 
            include_node_indices=False,
            cmap = 'grey_r',
            save_graphic_path = voltage_trace_path
        )

    # Create a named tuple for the network itself
    NetworkTuple = namedtuple('NetworkTuple', ['pos_array', 'edges', 'A'])
    
    return out, NetworkTuple(pos_array = pos_array, edges = edges, A = A)

def CreateSimulationResultsTupleFromFiles(
    array_id,
    array_file,
    num_time_steps,
    t_step_size,
    t_index,
):
    '''
    Parameters:
    ----------
        array_id: The unique id for the test.
        array_file: The pandas data frame that contains all permutations.

    Returns:
    -------
    '''

    SimulationResults = namedtuple(
        'SimulationResults', 
        [
            'voltages', 'us', 'injected_current',
            'spikes', 'q', 'num_time_steps', 't_step_size',
            't_index'
        ]
    )

    SimulationResults(
        voltages = np.loadtxt(f'outputs/voltages/voltages-array-id-{array_id}.csv', delimiter = ','),
        us = None,
        injected_current = np.loadtxt(f'outputs/injected-current/injected-current-array-id-{array_id}.csv', delimiter = ','),
        spikes = np.loadtxt(f'outputs/spikes/spikes-array-id-{array_id}.csv', delimiter = ','),
        q = q,
        num_time_steps = num_time_steps,
        t_step_size = t_step_size,
        t_index = t_index,
    )

def PCA(out, num_dim_pca, gaussian_filter_sigma = None, on_derivative = True):
    '''
    A function to calculte the PCA reduction of the voltage time series.

    Parameters:
    ----------
        out: NamedTuple. The returned tuple that has the voltages.
        num_dim_pca: The number of principle components to return.

    Returns:
    -------
        x_reduced: The PCA of the voltage derivatives.
        eig: The sorted eigenvalues of the covariance matrix.
        eig_vecs: The sorted eigenvectors of the covariance matrix.
    '''
    # Calculate based on the voltages over time
    x = out.voltages.copy()
    n = x.shape[1] # The number of columns is the number of neurons here 
    t_step_size = out.t_step_size # Already in ms

    # Apply a gaussian filter if a sigma value is provided, smooths the spikes so they're not as sharp
    if gaussian_filter_sigma:
        x = gaussian_filter1d(x, sigma=gaussian_filter_sigma, axis=0)
    
    # Normalize the voltages
    x_mean = np.mean(x, axis = 0)
    x_std = np.std(x, axis = 0)
    
    # Manually control for x_std == 0, neurons that didn't fire
    zero_std_idx = x_std == 0
    x[:, zero_std_idx] = np.zeros((x.shape[0], np.sum(zero_std_idx)), dtype = np.float64)
    x[:, ~zero_std_idx] = (x[:, ~zero_std_idx] - x_mean[~zero_std_idx]) / x_std[~zero_std_idx]
    
    # Manually calculate the change in the signal over time
    if on_derivative:
        x_d = np.concat(
            [
                np.zeros(n, dtype = np.float64).reshape(1, n), # Saying that at first time point it's a 0 derivative
                np.diff(x, axis = 0) / (t_step_size) # Calculating the derivative
            ]
        )
    
        x_cov = (1 / (x.shape[0] - 1)) * (x_d.T @ x_d)
    else:
        x_cov = (1 / (x.shape[0] - 1)) * (x.T @ x)
    
    # Calculate the eigenvectors of the covariance matrix (Our principle components)
    eig, eig_vecs = np.linalg.eig(x_cov)
    eig, eig_vecs = eig.real, eig_vecs.real # Remove imaginary portion in case of numerical instability
    
    # Sort the eigenvectors, negative so it's descending
    sort_idx = np.argsort(-eig)
    
    eig = eig[sort_idx]
    eig_vecs = eig_vecs[:, sort_idx]

    projection_vecs = eig_vecs[:, :num_dim_pca] # Select the eigenvectors for projection

    # Project into lower dimensional space
    if on_derivative:
        x_reduced = x_d @ projection_vecs
    else:
        x_reduced = x @ projection_vecs

    return x_reduced, eig, eig_vecs

# Need a way to get the NamedTuple back
def CreateNamedTuple(
    voltages,
    t_step_size, # Make sure in ms
    num_time_steps
):
    SimulationResults = namedtuple(
        'SimulationResults', 
        [
            'voltages', 'us', 'injected_current',
            'spikes', 'q', 'num_time_steps', 't_step_size',
            't_index'
        ]
    )

    out = SimulationResults(
        voltages = voltages,
        us = None,
        injected_current = None,
        spikes = None,
        q = None,
        num_time_steps = num_time_steps,
        t_step_size = t_step_size,
        t_index = None
    )

    return out

def GenerateNetworkTuple(
    config_dict,
    implement_nueron_modules_positions,
    n_dim,
    radius,
    n,
    n_per_group,
    seed,
    module_std,
    x1_mean,
    x2_mean,
    x3_mean,
    y1_mean,
    y2_mean,
    y3_mean,
    implement_neuron_modules_connections,
    adjacency_matrix_path,
    network_graph_path,
    distance_connection_probability_exponent_constant,
    num_rich_club_neurons_per_module,
    visualize_adjacency_matrix,
    visualize_connected_network,
    constant_connection_probability,
    implement_timing,
    constant_diameter,
    implement_constant_weights,
    constant_weight_value,
    mean_weight_val,
    std_weight_val,
    num_groups = 3,
    check_distance_matrix_symmetry = False
):
    
    # Define the positions array
    pos_array = GeneratePosArray(
            implement_nueron_modules_positions,
            n_dim = n_dim,
            radius = radius,
            n = n,
            num_groups = num_groups,
            n_per_group = n_per_group,
            seed = seed,
            std = module_std,
            x1_mean = x1_mean, 
            x2_mean = x2_mean,
            x3_mean = x3_mean,
            y1_mean = y1_mean,
            y2_mean = y2_mean,
            y3_mean = y3_mean, 
    )

    # Re-introduce n into the scope in case modules have been created
    n = pos_array.shape[0]

    # Calculate the euclidean distance between all nodes
    distance_matrix = GetDistanceMatrix(
        pos_array = pos_array,
        n_dim = n_dim,
        check_symmetry = check_distance_matrix_symmetry
    )

    # Define the adjacency matrix and edges of the network
    A, edges, rich_club_neurons = GetAdjacencyMatrix(
        implement_neuron_modules_connections = implement_neuron_modules_connections,
        distance_matrix = distance_matrix,
        pos_array = pos_array,
        radius = radius,
        n_dim = n_dim,
        adjacency_matrix_path = adjacency_matrix_path,
        network_graph_path = network_graph_path,
        n_per_group = n_per_group,
        num_groups = num_groups,
        distance_connection_probability_exponent_constant = distance_connection_probability_exponent_constant,
        num_rich_club_neurons_per_module = num_rich_club_neurons_per_module,
        visualize_adjacency_matrix = visualize_adjacency_matrix,
        visualize_connected_network = visualize_connected_network,
        constant_connection_probability = constant_connection_probability
    )

    if implement_timing:
        # Find axons that should be myelinated if constant diameter
        myelinated_index = np.zeros(edges.shape[0])

        # Myelinated the signals between modules
        if implement_neuron_modules_connections:
            mod_1_to_2 = np.argwhere(np.logical_and(edges[:,0] == 0, edges[:, 1] == n_per_group)).flatten()
            mod_2_to_3 = np.argwhere(np.logical_and(edges[:,0] == n_per_group, edges[:, 1] == n_per_group*2)).flatten()
            mode_3_to_2 = np.argwhere(np.logical_and(edges[:,0] == n_per_group*3, edges[:, 1] == 0)).flatten()
    
            myelinated_index[mod_1_to_2] = 1
            myelinated_index[mod_2_to_3] = 1
            myelinated_index[mode_3_to_2] = 1
        
        time_to_complete_ap = PosArray_to_SignalTiming(
            pos_array, edges, config_dict, 
            constant_diameter, myelinated_index
        )
    else:
        time_to_complete_ap = None
    
    if implement_constant_weights:
        weight_adjustement = np.ones((n,n)) * constant_weight_value
    else:
        weight_adjustement = np.random.normal(loc = mean_weight_val, scale = std_weight_val, size = (n,n))

    # Create a named tuple for the network itself
    NetworkTuple = namedtuple('NetworkTuple', ['pos_array', 'edges', 'A', 'time_to_complete_ap'])
    
    return NetworkTuple(pos_array = pos_array, edges = edges, A = A, time_to_complete_ap = time_to_complete_ap)

    

    
    
