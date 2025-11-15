# Purpose: A Python file to implmement the simulation with different permutations of the included featues. Also differences in the hyperparamters.
# Author: Ethan Nelson

# General Imports
import numpy as np
import pandas as pd
from collections import namedtuple
import time

# Own imports from utils file
from utils import *
from visualizations import *

### Values that don't change
# Load in the configuration dictionary
config_dict = LoadYAMLFile('config-file.yaml')

num_groups = 3

# Think about invidual tests with
array_id = 1

# Reproducibility
seed = config_dict['general']['seed']

### Values that can change
# Neurons and Their Physical Environment
n = config_dict['general']['n'] # Number of neurons, only matters if not dealing with modules
n_dim = config_dict['general']['n_dim'] # Number of dimensions for the physical space that the neurons inhabit
radius = config_dict['general']['radius'] # Width of the physical space that the neurons can inhabit in each dimension. Equidistant in all dimensions. 

# Modules of neurons
implement_nueron_modules_positions = True # Do you want to implement modules? 
n_per_group = config_dict['general']['n_per_group'] # Number of neurons in each module if that exists
module_std = config_dict['general']['module_std'] # The standard deviation of the node positions within a module
num_rich_club_neurons_per_module = config_dict['general']['num_rich_club_neurons_per_module']

x1_mean = config_dict['general']['x1_mean'] # Center in x dimension for 3 modules
x2_mean = config_dict['general']['x2_mean']
x3_mean = config_dict['general']['x3_mean']
y1_mean = config_dict['general']['y1_mean'] # Center in y dimension for 3 modules
y2_mean = config_dict['general']['y2_mean']
y3_mean = config_dict['general']['y3_mean']

# Connection Types
distance_connection_probability_exponent_constant = config_dict['general']['distance_connection_probability_exponent_constant']
constant_weight_value = config_dict['general']['constant_weight_value']


### Inputs to the Network ###
number_of_rich_club_neurons_to_stimulate = 1 # The number of rich club neurons to stimulate
q = config_dict['general']['q'] # The frequency of random spiking (seconds)


### Timing Parameters ###
num_time_steps = config_dict['timing']['num_time_steps']
t_step_size = float(config_dict['timing']['t_step_size'])


### Inidivudal neuron dynamics ###  
resting_voltage = config_dict['izhikevich']['rmv'] # Resting membrane potential to start
default_a = config_dict['izhikevich']['default_a'] # The default Izhikevich A value
default_b = config_dict['izhikevich']['default_b'] # The default Izhikevich B value
default_c = config_dict['izhikevich']['default_c'] # The default Izhikevich C value
default_d = config_dict['izhikevich']['default_d'] # The default Izhikevich D value
highest_injected_current = config_dict['izhikevich']['highest_injected_current'] # The amount of injected current
injected_current_decay_parameter = config_dict['general']['injected_current_decay_parameter'] # How the injected current is decaying

### Configuration Parameters ###
implement_neuron_modules_connections = True
include_constant_injected_current_to_rich_club = True
check_distance_matrix_symmetry = False
save_outputs = True
implement_timing = True
implement_constant_weights = True
vary_neuron_type = False

### Booleans for visualizations ###
visualize_nodal_positions_without_edges = False 
visualize_adjacency_matrix = True
visualize_connected_network = True
diagnostic_printing = False

### Image Paths ###
base_path = 'figures/'
adjacency_matrix_path = base_path + f'adjacency-matrix-heatmap-array-{array_id}.png'
network_graph_path = base_path + f'network-graph-{array_id}.png'

# Catching any last failures
if not distance_connection_probability_exponent_constant:
    distance_connection_probability_exponent_constant = None # Set to none if you want to use the default value depending on the result

# Try a simluation
start_time = time.time()
FullSimulation(
    config_dict = config_dict,
    array_id = array_id,
    seed = seed,
    n = n,
    n_per_group = n_per_group,
    n_dim = n_dim,
    radius = radius,
    implement_nueron_modules_positions = implement_nueron_modules_positions,
    implement_neuron_modules_connections = implement_neuron_modules_connections,
    module_std = module_std,
    distance_connection_probability_exponent_constant = distance_connection_probability_exponent_constant,
    num_rich_club_neurons_per_module = num_rich_club_neurons_per_module,
    q = q,
    injected_current_decay_parameter = injected_current_decay_parameter,
    include_constant_injected_current_to_rich_club = include_constant_injected_current_to_rich_club,
    x1_mean = x1_mean, 
    x2_mean = x2_mean,
    x3_mean = x3_mean,
    y1_mean = y1_mean,
    y2_mean = y2_mean,
    y3_mean = y3_mean,
    check_distance_matrix_symmetry = check_distance_matrix_symmetry,
    visualize_adjacency_matrix= visualize_adjacency_matrix,
    adjacency_matrix_path = adjacency_matrix_path,
    visualize_connected_network= visualize_connected_network,
    network_graph_path = network_graph_path,
    num_time_steps = num_time_steps,
    t_step_size = t_step_size,
    save_outputs = save_outputs,
    implement_timing = implement_timing,
    implement_constant_weights = implement_constant_weights,
    vary_neuron_type = vary_neuron_type,
    diagnostic_printing = diagnostic_printing,
    resting_voltage = resting_voltage,
    default_a = default_a,
    default_b = default_b,
    default_c = default_c,
    default_d = default_d,
    highest_injected_current = highest_injected_current,
    num_groups = num_groups,
    number_of_rich_club_neurons_to_stimulate = number_of_rich_club_neurons_to_stimulate,
    constant_weight_value = constant_weight_value
)

print(f'Success! Execution time: {time.time() - start_time:.2f} (s)')
    

    
    