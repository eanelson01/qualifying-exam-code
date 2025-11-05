# Imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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