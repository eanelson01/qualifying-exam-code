import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as c
from matplotlib.collections import LineCollection

def ShowNetwork(
    pos_array,
    edges,
    A,
    axis_limit,
    showticklabels = False,
    background_color = 'white',
    gridcolor='#FFFFFF',
    gridwidth=1,
    node_size = 4,
    title_font_color = 'black',
    font_size = 14,
    edge_width_constant = 2,
    node_colors = None,
    edge_colors = None,
    edge_widths = None,
    title = '',
    z_axis_lim_if_2d = 0.1,
    title_x = 0.5,
    xaxis_title= 'X',
    yaxis_title= 'Y',
    node_size_floor = 1,
    degree_based_size_scalar = 1
):
    '''
    Code to visualize the network. Adapted from my own research project.
    '''
    # Get the degrees of each neuron
    total_degree = np.sum(A, axis = 0) + np.sum(A, axis = 1)
    node_sizes = node_size_floor + (total_degree * degree_based_size_scalar)
    
    n_dim = pos_array.shape[1]
    if n_dim != 3:
        # Add a column of zeros to the positions array
        pos_array = np.column_stack([pos_array, np.zeros(pos_array.shape[0]).reshape(-1, 1)])

    if node_colors is None:
        node_colors = ['#1E90FF'] * pos_array.shape[0]

    if n_dim == 2:
        nodes=[
            dict(
                type='scatter',
                x=[pos_array[k][0]],
                y=[pos_array[k][1]],
                mode='markers',
                marker=dict(
                    size=node_sizes[k], color= e
                ),
                showlegend = False,
                text = f'Node Index: {k} \n Node Degree: {total_degree[k]}',
                hoverinfo = 'text'
            ) for k, e in enumerate(node_colors) 
        ]

    else:
        
        nodes=[
            dict(
                type='scatter3d',
                x=[pos_array[k][0]],
                y=[pos_array[k][1]],
                z=[pos_array[k][2]],
                mode='markers',
                marker=dict(
                    size=node_sizes[k], color= e
                ),
                showlegend = False,
                text = f'Node Index: {k}',
                hoverinfo = text,
            ) for k, e in enumerate(node_colors) 
        ]

    # Getting the edges
    xe, ye, ze = GetEdgePlotVectors(pos_array, edges)

    if edge_colors is None:
        edge_colors = ['grey'] * edges.shape[0]
    if edge_widths is None:
        edge_widths = [edge_width_constant] * edges.shape[0]

    if n_dim == 2:
        edge_trace = [
            dict(
                type='scatter',
                x=xe[(k*3):((k*3)+3)],
                y=ye[(k*3):((k*3)+3)],
                mode='lines',
                line=dict(color= e, width = edge_widths[k]),
                showlegend = False
            ) for k, e in enumerate(edge_colors)
        ]

    else:
        edge_trace = [
            dict(
                type='scatter3d',
                x=xe[(k*3):((k*3)+3)],
                y=ye[(k*3):((k*3)+3)],
                z=ze[(k*3):((k*3)+3)],
                mode='lines',
                line=dict(color= e, width = edge_widths[k]),
                showlegend = False
            ) for k, e in enumerate(edge_colors)
        ]

    plot_data = edge_trace + nodes
    
    axis = dict(
        showbackground=True,
        showline=True,
        zeroline=True,
        showgrid=True,
        showticklabels=showticklabels,
        title='',
        backgroundcolor = background_color,
        gridcolor = gridcolor,
        gridwidth = gridwidth,
        range=[-axis_limit, axis_limit]
    )

    if n_dim == 2:
        scene = dict(
            xaxis=dict(axis),
            yaxis=dict(axis)
        )
    else:
        scene = dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(zaxis)
        )
    
    layout = go.Layout(
        title={
            'text': title,
            'x': title_x
        },
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width=1000,
        height=1000,
        showlegend=False,
        scene=scene,
         margin=dict(
            t=100
        ),
        hovermode='closest',
        font = {
            'family': 'Arial',
            'color': title_font_color,
            'size': font_size
        }
    )

    fig = go.Figure(
        data = plot_data,
        layout = layout
    )

    if n_dim == 2:
        fig.update_layout(
            plot_bgcolor = 'white',
        )

    return fig

def GetEdgePlotVectors(pos_array, edges):
    '''
    
    A funciton to get just the edges for plotly graphing.
    
    '''
    xe = []
    ye = []
    ze = []
    for i in edges:
        xe += [pos_array[i[0], 0], pos_array[i[1], 0], None]
        ye += [pos_array[i[0], 1], pos_array[i[1], 1], None]
        ze += [pos_array[i[0], 2], pos_array[i[1], 2], None]
        
    return xe, ye, ze

def PlotVoltageTrace(
    out,
    cmap = 'bwr',
    cmap_center = 0.5,
    num_y_ticks = 10,
    save_graphic_path = None,
    flip_axes = False,
    include_node_indices = False,
    vmin = -90, vmax = 30
):

    if flip_axes == False:
        num_rows = out.voltages.shape[0]
        y_step = int(num_rows / num_y_ticks)
        y_labels = [f'{int(x)}' for x in (np.arange(num_rows + y_step, step = y_step) * (out.t_step_size))]
        
        ax = sns.heatmap(out.voltages, cmap = cmap, center = cmap_center, vmin = vmin, vmax = vmax)
        ax.set_xlabel('Node Index')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Voltage Activity of the Network')
        ax.set_yticks(np.arange(num_rows + y_step, step = y_step), labels = y_labels)
        if include_node_indices == False:
            ax.set_xticks([])
        ax.collections[0].colorbar.set_label('Voltage (mV)')

    else:
        num_rows = out.voltages.shape[0]
        y_step = int(num_rows / num_y_ticks)
        y_labels = [f'{int(x)}' for x in (np.arange(num_rows + y_step, step = y_step) * (out.t_step_size))]
        
        ax = sns.heatmap(out.voltages.T, cmap = cmap, center = cmap_center, vmin = vmin, vmax = vmax)
        ax.set_ylabel('Node Index')
        ax.set_xlabel('Time (ms)')
        ax.set_title('Voltage Activity of the Network')
        ax.set_xticks(np.arange(num_rows + y_step, step = y_step), labels = y_labels, rotation = 45)
        if include_node_indices == True:
            ax.set_yticks(np.arange(out.voltages.shape[1]) + 0.5, labels = np.arange(out.voltages.shape[1]))
        else:
            ax.set_yticks([])
        ax.collections[0].colorbar.set_label('Voltage (mV)')

    if save_graphic_path is None:
        plt.show()
    else:
        plt.savefig(save_graphic_path)
        plt.show()

def PlotSpikeTrace(
    out,
    cmap = 'grey_r',
    cmap_center = 0.5,
    num_y_ticks = 10,
    save_graphic_path = None,
    flip_axes = False,
    include_node_indices = False,
    vmin = -90, vmax = 30
):

    if flip_axes == False:
        num_rows = out.spikes.shape[0]
        y_step = int(num_rows / num_y_ticks)
        y_labels = [f'{int(x)}' for x in (np.arange(num_rows + y_step, step = y_step) * (out.t_step_size))]
        
        ax = sns.heatmap(out.spikes, cmap = cmap, center = cmap_center, vmin = vmin, vmax = vmax)
        ax.set_xlabel('Node Index')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Spiking Activity of the Network')
        ax.set_yticks(np.arange(num_rows + y_step, step = y_step), labels = y_labels)
        if include_node_indices == False:
            ax.set_xticks([])
        ax.collections[0].colorbar.set_label('Spike Indicator')

    else:
        num_rows = out.spikes.shape[0]
        y_step = int(num_rows / num_y_ticks)
        y_labels = [f'{int(x)}' for x in (np.arange(num_rows + y_step, step = y_step) * (out.t_step_size))]
        
        ax = sns.heatmap(out.spikes.T, cmap = cmap, center = cmap_center, vmin = vmin, vmax = vmax)
        ax.set_ylabel('Node Index')
        ax.set_xlabel('Time (ms)')
        ax.set_title('Spiking Activity of the Network')
        ax.set_xticks(np.arange(num_rows + y_step, step = y_step), labels = y_labels, rotation = 45)
        if include_node_indices == True:
            ax.set_yticks(np.arange(out.spikes.shape[1]) + 0.5, labels = np.arange(out.spikes.shape[1]))
        else:
            ax.set_yticks([])
        ax.collections[0].colorbar.set_label('Spike Indicator')

    if save_graphic_path is None:
        plt.show()
    else:
        plt.savefig(save_graphic_path)
        plt.show()

def PlotIndividualNodeTrace(
    out,
    node_index,
    color = 'dodgerblue'
):
    '''
    Quick function for plotting an individual node trace over time

    Parameters:
    ----------
        out: Named tuple. The 
        node_index: Int. The node index to plot.

    Returns:
    -------
        Plots the individual node trace.
    '''

    T = out.num_time_steps # The number of time steps
    voltage_trace = out.voltages[:, node_index] # The individual voltage trace
    
    plt.plot(np.arange(T) * out.t_step_size, voltage_trace, color = color)
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.yticks([-70, 0, 30])
    plt.title(f'Voltage Trace for Node Index: {node_index}')
    plt.show()

def PlotExplainedVariance(
    eig,
    markersize = 2,
    linewidth = 1, 
    xlabel = 'Principle Component Index',
    ylabel = 'Explained Variance',
    title = 'Explained Variance for Each Principle Component',
    file_save_path = None
):
    explained_variance = eig / np.sum(eig)
    plt.plot(
        np.arange(1, explained_variance.shape[0] + 1), explained_variance, 
        marker = 'o', color = 'dodgerblue', label = 'Explained Variance',
        markersize = markersize, linewidth = linewidth
    )
    plt.plot(
        np.arange(1, explained_variance.shape[0] + 1), np.cumsum(explained_variance), 
        marker = 'o', linestyle = '--', color = 'Black', label = 'Cumulative Explained Variance',
        markersize = markersize, linewidth = linewidth
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    if file_save_path is None:
        plt.show()
    else:
        plt.savefig(file_save_path)
        plt.show()

def PlotPCA(
    out,
    x_reduced,
    time_tick_interval = 1000,
    cmap = None,
    file_save_path = None,
    xlabel = 'PC1',
    ylabel = 'PC2',
    title = 'PCA of Voltage Time Derivatives'
):

    num_time_steps = out.num_time_steps
    t_step_size = out.t_step_size

    # Color mappings
    if cmap is None:
        cmap = cm.viridis

    cmap_norm = c.Normalize(vmin = 0, vmax = num_time_steps)
    colors = cmap(cmap_norm(np.arange(0, num_time_steps)))

    plt.scatter(x_reduced[:, 0], x_reduced[:, 1], color = colors)
    cbar = plt.colorbar(label = 'Time (ms)')
    cbar.set_ticks(
        np.arange(
            0, 1 + (time_tick_interval / num_time_steps), 
            step = (time_tick_interval / num_time_steps)
        ), 
        labels = np.arange(0, (num_time_steps + time_tick_interval) * (t_step_size), step = time_tick_interval * (t_step_size))
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if file_save_path is None:
        plt.show()
    else:
        plt.savefig(file_save_path)
        plt.show()

def PlotPCATrajectory2D(
    x_reduced,
    num_time_steps,
    t_step_size,
    xlabel = 'PC1',
    ylabel = 'PC2',
    title = 'Low Dimensional Trajectory',
    save_filepath = None,
    fixed_aspect_ratio = True
):
    '''
    A function to plot the trajectory of the system in lower dimension. Will be 2-dimensional
    Made with the help of ChatGPT

    Parameters:
    ----------
        x_reduced: The lower dimensional data.
        num_time_steps: Number of time steps simulated.
        t_step_size: The time step size in seconds.
        
    '''

    t = np.linspace(num_time_steps * t_step_size * 1000, t_step_size * 1000)
    points = x_reduced[:, :2].reshape(-1, 1, 2)

    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a LineCollection, color by time
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(t.min(), t.max()))
    lc.set_array(t)  # Color according to time
    lc.set_linewidth(2)
    
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.add_collection(lc)
    ax.autoscale()

    if fixed_aspect_ratio:
        ax.set_aspect('equal')
    
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label('Time (ms)')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if save_filepath is not None:
        plt.savefig(save_filepath)
        plt.show(renderer="png")
    else:
        plt.show(renderer="png")

# Colors
def PCATrajectoryOverTime(
    x_reduced,
    num_time_steps,
    save_filepath = None
):
    
    t = np.arange(num_time_steps) # * t_step_size * 1000)
    plot_data = go.Scatter3d(
        x=x_reduced[:, 0],
        y=x_reduced[:, 1],
        z=t, # x_reduced[:, 2],
        mode='lines',
        line=dict(
            color=t,         # color represents time
            colorscale='Viridis',
            width=5
        )
    )
    
    axis = dict(
        showbackground=True,
        showline=True,
        zeroline=True,
        showgrid=True,
        showticklabels=True,
        title='',
        backgroundcolor = '#FFFFFF',
        gridcolor = 'black',
        gridwidth = 2,
        # range=[-axis_limit, axis_limit]
    )
    
    scene = dict(
        xaxis=dict(axis),
        yaxis=dict(axis),
        zaxis=dict(axis)
    )
    
    layout = go.Layout(
        title={
            'text': 'PCA Trajectory',
            'x': 0.5
        },
        xaxis_title='PC1',
        yaxis_title='PC2',
        width=1000,
        height=1000,
        showlegend=False,
        scene=scene,
         margin=dict(
            t=100
        ),
        hovermode='closest',
        font = {
            'family': 'Arial',
            'color': 'black',
            'size': 12
        }
    )
    
    fig = go.Figure(
        data = plot_data,
        layout = layout
    )

    if save_filepath is not None:
        fig.write_image(save_filepath)
        fig.show(renderer="png")
        # return fig
    else:
        # return fig
        fig.show(renderer="png")
