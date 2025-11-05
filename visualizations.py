import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

def ShowNetwork(
    pos_array,
    edges,
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
    yaxis_title= 'Y'
):
    '''
    Code to visualize the network. Adapted from my own research project.
    '''

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
                    size=node_size, color= e
                ),
                showlegend = False
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
                    size=node_size, color= e
                ),
                showlegend = False,
                text = f'Node Index: {k}',
                hoverinfo = text,
            ) for k, e in enumerate(node_colors) 
        ]

    # Commenting this out for now
    # legend_data = [
    #         dict(
    #             type = 'scatter3d',
    #             x = [None],
    #             y = [None],
    #             z = [None],
    #             mode = 'markers',
    #             marker = dict(size = 10, color = legend_colors_dict[k]),
    #             name = k,
    #             showlegend = True,
    #         ) for k in legend_colors_dict
    # ]

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

    plot_data = nodes + edge_trace

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
    include_node_indices = False
):

    if flip_axes == False:
        num_rows = out.voltages.shape[0]
        y_step = int(num_rows / num_y_ticks)
        y_labels = [f'{int(x)}' for x in (np.arange(num_rows + y_step, step = y_step) * (out.t_step_size))]
        
        ax = sns.heatmap(out.voltages, cmap = cmap, center = cmap_center)
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
        
        ax = sns.heatmap(out.voltages.T, cmap = cmap, center = cmap_center)
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

def PlotSpikeTrace(
    out,
    cmap = 'grey_r',
    cmap_center = 0.5,
    num_y_ticks = 10,
    save_graphic_path = None,
    flip_axes = False,
    include_node_indices = False
):

    if flip_axes == False:
        num_rows = out.spikes.shape[0]
        y_step = int(num_rows / num_y_ticks)
        y_labels = [f'{int(x)}' for x in (np.arange(num_rows + y_step, step = y_step) * (out.t_step_size))]
        
        ax = sns.heatmap(out.spikes, cmap = cmap, center = cmap_center)
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
        
        ax = sns.heatmap(out.spikes.T, cmap = cmap, center = cmap_center)
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
