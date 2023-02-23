import plotly.graph_objects as go
import ipywidgets as widgets
from DRD.calculations import calculations
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import networkx as nx

def interactive_plot(embedding, color_points, coeffs, labels, h_connectivity_matrix, l_connectivity_matrix):
    '''
    This function creates an interactive plot for visualizing high-dimensional embeddings.

        Parameters:
            embedding (numpy.ndarray): A 2D array containing the low-dimensional embeddings.
            color_points (bool): Whether to color the data points based on the distortion coefficients.
            coeffs (numpy.ndarray): An array containing the distortion coefficients.
            labels (list): A list containing the labels of the data points.
            h_connectivity_matrix (numpy.ndarray): A 2D array containing the high-dimensional connectivity matrix.
            l_connectivity_matrix (numpy.ndarray): A 2D array containing the low-dimensional connectivity matrix.

        Returns:
            vbox (ipywidgets.VBox): A vertical box containing the interactive plot and output widget.
    '''
    f = go.FigureWidget([
        go.Scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            mode="markers", 
            marker=dict(
                colorbar=dict(
                    title="Distortion quotient",
                    orientation='h'
                ),
                showscale=True, 
                color=coeffs
            ), 
            
            hovertext = list(f'ID: {i}' for i in labels),
            name="Data point"
        ),
        go.Scatter(
            x=[],
            y=[], 
            mode="lines", 
            name="Lower Dimension"),
        go.Scatter(
            x=[],
            y=[],
            mode="lines", 
            line = dict(dash='dash'), 
            name="Higher Dimension")
    ])
    f.layout.hovermode = 'closest'
    ld_line = f.data[1]
    hd_line = f.data[2]
    
    global touch
    global pnts
    touch = 0
    pnts = []
    scatter = f.data[0]
    if color_points:
        scatter.marker.color = coeffs
    else:
        colors = ['#a3a7e4'] * 100
        scatter.marker.color = colors
    
    f.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
    )
    out = widgets.Output(layout={'border': '1px solid black'})

    # create our callback function
    @out.capture()
    def update_point(trace, points, selector):
        global touch
        global pnts
        if touch == 2:
            pnts = []
            touch = 0
            hd_line.update(x=[], y=[])
            ld_line.update(x=[], y=[])
        pnts.append(points.point_inds)
        touch += 1
        if touch == 2: 
            ld_path = calculations.build_path(*pnts[0], *pnts[1], l_connectivity_matrix, embedding)
            hd_path = calculations.build_path(*pnts[0], *pnts[1], h_connectivity_matrix, embedding)
            hd_line.update(x=hd_path[:, 0], y=hd_path[:, 1])
            ld_line.update(x=ld_path[:, 0], y=ld_path[:, 1])

    scatter.on_click(update_point)
    return widgets.VBox([widgets.VBox([f, out])])

def distortion_gradient(points, values, bounds=None, method='linear'):
    """
    Builds distortion gradient using griddata. 

        Parameters:
                points (numpy.ndarray): datapoints in 2D  
                values (numpy.ndarray): values of function in datapoints 
                bounds (tuple): bounds of gradient
                method: method of interpolation 

        Returns:
                k (int): Lookup in the description
    """
    plt.title('Distortion map')
    if bounds is None:
        bounds = calculations.infer_bounds(points)
    data = griddata(points, values, calculations.grid(bounds), method=method)
    plt.title('Distortion map')
    plt.imshow(data, origin='lower', extent=bounds, interpolation='spline16')

def distortion_gradient(points, values, bounds=None, method='linear'):
    """
    Calculates and plots the distortion gradient for a set of 2D datapoints.

        Parameters:
            points (numpy.ndarray): An array of 2D datapoints.
            values (numpy.ndarray): An array of function values at each datapoint.
            bounds (tuple): A tuple of the form (xmin, xmax, ymin, ymax) defining the bounds of the gradient. If None, the bounds are inferred from the input datapoints.
            method (str): The method of interpolation to use. Default is 'linear'.

        Returns:
            None
    """
    plt.title('Distortion map')
    if bounds is None:
        bounds = calculations.infer_bounds(points)
    data = griddata(points, values, calculations.grid(bounds), method=method)
    plt.title('Distortion map')
    plt.imshow(data, origin='lower', extent=bounds, interpolation='spline16')

def plot_distortion(embedding, coeffs, points = True, gradient = True, color_points = False):
    """
    Plots the distortion of a set of datapoints on a 2D plane.

        Parameters:
            embedding (numpy.ndarray): An array of datapoints with 2D coordinates.
            coeffs (numpy.ndarray): An array of distortion coefficients for each datapoint.
            points (bool), (optional): A boolean indicating whether to plot the datapoints.
            gradient (bool), (optional): A boolean indicating whether to plot the distortion gradient.
            color_points (bool), (optional): A boolean indicating whether to color the datapoints by their distortion coefficient.

        Returns:
            None
    """
    plt.figure(figsize=(30, 10))
    plt.title('Distortion')
    plt.xlabel('x')
    plt.ylabel('y')

    if points:
        if color_points:
            plt.scatter(embedding[:, 0], embedding[:, 1], c = coeffs, s=100, alpha=0.7)
        else:
            plt.scatter(embedding[:, 0], embedding[:, 1], c="black", marker='x')

    if gradient:

        distortion_gradient(embedding, coeffs, method='linear')
        
    clb = plt.colorbar()
    clb.ax.set_title('Distortion quotient')
    plt.show() 

def plot_graph(embedding, graph, distances_dict, color, alpha):
    """
    Plots a graph using a force-directed layout algorithm.

        Parameters:
            embedding (dict): A dictionary mapping each node to its position in 2D space.
            graph (networkx.Graph): A networkx Graph object representing the graph to be plotted.
            distances_dict (dict): A dictionary of dictionaries representing the distances between nodes.
                The keys of the outer dictionary should be the nodes of the graph, and the values should be
                inner dictionaries with keys that correspond to the nodes of the graph and values that
                correspond to the distances between them.
            color (str): A string representing the color to use for the edges of the graph.
            alpha (float): A float between 0 and 1 representing the opacity of the edges.

        Returns:
            None
    """
    distances_list = list(distances_dict[0].values())
    nx.draw_networkx_nodes(graph, embedding, node_size = 8, node_color="black")
    nx.draw_networkx_edges(graph, embedding, width=distances_list, edge_color=color, alpha=alpha)

def draw_graph_comparison(coords, graphs):
    """
    Draws a comparison of multiple graphs on the same coordinate system.

        Parameters:
            coords (numpy.ndarray): An array of coordinates for the graph.
            graphs (list): A list of dictionaries, where each dictionary describes a graph.
                Each dictionary should have the following keys:
                - "graph": An array of values for the graph.
                - "dists": A 1D array of distances from the origin for each value in the graph.
                - "color": A string representing the color of the graph.
                - "alpha": A float between 0 and 1 representing the opacity of the graph.

        Returns:
            None
    """
    for i in graphs:
        plot_graph(coords, i["graph"], i["dists"], i["color"], i["alpha"])