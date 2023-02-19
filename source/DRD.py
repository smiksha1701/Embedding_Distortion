#all imports for class
import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import networkx as nx

class DRD():
    def __init__(self, *argv):
        if len(argv)==3:
            self.X = argv[0]
            self.X_low = argv[1]
            self.k = argv[2]
    
    @staticmethod
    def grid(bounds, num_points=100):
        xs = np.linspace(bounds[0], bounds[1], num=num_points)
        ys = np.linspace(bounds[2], bounds[3], num=num_points)
        return tuple(np.meshgrid(xs, ys))
    
    @staticmethod
    def infer_bounds(points):
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        return mins[0], maxs[0], mins[1], maxs[1]
    
    @staticmethod
    def build_path(first, last, connectivity_m, X):
        """
        Builds path (list of coordinates of nodes) from one node to another by connectivity matrix.  

            Parameters:
                    dists (dict[Any, defaultdict[Any, float]): dictionary of distances between points 

            Returns:
                    path (numpy.ndarray): list of coordinates 
        """
        path = np.array([X[first]])
        head = first
        while head!=last:
            head = connectivity_m[last][head]
            path = np.concatenate((path, [X[head]]))
        return path
    
    @staticmethod
    def distance_matrix(dists):
        """
        Makes distance matrix out of distance dictionary.  

            Parameters:
                    dists: dictionary of distances between points (dict[Any, defaultdict[Any, float])

            Returns:
                    (numpy.ndarray): distortion coefficients
        """
        n = len(dists)
        return np.array([[dists[i][j] for j in range(n)] for i in range(n)])
    
    @staticmethod
    def find_min_k( X: np.ndarray):
        """
        Brut Force algorithm for searhing k  - smallest number of neareast datapoints needed 
        to build graph without independent parts.

            Parameters:
                    X (numpy.ndarray): dataset for which k is needed to be found. Shape(N, M).

            Returns:
                    k (int): Lookup in the description
        """
        def check(k):
            g = nx.Graph(kneighbors_graph(X, n_neighbors=k))
            return nx.is_connected(g)

        for k in range(1, X.shape[0]):
            if check(k):
                return k

    @staticmethod
    def shortest_paths(X: np.ndarray):
        """
        Calculates shortest paths between every pair of graph. 

            Parameters:
                    X: datapoints  (numpy.ndarray)

            Returns:
                    (numpy.ndarray): distortion coefficients
        """
        k = DRD.find_min_k(X)
        print(f'Min K found: {k}')
        graph = nx.Graph(kneighbors_graph(X, n_neighbors=k, mode='distance'))
        return graph, *nx.floyd_warshall_predecessor_and_distance(graph)

    @staticmethod
    
    def distortion_coef(X):
        """
        Calculates distortion coefficients of datapoints. 

            Parameters:
                    X: datapoints in 2D  (numpy.ndarray)

            Returns:
                    (numpy.ndarray): distortion coefficients
        """
        return X.mean(axis=0)

    @staticmethod
    def distortion_gradient(points, values, bounds=None, method='linear'):
        """
        Builds distortion gradient using griddata. 

            Parameters:
                    points: datapoints in 2D  (numpy.ndarray)
                    values: values of function in datapoints (numpy.ndarray)
                    bounds: bounds of gradient
                    method: method of interpolation 

            Returns:
                    k (int): Lookup in the description
        """
        plt.title('Distortion map')
        if bounds is None:
            bounds = DRD.infer_bounds(points)
        data = griddata(points, values, DRD.grid(bounds), method=method)
        plt.title('Distortion map')
        plt.imshow(data, origin='lower', extent=bounds, interpolation='spline16')

    def ReduceDimensionalityPCA(self, *argv) -> np.ndarray:
        """
        Returns transformed values of input data. PCA(Principal component analysis) with number of components to keep = 2 is used.

            Parameters:
                    X (numpy.ndarray): dataset to be transform. Shape(N, M).

            Returns:
                    self.X_low (numpy.ndarray): transformed values of input dataset. Shape(N, 2).
        """
        if argv:
            self.X = argv[0]
        self.X_low = PCA(n_components=2).fit_transform(self.X)
        return self.X_low
    
    def ReduceDimensionalitySVD(self, *argv) -> np.ndarray:
        """
        Returns transformed values of input data. PCA(Principal component analysis) with number of components to keep = 2 is used.

            Parameters:
                    X (numpy.ndarray): dataset to be transform. Shape(N, M).

            Returns:
                    self.X_low (numpy.ndarray): transformed values of input dataset. Shape(N, 2).
        """
        if argv:
            self.X = argv[0]
        self.X_low = TruncatedSVD(n_components=2).fit_transform(self.X)
        return self.X_low
    
    

    def distortion_matrix(self, dists_h, dists_l):
        mh = DRD.distance_matrix(dists_h)
        ml = DRD.distance_matrix(dists_l)
        ml[ml == 0] = np.nextafter(0, 1)
        return mh / ml

    def set_matrix(self, X, indexes):
        self.X = X
        self.indexes = indexes
        return self.X

    def run_experiment(self, *argv):
        if argv:
            self.X = argv[0]
            self.X_low = argv[1]
        print(f'Original shape: {self.X.shape}')
        self.graph_h, self.preds_h, self.dists_h = self.shortest_paths(self.X)

        print(f'Lowered shape: {self.X_low.shape}')
        self.graph_l, self.preds_l, self.dists_l = self.shortest_paths(self.X_low)

        m = self.distortion_matrix(self.dists_h, self.dists_l)
        self.coeffs = DRD.distortion_coef(m)

    def plot_graph(self):
        nx.draw_networkx_nodes(self.graph_l, self.X_low, node_size = 8, node_color="black")
        nx.draw_networkx_edges(self.graph_l, self.X_low,
                       width=self.dists_l,
                       edge_color='yellow',
                       alpha=0.6)
        nx.draw_networkx_edges(self.graph_h, self.X_low,
                       width=self.dists_h,
                       edge_color='red',
                       alpha=0.3)

    def plot_distortion(self, points = True, gradient = True, color_points = False):
        plt.figure(figsize=(30, 10))
        plt.title('Distortion')
        plt.xlabel('x')
        plt.ylabel('y')

        if points:
            if color_points:
                plt.scatter(self.X_low[:, 0], self.X_low[:, 1], c = self.coeffs, s=100, alpha=0.7)
            else:
                plt.scatter(self.X_low[:, 0], self.X_low[:, 1], c="black", marker='x')

        if gradient:
            DRD.distortion_gradient(self.X_low, self.coeffs, method='linear')
            
        clb = plt.colorbar()
        clb.ax.set_title('Distortion quotient')
        plt.show() 
    
    
    def interactive_plot(self, *argv):
        color_points = argv[0]
        f = go.FigureWidget([
            go.Scatter(
                x=self.X_low[:, 0],
                y=self.X_low[:, 1],
                mode="markers", 
                marker=dict(
                    colorbar=dict(
                        title="Distortion quotient",
                        orientation='h'
                    ),
                    showscale=True, 
                    color=self.coeffs
                ), 
                
                hovertext = list(f'ID: {i}' for i in self.indexes),
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
        self.touch = 0
        self.pnts = []
        scatter = f.data[0]
        if color_points:
            scatter.marker.color = self.coeffs
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
            touch = self.touch
            if touch == 2:
                self.pnts = []
                touch = 0
                hd_line.update(x=[], y=[])
                ld_line.update(x=[], y=[])
            self.pnts.append(points.point_inds)
            touch += 1
            if touch == 2: 
                ld_path = DRD.build_path(*self.pnts[0], *self.pnts[1], self.preds_l, self.X_low)
                hd_path = DRD.build_path(*self.pnts[0], *self.pnts[1], self.preds_h, self.X_low)
                hd_line.update(x=hd_path[:, 0], y=hd_path[:, 1])
                ld_line.update(x=ld_path[:, 0], y=ld_path[:, 1])
            self.touch = touch

        scatter.on_click(update_point)
        return widgets.VBox([widgets.VBox([f, out])])

        
    

        