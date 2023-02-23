import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import kneighbors_graph
import networkx as nx

def grid(bounds, num_points=100):
    """
    Generates a 2D grid of evenly spaced points within the given bounds.

        Parameters:
            bounds (tuple): A tuple of four floats (xmin, xmax, ymin, ymax) that represent the bounds of the rectangular area to generate a grid for.
            num_points (int), (optional): The number of points in each dimension of the grid. The default is 100.

        Returns:
            (tuple): A tuple of two 2D arrays representing the x-coordinates and y-coordinates of the grid points, respectively.
    """
    xs = np.linspace(bounds[0], bounds[1], num=num_points)
    ys = np.linspace(bounds[2], bounds[3], num=num_points)
    return tuple(np.meshgrid(xs, ys))

def infer_bounds(points):
    """
    Infer the bounds for a set of points in two dimensions.

        Parameters:
            points (numpy.ndarray): A two-dimensional numpy array of shape (n, 2)
                containing the points for which to infer the bounds.

        Returns:
            tuple: A tuple containing four elements: the minimum x-value, maximum x-value,
                minimum y-value, and maximum y-value of the points.
    """
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return mins[0], maxs[0], mins[1], maxs[1]

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

def shortest_paths(X: np.ndarray):
    """
    Calculates shortest paths between every pair of graph. 

        Parameters:
                X: datapoints  (numpy.ndarray)

        Returns:
                (numpy.ndarray): distortion coefficients
    """
    k = find_min_k(X)
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


def ReduceDimensionalityPCA(X) -> np.ndarray:
    """
    Returns transformed values of input data. PCA(Principal component analysis) with number of components to keep = 2 is used.

        Parameters:
                X (numpy.ndarray): dataset to be transform. Shape(N, M).

        Returns:
                self.X_low (numpy.ndarray): transformed values of input dataset. Shape(N, 2).
    """
    X_low = PCA(n_components=2).fit_transform(X)
    return X_low

def ReduceDimensionalitySVD(X) -> np.ndarray:
    """
    Returns transformed values of input data. PCA(Principal component analysis) with number of components to keep = 2 is used.

        Parameters:
                X (numpy.ndarray): dataset to be transform. Shape(N, M).

        Returns:
                self.X_low (numpy.ndarray): transformed values of input dataset. Shape(N, 2).
    """
    X_low = TruncatedSVD(n_components=2).fit_transform(X)
    return X_low



def distortion_matrix(dists_h, dists_l):
    """
    Calculate the distortion matrix given the pairwise distances in the high and low dimensions.

        Parameters:
            dists_h (numpy.ndarray): The pairwise distances of data points in the high-dimensional space.
            dists_l (numpy.ndarray): The pairwise distances of data points in the low-dimensional space.

        Returns:
            numpy.ndarray: The distortion matrix calculated as the element-wise division of the high-dimensional distance matrix 
                by the low-dimensional distance matrix. If an element in the low-dimensional distance matrix is zero, it is replaced by 
                the next smallest representable positive number before the division.
    """
    mh = distance_matrix(dists_h)
    ml = distance_matrix(dists_l)
    ml[ml == 0] = np.nextafter(0, 1)
    return mh / ml


def calc_dist_coeff(dists_h, dists_l):
    """
    Calculates the distortion coefficients for a given set of high-dimensional and low-dimensional distances.

        Parameters:
            dists_h (ndarray): A 1D array of high-dimensional distances.
            dists_l (ndarray): A 1D array of low-dimensional distances.

        Returns:
            ndarray: A 1D array of distortion coefficients.
    """
    m = distortion_matrix(dists_h, dists_l)
    coeffs = distortion_coef(m)
    return coeffs





