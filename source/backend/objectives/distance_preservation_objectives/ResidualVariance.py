import numpy
import scipy
import sklearn
from .DistancePreservationObjective import DistancePreservationObjective
import networkx


class ResidualVariance(DistancePreservationObjective):
    """
    Calculates residual variance.
    """
    def __init__(
            self,
            low_dimensional_data: numpy.ndarray,
            high_dimensional_data: numpy.ndarray,
            distance_metric: str = 'euclidean',
            use_geodesic_distances: bool = False
    ):
        """
        Initiates new instance of residual variance objective.
        :param low_dimensional_data:
        :param high_dimensional_data:
        :param distance_metric:
        :param use_geodesic_distances:
        """
        super().__init__(
            low_dimensional_data=low_dimensional_data,
            high_dimensional_data=high_dimensional_data,
            distance_metric=distance_metric,
            use_geodesic_distances=use_geodesic_distances
        )

    def compute(self):
        """
         This function allows to compute the residual variance as proposed in
         https://www.researchgate.net/publication/12204039_A_Global_Geometric_Framework_for_Nonlinear_Dimensionality_Reduction

         Args:
             + self._low_dimensional_data: the point cloud in state space
             + self._target_data: the point cloud in latent space
             + use_geodesic: Whether to use geodesic distance for state space
         """

        # We retrieve dimensions of the data
        n, m = self._low_dimensional_data.shape
        #  We compute distance matrices in both spaces
        if self._use_geodesic_distances:
            k = 2
            is_connex = False
            
            while is_connex is False:
                knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k)
                knn.fit(self._low_dimensional_data)
                M = knn.kneighbors_graph(self._low_dimensional_data, mode='distance')
                graph = networkx.from_scipy_sparse_matrix(M)
                is_connex = networkx.is_connected(graph)
                k += 1

            s_uni_distances = networkx.all_pairs_dijkstra_path_length(graph, cutoff=None, weight='weight')
            s_all_distances = numpy.array([numpy.array(a.items())[:, 1] for a in numpy.array(s_uni_distances.items())[:, 1]])
            s_all_distances = (s_all_distances + s_all_distances.T) / 2
            s_all_distances = s_all_distances.ravel()
        else:
            s_uni_distances = scipy.spatial.distance.pdist(self._low_dimensional_data)
            s_all_distances = scipy.spatial.distance.squareform(s_uni_distances).ravel()

        l_uni_distances = scipy.spatial.distance.pdist(self._target_data)
        l_all_distances = scipy.spatial.distance.squareform(l_uni_distances).ravel()

        # We compute the residual variance
        measure = sklearn.metrics.r2_score(s_all_distances, l_all_distances)

        return measure
