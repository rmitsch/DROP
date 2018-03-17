import abc
import numpy
from .DimensionalityReductionObjective import DimensionalityReductionObjective
from coranking.metrics import trustworthiness, continuity
from scipy.spatial import distance
import sklearn
import networkx as nx

Next up:
    - Move CorankingMatrix to separate class.
    - Split up embqual in relevant metrics.
    - Integrate metric calculation in data generation script.


class CorankingObjective(DimensionalityReductionObjective):
    """
    Abstract base class for calculation of coranking matrix-based objectives (e. g. trustworthiness, continuity, LCMC,
    ...).
    """

    def __init__(
            self,
            low_dimensional_data: numpy.ndarray,
            high_dimensional_data: numpy.ndarray,
            distance_metric: str = 'euclidean',
            coranking_matrix: numpy.ndarray = None,
            high_dimensional_neighbourhood_ranking: numpy.ndarray = None
    ):
        """
        Initializes coranking-based objectives.
        :param low_dimensional_data: Output of DR algorithm.
        :param high_dimensional_data: The original high-dimensional data set.
        :param coranking_matrix:
        :param distance_metric: Metric to use for calculation of coranking matrix (preferrably the same that's used
        for the distance matrix supplied to DR algorithm).
        :param high_dimensional_neighbourhood_ranking: Ranking of neighbours in original high-dimensional space. Only
        considered if coranking matrix is None. Calculated if neither coranking_matrix nor high_dimensional_neigh-
        bourhood ranking is supplied.
        """
        super().__init__(low_dimensional_data=low_dimensional_data, high_dimensional_data=high_dimensional_data)

        # Update coranking matrix with supplied value (or calculate, if none was supplied).
        self.coranking_matrix = \
            coranking_matrix if coranking_matrix is not None else \
            CorankingObjective.generate_coranking_matrix(
                high_dimensional_data=high_dimensional_data,
                low_dimensional_data=low_dimensional_data,
                distance_metric=distance_metric,
                high_dim_neighbourhood_ranking=high_dimensional_neighbourhood_ranking
            )

    @staticmethod
    def generate_neighbourhood_ranking(data: numpy.ndarray, distance_metric: str):
        """
        Generates neighbourhood ranking. Used by DimensionalityReductionObjective.generate_coranking_matrix().
        Might be used to generate a ranking once and supplying it to generate_coranking_matrix() instead of calculating
        it over and over.
        :param data:
        :param distance_metric:
        :return: ndarray of sorted and ranked neigbourhood similarity.
        """

        # Calculate distances, then sort by similarities.
        return distance.squareform(distance.pdist(data, distance_metric)).argsort(axis=1).argsort(axis=1)

    @staticmethod
    def generate_coranking_matrix_deprecated(
            high_dimensional_data: numpy.ndarray,
            low_dimensional_data: numpy.ndarray,
            distance_metric: str,
            high_dim_neighbourhood_ranking: numpy.ndarray = None
    ):
        """
        Calculates coranking matrix (might be used for calculating matrix only once if several coranking-based metrics
        are to be used).
        Uses code from https://github.com/samueljackson92/coranking/. Adaption: Parameter for choice of distance metric,
        miscellaneous refactoring.
        :param high_dimensional_data:
        :param low_dimensional_data:
        :param distance_metric: One of the distance metrics supported by scipy's cdist().
        :param high_dim_neighbourhood_ranking: Ranking of neighbourhood similarities. Calculated if none is supplied. If
        supplied, high_dimensional_data is not used.
        :return: Coranking matrix as 2-dim. ndarry.
        """

        n, m = high_dimensional_data.shape

        # Calculate ranking of neighbourhood similarities for high- and low-dimensional datasets.
        high_ranking = high_dim_neighbourhood_ranking if high_dim_neighbourhood_ranking is not None else \
            CorankingObjective.generate_neighbourhood_ranking(high_dimensional_data, distance_metric)
        low_ranking = \
            CorankingObjective.generate_neighbourhood_ranking(low_dimensional_data, distance_metric)

        # Aggregate coranking matrix.
        Q, xedges, yedges = numpy.histogram2d(high_ranking.flatten(), low_ranking.flatten(), bins=n)

        # Remove rankings which correspond to themselves, return coranking matrix.
        return Q[1:, 1:]

    @staticmethod
    def generate_coranking_matrix(
            high_dimensional_data: numpy.ndarray,
            low_dimensional_data: numpy.ndarray,
            distance_metric: str,
            high_dim_neighbourhood_ranking: numpy.ndarray = None,
            use_geodesic=False
    ):
        """
        This function allows to construct coranking matrix based on data in state and latent space. Based on implementation
        https://github.com/samueljackson92/coranking .

        :param high_dimensional_data: a point cloud in state space
        :param low_dimensional_data: a point cloud in latent space
        :param use_geodesic: Whether to use the geodesic distance for state space.
        """

        # We retrieve dimensions of the data
        n, m = high_dimensional_data.shape

        #  We compute distance matrices in both spaces
        # Calculate ranking in high dimensional space only if that hasn't been done yet.
        if high_dim_neighbourhood_ranking is None:
            if use_geodesic:
                k = 2
                is_connex = False
                while not is_connex:
                    knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k, metric=distance_metric)
                    knn.fit(high_dimensional_data)
                    M = knn.kneighbors_graph(high_dimensional_data, mode='distance')
                    graph = nx.from_scipy_sparse_matrix(M)
                    is_connex = nx.is_connected(graph)
                    k += 1
                high_distances = nx.all_pairs_dijkstra_path_length(graph, cutoff=None, weight='weight')
                high_distances = numpy.array([numpy.array(a.items())[:, 1] for a in numpy.array(high_distances.items())[:, 1]])
                high_distances = (high_distances + high_distances.T) / 2
            else:
                high_distances = distance.pdist(high_dimensional_data, metric=distance_metric)
                high_distances = distance.squareform(high_distances)

        # Calculate distances.
        low_distances = distance.pdist(low_dimensional_data, metric=distance_metric)
        low_distances = distance.squareform(low_distances)

        # For each point, we get rank of each point (take care of the weird way .argsort() works)
        high_ranking = high_dim_neighbourhood_ranking if high_dim_neighbourhood_ranking is not None else \
            high_distances.argsort(axis=1).argsort(axis=1)
        low_ranking = low_distances.argsort(axis=1).argsort(axis=1)

        # We compute the Coranking matrix
        Q, xedges, yedges = numpy.histogram2d(
            high_ranking.flatten(),
            low_ranking.flatten(),
            bins=n
        )

        # We remove the rankings corresponding to themselves
        Q = Q[1:, 1:]

        return Q

    @abc.abstractmethod
    def compute(self):
        pass
