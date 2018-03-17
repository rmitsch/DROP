import numpy
from .DimensionalityReductionObjective import DimensionalityReductionObjective
from coranking.metrics import trustworthiness, continuity
from scipy.spatial import distance
import sklearn
import networkx as nx

# Next up:
#     - Move CorankingMatrix to separate class.
#     - Split up embqual in relevant metrics.
#     - Integrate metric calculation in data generation script.


class CorankingMatrix:
    """
    Abstract base class for calculation of coranking matrix-based objectives (e. g. trustworthiness, continuity, LCMC,
    ...).
    """

    def __init__(
            self,
            low_dimensional_data: numpy.ndarray,
            high_dimensional_data: numpy.ndarray,
            distance_metric: str = 'euclidean',
            high_dimensional_neighbourhood_ranking: numpy.ndarray = None
    ):
        self._distance_metric = distance_metric

        self.matrix = self._generate_coranking_matrix(
            high_dimensional_data=high_dimensional_data,
            low_dimensional_data=low_dimensional_data,
            high_dim_neighbourhood_ranking=high_dimensional_neighbourhood_ranking
        )

    @staticmethod
    def _generate_neighbourhood_ranking(data: numpy.ndarray, distance_metric: str):
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

    def _generate_coranking_matrix(
            self,
            high_dimensional_data: numpy.ndarray,
            low_dimensional_data: numpy.ndarray,
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

        n, m = high_dimensional_data.shape

        # ------------------------------------------------------------------------------------
        # 1. Calculate ranking in high dimensional space only if that hasn't been done yet.
        # ------------------------------------------------------------------------------------

        if high_dim_neighbourhood_ranking is None:
            # Check use_geodesic to determine whether to use geodesic or spatial distances.
            high_ranking = \
                CorankingMatrix._calculate_geodesic_ranking(
                    data=high_dimensional_data,
                    distance_metric=self._distance_metric
                ) if use_geodesic else \
                CorankingMatrix._generate_neighbourhood_ranking(
                    high_dimensional_data,
                    distance_metric=self._distance_metric
                )

        # If high_dim_neighbourhood ranking was already supplied, use it instead of calculating anew.
        else:
            high_ranking = high_dim_neighbourhood_ranking

        # ------------------------------------------------------------------------------------
        # 2. Calculate ranking in low-dimensional space.
        # ------------------------------------------------------------------------------------

        # Calculate distances.
        low_ranking = CorankingMatrix._generate_neighbourhood_ranking(
            low_dimensional_data,
            distance_metric=self._distance_metric
        )

        # ------------------------------------------------------------------------------------
        # 3. Compute coranking matrix.
        # ------------------------------------------------------------------------------------

        Q, xedges, yedges = numpy.histogram2d(
            high_ranking.flatten(),
            low_ranking.flatten(),
            bins=n
        )

        # We remove the rankings corresponding to themselves
        return Q[1:, 1:]

    @staticmethod
    def _calculate_geodesic_ranking(data: numpy.ndarray, distance_metric: str):
        """
        Calculates neighbourhood ranking in data based on geodesic distances.
        :param data:
        :param distance_metric:
        :return:
        """
        k = 2
        is_connex = False
        while not is_connex:
            knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k, metric=distance_metric)
            knn.fit(data)
            M = knn.kneighbors_graph(data, mode='distance')
            graph = nx.from_scipy_sparse_matrix(M)
            is_connex = nx.is_connected(graph)
            k += 1
        distances = nx.all_pairs_dijkstra_path_length(graph, cutoff=None, weight='weight')
        distances = numpy.array(
            [numpy.array(a.items())[:, 1] for a in numpy.array(distances.items())[:, 1]])

        distances = (distances + distances.T) / 2

        # Generate rankings from distances.
        return distances.argsort(axis=1).argsort(axis=1)

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
            CorankingMatrix._generate_neighbourhood_ranking(high_dimensional_data, distance_metric)
        low_ranking = \
            CorankingMatrix._generate_neighbourhood_ranking(low_dimensional_data, distance_metric)

        # Aggregate coranking matrix.
        Q, xedges, yedges = numpy.histogram2d(high_ranking.flatten(), low_ranking.flatten(), bins=n)

        # Remove rankings which correspond to themselves, return coranking matrix.
        return Q[1:, 1:]

