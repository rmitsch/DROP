import copy

import numpy
from backend.objectives.DimensionalityReductionObjective import DimensionalityReductionObjective
from coranking.metrics import trustworthiness, continuity
from scipy.spatial import distance
import sklearn
import networkx as nx


class CorankingMatrix:
    """
    Contains representation of coranking matrix including auxiliary functions.
    """

    def __init__(
            self,
            low_dimensional_data: numpy.ndarray,
            high_dimensional_data: numpy.ndarray,
            distance_metric: str = 'euclidean',
            high_dimensional_neighbourhood_ranking: numpy.ndarray = None
    ):
        self._distance_metric = distance_metric
        self._high_dim_ranking = None
        self._low_dim_ranking = None

        self._matrix = self._generate_coranking_matrix(
            high_dimensional_data=high_dimensional_data,
            low_dimensional_data=low_dimensional_data,
            high_dim_neighbourhood_ranking=high_dimensional_neighbourhood_ranking
        )

    @staticmethod
    def generate_neighbourhood_ranking(distance_matrix: numpy.ndarray):
        """
        Generates neighbourhood ranking. Used externally, accepts pre-calculated distance matrix.
        Might be used to generate a ranking once and supplying it to generate_coranking_matrix() instead of calculating
        it over and over.
        :param distance_matrix
        :return: ndarray of sorted and ranked neigbourhood similarity.
        """

        # Original approach: Re-calculate with chosen data and distance metric.
        # Calculate distances, then sort by similarities.
        return distance_matrix.argsort(axis=1).argsort(axis=1)

    @staticmethod
    def _generate_neighbourhood_ranking(data: numpy.ndarray, distance_metric: str):
        """
        Generates neighbourhood ranking. Used internally, accepts original data and chosen distance metric.

        Might be used to generate a ranking once and supplying it to generate_coranking_matrix() instead of calculating
        it over and over.
        :param data:
        :param distance_metric:
        :return: ndarray of sorted and ranked neigbourhood similarity.
        """

        # Original approach: Re-calculate with chosen data and distance metric.
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
        This function allows to construct coranking matrix based on data in state and latent space. Based on
        implementations in https://github.com/samueljackson92/coranking and
        https://git.apere.me/apere/INRIA-Internship/src/1a73ba2619f13ec23cc860309dab7fe69fd18ab9/src/embqual.py.
        Adaption: Parameter for choice of distance metric,
        miscellaneous refactoring.

        :param high_dimensional_data:
        :param low_dimensional_data:
        :param use_geodesic: Whether to use the geodesic distance for state space.
        :param high_dim_neighbourhood_ranking: Ranking of neighbourhood similarities. Calculated if none is supplied. If
        supplied, high_dimensional_data is not used.
        :return: Coranking matrix as 2-dim. ndarry.
        """

        n, m = high_dimensional_data.shape

        # ------------------------------------------------------------------------------------
        # 1. Calculate ranking in high dimensional space only if that hasn't been done yet.
        # ------------------------------------------------------------------------------------

        if high_dim_neighbourhood_ranking is None:
            # Check use_geodesic to determine whether to use geodesic or spatial distances.
            self._high_dim_ranking = \
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
            self._high_dim_ranking = high_dim_neighbourhood_ranking

        # ------------------------------------------------------------------------------------
        # 2. Calculate ranking in low-dimensional space.
        # ------------------------------------------------------------------------------------

        # Calculate distances.
        self._low_dim_ranking = CorankingMatrix._generate_neighbourhood_ranking(
            low_dimensional_data,
            distance_metric=self._distance_metric
        )

        # ------------------------------------------------------------------------------------
        # 3. Compute coranking matrix.
        # ------------------------------------------------------------------------------------

        Q, xedges, yedges = numpy.histogram2d(
            self._high_dim_ranking.flatten(),
            self._low_dim_ranking.flatten(),
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

    def matrix(self):
        """
        Returns coranking matrix.
        :return: Coranking matrix.
        """
        return self._matrix

    def calculate_intrusion(self, k_values: list):
        """
        This method allows to compute the fraction of intrusion.

        Implementation based on https://github.com/gdkrmr/coRanking .

        :param Q: The coranking matrix
        :param k_values: The neighbourhood sizes to sample.
        :returns List of intrusion values for each k.
        """

        Q = self._matrix

        # We retrieve the number of points
        n = Q.shape[0]
        # We compute the values
        vals = Q

        measures = []
        for k in k_values:
            # We compute the mask.
            mask = numpy.zeros([n, n])
            mask[:k, :k] = numpy.triu(numpy.ones([k, k]))
            # We compute the normalization constant
            norm = k * (n + 1.)
            # We finally compute the measures
            measures.append((vals * mask).sum() / float(norm))

        return measures

    def calculate_extrusion(self, k_values: list):
        """
        This method allows to compute the fraction of extrusion.

        Implementation based on https://github.com/gdkrmr/coRanking .

        :param Q: The coranking matrix
        :param k_values: The neighbourhood sizes to sample.
        :returns List of extrusion values for each k.
        """

        Q = self._matrix

        # We retrieve the number of points
        n = Q.shape[0]
        # We compute the values
        vals = Q

        # We compute the mask
        mask = numpy.zeros([n, n])

        measures = []
        for k in k_values:
            mask[:k, :k] = numpy.tril(numpy.ones([k, k]))
            # We compute the normalization constant.
            norm = k * (n + 1.)
            # We finally compute the measures.
            measures.append((vals * mask).sum() / float(norm))

            # Reset mask.
            mask[:k, :k] = 0.

        return measures

    def high_dimensional_neighbourhood_ranking(self):
        """
        Returns ranking of neighbours in high-dimensional space.
        :return:
        """

        return self._high_dim_ranking

    def low_dimensional_neighbourhood_ranking(self):
        """
        Returns ranking of neighbours in low-dimensional space.
        :return:
        """

        return self._low_dim_ranking

    def create_pointwise_coranking_matrix_generator(self, indices: list = None):
        """
        Creates a generator yielding one pointwise
        (see section 4 in http://www.cs.rug.nl/biehl/Preprints/2012-esann-quality.pdf) coranking matrix Q per record.
        Applied to all records with index i in indices.
        :param indices:
        :return:
        """

        n, m = self._matrix.shape
        indices = indices if indices is not None else [i for i in range(0, n)]

        # Create pointwise coranking matrix for each index.
        for i in indices:
            # Initialize ranking matrices.
            low_dim_nh_ranking_i = numpy.zeros(self._low_dim_ranking.shape)
            high_dim_nh_ranking_i = numpy.zeros(self._high_dim_ranking.shape)

            # Copy row i.
            low_dim_nh_ranking_i[i] = self._low_dim_ranking[i]
            high_dim_nh_ranking_i[i] = self._high_dim_ranking[i]

            # Calculate coranking matrix.
            Q, xedges, yedges = numpy.histogram2d(
                high_dim_nh_ranking_i.flatten(),
                low_dim_nh_ranking_i.flatten(),
                bins=n
            )

            # Remove rankings which correspond to themselves, return coranking matrix.
            yield Q[1:, 1:]
