import numpy
from coranking.metrics import trustworthiness, continuity
from scipy.spatial import distance
from .CorankingObjective import CorankingObjective


class CorankingObjectiveBundle(CorankingObjective):
    """
    Class for calculation of all coranking matrix-based objectives (e. g. trustworthiness and continuity).
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
            CorankingObjectiveBundle.generate_coranking_matrix(
                high_dimensional_data=high_dimensional_data,
                low_dimensional_data=low_dimensional_data,
                distance_metric=distance_metric,
                high_dim_neighbourhood_ranking=high_dimensional_neighbourhood_ranking
            )

    def compute(self, k: int = 5):
        """
        Calculates coranking-based objectives.
        :param k: Number of nearest neighbours to consider.
        :return: Dictionary with one entry per coranking-based objective.
        """

        return {
            "trustworthiness": float(
                trustworthiness(self.coranking_matrix.astype(numpy.float16), min_k=k, max_k=(k + 1))[0]
            ),
            "continuity": float(
                continuity(self.coranking_matrix.astype(numpy.float16), min_k=k, max_k=(k + 1))[0]
            )
        }

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