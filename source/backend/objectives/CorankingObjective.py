import abc
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

    @abc.abstractmethod
    def compute(self):
        pass
