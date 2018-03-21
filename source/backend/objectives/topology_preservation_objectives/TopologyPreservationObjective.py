import abc
import numpy
from backend.objectives.DimensionalityReductionObjective import DimensionalityReductionObjective
from .CorankingMatrix import CorankingMatrix


class TopologyPreservationObjective(DimensionalityReductionObjective):
    """
    Abstract base class for calculation of coranking matrix-based objectives (e. g. trustworthiness, continuity, LCMC,
    ...).
    """

    def __init__(
            self,
            low_dimensional_data: numpy.ndarray,
            high_dimensional_data: numpy.ndarray,
            distance_metric: str = 'euclidean',
            coranking_matrix: CorankingMatrix = None,
            k_interval: tuple = (5, 5)
    ):
        """
        Initializes coranking-based objectives.
        :param low_dimensional_data: Output of DR algorithm.
        :param high_dimensional_data: The original high-dimensional data set.
        :param coranking_matrix:
        :param distance_metric: Metric to use for calculation of coranking matrix (preferrably the same that's used
        for the distance matrix supplied to DR algorithm).
        :param k_interval: Tuple containing interval describing which values of k to consider.
        """
        super().__init__(
            low_dimensional_data=low_dimensional_data,
            high_dimensional_data=high_dimensional_data,
            distance_metric=distance_metric
        )
        self._k_interval = k_interval

        # Update coranking matrix with supplied value (or calculate, if none was supplied).
        self._coranking_matrix = coranking_matrix if coranking_matrix is not None else CorankingMatrix(
            high_dimensional_data=high_dimensional_data,
            low_dimensional_data=low_dimensional_data,
            distance_metric=distance_metric,
            high_dimensional_neighbourhood_ranking=None
        )

    @abc.abstractmethod
    def compute(self):
        """
        Calculates objective.
        :return: Criterion averaged/scalarized over chosen k-ary neighbourhood.
        """
        pass
