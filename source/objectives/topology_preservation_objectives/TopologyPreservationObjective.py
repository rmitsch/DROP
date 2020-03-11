import abc
import numpy
from objectives.DimensionalityReductionObjective import DimensionalityReductionObjective
from .CorankingMatrix import CorankingMatrix


class TopologyPreservationObjective(DimensionalityReductionObjective):
    """
    Abstract base class for calculation of coranking matrix-based objectives (e. g. trustworthiness, continuity, LCMC,
    ...).
    """

    def __init__(
            self,
            low_dimensional_data: numpy.ndarray,
            distance_metric: str = None,
            high_dimensional_data: numpy.ndarray = None,
            coranking_matrix: CorankingMatrix = None
    ):
        """
        Initializes coranking-based objectives.
        :param low_dimensional_data: Output of DR algorithm.
        :param high_dimensional_data: The original high-dimensional data set.
        :param coranking_matrix:
        :param distance_metric: Metric to use for calculation of coranking matrix (preferrably the same that's used
        for the distance matrix supplied to DR algorithm).
        """

        assert high_dimensional_data is not None or coranking_matrix is not None
        assert coranking_matrix is not None or distance_metric, \
            "Distance metric must be specified when not passing high_dimensionsional_neighbourhood_rankings."

        super().__init__(
            low_dimensional_data=low_dimensional_data,
            high_dimensional_data=high_dimensional_data,
            distance_metric=distance_metric
        )

        # Update coranking matrix with supplied value (or calculate, if none was supplied).
        self._coranking_matrix: CorankingMatrix = coranking_matrix if coranking_matrix is not None else CorankingMatrix(
            high_dimensional_data=high_dimensional_data,
            low_dimensional_data=low_dimensional_data,
            distance_metric=distance_metric
        )

    @abc.abstractmethod
    def compute(self):
        """
        Calculates objective.
        :return: Criterion averaged/scalarized over chosen k-ary neighbourhood.
        """
        pass
