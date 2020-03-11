import abc
import numpy
from objectives.DimensionalityReductionObjective import DimensionalityReductionObjective


class DistancePreservationObjective(DimensionalityReductionObjective):
    """
    Abstract base class for calculation of objectives based on preservation of distances (e. g. Kruskal's stress).
    """

    def __init__(
            self,
            low_dimensional_data: numpy.ndarray,
            high_dimensional_data: numpy.ndarray,
            distance_metric: str = None,
            use_geodesic_distances: bool = False
    ):
        """
        Initializes new spatial objective.
        :param low_dimensional_data:
        :param high_dimensional_data:
        :param distance_metric:
        :param use_geodesic_distances: Determines whether geodesic distances are to be used. If false, spatial distances
        are used.
        """
        super().__init__(
            low_dimensional_data=low_dimensional_data,
            high_dimensional_data=high_dimensional_data,
            distance_metric=distance_metric
        )
        self._use_geodesic_distances = use_geodesic_distances

    @abc.abstractmethod
    def compute(self):
        """
        Calculates objective.
        :return: Criterion averaged/scalarized over chosen k-ary neighbourhood.
        """
        pass
