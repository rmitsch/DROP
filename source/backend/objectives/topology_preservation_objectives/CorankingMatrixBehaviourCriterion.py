import numpy
from .TopologyPreservationObjective import TopologyPreservationObjective
from .CorankingMatrix import CorankingMatrix


class CorankingMatrixBehaviourCriterion(TopologyPreservationObjective):
    """
    Calculates coranking matrix quality criterion (Q_nx).
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
        """

        super().__init__(
            low_dimensional_data=low_dimensional_data,
            high_dimensional_data=high_dimensional_data,
            distance_metric=distance_metric,
            coranking_matrix=coranking_matrix,
            k_interval=k_interval
        )

    def compute(self):
        """
        Calculates objective.
        Source: https://git.apere.me/apere/INRIA-Internship/src/1a73ba2619f13ec23cc860309dab7fe69fd18ab9/src/embqual.py.
        :return: Criterion averaged/scalarized over chosen k-ary neighbourhood.
        """

        # We compute the measure
        measure = self._coranking_matrix.calculate_intrusion(self._k_interval[0]) - \
                  self._coranking_matrix.calculate_extrusion(self._k_interval[0])

        # todo: Scalarize k-ary neighbourhood values.
        return measure
