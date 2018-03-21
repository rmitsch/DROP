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

        Q = self._coranking_matrix.matrix

        # We retrieve the number of points
        n = Q.shape[0]
        # We compute the values
        vals = Q
        # We compute the mask
        mask = numpy.zeros([n, n])

        objective_values = []
        for k in range(self._k_interval[0], self._k_interval[1]):
            mask[:k, :k] = 1.
            # We compute the normalization constant
            norm = k * (n + 1.)
            # We finally compute the measures.
            # Note: This calculates q_nx. r_nx = ( (N - 1) * q_nx(K) - K ) / (N - 1 - K)
            objective_values.append((vals * mask).sum() / float(norm))

        # todo: Scalarize k-ary neighbourhood values.
        measure = 0

        return measure

    def bnx(self, k):
        """
        This method allows to compute the coranking matrix behavior (intrusive if > 0, extrusive otherwise)

        Implementation based on https://github.com/gdkrmr/coRanking .

        :param Q: the coranking matrix
        :param k: the neighbourhood size
        """

        # We compute the measure
        measure = self._coranking_matrix.calculate_intrusion(k) - self._coranking_matrixcalculate_extrusion(k)

        return measure
