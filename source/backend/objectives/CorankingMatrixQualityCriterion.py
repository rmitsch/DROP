import numpy
from .CorankingObjective import CorankingObjective
from .CorankingMatrix import CorankingMatrix


class CorankingMatrixQualityCriterion(CorankingObjective):
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
            # We finally compute the measures
            objective_values.append((vals * mask).sum() / float(norm))

        # todo: Scalarize k-ary neighbourhood values.
        measure = 0

        return measure
