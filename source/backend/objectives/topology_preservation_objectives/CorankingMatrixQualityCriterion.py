import numpy
from .TopologyPreservationObjective import TopologyPreservationObjective
from .CorankingMatrix import CorankingMatrix


class CorankingMatrixQualityCriterion(TopologyPreservationObjective):
    """
    Calculates coranking matrix quality criterion (R_nx).
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

        Q = self._coranking_matrix.matrix()

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
            q_nx = (vals * mask).sum() / float(norm)
            r_nx = ((n - 1) * q_nx - k) / (n - 1 - k)
            # todo: Calculate area under the curve with a few (interspersed) values of k.
            # See p. 253, bottom right, on
            # https://www-sciencedirect-com.uaccess.univie.ac.at/science/article/pii/S0925231215003641 on equation.
            # Question: Values of k? Other paper used quaternary quantiles in range - might be an option; AUC
            # calculation weights small neighbourhoods higher.
            # Note that AUC here is just an average! Has to be divided by the number of k-ary neighbourhoods considered.
            auc_r_nx = 0
            # Append r_nx to list of values.
            objective_values.append(auc_r_nx)

        # todo: Scalarize k-ary neighbourhood values.
        measure = 0

        return measure
