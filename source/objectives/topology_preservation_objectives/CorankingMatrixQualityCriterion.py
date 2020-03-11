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
            distance_metric: str = None,
            high_dimensional_data: numpy.ndarray = None,
            coranking_matrix: CorankingMatrix = None
    ):
        """
        Initializes coranking-based objectives.
        """

        super().__init__(
            low_dimensional_data=low_dimensional_data,
            high_dimensional_data=high_dimensional_data,
            distance_metric=distance_metric,
            coranking_matrix=coranking_matrix
        )

    def compute(self):
        """
        Calculates objective.
        Source: https://git.apere.me/apere/INRIA-Internship/src/1a73ba2619f13ec23cc860309dab7fe69fd18ab9/src/embqual.py.
        :return: Criterion averaged/scalarized over chosen k-ary neighbourhood.
        """

        ########################################
        # 1. Prepare coranking matrix data.
        ########################################

        Q = self._coranking_matrix.matrix()
        # We retrieve the number of points
        n = Q.shape[0]
        # We compute the values
        vals = Q

        ########################################
        # 2. Pick k values to sample.
        ########################################

        # Pick immediate neighbourhood + 4 equidistant values of k beyond that for sampling purposes.
        k_samples = [1, 5, 10]
        k_samples.extend(numpy.linspace(
            start=1,
            stop=n - 2,
            num=4,
            endpoint=False,
            dtype=numpy.int
        ))

        ########################################
        # 3. Calculate AUC for r_nx.
        ########################################

        auc_r_nx = 0
        # We compute the mask.
        mask = numpy.zeros([n, n])

        # Range for k is (1, n - 2) - see https://www-sciencedirect-com/science/article/pii/S0925231215003641 for
        # derivation.
        for k in k_samples:
            mask[:k, :k] = 1.
            # We compute the normalization constant.
            norm = k * (n + 1.)

            # We finally compute the measures.
            # Note: This calculates q_nx.
            q_nx = (vals * mask).sum() / float(norm)

            # Calculate r_nx.
            r_nx = ((n - 1) * q_nx - k) / (n - 1 - k)

            # See p. 253, bottom right, on
            # https://www-sciencedirect-com.uaccess.univie.ac.at/science/article/pii/S0925231215003641 on equation.
            # Note that AUC here is just a weighted average! Has to be divided by the number of k-ary neighbourhoods
            # considered.
            auc_r_nx += r_nx / k

            # Reset mask.
            mask[:k, :k] = 0.

        return auc_r_nx / sum([1 / k for k in k_samples])
