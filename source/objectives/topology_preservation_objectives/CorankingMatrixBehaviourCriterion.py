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

        ########################################
        # 2. Pick k values to sample.
        ########################################

        # Pick immediate neighbourhood + 4 equidistant values of k beyond that for sampling purposes.
        # Pick 10-percentile neighbourhood for k (arbitray choice).
        k_samples = [1, 5, 10]
        k_samples.extend(numpy.linspace(
            start=1,
            stop=n - 2,
            num=4,
            endpoint=False,
            dtype=numpy.int
        ))

        ########################################
        # 3. Calculate AUC for b_nx.
        ########################################

        # Calculate intrusion and extrusion values.
        intrusion_scores = self._coranking_matrix.calculate_intrusion(k_samples)
        extrusion_scores = self._coranking_matrix.calculate_extrusion(k_samples)

        # Aggregate intrusion and extrusion scores.
        auc_b_nx = 0
        # Range for k is (1, n - 2) - see https://www-sciencedirect-com/science/article/pii/S0925231215003641 for
        # derivation.
        for i, k in enumerate(k_samples):
            auc_b_nx += intrusion_scores[i] / k - extrusion_scores[i] / k

        return auc_b_nx / sum([1 / k for k in k_samples])
