import numpy
from .TopologyPreservationObjective import TopologyPreservationObjective
from .CorankingMatrix import CorankingMatrix


class PointwiseCorankingMatrixQualityCriterion(TopologyPreservationObjective):
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

    def compute(self) -> float:
        """
        Calculates objective.
        Source: http://www.cs.rug.nl/biehl/Preprints/2012-esann-quality.pdf, section 4.
        :return: Vector with pointwise q_nx_i for k with weighted average.
        """

        ########################################
        # 1. Prepare auxiliary matrix data.
        ########################################

        # Nuber of points.
        num_points = self._low_dimensional_data.shape[0]
        # Counter.
        i = 0
        # k for which to compute q_nx(k).
        k_samples = [1, 5, 10]
        k_samples.extend(numpy.linspace(
            start=1,
            stop=num_points - 2,
            num=4,
            endpoint=False,
            dtype=numpy.int
        ))

        # Vector with all q_nx_i results for point with index i in (i).
        q_nx_i = numpy.zeros([num_points, 1])

        ########################################
        # 2. Compute pointwise q_nx_i.
        ########################################

        # Generate mask matrix.
        mask = numpy.zeros([num_points - 2, num_points - 2])

        # Compute q_nx_i(k).
        for Q_i in self._coranking_matrix.create_pointwise_coranking_matrix_generator():
            for k in k_samples:
                mask[:k, :k] = 1.

                # For equation see http://www.cs.rug.nl/biehl/Preprints/2012-esann-quality.pdf (section 4).
                q_nx_i[i] += (Q_i * mask).sum() / float(k)

                # Reset mask.
                mask[:k, :k] = 0
            i += 1

        return q_nx_i / len(k_samples)
