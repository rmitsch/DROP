import numpy
from coranking.metrics import trustworthiness, continuity
from .CorankingObjective import CorankingObjective
from .CorankingMatrix import CorankingMatrix


class CorankingObjectiveBundle(CorankingObjective):
    """
    Class for calculation of all coranking matrix-based objectives (e. g. trustworthiness and continuity).
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
        Calculates coranking-based objectives.
        :return: Dictionary with one entry per coranking-based objective.
        """

        # Note: Consider averageing/scalarization over k neighbours before returning a value.
        return {
            "trustworthiness": float(
                trustworthiness(
                    self.coranking_matrix.astype(numpy.float16),
                    min_k=self._k_interval[0],
                    max_k=self._k_interval[1] + 1
                )
            ),
            "continuity": float(
                continuity(
                    self.coranking_matrix.astype(numpy.float16),
                    min_k=self._k_interval[0],
                    max_k=self._k_interval[1] + 1
                )
            )
        }
