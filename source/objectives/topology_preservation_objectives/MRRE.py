import numpy
from .TopologyPreservationObjective import TopologyPreservationObjective
from .CorankingMatrix import CorankingMatrix


class MRRE(TopologyPreservationObjective):
    """
    Calculates mean relative rank error (MRRE).
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
        columns, rows = numpy.meshgrid(range(n), range(n))
        rank_c = columns+1
        rank_r = rows+1
        values = numpy.abs( (rank_r-rank_c)/rank_c*Q )

        objective_values = []
        for k in range(self._k_interval[0], self._k_interval[1]):
            # We finally compute the measures
            objective_values.append(
                2 - MRRE._mrre_trustworthiness(k, n, values) - MRRE._mrre_continuity(k, n, values)
            )

        # todo: Scalarize k-ary neighbourhood values.
        measure = 0

        return measure

    @staticmethod
    def _mrre_trustworthiness(k, n, values):
        """
        This method allows to compute the mean relative rank error Trustworthiness measure, as proposed in
        https://www.researchgate.net/publication/221165927_Rank-based_quality_assessment_of_nonlinear_dimensionality_reduction

        Implementation based on https://github.com/gdkrmr/coRanking .

        Source: https://git.apere.me/apere/INRIA-Internship/src/1a73ba2619f13ec23cc860309dab7fe69fd18ab9/src/embqual.py.

        :param k: The neighbourhood size
        :param n: Number of points.
        :param values:
        """

        # We compute the mask
        mask = numpy.zeros([n, n])
        mask[:k, :k] = 1.
        mask[k:, :k] = 1.
        # We compute the normalization constant
        norm = n * numpy.abs(n-2.*numpy.arange(1,k+1) / numpy.arange(1,k+1)).sum()
        # We finally compute the measures
        measure = (values * mask).sum() / float(norm)

        return measure

    @staticmethod
    def _mrre_continuity(k, n, values):
        """
        This method allows to compute the mean relative rank error Continuity measure, as proposed in
        https://www.researchgate.net/publication/221165927_Rank-based_quality_assessment_of_nonlinear_dimensionality_reduction

        Implementation based on https://github.com/gdkrmr/coRanking .

        Source: https://git.apere.me/apere/INRIA-Internship/src/1a73ba2619f13ec23cc860309dab7fe69fd18ab9/src/embqual.py.

        :param k: The neighbourhood size
        :param n: Number of points.
        :param values:
        """

        # We compute the mask
        mask = numpy.zeros([n, n])
        mask[:k, :k] = 1.
        mask[:k, k:] = 1.
        # We compute the normalization constant
        norm = n * numpy.abs(n-2.*numpy.arange(1,k+1) / numpy.arange(1,k+1)).sum()
        # We finally compute the measures
        measure = (values * mask).sum() / float(norm)

        return measure
