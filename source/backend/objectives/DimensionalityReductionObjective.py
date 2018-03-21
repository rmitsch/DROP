import abc
import numpy
from .Objective import Objective


class DimensionalityReductionObjective(Objective):
    """
    Abstract subclass of Objective used as superclass for all DR objectives.
    Note: Substantial parts of this implementation were derived or copied from
    https://git.apere.me/apere/INRIA-Internship/src/1a73ba2619f13ec23cc860309dab7fe69fd18ab9/src/embqual.py.
    """

    def __init__(
            self,
            low_dimensional_data: numpy.ndarray,
            high_dimensional_data: numpy.ndarray,
            distance_metric: str):
        """
        Initializes DR objective.
        :param low_dimensional_data: Output of DR algorithm.
        :param high_dimensional_data: The original high-dimensional data set.
        :param distance_metric: Distance metric to use.
        """
        super().__init__(
            low_dimensional_data=low_dimensional_data,
            target_data=high_dimensional_data
        )
        self._distance_metric = distance_metric

    @abc.abstractmethod
    def compute(self):
        pass
