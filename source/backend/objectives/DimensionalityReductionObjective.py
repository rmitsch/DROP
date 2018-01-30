import abc
import numpy
from .Objective import Objective


class DimensionalityReductionObjective(Objective):
    """
    Abstract subclass of Objective used as superclass for all DR objectives.
    """

    def __init__(self, low_dimensional_data: numpy.ndarray, high_dimensional_data: numpy.ndarray):
        """
        Initializes DR objective.
        :param low_dimensional_data: Output of DR algorithm.
        :param high_dimensional_data: The original high-dimensional data set.
        """
        super().__init__(low_dimensional_data=low_dimensional_data, target_data=high_dimensional_data)

    @abc.abstractmethod
    def compute(self):
        pass
