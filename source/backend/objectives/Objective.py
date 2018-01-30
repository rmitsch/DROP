import abc
import numpy


class Objective(abc.ABC):
    """
    Define Metric as abstract base class (ABC) for all dimensionsality reduction & target dataset objectives.
    """

    def __init__(self, low_dimensional_data: numpy.ndarray, target_data: numpy.ndarray):
        """
        Initializes objective.
        :param low_dimensional_data: Output of DR algorithm.
        :param target_data: Target data can be either (1) the original high-dimensional data set (if this is an DR ob-
        jective) or (2) the data that has to be predicted/matched by the low-dimensional data set, i. e. class/cluster
        labels or similar (if this is a ground truth objective).
        """
        super().__init__()

        self.target_data = target_data
        self.low_dimensional_data = low_dimensional_data

    @abc.abstractmethod
    def compute(self):
        pass
