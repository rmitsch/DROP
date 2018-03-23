import abc


class InputDataset:
    """
    Wrapper class for datasets used as input in DR process.
    Contains actual data as well information on ground truth.
    """

    def __init__(self):
        """
        Defines variables to be used in inheriting classes.
        """

        # Load or generate data.
        self._data = self._load_data()

    @abc.abstractmethod
    def _load_data(self):
        """
        Loads or generates data.
        :return:
        """
        pass

    @abc.abstractmethod
    def features(self):
        """
        Returns features in this dataset.
        :return:
        """
        pass

    @abc.abstractmethod
    def labels(self):
        """
        Returns ground truth labels in this dataset.
        :return:
        """
        pass

    @abc.abstractmethod
    def calculate_classification_accuracy(self):
        """
        Calculates classification accuracy with original dataset.
        :return:
        """
        pass