from backend.data_generation import InputDataset
from sklearn.datasets import load_wine


class WineDataset(InputDataset):
    """
    Loads wine dataset from
    ...).
    """

    def __init__(self):
        super().__init__()

    def _load_data(self):
        return load_wine()

    def features(self):
        return self._scaled_features

    def labels(self):
        return self._data.target
