from backend.data_generation import InputDataset
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


class WineDataset(InputDataset):
    """
    Loads wine dataset from sklearn.
    """

    def __init__(self, data=None, preprocessed_features=None, classification_accuracy=None):
        super().__init__(
            data=data,
            preprocessed_features=preprocessed_features,
            classification_accuracy=classification_accuracy
        )

    def _load_data(self):
        return load_wine()

    def features(self):
        return self._preprocessed_features

    def labels(self):
        return self._data.target

    def _preprocess_features(self):
        # Scale features.
        return StandardScaler().fit_transform(self._data.data)
