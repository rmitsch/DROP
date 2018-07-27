from backend.data_generation.datasets import InputDataset
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


class MNISTDataset(InputDataset):
    """
    Load MNIST datasets using sklearn.
    """

    def __init__(self):
        super().__init__()

    def _load_data(self):
        mnist_data = fetch_mldata('MNIST original', data_home="data")
        reduced_mnist_data = {
            "features": None,
            "labels": None
        }

        # MNIST with 70k entries requires too much memory for distance matrices for this machine.
        # Hence: We pick a subset of 10k as stratified sample.
        for train_indices, test_indices in StratifiedShuffleSplit(
                n_splits=1,
                test_size=1 / 10
        ).split(mnist_data.data, mnist_data.target):
            reduced_mnist_data["features"] = mnist_data.data[test_indices].astype(int)
            reduced_mnist_data["labels"] = mnist_data.target[test_indices].astype(int)

        return reduced_mnist_data

    def features(self):
        return self._data["features"]

    def labels(self):
        return self._data["labels"]

    def _preprocess_features(self):
        # Scale features.
        return StandardScaler().fit_transform(self._data["features"])
