from backend.data_generation import InputDataset
import sklearn
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
import psutil
from sklearn.model_selection import StratifiedShuffleSplit


class WineDataset(InputDataset):
    """
    Loads wine dataset from
    ...).
    """

    def __init__(self):
        super().__init__()

        # Scale features.
        self._scaled_features = StandardScaler().fit_transform(self._data.data)

    def _load_data(self):
        return load_wine()

    def features(self):
        return self._scaled_features

    def labels(self):
        return self._data.target

    def calculate_classification_accuracy(self):
        # Apply straightforward k-nearest neighbour w/o further preprocessing to predict class labels.
        clf = neighbors.KNeighborsClassifier(
            n_neighbors=3,
            weights="distance",
            n_jobs=psutil.cpu_count(logical=False)
        )

        # Loop through stratified splits, average prediction accuracy over all splits.
        accuracy = 0
        n_splits = 3
        for train_indices, test_indices in StratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=0.5
        ).split(self.features(), self.labels()):
            # Train model.
            clf.fit(self.features()[train_indices], self.labels()[train_indices])

            # Predict test set.
            predicted_labels = clf.predict(self.features()[test_indices])

            # Measure accuracy.
            accuracy += (predicted_labels == self.labels()[test_indices]).sum() / len(self._data)

        return accuracy / n_splits
