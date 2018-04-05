import abc
import sklearn.ensemble
from sklearn.model_selection import StratifiedShuffleSplit
import psutil
import numpy
from scipy.spatial.distance import cdist
from backend.utils import Utils


class InputDataset:
    """
    Wrapper class for datasets used as input in DR process.
    Contains actual data as well information on ground truth.
    """

    def __init__(self):
        """
        Defines variables to be used in inheriting classes.
        """

        # Get logger.
        self._logger = Utils.logger

        # Load or generate data.
        self._data = self._load_data()

        # Preprocess features.
        self._preprocessed_features = self._preprocess_features()

        # Calculate accuracy.
        self._classification_accuracy = self.calculate_classification_accuracy()

    @abc.abstractmethod
    def _load_data(self):
        """
        Loads or generates data for this dataset.
        :return:
        """
        pass

    @abc.abstractmethod
    def _preprocess_features(self):
        """
        Executes all necessary preprocessing.
        :return: Set of preprocessed features. Is stored in self._scaled_features.
        """
        pass

    @abc.abstractmethod
    def features(self):
        """
        Returns features in this dataset.
        :return:
        """
        pass

    def preprocessed_features(self):
        """
        Returns preprocssed features in this dataset.
        :return:
        """

        return self._preprocessed_features

    @abc.abstractmethod
    def labels(self):
        """
        Returns ground truth labels in this dataset.
        :return:
        """
        pass

    def calculate_classification_accuracy(self, features: numpy.ndarray = None):
        """
        Calculates classification accuracy with original dataset.
        Random forests are used as default classifier. Can be overwritten by children - important for comparison: Use
        same procedure for classification in both original and reduced dataset.
        :param features: Data to be used for classification. Labels are extract from original dataset.
        :return:
        """

        # Set features, if not specified in function call.
        features = self.preprocessed_features() if features is None else features
        labels = self.labels()

        # Apply straightforward k-nearest neighbour w/o further preprocessing to predict class labels.
        clf = sklearn.ensemble.RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            n_jobs=psutil.cpu_count(logical=False)
        )

        # Loop through stratified splits, average prediction accuracy over all splits.
        accuracy = 0
        n_splits = 3
        for train_indices, test_indices in StratifiedShuffleSplit(
                n_splits=n_splits,
                test_size=0.33
        ).split(features, labels):
            # Train model.
            clf.fit(features[train_indices], labels[train_indices])

            # Predict test set.
            predicted_labels = clf.predict(features[test_indices])

            # Measure accuracy.
            accuracy += (predicted_labels == labels[test_indices]).sum() / len(predicted_labels)

        return accuracy / n_splits

    def compute_distance_matrix(self, metric: str):
        """
        Compute distance matrix for all records in high-dimensional dataset.
        :param metric:
        :return: Distance matrix as numpy.ndarry.
        """

        return cdist(self._preprocessed_features(), self._preprocessed_features(), metric)
