import abc
import copy
import csv
import numpy
from scipy.spatial.distance import cdist
import sklearn.ensemble
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn.metrics
import hdbscan

from backend.utils import Utils


class InputDataset:
    """
    Wrapper class for datasets used as input in DR process.
    Contains actual data as well information on ground truth.
    """

    # Target domain performance value in original high-dimensional space.
    high_dim_TDP = None

    def __init__(self, data=None, preprocessed_features=None, classification_accuracy=None):
        """
        Defines variables to be used in inheriting classes.
        Takes some variable speeding up cloning an instance.
        :param data: Loaded primitive dataset.
        :param preprocessed_features: Preprocessed features.
        :param classification_accuracy: Classification accuracy of this dataset.
        """

        # Get logger.
        self._logger = Utils.logger

        # Set primitive data set.
        self._data = copy.deepcopy(data) if data is not None else self._load_data()

        # Preprocess features.
        self._preprocessed_features = copy.deepcopy(preprocessed_features) if preprocessed_features is not None \
            else self._preprocess_features()

        # Calculate accuracy.
        self._classification_accuracy = classification_accuracy if classification_accuracy is not None \
            else self.compute_TDP()

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

    def classification_accuracy(self):
        """
        Returns classification accuracy.
        :return:
        """
        return self._classification_accuracy

    @abc.abstractmethod
    def labels(self):
        """
        Returns ground truth labels in this dataset.
        :return:
        """
        pass

    def data(self):
        """
        Returns loaded data.
        :return:
        """
        return self._data

    @abc.abstractmethod
    def persist_records(self, directory: str):
        """
        Persists manifest properties of all of this dataset's records as .csv.
        Schema: [name, label, [features]].
        Note: Baseline code only works for datasets sticking exactly to the sklearn dataset pattern (i. e. self._data
        is a dictionary with one entry for "features" and one for "labels"); records are named by their index as they
        appear after being loaded.
        :param directory: Path to directory in which to store the file.
        """
        pass

    def compute_TDP(self, features: numpy.ndarray = None, relative: bool = False):
        """
        Calculates target domain performance for specified feature matrix.
        Random forests are used as default model. Can be overwritten by children - important for comparison: Use
        same procedure for prediction in both original and reduced dataset.
        :param features: Data to be used for prediction. Labels are extract from original dataset.
        :param relative: Indicates to compute RTDP instead of TDP.
        :return:
        """

        # Set features, if not specified in function call.
        features = self.preprocessed_features() if features is None else features
        labels = self.labels()

        # Loop through stratified splits, average prediction accuracy over all splits.
        accuracy = 0
        n_splits = 3

        # Apply random forest w/o further preprocessing to predict class labels.
        clf = sklearn.ensemble.RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            n_jobs=1  # psutil.cpu_count(logical=False)
        )

        for train_indices, test_indices in StratifiedShuffleSplit(
                n_splits=n_splits,
                test_size=0.3
        ).split(features, labels):
            # Train model.
            clf.fit(features[train_indices], labels[train_indices])

            # Predict test set.
            predicted_labels = clf.predict(features[test_indices])

            # Measure accuracy.
            accuracy += (predicted_labels == labels[test_indices]).sum() / len(predicted_labels)

        return accuracy / n_splits if not relative else accuracy / n_splits / InputDataset.high_dim_TDP

    def compute_distance_matrix(self, metric: str):
        """
        Compute distance matrix for all records in high-dimensional dataset.
        :param metric:
        :return: Distance matrix as numpy.ndarry.
        """

        return cdist(self.preprocessed_features(), self.preprocessed_features(), metric)

    def compute_separability_metric(self, features: numpy.ndarray) -> float:
        """
        Computes separability metric for this dataset.
        Note: Assumes classification as domain task.
        :param features: Coordinates of low-dimensional projection.
        :return: Normalized score between 0 and 1 indicating how well labels are separated in low-dim. projection.
        """

        ########################################################################
        # 1. Cluster projection with number of classes.
        ########################################################################
        # Alternative Approach: 1-kNN comparison - check if nearest neighbour is in same class.

        # Determine min_cluster_size as approximate min number of elements in a class
        unique, counts_per_class = numpy.unique(self.labels(), return_counts=True)

        # Create HDBSCAN instance and cluster data.
        clusterer = hdbscan.HDBSCAN(
            alpha=1.0,
            metric='euclidean',
            # Use approximate number of entries in least common class as minimal cluster size.
            min_cluster_size=int(counts_per_class.min() * 0.3),
            min_samples=None
        ).fit(features)

        # 2. Calculate Silhouette score based on true labels.
        # Distance matrix: 1 if labels are equal, 0 otherwise -> Hamming distance.
        try:
            silhouette_score = sklearn.metrics.silhouette_score(
                X=self.labels().reshape(-1, 1),
                metric='hamming',
                labels=clusterer.labels_
            )
            # Workaround: Use worst value if number is NaN - why does this happen?
            silhouette_score = -1 if numpy.isnan(silhouette_score) else silhouette_score
        # Silhouette score fails with only one label. Workaround: Set silhouette score to worst possible value in this
        # case. Actual solution: Force at least two clusters - diff. clustering algorithm?
        # See https://github.com/rmitsch/DROP/issues/49.
        except ValueError:
            silhouette_score = -1

        # Normalize to 0 <= x <= 1.
        return (silhouette_score + 1) / 2.0

    @staticmethod
    def check_dataset_name(parameter: str):
        """
        Checks whether supplied parameter is valid dataset name.
        :param parameter:
        :return: If exists: Dataset name. If does not exist: None.
        """

        if parameter in ("vis", "wine", "swiss_roll", "mnist", "happiness"):
            return parameter

        return None
