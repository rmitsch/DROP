import abc
import numpy as np
from scipy.spatial.distance import cdist
import sklearn.ensemble
import sklearn.metrics
import hdbscan
import pandas as pd
from utils import Utils
import logging
from enum import Enum


class InputDataset:
    """
    Wrapper class for datasets used as input in DR process.
    Contains actual data as well information on ground truth.
    """

    class DataSupertypes(Enum):
        NUMERICAL = "numerical"
        CATEGORICAL = "categorical"

    class DataSubtypes(Enum):
        CONTINOUS = "continous"
        DISCRETE = "discrete"
        NOMINAL = "nominal"

    def __init__(self, storage_path: str):
        """
        Defines variables to be used in inheriting classes.
        Takes some variable speeding up cloning an instance.
        :param storage_path: Path to folder storing data.
        """

        # Get logger.
        self._logger: logging.Logger = Utils.logger

        # Set storage path.
        self._storage_path: str = storage_path

        # Set primitive data set.
        self._data: dict = self._load_data()

        # Preprocess features.
        self._preprocessed_hd_features: np.ndarray = self._preprocess_hd_features()

        # Calculate accuracy for HD space, if not done yet.
        self._hd_target_domain_performance: float = self.compute_hd_target_domain_performance()

    @abc.abstractmethod
    def _load_data(self) -> dict:
        """
        Loads or generates data for this dataset.
        :return:
        """
        pass

    @abc.abstractmethod
    def _preprocess_hd_features(self) -> np.ndarray:
        """
        Executes all necessary preprocessing and transforms data for usage in sklearn-style models.
        :return: Set of preprocessed features as numpy array.
        """
        pass

    def features(self) -> pd.DataFrame:
        """
        Returns features in this dataset.
        :return:
        """
        return self._data["features"]

    def labels(self) -> pd.Series:
        """
        Returns ground truth labels in this dataset.
        :return:
        """
        return self._data["labels"]

    def preprocessed_features(self) -> np.ndarray:
        """
        Returns preprocssed features in this dataset.
        :return:
        """
        return self._preprocessed_hd_features

    def target_domain_performance(self) -> float:
        """
        Returns target domain performance accuracy.
        :return:
        """
        return self._hd_target_domain_performance

    def data(self) -> dict:
        """
        Returns loaded data.
        :return:
        """
        return self._data

    @abc.abstractmethod
    def persist_records(self):
        """
        Persists manifest properties of all of this dataset's records as .csv.
        Schema: [name, label, [features]].
        Note: Baseline code only works for datasets sticking to the sklearn dataset pattern (i. e. self._data
        is a dictionary with one entry for "features" and one for "labels"); records are named by their index as they
        appear after being loaded.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_attributes_data_types() -> dict:
        """
        Describes dataset's columns and their typification. Supertypes/types:
          - Categorical
            - Nominal
            - Ordinal
          - Numerical
            - Discrete
            - Continous
            - Interval
            - Ratio
        :return: Dictionary describing columns' types. {attribute -> {supertype: x, type: y[, ordering: []]}}. Ordering
        is only included for ordinal attributes.
        """
        pass

    @staticmethod
    def get_pointwise_metrics_data_types() -> dict:
        """
        Similar to get_attributes_data_types(), except this specifies the data types for dataset-agnostic pointwise DR
        quality metrics passed on to frontend.
        :return: Dictionary describing pointwise quality metrics' types. {metric -> {supertype: x, type: y}}.
        """

        supertypes: Enum = InputDataset.DataSupertypes
        subtypes: Enum = InputDataset.DataSubtypes

        return {
            "q_nx_i": {"supertype": supertypes.NUMERICAL.value, "type": subtypes.DISCRETE.value}
        }

    def compute_hd_target_domain_performance(self) -> float:
        """
        Calculates target domain performance for feature matrix in original, high-dimensional dataset.
        :return: Target domain performance for original, high-dimensional dataset.
        """
        pass

    def compute_relative_target_domain_performance(self, features: np.ndarray) -> float:
        """
        Calculates relative target domain performance for specified feature matrix.
        :param features: Data to be used for prediction. Labels are extracted from original dataset.
        :return: Relative target domain performance (i.e. TDP divided by TDP in original/HD space).
        """
        pass

    def compute_distance_matrix(self) -> np.ndarray:
        """
        Compute distance matrix for all records in high-dimensional dataset.
        :return: Distance matrix as numpy.ndarry.
        """

        return cdist(self._preprocessed_hd_features, self._preprocessed_hd_features, "euclidean")

    def compute_separability_metric(
            self,
            features: np.ndarray,
            cluster_metric: str = "euclidean",
            silhouette_metric: str = "hamming",
            min_cluster_size: int = 2
    ) -> float:
        """
        Computes separability metric for this dataset.
        Note: Assumes classification as domain task.
        :param features: Coordinates of low-dimensional projection.
        :param cluster_metric: Metric to use in clustering.
        :param silhouette_metric: Metric to use when computing Silhouette score.
        :param min_cluster_size: Minimal number of records in cluster.
        :return: Normalized score between 0 and 1 indicating cluster consistency in feature space.
        """

        ########################################################################
        # 1. Cluster projection with number of classes.
        ########################################################################

        # Create HDBSCAN instance and cluster data.
        # noinspection PyTypeChecker
        clusterer: hdbscan.HDBSCAN = hdbscan.HDBSCAN(
            alpha=1.0,
            metric=cluster_metric,
            # Use approximate number of entries in least common class as minimal cluster size.
            min_cluster_size=min_cluster_size,
            min_samples=None
        ).fit(features)

        # 2. Calculate Silhouette score based on true labels.
        # Distance matrix: 1 if labels are equal, 0 otherwise -> Hamming distance.
        try:
            silhouette_score: float = sklearn.metrics.silhouette_score(
                X=self.labels().values.reshape(-1, 1),
                metric=silhouette_metric,
                labels=clusterer.labels_
            )
            # Workaround: Use worst value if number is NaN - why does this happen?
            silhouette_score = -1 if np.isnan(silhouette_score) else silhouette_score
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

        if parameter in ("movie", "happiness"):
            return parameter

        return None

    @staticmethod
    @abc.abstractmethod
    def sort_dataframe_columns_for_frontend(df: pd.DataFrame) -> pd.DataFrame:
        """
        Sorts dataframe in desired sequence for frontend.
        :param df:
        :return:
        """
        pass
