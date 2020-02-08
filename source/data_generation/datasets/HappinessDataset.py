import csv
import math
import os
import warnings

import hdbscan
import psutil
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import re
from sklearn.preprocessing import StandardScaler
import sklearn
from data_generation.datasets import InputDataset
import xgboost


class HappinessDataset(InputDataset):
    """
    Extended Kaggle happiness dataset.
    """

    # Measured TDP for Happiness dataset with regression task in terms of RMSE.
    high_dim_TDP = 0.31

    def __init__(self, storage_path: str):
        self._df = None
        super().__init__(storage_path=storage_path)

    def _load_data(self):
        df = pd.read_csv(filepath_or_buffer=self._storage_path + "/happiness_2017.csv").drop(
            ["map_reference", "biggest_official_language", "gdp_per_capita[$]"], axis=1
        ).set_index("country")
        df = df.rename(columns={col: re.sub(r'\[.*\]', '', col) for col in df.columns})
        df = df.dropna(axis='columns')
        self._df = df

        return {
            "features": df.drop(["happiness_score", "happiness_rank"], axis=1),
            "labels": df.happiness_score
        }

    def features(self):
        return self._data["features"]

    def labels(self):
        return self._data["labels"]

    def _preprocess_features(self):
        return StandardScaler().fit_transform(self._data["features"].values)

    def persist_records(self):
        filepath = self._storage_path + '/happiness_records.csv'

        if not os.path.isfile(filepath):
            features_df = self._df.copy(deep=True)
            features_df["record_name"] = features_df.index.values
            features_df = features_df.rename(columns={"happiness_level": "target_label"})
            features_df.to_csv(path_or_buf=filepath, index=False)

    def compute_target_domain_performance(self, features: np.ndarray = None, relative: bool = False):
        # Set features, if not specified in function call.
        features = self.preprocessed_features() if features is None else features
        labels = np.reshape(self.labels().values, (self.labels().values.shape[0], 1))

        # Loop through stratified splits, average prediction accuracy over all splits.
        accuracy = 0
        n_splits = 100

        # Apply random forest w/o further preprocessing to predict class labels.
        reg = xgboost.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.08,
            subsample=0.75,
            colsample_bytree=1,
            max_depth=7
        )

        for i in range(0, n_splits):
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(features, labels, test_size=0.5)
            reg.fit(x_train, y_train)

            # Measure accuracy.
            y_pred = reg.predict(x_test)
            y_pred = np.reshape(y_pred, (y_pred.shape[0], 1))
            accuracy += np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred))

        return accuracy / n_splits if not relative else HappinessDataset.high_dim_TDP / (accuracy / n_splits)

    def compute_separability_metric(self, features: np.ndarray) -> float:
        """
        Computes separability metric for this dataset.
        Note: Assumes classification as domain task.
        :param features: Coordinates of low-dimensional projection.
        :return: Normalized score between 0 and 1 indicating how well labels are separated in low-dim. projection.
        """

        # 1. Cluster projection with number of classes.
        clusterer = hdbscan.HDBSCAN(alpha=1.0, metric='euclidean').fit(features)

        # 2. Calculate Silhouette score.
        try:
            silhouette_score = sklearn.metrics.silhouette_score(
                X=self.labels().values.reshape(-1, 1), metric='euclidean', labels=clusterer.labels_
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
    def get_attributes_data_types() -> dict:
        return {
            "happiness_rank": {"supertype": "numerical", "type": "discrete"},
            "happiness_score": {"supertype": "numerical", "type": "continous"},
            "economy": {"supertype": "numerical", "type": "continous"},
            "family": {"supertype": "numerical", "type": "continous"},
            "health": {"supertype": "numerical", "type": "continous"},
            "freedom": {"supertype": "numerical", "type": "continous"},
            "generosity": {"supertype": "numerical", "type": "continous"},
            "corruption": {"supertype": "numerical", "type": "continous"},
            "dystopia_residual": {"supertype": "numerical", "type": "continous"},
            "cellular_subscriptions": {"supertype": "numerical", "type": "discrete"},
            "surplus_deficit_gdp": {"supertype": "numerical", "type": "continous"},
            "inflation_rate": {"supertype": "numerical", "type": "continous"},
            "population": {"supertype": "numerical", "type": "discrete"},
            "record_name": {"supertype": "categorical", "type": "nominal"}
        }

    @staticmethod
    def sort_dataframe_columns_for_frontend(df: pd.DataFrame) -> pd.DataFrame:
        df_sorted: pd.DataFrame = df.reindex(sorted(df.columns), axis=1)
        df_sorted = df_sorted[['record_name'] + [col_name for col_name in df if col_name not in ['record_name']]]

        return df_sorted
