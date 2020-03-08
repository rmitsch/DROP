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
    Extended World Happiness Report dataset.
    Sources: https://www.kaggle.com/unsdsn/world-happiness, Christoph Kralj.
    """

    # Measured TDP for Happiness dataset with regression task in terms of RMSE.
    high_dim_TDP = 0.31

    def __init__(self, storage_path: str):
        self._df: pd.DataFrame = None
        super().__init__(storage_path=storage_path)

    def _load_data(self) -> dict:
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

    def _preprocess_features(self) -> np.ndarray:
        return StandardScaler().fit_transform(self._data["features"].values)

    def persist_records(self):
        filepath: str = self._storage_path + '/happiness_records.csv'

        if not os.path.isfile(filepath):
            df: pd.DataFrame = self._df.copy(deep=True)
            df["record_name"] = df.index.values
            df.to_csv(path_or_buf=filepath, index=False)

    def compute_target_domain_performance(self, features: np.ndarray) -> float:
        """
        Auxiliary function providing TDP computation for both HD/original and relative case, since for HappinessDataset
        both implementations are identical (both rely exclusively on numerical features).
        :param features: Feature matrix as numeric numpy array.
        :return: Target domain performance.
        """

        labels: np.ndarray = np.reshape(self.labels().values, (-1, 1))

        # Loop through stratified splits, average prediction accuracy over all splits.
        accuracy: float = 0
        n_splits: int = 100

        # Apply boosting w/o further preprocessing to predict target values.
        reg: xgboost.XGBRegressor = xgboost.XGBRegressor(
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
            y_pred: np.ndarray = reg.predict(x_test)
            y_pred: np.ndarray = np.reshape(y_pred, (y_pred.shape[0], 1))
            accuracy += np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred))

        return accuracy / n_splits

    def compute_hd_target_domain_performance(self) -> float:
        return self.compute_target_domain_performance(self._preprocessed_features)

    def compute_relative_target_domain_performance(self, features: np.ndarray) -> float:
        return self.compute_target_domain_performance(features) / self._hd_target_domain_performance

    def compute_separability_metric(
            self,
            features: np.ndarray,
            cluster_metric: str = "euclidean",
            silhouette_metric: str = "euclidean",
            min_cluster_size: int = 2
    ) -> float:
        return super().compute_separability_metric(features, cluster_metric, silhouette_metric, min_cluster_size)

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
