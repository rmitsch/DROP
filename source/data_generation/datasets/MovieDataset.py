import csv
import math
import os
import warnings

import hdbscan
import psutil
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import sklearn
from data_generation.datasets import InputDataset
import xgboost
import ast
import datetime


class MovieDataset(InputDataset):
    """
    Data on
    """

    genres: list = [
        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign',
        'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'
    ]

    def __init__(self, storage_path: str):
        self._df = None
        super().__init__(storage_path=storage_path)

    def _load_data(self):
        target_col: str = "vote_average"

        # Load and prepare data.
        movies_metadata: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=self._storage_path + "/movies_metadata.csv", low_memory=False,
            dtype={"id": int, "popularity": float}
        ).drop(columns=[
            "belongs_to_collection", "homepage", "imdb_id", "poster_path", "status", "video", "original_title",
            "original_language"
        ]).set_index("id").sort_values(by="popularity", ascending=False)

        # Simplify JSON lists - dispose of IDs.
        def convert(x: str) -> list:
            res = ast.literal_eval(x)
            return res if type(res) not in (float, int, bool) else []
        for col in ("genres", "production_companies", "production_countries", "spoken_languages"):
            movies_metadata[col] = movies_metadata[col].astype("str").str.replace("nan", "[]").apply(
                lambda vals: [val["name"] for val in convert(vals)]
            )

        # Move genres into separate column.
        movies_metadata = movies_metadata.join(
            pd.get_dummies(movies_metadata.genres.apply(pd.Series).stack()).sum(level=0), how="inner"
        ).drop(columns=["genres"])

        # Limit to most popular movies.
        movies_metadata = movies_metadata.head(1000)

        # Join with story keywords.
        movies_keywords: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=self._storage_path + "/movies_keywords.csv", low_memory=False, error_bad_lines=False,
            dtype={"id": int}
        ).set_index("id")
        movies_keywords.keywords = movies_keywords.keywords.astype("str").str.replace("nan", "[]").apply(
            lambda vals: [val["name"] for val in convert(vals)]
        )
        movies_metadata = movies_metadata.join(movies_keywords)
        self._df = movies_metadata

        # todo
        #  - finish MovieDataset implementation in backend
        #    - drop production companies?
        #    - next step: TDP computation
        #  - generate embeddings
        #  - integrate in frontend
        #  - evaluate
        # approach: keep data in frontend format (i.e. not all categorical columns expanded into normalized columns) and
        # persists like this. in consequence:
        # prepare data for classification and distance matrix computation in preprocess_data() -> normalizing, one-hot
        # encoding etc. use weighting after one-hot encoding! -> important for suitable distance matrixes.
        # alternatively: apply weighting in compute_distance_matrix() - feature weighting not necessary for TDP
        # computation.
        # hence: implement compute_target_domain_performance(), compute_separability_matrix() and
        # compute_distance_matrix() in MovieDataset.

        return {
            "features": movies_metadata.drop([target_col], axis=1),
            "labels": movies_metadata[target_col]
        }

    def _preprocess_hd_features(self):
        features: pd.DataFrame = self._data["features"].copy(deep=True)
        features["age"] = (
            datetime.datetime.now() - pd.to_datetime(features.release_date)
        ).dt.total_seconds() / (24 * 60 * 60)
        features = features.drop(columns="release_date")

        for cat_col in ("production_companies", "production_countries", "spoken_languages", "keywords"):
            features = features.join(
                pd.get_dummies(features[cat_col].apply(pd.Series).stack(), prefix=cat_col).sum(level=0),
                how="left"
            ).drop(columns=[cat_col])

        return StandardScaler().fit_transform(features.drop(columns=["overview", "tagline", "title"]).values)

    def persist_records(self):
        filepath: str = self._storage_path + '/movie_records.csv'

        if not os.path.isfile(filepath):
            df: pd.DataFrame = self._df.copy(deep=True)
            df["record_name"] = df.title
            df.to_csv(path_or_buf=filepath, index=False)

    def _compute_target_domain_performance(self, features: np.ndarray) -> float:
        """
        Auxiliary function providing TDP computation for both HD/original and relative case, since for HappinessDataset
        both implementations are identical (both rely exclusively on numerical features).
        :param features: Feature matrix as numeric numpy array.
        :return: Target domain performance.
        """

        labels: np.ndarray = np.reshape(self.labels().values, (-1, 1))

        # Loop through stratified splits, average prediction accuracy over all splits.
        relative_error: float = 0
        n_splits: int = 10

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
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(features, labels, test_size=0.3)
            reg.fit(x_train, y_train)

            # Measure accuracy.
            y_pred: np.ndarray = np.reshape(reg.predict(x_test), (-1, 1))
            relative_error += np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred)) / y_pred.mean()

        return relative_error / n_splits

    def compute_hd_target_domain_performance(self) -> float:
        return self._compute_target_domain_performance(np.nan_to_num(self._preprocessed_hd_features))

    def compute_relative_target_domain_performance(self, features: np.ndarray):
        # TDP is measured as relative error here, so we divide performance in HD by that in LD space.
        return self._hd_target_domain_performance / self._compute_target_domain_performance(features)

    @staticmethod
    def get_attributes_data_types() -> dict:
        pass

    @staticmethod
    def sort_dataframe_columns_for_frontend(df: pd.DataFrame) -> pd.DataFrame:
        pass
