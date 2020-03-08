import csv
import math
import os
import warnings

import hdbscan
import psutil
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import sklearn
import json
from data_generation.datasets import InputDataset
import xgboost
import ast


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
        # Load and prepare data. Limit to 500 most popular movies.
        movies_metadata: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=self._storage_path + "/movies_metadata.csv", low_memory=False,
            dtype={"id": int, "popularity": float}
        ).drop(columns=[
            "belongs_to_collection", "homepage", "imdb_id", "poster_path", "status", "video", "original_title",
            "original_language"
        ]).set_index("id").sort_values(by="popularity", ascending=False).head(500)

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
            "features": movies_metadata.drop(["revenue"], axis=1),
            "labels": movies_metadata.revenue
        }

    def _preprocess_features(self):
        pass

    def features(self):
        pass

    def labels(self):
        pass

    def persist_records(self):
        filepath: str = self._storage_path + '/movie_records.csv'

        if not os.path.isfile(filepath):
            df: pd.DataFrame = self._df.copy(deep=True)
            df["record_name"] = df.title
            df.to_csv(path_or_buf=filepath, index=False)

    def compute_hd_target_domain_performance(self) -> float:
        pass

    def compute_relative_target_domain_performance(self, features: np.ndarray):
        pass

    @staticmethod
    def get_attributes_data_types() -> dict:
        pass

    @staticmethod
    def sort_dataframe_columns_for_frontend(df: pd.DataFrame) -> pd.DataFrame:
        pass
