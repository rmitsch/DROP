import os
from enum import Enum
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np
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
        self._df: pd.DataFrame = None
        self._preprocessed_feature_cols: list = None

        super().__init__(storage_path=storage_path)

    def _load_data(self):
        target_col: str = "vote_average"

        # Load and prepare data.
        movies_metadata: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=self._storage_path + "/movies_metadata.csv", low_memory=False,
            dtype={"id": int, "popularity": float}
        )

        # Wrap title in link to IMDB page.
        movies_metadata["title"] = (
            "<a target=_blank " +
            "href='" + "https://www.imdb.com/title/" + movies_metadata.imdb_id + "/'>" +
            movies_metadata.title +
            "</a>"
        )

        # Discard unnecessary columns.
        movies_metadata = movies_metadata.drop(columns=[
            "belongs_to_collection", "homepage", "imdb_id", "poster_path", "status", "video", "original_title",
            "original_language", "vote_count", "overview"
        ]).set_index("id")

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
        ).drop(columns=["genres"]).sort_values(by="revenue", ascending=False)

        # Limit to most popular movies.
        movies_metadata = movies_metadata.head(300)

        # Join with story keywords.
        movies_keywords: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=self._storage_path + "/movies_keywords.csv", low_memory=False, error_bad_lines=False,
            dtype={"id": int}
        ).set_index("id")
        movies_keywords.keywords = movies_keywords.keywords.astype("str").str.replace("nan", "[]").apply(
            lambda vals: [val["name"] for val in convert(vals)]
        )
        movies_metadata = movies_metadata.join(movies_keywords)

        # Compute movie age in years.
        movies_metadata["age"] = (
            datetime.datetime.now() - pd.to_datetime(movies_metadata.release_date)
        ).dt.total_seconds() / (24 * 60 * 60) / 365.25
        movies_metadata = movies_metadata.drop(columns="release_date")

        self._df = movies_metadata
        return {
            "features": movies_metadata.drop([target_col], axis=1),
            "labels": movies_metadata[target_col]
        }

    def _preprocess_hd_features(self):
        features: pd.DataFrame = self._data["features"].copy(deep=True).drop(columns=[
            # Drop textual data since we rely only on numerical on categorical data (for now).
            "tagline", "title"
        ])

        for cat_col in ("production_companies", "production_countries", "spoken_languages", "keywords"):
            features = features.join(
                pd.get_dummies(features[cat_col].apply(pd.Series).stack(), prefix=cat_col).sum(level=0),
                how="left"
            ).drop(columns=[cat_col])

        self._preprocessed_feature_cols = features.columns.values.tolist()

        return StandardScaler().fit_transform(features.values)

    def persist_records(self):
        filepath: str = self._storage_path + '/records.csv'

        if not os.path.isfile(filepath):
            df: pd.DataFrame = self._df.copy(deep=True)
            df["record_name"] = df.title
            df.drop(
                columns=["spoken_languages", "title", "tagline"]
            ).to_csv(
                path_or_buf=filepath, index=False
            )

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
        n_splits: int = 3

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

    def compute_distance_matrix(self) -> np.ndarray:
        """
        Compute distance matrix for all records in high-dimensional dataset.
        Note that some subjective decisions regarding feature weights are made in order to achieve appropriate distance
        computations.
        :return: Distance matrix as numpy.ndarry.
        """

        features: np.ndarray = self._preprocessed_hd_features
        feature_cols: list = self._preprocessed_feature_cols

        ####################################################
        # Compute distances for real-valued columns.
        ####################################################

        col_idx: list = [feature_cols.index(col) for col in ["popularity", "revenue", "age"]]
        distances: np.ndarray = cdist(features[:, col_idx], features[:, col_idx], "euclidean")
        # Normalize distances as workaround for merging with distances computed with other metrics (specifically
        # Jaccard).
        distances /= np.max(distances)
        distances *= 0.5

        ####################################################
        # Compute distances for content keywords.
        ####################################################

        col_idx = [
            feature_cols.index(col) for col in
            [col for col in feature_cols if col.startswith("keywords")]
        ]
        distances += cdist(features[:, col_idx], features[:, col_idx], "jaccard") * 1

        ####################################################
        # Compute distances for genres.
        ####################################################

        col_idx = [feature_cols.index(col) for col in MovieDataset.genres]
        distances += cdist(features[:, col_idx], features[:, col_idx], "jaccard") * 1

        return distances

    @staticmethod
    def get_attributes_data_types() -> dict:
        supertypes: Enum = InputDataset.DataSupertypes
        subtypes: Enum = InputDataset.DataSubtypes

        return {
            **{
                "budget": {"supertype": supertypes.NUMERICAL.value, "type": subtypes.DISCRETE.value},
                "popularity": {"supertype": supertypes.NUMERICAL.value, "type": subtypes.CONTINOUS.value},
                "production_companies": {"supertype": supertypes.CATEGORICAL.value, "type": subtypes.NOMINAL.value},
                "production_countries": {"supertype": supertypes.CATEGORICAL.value, "type": subtypes.NOMINAL.value},
                "age": {"supertype": supertypes.NUMERICAL.value, "type": subtypes.CONTINOUS.value},
                "revenue": {"supertype": supertypes.NUMERICAL.value, "type": subtypes.CONTINOUS.value},
                "runtime": {"supertype": supertypes.NUMERICAL.value, "type": subtypes.CONTINOUS.value},
                "spoken_languages": {"supertype": supertypes.CATEGORICAL.value, "type": subtypes.NOMINAL.value},
                "tagline": {"supertype": supertypes.CATEGORICAL.value, "type": subtypes.NOMINAL.value},
                "title": {"supertype": supertypes.CATEGORICAL.value, "type": subtypes.NOMINAL.value},
                "record_name": {"supertype": supertypes.CATEGORICAL.value, "type": subtypes.NOMINAL.value},
                "vote_average": {"supertype": supertypes.NUMERICAL.value, "type": subtypes.CONTINOUS.value},
                "keywords": {"supertype": supertypes.CATEGORICAL.value, "type": subtypes.NOMINAL.value}
            },
            **{
                genre: {"supertype": supertypes.NUMERICAL.value, "type": subtypes.DISCRETE.value}
                for genre in MovieDataset.genres
            }
        }

    @staticmethod
    def sort_dataframe_columns_for_frontend(df: pd.DataFrame) -> pd.DataFrame:
        df_sorted: pd.DataFrame = df.reindex(sorted(df.columns), axis=1)
        df_sorted = df_sorted[['record_name'] + [col_name for col_name in df if col_name not in ['record_name']]]

        return df_sorted
