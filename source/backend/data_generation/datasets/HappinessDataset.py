import csv
import math
import os
import warnings

import psutil
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import re
from sklearn.preprocessing import StandardScaler
import sklearn
from backend.data_generation.datasets import InputDataset
import xgboost


class HappinessDataset(InputDataset):
    """
    Extended Kaggle happiness dataset.
    """

    def __init__(self):
        self._df = None
        super().__init__()

    def _load_data(self):
        df = pd.read_csv(filepath_or_buffer="../../../data/happiness_2017.csv").drop(
            ["map_reference", "biggest_official_language", "gdp_per_capita[$]"], axis=1
        ).set_index("country")
        df = df.rename(columns={col: re.sub(r'\[.*\]', '', col) for col in df.columns})
        df = df.dropna(axis='columns')
        df["happiness_level"] = df.happiness_score.apply(lambda x: math.ceil(max(x - 4, 0)))
        self._df = df

        return {
            "features": df.drop(["happiness_score", "happiness_level", "happiness_rank"], axis=1),
            "labels": df.happiness_level
        }

    def features(self):
        return self._data["features"]

    def labels(self):
        return self._data["labels"]

    def _preprocess_features(self):
        return StandardScaler().fit_transform(self._data["features"].values)

    def persist_records(self, directory: str):
        filepath = directory + '/happiness_records.csv'

        if not os.path.isfile(filepath):
            features_df = self._df.copy(deep=True)
            features_df["record_name"] = features_df.index.values
            features_df = features_df.rename(columns={"happiness_level": "target_label"})
            features_df = features_df.drop(["happiness_score", "happiness_rank"], axis=1)
            features_df.to_csv(path_or_buf=filepath, index=False)

    def calculate_classification_accuracy(self, features: np.ndarray = None):
        # Set features, if not specified in function call.
        features = self.preprocessed_features() if features is None else features
        labels = self.labels()

        # Loop through stratified splits, average prediction accuracy over all splits.
        accuracy = 0
        n_splits = 20

        # Apply random forest w/o further preprocessing to predict class labels.
        clf = xgboost.XGBClassifier(
            n_estimators=1, n_jobs=1, nthread=1
        )

        for train_indices, test_indices in StratifiedShuffleSplit(
                n_splits=n_splits, test_size=0.7
        ).split(features, labels):
            # Train model.
            clf.fit(features[train_indices], labels[train_indices])

            # Predict test set.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predicted_labels = clf.predict(features[test_indices])

            # Measure accuracy.
            accuracy += sklearn.metrics.f1_score(labels[test_indices], predicted_labels, average="macro")

        print(accuracy / n_splits)

        return accuracy / n_splits
