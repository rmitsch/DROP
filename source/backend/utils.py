import logging
from contextlib import contextmanager
import sys
import os

import lime
import numpy
import pandas
import psutil
import sklearn
from flask import Flask
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import lime.lime_tabular


# Class for various, non-essential tasks.
class Utils:
    # Inactive constructor.
    def __init__(self):
        print("Constructing Utils. Shouldn't happen")

    # Construct logging object.
    @staticmethod
    def create_logger():
        # Logger set-up.
        # Source: https://docs.python.org/3/howto/logging-cookbook.html
        # Create global logger.
        Utils.logger = logging.getLogger('DROP')
        Utils.logger.setLevel(logging.DEBUG)
        # Create console handler.
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # Create formatter and add it to handlers.
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s | %(message)s')
        ch.setFormatter(formatter)
        # Add handlers to logger.
        Utils.logger.addHandler(ch)

        # Return logger object.
        return Utils.logger

    @staticmethod
    def extract_decision_tree_structure(clf: DecisionTreeRegressor, features: list, labels: list, node_index: int = 0):
        """
        Return textual structure for generated decision tree.
        Source: https://planspace.org/20151129-see_sklearn_trees_with_d3/.
        :return: Textual representation of regression tree.
        """

        node = {}

        if clf.tree_.children_left[node_index] == -1:  # indicates leaf
            node['name'] = " | ".join([
                                          label + ": " +
                                          str(round(clf.tree_.value[node_index][i][0], 3))
                                          for i, label in enumerate(labels[0])
                                          ])

        else:
            feature = features[clf.tree_.feature[node_index]]
            threshold = clf.tree_.threshold[node_index]
            node['name'] = '{} > {}'.format(feature, round(threshold, 3))
            left_index = clf.tree_.children_left[node_index]
            right_index = clf.tree_.children_right[node_index]

            node['children'] = [
                Utils.extract_decision_tree_structure(clf, features, labels, right_index),
                Utils.extract_decision_tree_structure(clf, features, labels, left_index)
            ]

        return node

    @staticmethod
    @contextmanager
    def suppress_stdout():
        """
        Surpresses output by creating a context object.
        Source: https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
        :return:
        """
        with open(os.devnull, "w") as devnull:
            old_stderr = sys.stderr
            old_stdout = sys.stdout
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    @staticmethod
    def preprocess_embedding_metadata_for_predictor(metadata_template: dict, embeddings_metadata: pandas.DataFrame):
        """
        Preprocesses embedding metadata for use in sklearn-style predictors.
        :param metadata_template:
        :param embeddings_metadata:
        :return: (1) Data frame holding all features (categorical ones are coded numerically); (2) data frame holding
        labels.
        """

        features_names = [item["name"] for item in metadata_template["hyperparameters"]]
        features_df = embeddings_metadata[features_names]

        # Encode categorical values numerically.
        for feature in metadata_template["hyperparameters"]:
            if feature["type"] != "numeric":
                features_df[feature["name"]] = LabelEncoder().fit_transform(
                    features_df[feature["name"]].values
                )

        return features_df, embeddings_metadata[metadata_template["objectives"]]

    @staticmethod
    def fit_random_forest_regressors(metadata_template: dict, embeddings_metadata: pandas.DataFrame):
        """
        Fits a sklearn random forest regressor to specified dataframe holding embedding data for all objectives defined
        in metadata_template["objectives"].
        :param metadata_template:
        :param embeddings_metadata:
        :return:
        """
        results = {}

        # Prepare data frame for sklearn predictors.
        features_df, labels_df = Utils.preprocess_embedding_metadata_for_predictor(
            metadata_template=metadata_template, embeddings_metadata=embeddings_metadata
        )

        # Compute global surrogate model as basis for LIME's local explainers.
        for objective in metadata_template["objectives"]:
            results[objective] = sklearn.ensemble.RandomForestRegressor(
                n_estimators=100,
                n_jobs=psutil.cpu_count(logical=False)
            ).fit(
                X=features_df,
                y=labels_df[objective]
            )

        return results

    @staticmethod
    def initialize_lime_explainer(metadata_template: dict, embeddings_metadata: pandas.DataFrame):
        """
        Initialize LIME explainer with global surrogate model (as produced by Utils.fit_random_forest_regressors).
        :param metadata_template:
        :param embeddings_metadata:
        :return:
        """
        # Prepare data frame for LIME.
        features_df, labels_df = Utils.preprocess_embedding_metadata_for_predictor(
            metadata_template=metadata_template, embeddings_metadata=embeddings_metadata
        )

        return lime.lime_tabular.LimeTabularExplainer(
            training_data=features_df.values,
            feature_names=[param["name"] for param in metadata_template["hyperparameters"]],
            class_names=metadata_template["objectives"],
            discretize_continuous=True,
            discretizer='decile',
            verbose=True,
            mode='regression'
        )

    @staticmethod
    def you_suck(rafi=[], ich=[]):
        complaints = ["i am tired", "i am hungry", "i suck so much!!!!!!", "i am a snob"]
        if "hungry" in complaints:
            ich.append(1)
        rafi.append(1)
        if len(rafi) > 1:
            print("Rafi sucks")
        else:
            print("I am the best")


if __name__ == '__main__':
    Utils.you_suck()
