import logging
from contextlib import contextmanager
import sys
import os

import pandas as pd
import lime
import numpy as np
import pandas
import psutil
import sklearn
from flask import Flask
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from skrules import SkopeRules
import lime.lime_tabular


class Utils:
    """
    Class for various, non-essential tasks.
    """

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
        Preprocesses embedding metadata for use in sklearn-style predictors and LIME.
        :param metadata_template:
        :param embeddings_metadata:
        :return: (1) Data frame holding all features (categorical ones are coded numerically); (2) data frame holding
        labels; (3) list of lists translating encoding indices to original category names.
        """

        features_names = [item["name"] for item in metadata_template["hyperparameters"]]
        features_df = embeddings_metadata[features_names].copy(deep=True)

        # Encode categorical values numerically.
        encoded_categorical_values_to_category_names = {}
        for i, feature in enumerate(metadata_template["hyperparameters"]):
            if feature["type"] == "categorical":
                # Map encoded numerical value to original categorical value.
                encoded_categorical_values_to_category_names[i] = [
                    value.decode('UTF-8') for value in features_df[feature["name"]].unique()
                ]
                # Encode categorical values numerically.
                features_df[feature["name"]] = LabelEncoder().fit_transform(
                    features_df[feature["name"]].values
                )

                # Encode as one-hot (note that 0 and 1 might be reverse (i. e. 0 used for active and 1 for inactive).
                # Doesn't make a difference for models though as long as this encoding used in a consistent manner.
                features_df = pandas.concat([
                    features_df,
                    pandas.get_dummies(
                        features_df[feature["name"]],
                        prefix="cat_" + feature["name"]
                    )
                ], axis=1).drop([feature["name"]], axis=1)

                # Rename columns according to actual values.
                features_df = features_df.rename(
                    index=str,
                    columns={
                        "cat_" + feature["name"] + "_" + str(i): feature["name"] + "_" + value
                        for i, value in enumerate(encoded_categorical_values_to_category_names[i])
                    }
                )

        return features_df, \
               embeddings_metadata[metadata_template["objectives"]], \
               encoded_categorical_values_to_category_names

    @staticmethod
    def fit_random_forest_regressors(
            metadata_template: dict, features_df: pandas.DataFrame, labels_df: pandas.DataFrame
    ):
        """
        Fits sklearn random forest regressors to specified dataframe holding embedding data for each objective defined.
        in metadata_template["objectives"].
        :param metadata_template:
        :param features_df: Preprocessed dataframes with feature values.
        :param labels_df: Preprocessed dataframes with label values.
        :return: Dictionary with objective -> regressor.
        """

        # Compute global surrogate models as basis for LIME's local explainers.
        return {
            objective: sklearn.ensemble.RandomForestRegressor(
                n_estimators=100,
                n_jobs=psutil.cpu_count(logical=False)
            ).fit(
                X=features_df,
                y=labels_df[objective]
            )
            for objective in metadata_template["objectives"]
        }

    @staticmethod
    def initialize_lime_explainers(metadata_template: dict, features_df: pandas.DataFrame):
        """
        Initialize LIME explainers with global surrogate models (as produced by Utils.fit_random_forest_regressors).
        :param metadata_template:
        :param features_df: Preprocessed dataframes with feature values.
        :return: Dictionary with objective -> LIME explainer for this objective.
        """
        ##################################
        # 1. Prepare information on
        # categorical values for LIME.
        ##################################

        # Fetch names of categorical hyperparameters.
        categorical_hyperparameters = [
            dict["name"]
            for index, dict in enumerate(metadata_template["hyperparameters"])
            if dict["type"] == "categorical"
        ]

        # A bit hacky: Find categorical attributes by picking columns starting with "CATEGORICAL ATTRIBUTE"_X.
        categorical_feature_indices = [
            i for i, col_name in enumerate(features_df.columns)
            # col_name_parts[0] is the prefix, i. e. the name of the original (potentially categorical) hyperparameter.
            if len(col_name.split("_")) == 2 and col_name.split("_")[0] in categorical_hyperparameters
        ]

        ##################################
        # 2. Initialize explainers for
        # each objective.
        ##################################

        return {
            objective: lime.lime_tabular.LimeTabularExplainer(
                training_data=features_df.values,
                feature_names=features_df.columns,
                class_names=metadata_template["objectives"],
                # discretize_continuous=True,
                # discretizer='decile',
                # Specify categorical hyperparameters.
                categorical_features=categorical_feature_indices,
                verbose=False,
                mode='regression'
            )
            for objective in metadata_template["objectives"]
        }

    @staticmethod
    def init_flask_app(frontend_path: str):
        """
        Initialize Flask app.
        :param frontend_path: Path to directory containing frontend project.
        :return: App object.
        """
        flask_app = Flask(
            __name__,
            template_folder=os.path.join(frontend_path, 'source/templates'),
            static_folder=os.path.join(frontend_path, 'source/static')
        )

        # Define version.
        flask_app.config["VERSION"] = "0.23.2"

        # Store metadata template. Is assembled once in /get_metadata.
        flask_app.config["METADATA_TEMPLATE"] = None

        # Store dataframe holding embedding metadata and related data.
        flask_app.config["EMBEDDING_METADATA"] = {
            "original": None,
            "features_preprocessed": None,
            "features_categorical_encoding_translation": None,
            "labels": None
        }

        # Store name of current dataset and kernel. Note that these values is only changed at call of /get_metadata.
        # Use t-SNE on happiness dataset as default.
        flask_app.config["DATASET_NAME"] = None
        flask_app.config["DR_KERNEL_NAME"] = "tsne"
        flask_app.config["FULL_FILE_NAME"] = "happiness"

        # For storage of global, unrestricted model used by LIME for local explanations.
        # Has one global regressor for each possible objective.
        flask_app.config["GLOBAL_SURROGATE_MODEL"] = {}
        flask_app.config["LIME_EXPLAINER"] = None

        return flask_app

    @staticmethod
    def extract_rules(
            bin_label: pd.Interval, features_df: pd.DataFrame, class_encodings: pd.DataFrame, objective_name: str
    ) -> list:
        """
        Extract rules with given data and bin label.
        :param bin_label:
        :param features_df:
        :param class_encodings:
        :param objective_name:
        :return: List of extracted rules: (rule, precision, recall, support, result from, result to).
        """
        rules_clf = SkopeRules(
            max_depth_duplication=None,
            n_estimators=30,
            precision_min=0.2,
            recall_min=0.01,
            feature_names=features_df.columns.values,
            n_jobs=1
        )
        rules_clf.fit(features_df.values, class_encodings[objective_name] == bin_label)

        return [
            (rule[0], rule[1][0], rule[1][1], rule[1][2], bin_label.left, bin_label.right)
            for rule in rules_clf.rules_
        ]

    @staticmethod
    def prepare_binned_original_dataset(dataset_name: str, bin_count: int = 5) -> pd.DataFrame:
        """
        Prepares original dataset for detail view in frontend by loading and binning it.
        :param dataset_name:
        :param bin_count:
        :return: Dataframe with all original attributes plus binned versions of it for numerical attributes.
        """

        df: pd.DataFrame = pandas.read_csv(
            filepath_or_buffer=os.getcwd() + "/../data/" + dataset_name + "_records.csv",
            delimiter=',',
            quotechar='"'
        )

        # Bin columns for histograms.
        for attribute in df.select_dtypes(include=[np.number]).columns:
            df[attribute + "#histogram"] = pd.cut(df[attribute], bins=bin_count).apply(lambda x: x.left)

        # Round float values.
        for numerical_col in df.select_dtypes(include='float64').columns:
            df[numerical_col] = df[numerical_col].round(decimals=3)

        return df

    @staticmethod
    def get_active_col_indices(metadata: pd.DataFrame, metadata_config: dict, embedding_id: int) -> list:
        """
        Auxiliary method for picking only those columns in metadata dataframe that are either numerical or "active" -
        i. e. currently: A metric column.
        :param metadata:
        :param metadata_config:
        :param embedding_id:
        :return: List of active columns in metadata dataframe.
        """

        cols = metadata.columns.values
        return [
            i for i
            in range(len(cols))
            if "metric_" not in cols[i] or
               cols[i] == "metric_" + str(
                metadata_config["original"].loc[[embedding_id]].metric.values[0]
            )[2:-1]
        ]

    @staticmethod
    def get_metadata_template(dr_kernel_config: dict) -> dict:
        """
        Assembles metadata template (i. e. which hyperparameters and objectives are available).
        :param dr_kernel_config:
        :return: Dictionary: {"hyperparameters": [...], "objectives": [...]}
        """

        return {
            "hyperparameters": dr_kernel_config["parameters"],
            "objectives": [
                "runtime",
                "r_nx",
                "stress",
                "classification_accuracy",
                "separability_metric"
            ]
        }