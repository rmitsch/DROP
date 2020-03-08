import logging
from contextlib import contextmanager
import sys
import os
import dropbox
from dropbox.files import WriteMode as DropboxWriteMode
from typing import List

import pandas as pd
import numpy as np
import pandas
from flask import Flask
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgbm
from skrules import SkopeRules


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
        Utils.logger = logging.getLogger('TALE')
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
        Preprocesses embedding metadata for use in sklearn-style predictors and explainers.
        :param metadata_template:
        :param embeddings_metadata:
        :return: (1) Data frame holding all features (categorical ones are coded numerically); (2) data frame holding
        labels; (3) list of lists translating encoding indices to original category names.
        """

        features_names: list = [item["name"] for item in metadata_template["hyperparameters"]]
        features_df: pd.DataFrame = embeddings_metadata[features_names].copy(deep=True)

        # Encode categorical values numerically.
        encoded_categorical_values_to_category_names = {}
        for i, feature in enumerate(metadata_template["hyperparameters"]):
            if feature["type"] == "categorical":
                # Map encoded numerical value to original categorical value.
                encoded_categorical_values_to_category_names[i] = [
                    value.decode('UTF-8') for value in features_df[feature["name"]].unique()
                ]
                # Encode categorical values numerically.
                features_df[feature["name"]] = LabelEncoder().fit_transform(features_df[feature["name"]].values)

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

        return (
            features_df,
            embeddings_metadata[metadata_template["objectives"]],
            encoded_categorical_values_to_category_names
        )

    @staticmethod
    def fit_surrogate_regressor(
            metadata_template: dict, features_df: pandas.DataFrame, labels_df: pandas.DataFrame
    ):
        """
        Fits regressors to specified dataframe holding embedding data for each objective defined in
        metadata_template["objectives"].
        :param metadata_template:
        :param features_df: Preprocessed dataframes with feature values.
        :param labels_df: Preprocessed dataframes with label values.
        :return: Dictionary with objective -> regressor.
        """

        objectives: list = list(metadata_template["objectives"])

        df: pd.DataFrame = pd.concat(
            [features_df.reset_index(), labels_df.reset_index().drop(columns=["id"])], axis=1
        ).set_index("id").sample(frac=1)
        n_train: int = int(len(df) * 0.8)
        train_df: pd.DataFrame = df.head(n_train)
        test_df: pd.DataFrame = df.tail(len(df) - n_train)

        # Compute global surrogate models as basis for local explainers.
        return {
            objective: lgbm.LGBMRegressor(
                boosting_type="gbdt",
                n_estimators=100,
                num_iterations=5000,
                learning_rate=10e-4,
                silent=True
            ).fit(
                train_df.drop(columns=objectives),
                train_df[objective],
                eval_set=[(test_df.drop(columns=objectives), test_df[objective])],
                eval_metric='l2',
                early_stopping_rounds=1000,
                verbose=False
            )
            for objective in metadata_template["objectives"]
        }

    @staticmethod
    def init_flask_app(argv: List[str]):
        """
        Initialize Flask app.
        :param argv: List of passed arguments.
        :return: App object.
        """

        # Get path to frontend folder.
        assert len(argv) >= 3, "Paths to frontend and data storage have to be passed."
        frontend_path: str = sys.argv[1]

        # Init Flask app.
        flask_app: Flask = Flask(
            __name__,
            template_folder=os.path.join(frontend_path, 'source/templates'),
            static_folder=os.path.join(frontend_path, 'source/static')
        )

        # Define version.
        flask_app.config["VERSION"] = "0.25.2"

        # Store path to data storage location.
        flask_app.config["STORAGE_PATH"] = sys.argv[2] + "/"

        # Initialize Dropbox access, if token was provided.
        if len(sys.argv) == 4:
            print(
                "Warning: For persistent result storage both the experiment name and the Dropbox OAuth 2 token have to "
                "be provided."
            )
        if len(sys.argv) >= 5:
            flask_app.config["EXPERIMENT_NAME"] = argv[3]
            flask_app.config["DROPBOX"]: DropboxAccount = DropboxAccount(argv[4])

        # Store metadata template. Is assembled once in /get_metadata.
        flask_app.config["METADATA_TEMPLATE"] = None

        # Set cache directory.
        flask_app.config["CACHE_ROOT"] = "/tmp"

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

        # For storage of global, unrestricted model used by local explanations.
        # Has one global regressor for each possible objective.
        flask_app.config["GLOBAL_SURROGATE_MODEL"] = {}

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
        rules_clf: SkopeRules = SkopeRules(
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
    def prepare_binned_original_dataset(storage_path: str, dataset_name: str, bin_count: int = 5) -> pd.DataFrame:
        """
        Prepares original dataset for detail view in frontend by loading and binning it.
        :param storage_path:
        :param dataset_name:
        :param bin_count:
        :return: Dataframe with all original attributes plus binned versions of it for numerical attributes.
        """

        df: pd.DataFrame = pandas.read_csv(
            filepath_or_buffer=storage_path + "/" + dataset_name + "_records.csv",
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
        Auxiliary method for picking only those columns in metadata dataframe that are either numerical or "active" for
        a specific embedding. Currently variant: Distance metric columns.
        :param metadata:
        :param metadata_config:
        :param embedding_id:
        :return: List of active columns in metadata dataframe.
        """

        cols: list = metadata.columns.values
        return [
            i for i
            in range(len(cols))
            if "metric_" not in cols[i] or
            cols[i] == "metric_" + str(
                metadata_config["original"].loc[[embedding_id]].metric.values[0]
            )[2:-1]
        ]

    @staticmethod
    def gather_column_activity(features: pd.DataFrame, dim_red_kernel_parameters: dict) -> pd.DataFrame:
        """
        Auxiliary method for picking only those columns in metadata dataframe that are either numerical or "active" for
        all embeddings. Currently variant: Distance metric columns.
        :param features: Dataframe containing dataset features (objectives not required).
        :param dim_red_kernel_parameters:
        :return: Dataframe with active column indices and names per record.
        """

        # Compose indices of active columns as set of indices of numerical columns + set of indices of active one-hot
        # encoded categorical columns.
        cols: list = list(features.columns)
        # Get names and indices of numeric columns.
        num_cols: list = [col for col in dim_red_kernel_parameters if col["type"] == "numeric"]
        num_col_indices: list = [cols.index(feat["name"]) for feat in num_cols]

        # Get names of categorical columns. Categorical columns' activity depends on the record (e.g. whether distance
        # metric == 'euclidean' or 'cosine', hence we compile the corresponding columns and names for each record.
        cat_cols: list = [col for col in dim_red_kernel_parameters if col["type"] == "categorical"]

        # Initialize dataframe for feature activities.
        active_columns: pd.DataFrame = pd.DataFrame({
            **{"id": features.index.values},
            **{col["name"]: None for col in cat_cols},
            **{col["name"] + "_idx": None for col in cat_cols}
        }).set_index("id")

        # Assume one-hot encoding with '_' as concatenating element; transform one-hot encoded columns into a single
        # one.
        for col in cat_cols:
            for val in col["values"]:
                onehot_colname: str = col["name"] + "_" + val
                active_columns.loc[features[onehot_colname] == 1, col["name"]] = val
                active_columns.loc[features[onehot_colname] == 1, col["name"] + "_idx"] = cols.index(onehot_colname)

        # Merge all _idx columns into one; add numerical column indices.
        active_columns["idx"] = active_columns[
            [col for col in active_columns if "idx" in col]
        ].apply(list, axis=1).apply(lambda x: [*num_col_indices, *x])
        active_columns["cols"] = active_columns.idx.apply(lambda x: [cols[i] for i in x])

        # Change ID type to int.
        active_columns = active_columns.reset_index()
        active_columns.id = active_columns.id.astype(int)
        active_columns = active_columns.set_index("id")

        return active_columns[["idx", "cols"]]

    @staticmethod
    def get_active_col_indices_for_all_embeddings(available_cols: list, original_df: pd.DataFrame) -> list:
        """
        Auxiliary method for picking only those columns in metadata dataframe that are either numerical or "active" for
        all embeddings. Currently variant: Distance metric columns.
        :param available_cols:
        :param original_df:
        :return: List of active columns in metadata dataframe.
        """
        pass

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
                "target_domain_performance",
                "separability_metric"
            ]
        }


class DropboxAccount:
    """
    Providing tools for synchronizing files (specifically files holding test results) to Dropbox account.
    Based on https://stackoverflow.com/a/36851978.
    """
    def __init__(self, access_token: str):
        self._dbx: dropbox.Dropbox = dropbox.Dropbox(access_token)

    def upload_file(self, local_filepath: str, dropbox_filename: str):
        """
        Uploads a file to Dropbox using API v2.
        :param local_filepath: Local filepath.
        :param dropbox_filename: File's name in Dropbox.
        """
        with open(local_filepath, 'rb') as f:
            self._dbx.files_upload(f.read(), "/TALE-study/" + dropbox_filename, mode=DropboxWriteMode.overwrite)
