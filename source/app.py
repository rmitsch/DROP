import os
import sys

import sklearn
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import tables
import pandas
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from tables import *
import numpy
from sklearn.datasets import load_boston

from data_generation.datasets.InputDataset import InputDataset
from data_generation.dimensionality_reduction import DimensionalityReductionKernel
from utils import Utils
import dcor
import shap

frontend_path = sys.argv[1]
# Initialize logger.
logger = Utils.create_logger()
# Initialize flask app.
app = Utils.init_flask_app(frontend_path)


# root: Render HTML for start menu.
@app.route("/")
def index():
    return render_template("index.html", version=app.config["VERSION"])


@app.route('/get_metadata', methods=["GET"])
def get_metadata():
    """
    Reads metadata content (i. e. model parametrizations and objectives) of specified .h5 file.
    GET parameters:
        - datasetName with "dataset".
        - drKernelName with "drk".
    :return:
    """
    app.config["DATASET_NAME"] = InputDataset.check_dataset_name(request.args.get('datasetName'))
    app.config["DR_KERNEL_NAME"] = DimensionalityReductionKernel.check_kernel_name(request.args.get('drKernelName'))
    # Compile metadata template.
    if app.config["METADATA_TEMPLATE"] is None:
        get_metadata_template()

    # Build file name.
    file_name = os.getcwd() + "/../data/drop_" + app.config["DATASET_NAME"] + "_" + app.config["DR_KERNEL_NAME"] + ".h5"
    app.config["FULL_FILE_NAME"] = file_name

    # Open .h5 file, if dataset name and DR kernel name are valid and file exists.
    if app.config["DATASET_NAME"] is not None and \
            app.config["DR_KERNEL_NAME"] is not None and \
            os.path.isfile(file_name):
        ###################################################
        # Load dataset.
        ###################################################

        h5file = tables.open_file(filename=file_name, mode="r")
        # Cast to dataframe, then return as JSON.
        df = pandas.DataFrame(h5file.root.metadata[:]).set_index("id")
        # Close file.
        h5file.close()

        ###################################################
        # Preprocess and cache dataset.
        ###################################################

        app.config["EMBEDDING_METADATA"]["original"] = df
        app.config["EMBEDDING_METADATA"]["features_preprocessed"], \
        app.config["EMBEDDING_METADATA"]["labels"], \
        app.config["EMBEDDING_METADATA"]["features_categorical_encoding_translation"] = \
            Utils.preprocess_embedding_metadata_for_predictor(
                metadata_template=app.config["METADATA_TEMPLATE"], embeddings_metadata=df
            )

        ###################################################
        # Compute global surrogate models and initialize
        # corresponding LIME explainers.
        ###################################################

        # Compute regressor for each objective.
        app.config["GLOBAL_SURROGATE_MODELS"] = Utils.fit_random_forest_regressors(
            metadata_template=app.config["METADATA_TEMPLATE"],
            features_df=app.config["EMBEDDING_METADATA"]["features_preprocessed"],
            labels_df=app.config["EMBEDDING_METADATA"]["labels"]
        )

        # Initialize LIME explainers for each objective.
        app.config["EXPLAINERS"] = Utils.initialize_lime_explainers(
            metadata_template=app.config["METADATA_TEMPLATE"],
            features_df=app.config["EMBEDDING_METADATA"]["features_preprocessed"]
        )

        # Return JSON-formatted embedding data.
        return jsonify(df.drop(["b_nx"], axis=1).to_json(orient='index'))

    else:
        return "File/kernel does not exist.", 400


@app.route('/get_metadata_template', methods=["GET"])
def get_metadata_template():
    """
    Assembles metadata template (i. e. which hyperparameters and objectives are available).
    :return: Dictionary: {"hyperparameters": [...], "objectives": [...]}
    """

    app.config["METADATA_TEMPLATE"] = {
        "hyperparameters": DimensionalityReductionKernel.
            DIM_RED_KERNELS[app.config["DR_KERNEL_NAME"].upper()]["parameters"],
        "objectives": [
            "runtime",
            "r_nx",
            # "b_nx",
            "stress",
            "classification_accuracy",
            "separability_metric"
        ]
    }

    return jsonify(app.config["METADATA_TEMPLATE"])


@app.route('/get_surrogate_model_data', methods=["GET"])
def get_surrogate_model_data():
    """
    Yields structural data for surrogate model.
    GET parameters:
        - "modeltype": Model type can be specified with GET param (currently only decision tree with "tree" supported).
        - "objs": Objectives with objs=alpha,beta,...
        - "depth": Max. depth of decision tree with depth=x.
        - "ids": List of embedding IDs to consider, with ids=1,2,3,... Note: If "ids" is not specified, all embeddings
          are used to construct surrogate model.
    :return: Jsonified structure of surrogate model for DR metadata.
    """
    metadata_template = app.config["METADATA_TEMPLATE"]
    surrogate_model_type = request.args["modeltype"]
    objective_names = request.args["objs"].split(",")
    depth = int(request.args["depth"])
    ids = request.args.get("ids")

    # ------------------------------------------------------
    # 1. Check for mistakes in parameters.
    # ------------------------------------------------------

    if surrogate_model_type not in ["tree"]:
        return "Surrogate model " + surrogate_model_type + " is not supported.", 400

    for obj_name in objective_names:
        if obj_name not in metadata_template["objectives"]:
            return "Objective " + obj_name + " is not supported.", 400

    # ------------------------------------------------------
    # 2. Pre-select embeddings to use for surrogate model.
    # ------------------------------------------------------

    ids = list(map(int, ids.split(","))) if ids is not None else None
    features_df = app.config["EMBEDDING_METADATA"]["features_preprocessed"]
    labels_df = app.config["EMBEDDING_METADATA"]["labels"]

    import pandas as pd
    from skrules import SkopeRules
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(features_df.head(10))
        print(labels_df.head(10))

    clf = SkopeRules(
        max_depth_duplication=None,
        n_estimators=30,
        precision_min=0.2,
        recall_min=0.01,
        feature_names=features_df.columns.values
    )

    for label_name in labels_df.columns:
        print(label_name)

        class_encodings = pd.DataFrame(pd.qcut(labels_df[label_name], 5, range(5)).astype("int"))

        for i in range(1, 5):
            class_encodings[i] = 0
            vals = class_encodings[label_name] == i
            clf.fit(features_df.values, vals)

            # rule: (rule, precision, recall, number of affected records).
            print(i)
            for rule in clf.rules_:
                print(rule)
            print("***")

        print("------------------------")

    if ids is not None:
        features_df = features_df.iloc[ids]
        labels_df = labels_df.iloc[ids]

    # ------------------------------------------------------
    # 2. Build regressor.
    # ------------------------------------------------------

    # Fit decision tree.
    tree = DecisionTreeRegressor(max_depth=depth)
    tree.fit(features_df, labels_df)

    # ------------------------------------------------------
    # 3. Extract tree structure and return result.
    # ------------------------------------------------------

    # Extract tree structure and return as JSON.
    tree_structure = Utils.extract_decision_tree_structure(
        tree, features_df.columns, [objective_names]
    )
    return jsonify(tree_structure)


@app.route('/get_sample_dissonance', methods=["GET"])
def get_sample_dissonance():
    """
    Calculates and fetches variance/divergence of individual samples over all DR model parametrizations.
    GET parameters:
        - Distance function to use for determining neighbourhoods (not supported yet).
    :return:
    """
    file_name = app.config["FULL_FILE_NAME"]

    if os.path.isfile(file_name):
        h5file = open_file(filename=file_name, mode="r+")

        # ------------------------------------------------------
        # 1. Get metadata on numbers of models and samples.
        # ------------------------------------------------------

        # Use arbitrary model to fetch number of records/points in model.
        num_records = h5file.root.metadata[0][1]
        # Initialize numpy matrix for pointwise qualities.
        pointwise_qualities = numpy.zeros([h5file.root.metadata.nrows, num_records])

        # ------------------------------------------------------
        # 2. Iterate over models.
        # ------------------------------------------------------

        for model_pointwise_quality_leaf in h5file.walk_nodes("/pointwise_quality/", classname="CArray"):
            model_id = int(model_pointwise_quality_leaf._v_name[5:])
            pointwise_qualities[model_id - 1] = model_pointwise_quality_leaf.read().flatten()

        # Close file.
        h5file.close()

        # Reshape data to desired model_id:sample_id:value format.
        df = pandas.DataFrame(pointwise_qualities)
        df["model_id"] = df.index
        df = df.melt("model_id", var_name='sample_id', value_name="measure")

        # Return jsonified version of model x sample quality matrix.
        return df.to_json(orient='records')

    else:
        return "File does not exist.", 400


@app.route('/get_dr_model_details', methods=["GET"])
def get_dr_model_details():
    """
    Fetches data for DR model with specifie ID.
    GET parameters:
        - "id" for ID of DR embedding.
    :return: Jsonified structure of surrogate model for DR metadata.
    """

    embedding_id = int(request.args["id"])
    file_name = app.config["FULL_FILE_NAME"]
    high_dim_file_name = os.getcwd() + "/../data/" + app.config["DATASET_NAME"] + "_records.csv"

    if not os.path.isfile(file_name):
        return "File " + file_name + "does not exist.", 400
    if not os.path.isfile(high_dim_file_name):
        return "File " + high_dim_file_name + "does not exist.", 400

    # Open file containing information on low-dimensional projections.
    h5file = open_file(filename=file_name, mode="r+")

    # Fetch dataframe with preprocessed features.
    embedding_metadata_feat_df = app.config["EMBEDDING_METADATA"]["features_preprocessed"].loc[[str(embedding_id)]]

    # Drop index for categorical variables that are inactive for this record.
    # Note: Currently hardcoded for metric only.
    cols = embedding_metadata_feat_df.columns.values
    param_indices = [
        i for i
        in range(len(cols))
        if "metric_" not in cols[i] or
           cols[i] == "metric_" + str(
            app.config["EMBEDDING_METADATA"]["original"].loc[[embedding_id]].metric.values[0]
        )[2:-1]
    ]

    # Let SHAP estimate influence of hyperparameter values for each objective.
    # See https://github.com/slundberg/shap/issues/392 on to verify predicted SHAP values.
    explanations = {
        objective: shap.TreeExplainer(
            app.config["GLOBAL_SURROGATE_MODELS"][objective]
        ).shap_values(embedding_metadata_feat_df.values[0], approximate=False)[param_indices].tolist()
        for objective in app.config["METADATA_TEMPLATE"]["objectives"]
    }
    # Transform SHAP values of objectives w/o upper bounds into [0, 1]-interval by dividing values for unbounded
    # objectives  through the maximum for this objective.
    # Note that we assume all objectives, including those w/o upper bounds, to be [0, x] where x is either 1 or an
    # arbitrary real number.
    # Hence we iterate over upper-unbounded objectives, get their max, divide values in explanations through the maximum
    # of that objective. This yields [0, 1]-intervals for all objectives.
    for obj in DimensionalityReductionKernel.OBJECTIVES_WO_UPPER_BOUND:
        explanations[obj] = (explanations[obj] / app.config["EMBEDDING_METADATA"]["original"][obj].max()).tolist()

    # Assemble result object.
    result = {
        # --------------------------------------------------------
        # Retrieve data from low-dim. dataset.
        # --------------------------------------------------------

        # Transform node with this model into a dataframe so we can easily retain column names.
        "model_metadata": app.config["EMBEDDING_METADATA"]["original"].to_json(orient='index'),
        # Fetch projection record by node name.
        "low_dim_projection": h5file.root.projection_coordinates._f_get_child("model" + str(embedding_id)).read().tolist(),
        # Get dissonances of this model's samples.
        "sample_dissonances": h5file.root.pointwise_quality._f_get_child("model" + str(embedding_id)).read().tolist(),

        # --------------------------------------------------------
        # Retrieve data from original, high-dim. dataset.
        # --------------------------------------------------------

        # Fetch record names/titles, labels, original features.
        "original_dataset": pandas.read_csv(
            filepath_or_buffer=os.getcwd() + "/../data/" + app.config["DATASET_NAME"] + "_records.csv",
            delimiter=',',
            quotechar='"'
        ).to_json(orient='index'),

        # --------------------------------------------------------
        # Explain embedding value with LIME.
        # --------------------------------------------------------

        "explanation_columns": [
            # Hardcoded workaround for one-hot encoded category attribute: Rename to "metric".
            col if "metric_" not in col else "metric" for col
            in app.config["EMBEDDING_METADATA"]["features_preprocessed"].columns.values[param_indices]
        ],
        "explanations": explanations
    }

    # Close file with low-dim. data.
    h5file.close()

    return jsonify(result)


@app.route('/compute_correlation_strength', methods=["GET"])
def compute_correlation_strength():
    """
    Computes correlation strengths between pairs of attributes.
    Works on currently loaded dataset.
        GET parameters:
        - "ids": List of embedding IDs to consider, with ids=1,2,3,... Note: If "ids" is not specified, all embeddings
                 are taken into account.
    :return:
    """

    df = app.config["EMBEDDING_METADATA"]["original"].drop(["num_records"], axis=1)
    ids = request.args.get("ids")
    ids = list(map(int, ids.split(","))) if ids is not None else None

    if ids is not None:
        df = df.iloc[ids]

    df.metric = df.metric.astype("category").cat.codes

    return df.corr(method=lambda x, y: dcor.distance_correlation(x, y)).to_json(orient='index')


# Launch on :2483.
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2483, debug=True)
