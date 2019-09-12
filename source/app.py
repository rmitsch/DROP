import os
import sys

from functools import partial
from flask import render_template
from flask import request
from flask import jsonify
import tables
import pandas
import math
from tables import *
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm
import dcor
import shap
import pickle
from multiprocessing.pool import Pool as ProcessPool

from data_generation.datasets import *
from data_generation.dimensionality_reduction import DimensionalityReductionKernel
from objectives.topology_preservation_objectives import CorankingMatrix
from utils import Utils


frontend_path = sys.argv[1]
# Initialize logger.
logger = Utils.create_logger()
# Initialize flask app.
app = Utils.init_flask_app(frontend_path)
# Define data storage location.
storage_path = os.getcwd() + "/../data/"


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
    app.config["CACHE_ROOT"] = "/tmp"
    app.config["DATASET_NAME"] = InputDataset.check_dataset_name(request.args.get('datasetName'))
    app.config["DR_KERNEL_NAME"] = DimensionalityReductionKernel.check_kernel_name(request.args.get('drKernelName'))
    dataset_name_class_links = {
        "vis": VISPaperDataset,
        "wine": WineDataset,
        "swiss_roll": SwissRollDataset,
        "mnist": MNISTDataset,
        "happiness": HappinessDataset
    }
    app.config["DATASET_CLASS"] = dataset_name_class_links[app.config["DATASET_NAME"]]

    # Currently only Happiness dataset is fully supported.
    assert app.config["DATASET_NAME"] == "happiness", "Only Happiness dataset currently supported."

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
        # Compute global surrogate models.
        ###################################################

        # Compute regressor for each objective.
        cached_surrogate_models_filepath = os.path.join(app.config["CACHE_ROOT"], "global_surrogate_models.pkl")
        if os.path.isfile(cached_surrogate_models_filepath):
            with open(cached_surrogate_models_filepath, "rb") as file:
                app.config["GLOBAL_SURROGATE_MODELS"] = pickle.load(file)
        else:
            app.config["GLOBAL_SURROGATE_MODELS"] = Utils.fit_random_forest_regressors(
                metadata_template=app.config["METADATA_TEMPLATE"],
                features_df=app.config["EMBEDDING_METADATA"]["features_preprocessed"],
                labels_df=app.config["EMBEDDING_METADATA"]["labels"]
            )
            with open(cached_surrogate_models_filepath, "wb") as file:
                pickle.dump(app.config["GLOBAL_SURROGATE_MODELS"], file)

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
        - "modeltype": Model type can be specified with GET param (currently only decision tree with "rules" supported).
        - "objs": Objectives with objs=alpha,beta,...
        - "ids": List of embedding IDs to consider, with ids=1,2,3,... Note: If "ids" is not specified, all embeddings
          are used to construct surrogate model.
        - "n_bins": Number of bins to use for surrogate model's predictions.
    :return: Jsonified structure of surrogate model for DR metadata.
    """

    metadata_template = app.config["METADATA_TEMPLATE"]
    surrogate_model_type = request.args["modeltype"]
    objective_name = request.args["objs"]
    number_of_bins = int(request.args["n_bins"]) if request.args["n_bins"] is not None else 5
    ids = request.args.get("ids")

    # ------------------------------------------------------
    # 1. Check for mistakes in parameters.
    # ------------------------------------------------------

    if surrogate_model_type not in ["rules"]:
        return "Surrogate model " + surrogate_model_type + " is not supported.", 400

    if objective_name not in metadata_template["objectives"]:
        return "Objective " + objective_name + " is not supported.", 400

    # ------------------------------------------------------
    # 2. Pre-select embeddings to use for surrogate model.
    # ------------------------------------------------------

    ids = list(map(int, ids.split(","))) if ids is not None else None
    features_df = app.config["EMBEDDING_METADATA"]["features_preprocessed"]
    labels_df = app.config["EMBEDDING_METADATA"]["labels"]

    # Consider filtered IDs before creating model(s).
    if ids is not None:
        features_df = features_df.iloc[ids]
        labels_df = labels_df.iloc[ids]

    class_encodings = pd.DataFrame(pd.cut(labels_df[objective_name], number_of_bins))
    bin_labels = class_encodings[objective_name].unique()
    with ProcessPool(math.floor(psutil.cpu_count(logical=True))) as pool:
        rule_data = list(
            tqdm(
                pool.imap(
                    partial(
                        Utils.extract_rules,
                        features_df=features_df,
                        class_encodings=class_encodings,
                        objective_name=objective_name
                    ),
                    bin_labels
                ),
                total=len(bin_labels)
            )
        )

    rule_data = pd.DataFrame(
        [
            item
            for sublist in rule_data
            for item in sublist
        ],
        columns=["rule", "precision", "recall", "support", "from", "to"]
    )

    # Bin data for frontend.
    for attribute in ["precision", "recall", "support"]:
        quantiles = pd.cut(rule_data[attribute], number_of_bins)
        rule_data[attribute + "#histogram"] = quantiles.apply(lambda x: x.left)
    rule_data["from#histogram"] = rule_data["from"]
    rule_data["to#histogram"] = rule_data["to"]
    rule_data.rule = rule_data.rule.str.replace(" and ", "<br>")

    return rule_data.to_json(orient='records')


@app.route('/get_sample_dissonance', methods=["GET"])
def get_sample_dissonance():
    """
    Calculates and fetches variance/divergence of individual samples over all DR model parametrizations.
    GET parameters:
        - Distance function to use for determining neighbourhoods (not supported yet).
    :return:
    """
    file_name = app.config["FULL_FILE_NAME"]
    cached_file_path = os.path.join(app.config["CACHE_ROOT"], "sample_dissonance.pkl")

    if os.path.isfile(file_name):
        if os.path.isfile(cached_file_path):
            return pd.read_pickle(cached_file_path).to_json(orient='records')

        h5file = open_file(filename=file_name, mode="r+")

        # ------------------------------------------------------
        # 1. Get metadata on numbers of models and samples.
        # ------------------------------------------------------

        # Use arbitrary model to fetch number of records/points in model.
        num_records = h5file.root.metadata[0][1]
        # Initialize numpy matrix for pointwise qualities.
        pointwise_qualities = np.zeros([h5file.root.metadata.nrows, num_records])

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

        # Cache result as file.
        df.to_pickle(cached_file_path)

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
    high_dim_file_name = storage_path + app.config["DATASET_NAME"] + "_records.csv"
    high_dim_neighbour_ranking_file_name = storage_path + app.config["DATASET_NAME"] + "_neighbourhood_ranking.pkl"
    high_dim_distance_matrices_file_name = storage_path + app.config["DATASET_NAME"] + "_distance_matrices.pkl"

    # Make sure all files exist.
    for fn in (
            file_name, high_dim_file_name, high_dim_neighbour_ranking_file_name,
            high_dim_distance_matrices_file_name
    ):
        if not os.path.isfile(fn):
            return "File " + fn + " does not exist.", 400

    # Open file containing information on low-dimensional projections.
    h5file = open_file(filename=file_name, mode="r+")

    # Read coordinates for low-dimensional projection of this embedding.
    low_dim_projection = h5file.root.projection_coordinates._f_get_child("model" + str(embedding_id)).read()

    # Merge original high-dim. dataset with low-dim. coordinates.
    original_dataset: pd.DataFrame = Utils.prepare_binned_original_dataset(app.config["DATASET_NAME"])
    # Append low.-dim. coordinates.
    for dim_idx in range(low_dim_projection.shape[1]):
        original_dataset[dim_idx] = low_dim_projection[:, dim_idx]
    # Pad low-dim. coordinates with a 0-dimension if they are just one-dimensional.
    if low_dim_projection.shape[1] == 1:
        original_dataset[1] = 0
    # Add ID for table in frontend (needed by jquery's datatable library).
    original_dataset.insert(loc=0, column="id", value=[_ for _ in range(len(original_dataset))])

    # Compute pairwise displacement data.
    pairwise_displacement_data: pd.DataFrame = CorankingMatrix.compute_pairwise_displacement_data(
        high_dim_distance_matrices_file_name,
        high_dim_neighbour_ranking_file_name,
        low_dim_projection
    )

    # Fetch dataframe with preprocessed features.
    embedding_metadata_feat_df = app.config["EMBEDDING_METADATA"]["features_preprocessed"].loc[[str(embedding_id)]]

    # Drop index for categorical variables that are inactive for this record.
    # Note: Currently hardcoded for metric only.
    param_indices = Utils.get_active_col_indices(
        embedding_metadata_feat_df, app.config["EMBEDDING_METADATA"], embedding_id
    )

    # Let SHAP estimate influence of hyperparameter values for each objective.
    # See https://github.com/slundberg/shap/issues/392 on how to verify predicted SHAP values.
    explanations: dict = {
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
        # Retrieve model metadata.
        # --------------------------------------------------------

        # Transform node with this model into a dataframe so we can easily retain column names.
        "model_metadata": app.config["EMBEDDING_METADATA"]["original"].to_json(orient='index'),

        # --------------------------------------------------------
        # Retrieve data from original, high-dim. dataset.
        # --------------------------------------------------------

        # Fetch record names/titles, labels, original features.
        "original_dataset": original_dataset.to_json(orient='records'),
        # Get attributes' data types.
        "attribute_data_types": app.config["DATASET_CLASS"].get_attributes_data_types(),

        # --------------------------------------------------------
        # Relative positional data.
        # --------------------------------------------------------

        "pairwise_displacement_data": pairwise_displacement_data.drop(columns=[
            "high_dim_neighbour_rank", "low_dim_neighbour_rank"
        ]).to_dict(orient="records"),
        # Bin data for co-ranking matrix.
        "coranking_matrix_data": CorankingMatrix.bin_coranking_matrix_data(
            pairwise_displacement_data
        ).to_dict(orient="records"),

        # --------------------------------------------------------
        # Explain embedding value with SHAP.
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
    app.run(host='0.0.0.0', port=2483, threaded=True, debug=True)
