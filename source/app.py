import os

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

from backend.data_generation.datasets.InputDataset import InputDataset
from backend.data_generation.dimensionality_reduction import DimensionalityReductionKernel
from backend.utils import Utils


def init_flask_app():
    """
    Initialize Flask app.
    :return: App object.
    """
    flask_app = Flask(
        __name__,
        template_folder='frontend/templates',
        static_folder='frontend/static'
    )

    # Define version.
    flask_app.config["VERSION"] = "0.6"

    # Store metadata template. Is assembled once in /get_metadata.
    flask_app.config["METADATA_TEMPLATE"] = None

    # Store name of current dataset and kernel. Note that these values is only changed at call of /get_metadata.
    # Use t-SNE on wine dataset as default.
    flask_app.config["DATASET_NAME"] = None
    flask_app.config["DR_KERNEL_NAME"] = "tsne"
    flask_app.config["FULL_FILE_NAME"] = "wine"

    # For storage of global, unrestricted model used by LIME for local explanations.
    # Has one global regressor for each possible objective.
    flask_app.config["GLOBAL_SURROGATE_MODEL"] = {}
    flask_app.config["LIME_EXPLAINER"] = None

    return flask_app


# Initialize logger.
logger = Utils.create_logger()
# Initialize flask app.
app = init_flask_app()


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
        # Compute global surrogate models and initialize
        # corresponding LIME explainers.
        ###################################################

        # Compute regressor for each objective.
        app.config["GLOBAL_SURROGATE_MODELS"] = Utils.fit_random_forest_regressors(
            metadata_template=app.config["METADATA_TEMPLATE"],
            embeddings_metadata=df
        )

        # Initialize LIME explainers for each objective.
        app.config["LIME_EXPLAINERS"] = Utils.initialize_lime_explainers(
            metadata_template=app.config["METADATA_TEMPLATE"],
            embeddings_metadata=df
        )

        # Return JSON-formatted embedding data.
        return jsonify(df.to_json(orient='index'))

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
            "b_nx",
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
        - Model type can be specified with GET param. "modeltype" (currently only decision tree with "tree").
        - Objectives with objs=alpha,beta,...
        - Max. depth of decision tree with depth=x.
    :return: Jsonified structure of surrogate model for DR metadata.
    """
    metadata_template = app.config["METADATA_TEMPLATE"]
    surrogate_model_type = request.args["modeltype"]
    objective_names = request.args["objs"].split(",")
    depth = int(request.args["depth"])

    # ------------------------------------------------------
    # 1. Check for mistakes in parameters.
    # ------------------------------------------------------

    if surrogate_model_type not in ["tree"]:
        return "Surrogate model " + surrogate_model_type + " is not supported.", 400

    for obj_name in objective_names:
        if obj_name not in metadata_template["objectives"]:
            return "Objective " + obj_name + " is not supported.", 400

    # ------------------------------------------------------
    # 2. Fetch data.
    # ------------------------------------------------------

    # Open .h5 file.
    file_name = app.config["FULL_FILE_NAME"]
    if os.path.isfile(file_name):
        h5file = tables.open_file(filename=file_name, mode="r")
        # Cast to dataframe, then return as JSON.
        df = pandas.DataFrame(h5file.root.metadata[:]).set_index("id")
        # Close file.
        h5file.close()

        # ------------------------------------------------------
        # 3. Build model.
        # ------------------------------------------------------

        # Create one decision tree model for each metric.
        features_names = [item["name"] for item in metadata_template["hyperparameters"]]
        features_df = df[features_names]

        # Encode categorical values numerically.
        for feature in metadata_template["hyperparameters"]:
            if feature["type"] != "numeric":
                features_df[feature["name"]] = LabelEncoder().fit_transform(
                    features_df[feature["name"]].values
                )

        # Fit decision tree.
        tree = DecisionTreeRegressor(max_depth=depth)
        tree.fit(features_df, df[objective_names])

        # ------------------------------------------------------
        # 4. Extract tree structure and return result.
        # ------------------------------------------------------

        # Extract tree structure and return as JSON.
        tree_structure = Utils.extract_decision_tree_structure(tree, features_names, [objective_names])
        return jsonify(tree_structure)

    else:
        return "File does not exist.", 400


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
            # Gather all pointwise qualities in file.
            pointwise_qualities[model_id] = model_pointwise_quality_leaf.read().flatten()

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

    # Prepare dataframe with embedding metadata.
    embedding_metadata_df = pandas.DataFrame.from_records(
        h5file.root.metadata.read_where("(id == " + str(embedding_id) + ")")
    ).set_index("id")

    boston = load_boston()
    train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(boston.data, boston.target,
                                                                                      train_size=0.80, test_size=0.20)

    # Preprocess dataframe.
    embedding_metadata_feat_df, embedding_metadata_labels_df = Utils.preprocess_embedding_metadata_for_predictor(
        metadata_template=app.config["METADATA_TEMPLATE"],
        embeddings_metadata=embedding_metadata_df
    )

    print(test[0])
    print(test[0].shape)
    print(embedding_metadata_feat_df.iloc[0].values.reshape(1, -1).shape)
    print(embedding_metadata_feat_df.iloc[0].values.shape)
    CONTINUE HERE: Get and return results.
    print(app.config["GLOBAL_SURROGATE_MODEL"].predict(embedding_metadata_feat_df.iloc[0].values.reshape(1, -1)))
    explanation = app.config["LIME_EXPLAINER"].explain_instance(
        data_row=embedding_metadata_feat_df.iloc[0].values.reshape(1, -1),
        predict_fn=app.config["GLOBAL_SURROGATE_MODEL"].predict,
        labels=(obj for obj in app.config["METADATA_TEMPLATE"]["objectives"]),
        # num_features=len(app.config["METADATA_TEMPLATE"]["hyperparameters"])
    )
    print(explanation.as_list())


    # Assemble result object.
    result = {
        # --------------------------------------------------------
        # Retrieve data from low-dim. dataset.
        # --------------------------------------------------------

        # Transform node with this model into a dataframe so we can easily retain column names.
        "model_metadata": embedding_metadata_df.to_json(orient='index'),
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
            delimiter=';',
            quotechar='"'
        ).to_json(orient='index'),

        # --------------------------------------------------------
        # Explain embedding value with LIME.
        # --------------------------------------------------------

        "lime_explanation": 0
    }

    # Close file with low-dim. data.
    h5file.close()

    return jsonify(result)


# Launch on :2483.
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2483, debug=True)
