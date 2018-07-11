from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import json
import os
import tables
import pandas
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from tables import *
import numpy

from backend.data_generation.datasets.WineDataset import WineDataset
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
    flask_app.config["VERSION"] = "0.2.3"

    # Limit of 100 MB for upload.
    flask_app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

    flask_app.config["METADATA_TEMPLATE"] = {
        "hyperparameters": [
                {"name": "n_components", "type": "numeric"},
                {"name": "perplexity", "type": "numeric"},
                {"name": "early_exaggeration", "type": "numeric"},
                {"name": "learning_rate", "type": "numeric"},
                {"name": "n_iter", "type": "numeric"},
                {"name": "angle", "type": "numeric"},
                {"name": "metric", "type": "categorical"}
            ],

        "objectives": [
            "runtime",
            "r_nx",
            "b_nx",
            "stress",
            "classification_accuracy",
            "separability_metric"
        ]
    }

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
    :return:
    """
    # todo Should be customized (e. g. filename should be selected in UI).

    # Open .h5 file.
    file_name = os.getcwd() + "/../data/drop_wine.h5"
    if os.path.isfile(file_name):
        h5file = tables.open_file(filename=file_name, mode="r")
        # Cast to dataframe, then return as JSON.
        df = pandas.DataFrame(h5file.root.metadata[:]).set_index("id")
        # Close file.
        h5file.close()

        return jsonify(df.to_json(orient='index'))

    else:
        return "File does not exist.", 400


@app.route('/get_metadata_template', methods=["GET"])
def get_metadata_template():
    """
    Assembles metadata template (i. e. which hyperparameters and objectives are available).
    :return: Dictionary: {"hyperparameters": [...], "objectives": [...]}
    """
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
    # todo Add file name as parameter.

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
    file_name = os.getcwd() + "/../data/drop_wine.h5"
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
        - File name (not supported yet).
        - Distance function to use for determining neighbourhoods (not supported yet).
    :return:
    """
    # todo Add filename as GET parameter/make dataset-variant, load file according to param.
    # todo Store distance matrices in file.
    file_name = os.getcwd() + "/../data/drop_wine.h5"

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

            pointwise_qualities[model_id] = model_pointwise_quality_leaf.read().flatten()

        # Close file.
        h5file.close()

        # Reshape data to desired model_id:sample_id:value format.
        df = pandas.DataFrame(pointwise_qualities)
        df["model_id"] = df.index
        df = df.melt("model_id", var_name='sample_id', value_name="measure")

        # Return jsonified version of model x sample quality matrix.
        return df.head(n=100).to_json(orient='records')

    else:
        return "File does not exist.", 400

# Launch on :2483.
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2483, debug=True)
