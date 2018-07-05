from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import json
from backend.utils import Utils
import os
import tables
import pandas
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris


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
    Yields structural data for surrogate model. Parameters:
        - Model type can be specified with GET param. "modeltype" (currently only decision tree with "tree").
        - Objectives with objs=alpha,beta,...
        - Max. depth of decision tree with depth=x.
    :return:
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

    # Open .h5 file.
    file_name = os.getcwd() + "/../data/drop_wine.h5"
    if os.path.isfile(file_name):
        # ------------------------------------------------------
        # 2. Fetch data.
        # ------------------------------------------------------

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
        tree_structure = extract_decision_tree_structure(tree, features_names, [objective_names])
        # todo Fetch/generate correct decision tree for this dataset and model type.
        return jsonify(tree_structure)

    else:
        return "File does not exist.", 400


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
            extract_decision_tree_structure(clf, features, labels, right_index),
            extract_decision_tree_structure(clf, features, labels, left_index)
        ]

    return node

# Launch on :2483.
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2483, debug=True)
