from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from backend.utils import Utils
import os
import tables
import pandas


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
    # Open .h5 file.
    # todo Should be customized (e. g. filename should be selected in UI).
    file_name = os.getcwd() + "/../data/drop_wine.h5"
    print(file_name)
    if os.path.isfile(file_name):
        h5file = tables.open_file(filename=file_name, mode="r")
        # Cast to dataframe, then return as JSON.
        df = pandas.DataFrame(h5file.root.metadata[:]).set_index("id")

        return jsonify(df.to_json(orient='index'))

    else:
        return "File does not exist.", 400


@app.route('/get_metadata_template', methods=["GET"])
def get_metadata_template():
    """
    Assembles metadata template (i. e. which hyperparameters and objectives are available).
    :return: Dictionary: {"hyperparameters": [...], "objectives": [...]}
    """
    # todo Should be customized (e. g. filename should be selected in UI).

    return jsonify({
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
    })

# Launch on :2483.
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2483, debug=True)
