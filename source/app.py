from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from backend.utils import Utils


def init_flask_app():
    """
    Initialize Flask app.
    :return: App object.
    """
    app = Flask(__name__,
                template_folder='frontend/templates',
                static_folder='frontend/static')

    # Define version.
    app.config["VERSION"] = "0.1"

    return app


# Initialize logger.
logger = Utils.create_logger()
# Initialize flask app.
app = init_flask_app()


# root: Render HTML for start menu.
@app.route("/")
def index():
    return render_template("index.html", version=app.config["VERSION"], entrypoint="home")


# Launch on :2483.
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2483, debug=True)
