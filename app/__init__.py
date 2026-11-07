"""VisionGuard Flask app (Tasks 4 and 5).

create_app() builds the app once: it reads settings from the environment, connects to
the database and loads the model a single time. Tests pass in a fake predictor.
"""
import logging

from flask import Flask, render_template

from app.config import Settings
from app.db import Database


def create_app(settings: Settings | None = None, predictor=None) -> Flask:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = settings or Settings.from_env()

    app = Flask(__name__)
    # Requests larger than this are rejected by Flask with 413 before our code runs.
    app.config["MAX_CONTENT_LENGTH"] = int(settings.max_upload_mb * 1024 * 1024)
    app.config["VG_SETTINGS"] = settings

    db = Database(settings.database_url)
    db.create_tables()

    if predictor is None and settings.model_api_url:
        # Task 8: send images to the cloud model API (Task 7) instead of loading TensorFlow.
        from app.remote import RemotePredictor
        predictor = RemotePredictor(settings.model_api_url, settings.model_api_key)
    elif predictor is None:
        from app.inference import Predictor
        predictor = Predictor(settings.model_path, settings.threshold,
                              settings.model_name, settings.model_version)

    app.extensions["vg_db"] = db
    app.extensions["vg_predictor"] = predictor

    from app.routes import bp
    app.register_blueprint(bp)
    _register_error_pages(app, settings)
    return app


def _register_error_pages(app: Flask, settings: Settings) -> None:
    def page(status, title, message):
        return render_template("error.html", title=title, message=message, status=status), status

    @app.errorhandler(404)
    def not_found(_e):
        return page(404, "Page not found", "That page does not exist.")

    @app.errorhandler(405)
    def wrong_method(_e):
        return page(405, "Not allowed", "Use the upload form to submit an image.")

    @app.errorhandler(413)
    def too_large(_e):
        return page(413, "File too large",
                    f"Images must be {settings.max_upload_mb:g} MB or smaller.")

    @app.errorhandler(500)
    def server_error(_e):
        return page(500, "Something went wrong",
                    "The server hit an unexpected error. Please try again.")
