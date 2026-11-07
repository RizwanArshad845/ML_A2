"""Task 7: VisionGuard model API (deployed to Hugging Face Spaces).

A small JSON service around the same Predictor the web app uses:

    GET  /         service info (no key)
    GET  /health   liveness + model version (no key)
    POST /predict  multipart field "image", header "X-API-Key: <key>"  -> JSON verdict

The API key comes from the API_KEY environment variable (a Space secret in the cloud).
The service refuses to start without one, so it can never run unprotected by accident.
The image is validated exactly like the web app (type, size, decodability) and kept
in memory only.
"""
import hmac
import logging
import os

from flask import Flask, jsonify, request

from app.validation import UploadError, validate_upload

log = logging.getLogger(__name__)


def create_api(predictor=None, api_key: str | None = None, max_upload_mb: float | None = None) -> Flask:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    api_key = api_key if api_key is not None else os.getenv("API_KEY", "")
    if len(api_key) < 16:
        raise RuntimeError("Set API_KEY (at least 16 characters) before starting the model API.")
    max_upload_mb = max_upload_mb or float(os.getenv("MAX_UPLOAD_MB", "5"))

    if predictor is None:
        from app.config import Settings
        from app.inference import Predictor
        s = Settings.from_env()
        predictor = Predictor(s.model_path, s.threshold, s.model_name, s.model_version)

    api = Flask(__name__)
    api.config["MAX_CONTENT_LENGTH"] = int(max_upload_mb * 1024 * 1024)

    def error(status: int, title: str, message: str):
        return jsonify(error=title, message=message, status=status), status

    @api.get("/")
    def info():
        return jsonify(service="VisionGuard model API", model=predictor.model_name,
                       model_version=predictor.model_version,
                       endpoints={"GET /health": "status (no key)",
                                  "POST /predict": "multipart 'image' + header X-API-Key"})

    @api.get("/health")
    def health():
        return jsonify(status="ok", model_loaded=True, model_name=predictor.model_name,
                       model_version=predictor.model_version, threshold=predictor.threshold)

    @api.post("/predict")
    def predict():
        sent = request.headers.get("X-API-Key", "")
        # compare_digest takes the same time whether the first or last character differs,
        # so the key cannot be guessed character by character from response times.
        if not sent or not hmac.compare_digest(sent.encode(), api_key.encode()):
            log.warning("rejected request without a valid API key from %s", request.remote_addr)
            return error(401, "Unauthorized", "Missing or invalid X-API-Key header.")
        try:
            upload = validate_upload(request.files.get("image"))
            pred = predictor.predict(upload.raw)
        except UploadError as e:
            return error(e.status, e.title, e.message)
        except ValueError:
            return error(422, "Image could not be read", "The model could not decode this image.")
        log.info("predict %s: %s (P(AI)=%.3f, %.0f ms)", upload.filename, pred.label, pred.prob_ai,
                 pred.latency_ms)
        return jsonify(label=pred.label, prob_ai=round(pred.prob_ai, 4),
                       confidence=round(pred.confidence, 4), threshold=pred.threshold,
                       model_name=pred.model_name, model_version=pred.model_version,
                       latency_ms=round(pred.latency_ms, 1), filename=upload.filename,
                       width=upload.width, height=upload.height, file_size=upload.size,
                       sha256=upload.sha256)

    @api.errorhandler(413)
    def too_large(_e):
        return error(413, "File too large", f"Images must be {max_upload_mb:g} MB or smaller.")

    @api.errorhandler(404)
    def not_found(_e):
        return error(404, "Not found", "Unknown endpoint. See GET /.")

    @api.errorhandler(405)
    def wrong_method(_e):
        return error(405, "Method not allowed", "Check the HTTP method. See GET /.")

    return api
