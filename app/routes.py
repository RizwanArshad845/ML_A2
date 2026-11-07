"""Task 4.2 routes: upload, result, history and health check."""
import logging

from flask import Blueprint, abort, current_app, jsonify, redirect, render_template, request, url_for

from app.remote import RemoteModelError
from app.validation import ALLOWED_EXTENSIONS, UploadError, validate_upload

log = logging.getLogger(__name__)
bp = Blueprint("main", __name__)


def _db():
    return current_app.extensions["vg_db"]


def _predictor():
    return current_app.extensions["vg_predictor"]


@bp.get("/")
def index():
    s = current_app.config["VG_SETTINGS"]
    return render_template("index.html", max_mb=s.max_upload_mb,
                           allowed=", ".join(sorted(ALLOWED_EXTENSIONS)))


@bp.post("/predict")
def predict():
    try:
        upload = validate_upload(request.files.get("image"))
        pred = _predictor().predict(upload.raw)
    except UploadError as e:
        log.info("rejected upload: %s", e.message)
        return render_template("error.html", title=e.title, message=e.message,
                               status=e.status), e.status
    except ValueError:
        return render_template("error.html", title="Image could not be read",
                               message="The model could not decode this image.",
                               status=422), 422
    except RemoteModelError as e:
        return render_template("error.html", title="Model service unavailable",
                               message=str(e), status=503), 503

    rec = _db().add(
        filename=upload.filename, file_sha256=upload.sha256, file_size=upload.size,
        width=upload.width, height=upload.height,
        label=pred.label, prob_ai=pred.prob_ai, confidence=pred.confidence,
        threshold=pred.threshold, model_name=pred.model_name,
        model_version=pred.model_version, latency_ms=pred.latency_ms,
    )
    log.info("prediction #%d %s: %s (%.3f)", rec.id, upload.filename, pred.label, pred.confidence)
    # Post/Redirect/Get: refreshing the result page does not resubmit the upload.
    return redirect(url_for("main.result", record_id=rec.id), code=303)


@bp.get("/result/<int:record_id>")
def result(record_id: int):
    rec = _db().get(record_id)
    if rec is None:
        abort(404)
    return render_template("result.html", rec=rec)


@bp.get("/history")
def history():
    limit = min(max(request.args.get("limit", 100, type=int), 1), 500)
    db = _db()
    return render_template("history.html", records=db.recent(limit), total=db.count(),
                           limit=limit)


@bp.get("/health")
def health():
    p = _predictor()
    db_ok = _db().ping()
    body = {
        "status": "ok" if db_ok else "degraded",
        "model_loaded": p is not None,
        "model_backend": getattr(p, "backend", "local"),
        "model_name": p.model_name,
        "model_version": p.model_version,
        "threshold": p.threshold,
        "database": "ok" if db_ok else "unreachable",
    }
    return jsonify(body), 200 if db_ok else 503
