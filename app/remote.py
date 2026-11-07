"""Task 8: call the cloud model API (Task 7, on Vercel) instead of loading the model locally.

The Flask app picks this predictor when MODEL_API_URL is set. It has the same
predict(raw) -> Prediction interface as app.inference.Predictor, so the routes, the
database trail and the history page do not change at all.

The API key is read from MODEL_API_KEY (in .env, never in code) and sent in the
X-API-Key header over HTTPS.
"""
import logging
import time

import requests

from app.inference import Prediction

log = logging.getLogger(__name__)


class RemoteModelError(Exception):
    """The model API could not be reached or did not give a usable answer."""


class RemotePredictor:
    backend = "remote"

    def __init__(self, api_url: str, api_key: str, timeout: float = 60.0):
        if not api_key:
            raise RuntimeError("MODEL_API_URL is set but MODEL_API_KEY is empty (add it to .env).")
        self.api_url = api_url.rstrip("/")
        self._headers = {"X-API-Key": api_key}
        self.timeout = timeout
        self.session = requests.Session()
        # Ask the API which model it serves, for the health page and the history records.
        self.model_name, self.model_version, self.threshold = "visionguard", "unknown", None
        try:
            info = self.session.get(f"{self.api_url}/health", timeout=timeout).json()
            self.model_name = info.get("model_name", self.model_name)
            self.model_version = str(info.get("model_version", self.model_version))
            self.threshold = info.get("threshold")
            log.info("Using cloud model API %s (model %s v%s)", self.api_url, self.model_name,
                     self.model_version)
        except (requests.RequestException, ValueError) as e:
            # Do not crash the web app: the API may just be cold. Requests will retry it.
            log.warning("Model API %s not reachable at start-up: %s", self.api_url, e)

    def predict(self, raw: bytes) -> Prediction:
        name = "upload.png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else "upload.jpg"
        start = time.perf_counter()
        try:
            r = self.session.post(f"{self.api_url}/predict", headers=self._headers,
                                  files={"image": (name, raw)}, timeout=self.timeout)
        except requests.RequestException as e:
            log.error("Model API request failed: %s", e)
            raise RemoteModelError("The model service could not be reached. Please try again.") from e
        round_trip_ms = (time.perf_counter() - start) * 1000

        if r.status_code == 401:
            log.error("Model API rejected our key (401): check MODEL_API_KEY")
            raise RemoteModelError("The model service rejected this server's API key.")
        try:
            body = r.json()
        except ValueError as e:
            raise RemoteModelError(f"Invalid response from the model service ({r.status_code}).") from e
        if r.status_code == 422:
            raise ValueError(body.get("message", "image could not be decoded"))
        if r.status_code != 200:
            raise RemoteModelError(body.get("message", f"Model service error ({r.status_code})."))

        self.model_version = str(body["model_version"])
        self.threshold = body["threshold"]
        log.info("Model API answered in %.0f ms (model %.0f ms)", round_trip_ms, body["latency_ms"])
        return Prediction(label=body["label"], prob_ai=body["prob_ai"],
                          confidence=body["confidence"], threshold=body["threshold"],
                          model_name=f"{body['model_name']} (cloud API)",
                          model_version=self.model_version, latency_ms=round_trip_ms)
