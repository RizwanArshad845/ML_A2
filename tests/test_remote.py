"""Task 8: the Flask app in remote mode, with the cloud API replaced by a fake HTTP layer."""
import pytest
import requests

from app import create_app
from app.config import Settings
from app.remote import RemotePredictor
from tests.test_app import png_bytes, upload


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class FakeSession:
    """Stands in for requests.Session: records calls, returns canned API answers."""

    def __init__(self, predict_response=None, fail=False):
        self.predict_response = predict_response or FakeResponse(200, {
            "label": "Real", "prob_ai": 0.2, "confidence": 0.8, "threshold": 0.68,
            "model_name": "visionguard", "model_version": "2", "latency_ms": 40.0})
        self.fail = fail
        self.posts = []

    def get(self, url, timeout):
        return FakeResponse(200, {"model_name": "visionguard", "model_version": "2", "threshold": 0.68})

    def post(self, url, headers, files, timeout):
        if self.fail:
            raise requests.ConnectionError("network down")
        self.posts.append((url, headers, files))
        return self.predict_response


def remote_client(tmp_path, monkeypatch, session):
    monkeypatch.setattr("app.remote.requests.Session", lambda: session)
    settings = Settings(model_path="unused", threshold=None,
                        database_url=f"sqlite:///{(tmp_path / 'r.db').as_posix()}",
                        max_upload_mb=1, model_name="visionguard", model_version="",
                        host="127.0.0.1", port=5000,
                        model_api_url="https://example.vercel.app/", model_api_key="k" * 20)
    app = create_app(settings)
    app.config["TESTING"] = True
    return app.test_client()


def test_remote_prediction_is_sent_with_key_and_logged(tmp_path, monkeypatch):
    session = FakeSession()
    client = remote_client(tmp_path, monkeypatch, session)
    assert client.get("/health").get_json()["model_backend"] == "remote"
    r = upload(client, png_bytes(), "cat.png")
    assert r.status_code == 303
    url, headers, files = session.posts[0]
    assert url == "https://example.vercel.app/predict" and headers == {"X-API-Key": "k" * 20}
    history = client.get("/history").get_data(as_text=True)
    assert "cat.png" in history and "cloud API" in history


def test_api_down_shows_friendly_503(tmp_path, monkeypatch):
    client = remote_client(tmp_path, monkeypatch, FakeSession(fail=True))
    r = upload(client, png_bytes(), "cat.png")
    assert r.status_code == 503 and b"Model service unavailable" in r.data


def test_wrong_key_is_reported_not_crashed(tmp_path, monkeypatch):
    session = FakeSession(predict_response=FakeResponse(401, {"error": "Unauthorized"}))
    r = upload(remote_client(tmp_path, monkeypatch, session), png_bytes(), "cat.png")
    assert r.status_code == 503 and b"rejected" in r.data


def test_remote_needs_a_key():
    with pytest.raises(RuntimeError):
        RemotePredictor("https://example.vercel.app", "")
