"""Flask app tests. A fake predictor replaces the Keras model so tests run in seconds
and do not need TensorFlow or a trained model (also used later by GitHub Actions CI)."""
from io import BytesIO

import pytest
from PIL import Image

from app import create_app
from app.config import Settings
from app.inference import AI_LABEL, Prediction


class FakePredictor:
    model_name = "fake-model"
    model_version = "test"
    threshold = 0.5

    def __init__(self, prob_ai=0.9):
        self.prob_ai = prob_ai
        self.calls = 0

    def predict(self, raw: bytes) -> Prediction:
        self.calls += 1
        return Prediction(label=AI_LABEL, prob_ai=self.prob_ai, confidence=self.prob_ai,
                          threshold=self.threshold, model_name=self.model_name,
                          model_version=self.model_version, latency_ms=1.0)


def png_bytes(size=(32, 24)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, (120, 40, 200)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def predictor():
    return FakePredictor()


@pytest.fixture
def client(tmp_path, predictor):
    settings = Settings(model_path="unused", threshold=None,
                        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
                        max_upload_mb=1, model_name="fake-model", model_version="test",
                        host="127.0.0.1", port=5000)
    app = create_app(settings, predictor=predictor)
    app.config["TESTING"] = True
    return app.test_client()


def upload(client, data: bytes, name: str):
    return client.post("/predict", data={"image": (BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_health_reports_model_and_database(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok" and body["database"] == "ok" and body["model_loaded"]


def test_pages_render(client):
    assert client.get("/").status_code == 200
    assert client.get("/history").status_code == 200


def test_valid_upload_is_predicted_logged_and_shown(client, predictor):
    r = upload(client, png_bytes(), "cat.png")
    assert r.status_code == 303 and "/result/1" in r.headers["Location"]
    page = client.get("/result/1").get_data(as_text=True)
    assert "Likely AI-generated" in page and "90.0% confidence" in page
    history = client.get("/history").get_data(as_text=True)
    assert "cat.png" in history and "fake-model" in history
    assert predictor.calls == 1


def test_wrong_file_type_rejected(client, predictor):
    r = upload(client, b"hello", "notes.txt")
    assert r.status_code == 415
    assert "not accepted" in r.get_data(as_text=True)
    assert predictor.calls == 0


def test_file_over_size_limit_rejected(client, predictor):
    r = upload(client, b"\xff" * (2 * 1024 * 1024), "huge.jpg")  # limit is 1 MB in tests
    assert r.status_code == 413
    assert "too large" in r.get_data(as_text=True).lower()
    assert predictor.calls == 0


def test_fake_image_that_cannot_be_decoded_rejected(client, predictor):
    r = upload(client, b"this is not really a jpeg", "photo.jpg")
    assert r.status_code == 422
    assert "could not be decoded" in r.get_data(as_text=True)
    assert predictor.calls == 0


def test_missing_file_rejected(client):
    r = client.post("/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_filename_is_sanitized(client):
    upload(client, png_bytes(), "../../etc/evil name.png")
    history = client.get("/history").get_data(as_text=True)
    assert "etc_evil_name.png" in history and "../" not in history


def test_unknown_result_is_404(client):
    assert client.get("/result/999").status_code == 404


def test_upload_is_not_written_to_disk(client, tmp_path):
    upload(client, png_bytes(), "keep_me_off_disk.png")
    assert {p.name for p in tmp_path.iterdir()} == {"test.db"}
