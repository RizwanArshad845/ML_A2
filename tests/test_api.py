"""Model API tests (Task 7) with the fake predictor, so no TensorFlow is needed."""
from io import BytesIO

import pytest

from api import create_api
from tests.test_app import FakePredictor, png_bytes

KEY = "test-key-0123456789abcdef"


@pytest.fixture
def client():
    return create_api(predictor=FakePredictor(), api_key=KEY, max_upload_mb=1).test_client()


def post(client, data=None, key=KEY, name="cat.png"):
    headers = {"X-API-Key": key} if key else {}
    data = png_bytes() if data is None else data
    return client.post("/predict", headers=headers,
                       data={"image": (BytesIO(data), name)}, content_type="multipart/form-data")


def test_refuses_to_start_without_key():
    with pytest.raises(RuntimeError):
        create_api(predictor=FakePredictor(), api_key="")


def test_health_needs_no_key(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json["status"] == "ok"


def test_predict_with_key(client):
    r = post(client)
    assert r.status_code == 200
    assert r.json["label"] == "AI-generated" and r.json["model_version"] == "test"


@pytest.mark.parametrize("key", [None, "wrong-key-0123456789abcdef"])
def test_predict_rejects_missing_or_wrong_key(client, key):
    r = post(client, key=key)
    assert r.status_code == 401 and r.json["error"] == "Unauthorized"


def test_predict_validation_errors_are_json(client):
    assert post(client, name="notes.txt").status_code == 415
    assert post(client, data=b"not an image", name="fake.jpg").status_code == 422
    assert post(client, data=b"x" * (2 * 1024 * 1024), name="big.jpg").status_code == 413
