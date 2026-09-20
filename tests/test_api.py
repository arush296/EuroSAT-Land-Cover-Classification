from io import BytesIO

import pytest
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import main


class FakeInput:
    name = "images"


class FakeModel:
    """Small deterministic ONNX-session stand-in for either trained model."""

    def __init__(self, predicted_class_index: int):
        self.predicted_class_index = predicted_class_index

    def get_inputs(self):
        return [FakeInput()]

    def run(self, output_names, inputs):
        images = inputs["images"]
        logits = np.zeros((images.shape[0], len(main.CLASS_NAMES)), dtype=np.float32)
        logits[:, self.predicted_class_index] = 10
        return [logits]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """Start the API without loading the real checkpoint."""
    monkeypatch.setattr(
        main,
        "load_model",
        lambda checkpoint, *args, **kwargs: FakeModel(
            0 if "custom_cnn" in str(checkpoint) else 1
        ),
    )

    with TestClient(main.app) as test_client:
        yield test_client


def create_test_image() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 64), color="green").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "models_loaded": {
            "custom_cnn": True,
            "resnet18": True,
        },
    }


def test_model_info(client: TestClient):
    response = client.get("/model-info")
    body = response.json()

    assert response.status_code == 200
    assert body["input_size"] == [64, 64]
    assert body["classes"] == list(main.CLASS_NAMES)
    assert set(body["models"]) == {"custom_cnn", "resnet18"}
    assert body["models"]["custom_cnn"]["test_accuracy"] == pytest.approx(
        2485 / 2700
    )
    assert body["models"]["resnet18"]["test_accuracy"] == pytest.approx(
        2617 / 2700
    )


def test_cors_allows_local_frontend(client: TestClient):
    response = client.options(
        "/compare",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_predict_image(client: TestClient):
    response = client.post(
        "/predict?top_k=3",
        files={
            "file": (
                "satellite.png",
                create_test_image(),
                "image/png",
            )
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["filename"] == "satellite.png"
    assert body["predicted_class"] == "Forest"
    assert body["top_predictions"][0]["class_name"] == "Forest"
    assert len(body["top_predictions"]) == 3


def test_top_k_controls_number_of_predictions(client: TestClient):
    response = client.post(
        "/predict?top_k=1",
        files={"file": ("satellite.png", create_test_image(), "image/png")},
    )

    assert response.status_code == 200
    assert len(response.json()["top_predictions"]) == 1


def test_compare_runs_both_models(client: TestClient):
    response = client.post(
        "/compare?top_k=3",
        files={"file": ("satellite.png", create_test_image(), "image/png")},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["filename"] == "satellite.png"
    assert body["custom_cnn"]["model_name"] == "Custom CNN"
    assert body["custom_cnn"]["predicted_class"] == "AnnualCrop"
    assert body["resnet18"]["model_name"] == "ResNet18"
    assert body["resnet18"]["predicted_class"] == "Forest"
    assert len(body["custom_cnn"]["top_predictions"]) == 3
    assert len(body["resnet18"]["top_predictions"]) == 3


@pytest.mark.parametrize("top_k", [0, 11])
def test_rejects_invalid_top_k(client: TestClient, top_k: int):
    response = client.post(
        f"/predict?top_k={top_k}",
        files={"file": ("satellite.png", create_test_image(), "image/png")},
    )

    assert response.status_code == 422


def test_rejects_unsupported_file_type(client: TestClient):
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415


def test_rejects_corrupt_image(client: TestClient):
    response = client.post(
        "/predict",
        files={"file": ("broken.png", b"not an image", "image/png")},
    )

    assert response.status_code == 400


def test_rejects_empty_upload(client: TestClient):
    response = client.post(
        "/predict",
        files={"file": ("empty.png", b"", "image/png")},
    )

    assert response.status_code == 400


def test_rejects_oversized_upload(client: TestClient):
    oversized_content = b"0" * (main.MAX_UPLOAD_BYTES + 1)
    response = client.post(
        "/predict",
        files={"file": ("large.png", oversized_content, "image/png")},
    )

    assert response.status_code == 413
