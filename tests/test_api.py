from io import BytesIO

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn

import main


class FakeModel(nn.Module):
    """Small deterministic stand-in for the trained ResNet18."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(images.shape[0], len(main.CLASS_NAMES))
        logits[:, 1] = 10  # Always predict Forest.
        return logits


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """Start the API without loading the real checkpoint."""
    monkeypatch.setattr(
        main,
        "load_model",
        lambda *args, **kwargs: FakeModel(),
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
    }


def test_model_info(client: TestClient):
    response = client.get("/model-info")
    body = response.json()

    assert response.status_code == 200
    assert body["model"] == "resnet18"
    assert body["input_size"] == [64, 64]
    assert body["classes"] == list(main.CLASS_NAMES)


def test_cors_allows_local_frontend(client: TestClient):
    response = client.options(
        "/predict",
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
