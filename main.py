from __future__ import annotations

import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from schema import ComparisonResponse, ModelPrediction, PredictionResponse

from inference import (
    CLASS_NAMES,
    DEFAULT_CUSTOM_CNN_CHECKPOINT,
    DEFAULT_RESNET_CHECKPOINT,
    INPUT_SIZE,
    load_model,
    predict_image,
)


RESNET_CHECKPOINT_PATH = Path(
    os.getenv(
        "EUROSAT_RESNET_CHECKPOINT",
        os.getenv("EUROSAT_CHECKPOINT", str(DEFAULT_RESNET_CHECKPOINT)),
    )
)
CUSTOM_CNN_CHECKPOINT_PATH = Path(
    os.getenv("EUROSAT_CUSTOM_CNN_CHECKPOINT", str(DEFAULT_CUSTOM_CNN_CHECKPOINT))
)
CUSTOM_CNN_MODEL_URL = os.getenv(
    "EUROSAT_CUSTOM_CNN_MODEL_URL",
    "https://github.com/arush296/EuroSAT-Land-Cover-Classification/"
    "releases/download/model-v1/custom_cnn_eurosat.onnx",
)
DEVICE = os.getenv("EUROSAT_DEVICE", "cpu")
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
SUPPORTED_MEDIA_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/tiff",
    "image/bmp",
    "application/octet-stream",
}

MODEL_METADATA = {
    "custom_cnn": {
        "model_name": "Custom CNN",
        "test_accuracy": 2485 / 2700,
        "training_method": "Trained from scratch with augmentation",
    },
    "resnet18": {
        "model_name": "ResNet18",
        "test_accuracy": 2617 / 2700,
        "training_method": "ImageNet pretrained and fully fine-tuned",
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both deployment models once when the API starts."""
    app.state.models = {
        "custom_cnn": load_model(
            CUSTOM_CNN_CHECKPOINT_PATH,
            DEVICE,
            download_url=CUSTOM_CNN_MODEL_URL,
        ),
        "resnet18": load_model(RESNET_CHECKPOINT_PATH, DEVICE),
    }
    yield
    app.state.models = {}


app = FastAPI(
    title="EuroSAT Land-Cover Classifier",
    description="Compare two models that classify EuroSAT satellite image patches.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Render sets this to the Vercel origin; local frontend origins are defaults.
    allow_origins=FRONTEND_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    models = getattr(request.app.state, "models", {})
    models_loaded = {
        name: models.get(name) is not None for name in ("custom_cnn", "resnet18")
    }
    return {
        "status": "ok",
        "model_loaded": all(models_loaded.values()),
        "models_loaded": models_loaded,
    }


@app.get("/model-info")
def model_info() -> dict[str, object]:
    return {
        "device": DEVICE,
        "input_size": list(INPUT_SIZE),
        "classes": list(CLASS_NAMES),
        "models": {
            "custom_cnn": {
                **MODEL_METADATA["custom_cnn"],
                "checkpoint": CUSTOM_CNN_CHECKPOINT_PATH.name,
                "preprocessing": "RGB values scaled to [0, 1]",
            },
            "resnet18": {
                **MODEL_METADATA["resnet18"],
                "checkpoint": RESNET_CHECKPOINT_PATH.name,
                "preprocessing": "ImageNet mean and standard deviation normalization",
            },
        },
    }


def validated_image(file: UploadFile) -> Image.Image:
    """Validate one upload and return a detached RGB image."""
    if file.content_type and file.content_type not in SUPPORTED_MEDIA_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload a JPEG, PNG, WebP, TIFF, or BMP image.",
        )

    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="The uploaded image must be 10 MB or smaller.",
        )

    try:
        with Image.open(BytesIO(contents)) as image:
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="The uploaded image has too many pixels.",
                )
            image.load()
            return image.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is not a valid readable image.",
        ) from error


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: Request,
    file: Annotated[UploadFile, File(description="Satellite image to classify")],
    top_k: Annotated[int, Query(ge=1, le=len(CLASS_NAMES))] = 3,
) -> PredictionResponse:
    """Return the ResNet18 result for backward compatibility."""
    image = validated_image(file)
    result = predict_image(
        image,
        request.app.state.models["resnet18"],
        device=DEVICE,
        top_k=top_k,
    )

    return PredictionResponse(filename=file.filename, **result)


@app.post("/compare", response_model=ComparisonResponse)
def compare(
    request: Request,
    file: Annotated[UploadFile, File(description="Satellite image to classify")],
    top_k: Annotated[int, Query(ge=1, le=len(CLASS_NAMES))] = 3,
) -> ComparisonResponse:
    """Run the same uploaded image through CustomCNN and ResNet18."""
    image = validated_image(file)
    custom_result = predict_image(
        image,
        request.app.state.models["custom_cnn"],
        device=DEVICE,
        top_k=top_k,
        normalize=False,
    )
    resnet_result = predict_image(
        image,
        request.app.state.models["resnet18"],
        device=DEVICE,
        top_k=top_k,
    )

    return ComparisonResponse(
        filename=file.filename,
        custom_cnn=ModelPrediction(**MODEL_METADATA["custom_cnn"], **custom_result),
        resnet18=ModelPrediction(**MODEL_METADATA["resnet18"], **resnet_result),
    )
