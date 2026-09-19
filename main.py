from __future__ import annotations

import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from schema import PredictionResponse, PredictionItem

from inference import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT,
    INPUT_SIZE,
    load_model,
    predict_image,
)


CHECKPOINT_PATH = Path(os.getenv("EUROSAT_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
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



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the API starts."""
    app.state.model = load_model(CHECKPOINT_PATH, DEVICE)
    yield
    app.state.model = None


app = FastAPI(
    title="EuroSAT Land-Cover Classifier",
    description="Classify satellite image patches into one of ten EuroSAT classes.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health(request: Request) -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_loaded": request.app.state.model is not None,
    }


@app.get("/model-info")
def model_info() -> dict[str, object]:
    return {
        "model": "resnet18",
        "checkpoint": CHECKPOINT_PATH.name,
        "device": DEVICE,
        "input_size": list(INPUT_SIZE),
        "classes": list(CLASS_NAMES),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: Request,
    file: Annotated[UploadFile, File(description="Satellite image to classify")],
    top_k: Annotated[int, Query(ge=1, le=len(CLASS_NAMES))] = 3,
) -> PredictionResponse:
    """Validate an uploaded image and return its highest-scoring classes."""
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
            result = predict_image(
                image,
                request.app.state.model,
                device=DEVICE,
                top_k=top_k,
            )
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is not a valid readable image.",
        ) from error

    return PredictionResponse(filename=file.filename, **result)
