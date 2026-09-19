"""Inference utilities for the exported EuroSAT ResNet18 ONNX model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "resnet18_eurosat.onnx"

CLASS_NAMES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = (64, 64)


def load_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: str = "cpu",
) -> ort.InferenceSession:
    """Load the ONNX model with memory-conscious CPU settings."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if device != "cpu":
        raise ValueError("The ONNX deployment currently supports CPU inference only.")

    options = ort.SessionOptions()
    options.intra_op_num_threads = int(os.getenv("ORT_INTRA_OP_THREADS", "1"))
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return ort.InferenceSession(
        str(checkpoint_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert one image into a normalized batch with shape [1, 3, 64, 64]."""
    image = image.convert("RGB").resize(INPUT_SIZE, Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    batch = np.transpose(pixels, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(batch, dtype=np.float32)


def predict_image(
    image: Image.Image | str | Path,
    model: ort.InferenceSession,
    device: str = "cpu",
    top_k: int = 3,
) -> dict[str, Any]:
    """Predict a EuroSAT class and return the top scoring classes."""
    if not 1 <= top_k <= len(CLASS_NAMES):
        raise ValueError(f"top_k must be between 1 and {len(CLASS_NAMES)}")
    if device != "cpu":
        raise ValueError("The ONNX deployment currently supports CPU inference only.")

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened_image:
            batch = preprocess_image(opened_image)
    else:
        batch = preprocess_image(image)

    input_name = model.get_inputs()[0].name
    logits = np.asarray(model.run(None, {input_name: batch})[0])[0]
    exponentials = np.exp(logits - np.max(logits))
    scores = exponentials / exponentials.sum()
    top_indices = np.argsort(scores)[::-1][:top_k]

    top_predictions = [
        {
            "class_name": CLASS_NAMES[int(index)],
            "score": float(scores[index]),
        }
        for index in top_indices
    ]

    return {
        "predicted_class": top_predictions[0]["class_name"],
        "score": top_predictions[0]["score"],
        "top_predictions": top_predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a EuroSAT image.")
    parser.add_argument("image", type=Path, help="Path to an image file")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the exported ONNX model",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    prediction = predict_image(
        args.image,
        model,
        device=args.device,
        top_k=args.top_k,
    )
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
