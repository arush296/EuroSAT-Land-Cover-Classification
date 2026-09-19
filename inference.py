"""Inference utilities for the fine-tuned EuroSAT ResNet18 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import models
from torchvision.transforms import v2


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "resnet_fully_fine_tuned.pth"

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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = (64, 64)

# transforming required image as per model input
INFERENCE_TRANSFORM = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize(INPUT_SIZE, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# initialising resnet18 but without the weights
def build_model() -> nn.Module:
    """Create the same 10-class ResNet18 architecture used during training."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model

#  loading .pth and putting the weights into the resnet18 model
def load_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: str | torch.device = "cpu",
) -> nn.Module:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(device)
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    # Also support a future checkpoint saved as {"model_state_dict": ...}.
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model = build_model()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert one image into a normalized batch with shape [1, 3, 64, 64]."""
    image = image.convert("RGB")
    tensor = INFERENCE_TRANSFORM(image)
    return tensor.unsqueeze(0)


def predict_image(
    image: Image.Image | str | Path,
    model: nn.Module,
    device: str | torch.device = "cpu",
    top_k: int = 3,
) -> dict[str, Any]:
    """Predict a EuroSAT class and return the top scoring classes."""
    if not 1 <= top_k <= len(CLASS_NAMES):
        raise ValueError(f"top_k must be between 1 and {len(CLASS_NAMES)}")

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened_image:
            batch = preprocess_image(opened_image)
    else:
        batch = preprocess_image(image)

    device = torch.device(device)
    batch = batch.to(device)

    with torch.inference_mode():
        logits = model(batch)
        scores = torch.softmax(logits, dim=1)[0]
        top_scores, top_indices = torch.topk(scores, k=top_k)

    top_predictions = [
        {
            "class_name": CLASS_NAMES[index],
            "score": float(score),
        }
        for score, index in zip(
            top_scores.cpu().tolist(),
            top_indices.cpu().tolist(),
        )
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
        help="Path to the trained ResNet18 checkpoint",
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
