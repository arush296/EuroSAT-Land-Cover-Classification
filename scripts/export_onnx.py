"""Export the trained PyTorch ResNet18 checkpoint for lightweight deployment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision import models

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from inference import CLASS_NAMES, INPUT_SIZE  # noqa: E402


DEFAULT_SOURCE = PROJECT_DIR / "resnet_fully_fine_tuned.pth"
DEFAULT_OUTPUT = PROJECT_DIR / "resnet18_eurosat.onnx"


def build_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def export_model(source: Path, output: Path) -> None:
    state_dict = torch.load(source, map_location="cpu", weights_only=True)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model = build_model()
    model.load_state_dict(state_dict)
    model.eval()

    example_input = torch.randn(1, 3, *INPUT_SIZE)
    onnx_program = torch.onnx.export(
        model,
        (example_input,),
        input_names=["images"],
        output_names=["logits"],
        dynamo=True,
        external_data=False,
        verify=True,
    )
    onnx_program.save(output)
    print(f"Exported ONNX model to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export_model(args.source, args.output)


if __name__ == "__main__":
    main()
