"""Export the trained CustomCNN checkpoint for lightweight deployment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from inference import CLASS_NAMES, INPUT_SIZE  # noqa: E402


DEFAULT_SOURCE = PROJECT_DIR / "custom_cnn_clean.pth"
DEFAULT_OUTPUT = PROJECT_DIR / "custom_cnn_eurosat.onnx"


class CustomCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flattened_features = 128 * (INPUT_SIZE[0] // 8) * (INPUT_SIZE[1] // 8)
        self.classifier = nn.Sequential(
            nn.Linear(flattened_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(CLASS_NAMES)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(torch.flatten(features, 1))


def export_model(source: Path, output: Path) -> None:
    model = CustomCNN()
    model.load_state_dict(torch.load(source, map_location="cpu", weights_only=True))
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
