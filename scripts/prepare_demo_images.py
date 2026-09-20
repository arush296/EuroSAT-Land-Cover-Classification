"""Build a small, class-balanced frontend demo pool from the test split."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from torchvision.datasets import EuroSAT


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_DIR / "data"
DEFAULT_SPLIT_PATH = PROJECT_DIR / "split_indices_seed42.pth"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "frontend" / "examples"


def slugify(class_name: str) -> str:
    separated = re.sub(r"(?<!^)(?=[A-Z])", "-", class_name)
    return separated.lower()


def prepare_examples(
    data_root: Path,
    split_path: Path,
    output_dir: Path,
    examples_per_class: int,
    seed: int,
) -> None:
    dataset = EuroSAT(root=data_root, download=False)
    test_indices = torch.load(split_path, weights_only=True)["test"]

    candidates: dict[str, list[Path]] = defaultdict(list)
    for index in test_indices:
        image_path, class_index = dataset.samples[index]
        candidates[dataset.classes[class_index]].append(Path(image_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_image in output_dir.glob("*.jpg"):
        old_image.unlink()

    generator = random.Random(seed)
    manifest = []

    for class_name in dataset.classes:
        class_candidates = candidates[class_name]
        if len(class_candidates) < examples_per_class:
            raise ValueError(
                f"Only {len(class_candidates)} test images available for {class_name}."
            )

        for position, source in enumerate(
            generator.sample(class_candidates, examples_per_class),
            start=1,
        ):
            destination_name = f"{slugify(class_name)}-{position}.jpg"
            shutil.copy2(source, output_dir / destination_name)
            manifest.append(
                {
                    "class_name": class_name,
                    "file": f"examples/{destination_name}",
                    "source_filename": source.name,
                }
            )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source": "EuroSAT RGB",
                "split": "test",
                "selection_seed": seed,
                "examples_per_class": examples_per_class,
                "examples": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Prepared {len(manifest)} held-out examples "
        f"({examples_per_class} per class) in {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--examples-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    prepare_examples(
        data_root=args.data_root,
        split_path=args.split_path,
        output_dir=args.output_dir,
        examples_per_class=args.examples_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
