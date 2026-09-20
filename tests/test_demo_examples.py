import json
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_DIR / "frontend" / "examples"
EXPECTED_CLASSES = {
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
}


def test_demo_manifest_is_class_balanced_and_complete():
    manifest = json.loads((EXAMPLES_DIR / "manifest.json").read_text())
    examples = manifest["examples"]
    class_counts = Counter(example["class_name"] for example in examples)

    assert manifest["split"] == "test"
    assert manifest["examples_per_class"] == 5
    assert set(class_counts) == EXPECTED_CLASSES
    assert set(class_counts.values()) == {5}
    assert len(examples) == 50

    for example in examples:
        relative_file = Path(example["file"])
        assert relative_file.parts[0] == "examples"
        assert (PROJECT_DIR / "frontend" / relative_file).is_file()
