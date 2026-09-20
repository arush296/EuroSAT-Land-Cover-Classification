# Demo images

These images are a reproducible, class-balanced sample from the project's
saved EuroSAT test split. They are not part of either the training or validation
split.

The pool is generated from local data with:

```bash
python scripts/prepare_demo_images.py
```

The frontend reads `manifest.json` and randomly shows one image from each of
the ten classes on every page load or shuffle.

Source: [EuroSAT](https://github.com/phelber/EuroSAT) by Patrick Helber,
Benjamin Bischke, Andreas Dengel, and Damian Borth. EuroSAT is derived from
Copernicus Sentinel-2 imagery and distributed under the MIT License included
in this directory.
