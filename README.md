# EuroSAT Land-Cover Classification

An end-to-end deep-learning project that classifies RGB satellite patches into ten land-cover classes. It compares a CNN trained from scratch with an ImageNet-pretrained ResNet18, then serves both models through FastAPI so their predictions can be inspected side by side in a small web app.

[Live demo](https://eurosat-classifier.vercel.app) · [API docs](https://eurosat-api.onrender.com/docs) · [Model release](https://github.com/arush296/EuroSAT-Land-Cover-Classification/releases/tag/model-v1)

## Project at a glance

- Trained and evaluated two PyTorch classifiers on the same reproducible split of 27,000 EuroSAT images.
- Explored augmentation, dropout, frozen-feature transfer learning, and full ResNet18 fine-tuning.
- Analysed learning curves, confusion matrices, per-class precision/recall/F1, and misclassified predictions.
- Exported the final checkpoints to ONNX and built a tested FastAPI inference service.
- Deployed a Vercel frontend that compares both models on uploads or shuffled, held-out examples.

## Final results

Both deployed checkpoints were re-evaluated on the same saved 2,700-image test split.

| Model | Training approach | Correct | Test accuracy |
| --- | --- | ---: | ---: |
| Custom CNN | Trained from scratch with augmentation | 2,485 / 2,700 | 92.04% |
| ResNet18 | ImageNet initialization; all layers fine-tuned | **2,617 / 2,700** | **96.93%** |

ResNet18 improved test accuracy by **4.89 percentage points** and correctly classified **132 more images**. Its macro F1-score was 0.969; `SeaLake` was the strongest class (0.989 F1), while `River` was the weakest (0.940 F1).

## Experiments and observations

### 1. Custom CNN baseline

The custom network contains three `Conv2d → BatchNorm → ReLU → MaxPool` blocks followed by a 256-unit fully connected layer, 0.3 dropout, and a ten-class output layer. It was trained from scratch with Adam (`lr=1e-3`) and cross-entropy loss.

The main regularization experiment compared otherwise similar 30-epoch runs:

| Custom CNN run | Final train accuracy | Best validation accuracy | Observation |
| --- | ---: | ---: | --- |
| With 0.3 dropout | 96.95% | **93.22%** | Better validation peak and smaller train/validation gap |
| Without dropout | 98.67% | 91.31% | Higher training accuracy but stronger overfitting |

Dropout was therefore retained. Horizontal and vertical flips plus rotations up to 20° were also added to the training pipeline. Validation performance was noticeably less stable than training performance, so the final 40-epoch run saved the checkpoint with the lowest validation loss instead of simply using the last epoch.

### 2. ResNet18 transfer learning

ResNet18's 1,000-class head was replaced with a ten-class linear layer. An early transfer-learning comparison produced:

| ResNet18 run | Test accuracy |
| --- | ---: |
| Frozen ImageNet backbone; only the new head trained | 81.41% |
| Full-network fine-tuning | 96.41% |

Freezing the backbone reduced the number of trainable parameters but transferred poorly to 64 × 64 satellite imagery. The final experiment therefore fine-tuned every layer for 20 epochs using SGD (`lr=0.001`, momentum `0.9`) and reached **96.93%** on the fixed test split. Its best saved checkpoint was selected by validation loss; the highest observed validation accuracy was 97.39%.

### 3. What the comparison showed

- Transfer learning provided a clear advantage even though the source domain was natural imagery rather than satellite imagery.
- Full fine-tuning mattered substantially more than using ResNet18 as a fixed feature extractor.
- The custom CNN remained a useful lightweight baseline and made the deployed demo more informative than presenting one score in isolation.
- Confidence is shown as the model's softmax score, not a calibrated probability.

## Dataset and evaluation protocol

The project uses the RGB version of [EuroSAT](https://github.com/phelber/EuroSAT): 27,000 Sentinel-2 image patches at `64 × 64` pixels across:

`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, and `SeaLake`.

| Split | Images | Share |
| --- | ---: | ---: |
| Training | 18,900 | 70% |
| Validation | 5,400 | 20% |
| Test | 2,700 | 10% |

The split was generated with seed `42`, saved to `split_indices_seed42.pth`, and reused by both notebooks. Augmentation is applied only to training images; validation and test images are left unaugmented. CustomCNN receives RGB values scaled to `[0, 1]`, while ResNet18 additionally uses ImageNet mean and standard-deviation normalization. Images are kept at their native `64 × 64` size.

## Deployed system

The browser sends one image to `POST /compare`. FastAPI validates it, applies each model's own preprocessing, runs two ONNX Runtime sessions, and returns each model's top prediction and top-three scores.

The models were converted from PyTorch `.pth` checkpoints to ONNX for CPU-only deployment, with the export scripts verifying the converted graphs. This removed the PyTorch runtime from the production API and kept the two-model service within Render's free-tier memory limit. The exported CustomCNN and ResNet18 files are approximately 8.4 MB and 43 MB; both sessions together use about 196 MB of memory in the deployed process.

The frontend is hosted on Vercel and the API on Render. It includes 50 examples drawn from the saved test split—five per class—and displays a shuffled, class-balanced set of ten. Because these examples have known labels, the UI can show Correct/Incorrect badges; no such claim is made for a user upload.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Reports whether both models loaded |
| `GET` | `/model-info` | Returns classes, preprocessing, and model metadata |
| `POST` | `/compare?top_k=3` | Runs CustomCNN and ResNet18 on one image |
| `POST` | `/predict?top_k=3` | Backward-compatible ResNet18-only inference |

Uploads are limited to 10 MB and 25 million pixels. JPEG, PNG, WebP, TIFF, and BMP are accepted. CORS is restricted to the configured frontend origins.

## Repository structure

```text
.
├── notebooks/                 # Preprocessing, training and analysis
├── scripts/                   # ONNX export and demo-image preparation
├── frontend/                  # Static comparison UI and test examples
├── tests/                     # API and demo-data tests
├── inference.py               # ONNX preprocessing and prediction
├── main.py                    # FastAPI routes and upload validation
├── schema.py                  # Response models
├── requirements-api.txt       # Lightweight production dependencies
└── requirements.txt           # Training and notebook dependencies
```

The dataset, generated split file, `.pth` checkpoints, and `.onnx` models are intentionally excluded from Git. Deployment models are published as assets in the [`model-v1` release](https://github.com/arush296/EuroSAT-Land-Cover-Classification/releases/tag/model-v1).

## Quick start

```bash
git clone https://github.com/arush296/EuroSAT-Land-Cover-Classification.git
cd EuroSAT-Land-Cover-Classification
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "from torchvision.datasets import EuroSAT; EuroSAT(root='data', download=True)"
jupyter lab
```

To run the web app locally, download both ONNX assets from the model release into the repository root, then start the backend and frontend in separate terminals:

```bash
uvicorn main:app --reload
python -m http.server 3000 --directory frontend
```

Open <http://127.0.0.1:3000>. The API documentation is at <http://127.0.0.1:8000/docs>.

## Tests and reproducibility

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The API suite replaces the real ONNX sessions with deterministic test doubles, allowing validation, CORS, health checks, response contracts, `top_k`, and error handling to be tested without loading model weights. A separate test checks that the bundled example manifest is complete and class-balanced.

After retraining, regenerate the deployment models with:

```bash
python scripts/export_onnx.py
python scripts/export_custom_cnn_onnx.py
```

## Limitations and next steps

- The current random split can place visually related images from the same geographic area in different subsets, so it does not measure geographic generalization.
- Softmax confidence has not been calibrated and should not be interpreted as certainty.
- The classifier expects imagery similar to EuroSAT RGB patches; it is not a general satellite-scene classifier.
- Render's free service sleeps after inactivity, so the first request can be slow while the backend wakes up.

Useful next experiments are confidence calibration, geographically disjoint evaluation, and measuring how accuracy changes when only 10%, 25%, 50%, or 100% of the training set is available.
