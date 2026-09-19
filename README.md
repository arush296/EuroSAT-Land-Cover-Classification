# EuroSAT Land-Cover Classification

An end-to-end image-classification project for identifying land-cover types in EuroSAT satellite imagery. The project compares a custom convolutional neural network with a fully fine-tuned ResNet18, exposes the selected model through FastAPI, and provides a small browser frontend.

[Live demo](https://eurosat-classifier.vercel.app) · [API health](https://eurosat-api.onrender.com/health) · [Interactive API documentation](https://eurosat-api.onrender.com/docs)

## Results

Both models use the same saved train, validation, and test indices so their results are directly comparable.

| Model | Correct predictions | Test accuracy | Notes |
| --- | ---: | ---: | --- |
| Custom CNN | 2,485 / 2,700 | 92.04% | Trained from scratch with augmentation |
| ResNet18 | **2,617 / 2,700** | **96.93%** | ImageNet initialization; all layers fine-tuned |

These values were reproduced directly from the saved checkpoints on the fixed test split.

The selected ResNet18 model is exported to ONNX for deployment. Its ONNX and PyTorch outputs were checked locally and produced the same predicted class with a maximum score difference of approximately `2.6e-9` on the verification image.

## Dataset and preprocessing

The project uses the RGB version of [EuroSAT](https://github.com/phelber/EuroSAT), containing 27,000 images across ten classes:

`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, and `SeaLake`.

The dataset is split once with seed `42`, and the generated indices are reused by both notebooks:

| Split | Images | Percentage |
| --- | ---: | ---: |
| Training | 18,900 | 70% |
| Validation | 5,400 | 20% |
| Test | 2,700 | 10% |

Training images receive random horizontal flips, vertical flips, and rotations. Evaluation images do not receive augmentation. ResNet18 inputs additionally use ImageNet mean and standard-deviation normalization. Images remain at their native `64 × 64` resolution.

## Project structure

```text
.
├── EuroSAT.ipynb              # Custom CNN training and evaluation
├── resnet18.ipynb             # ResNet18 fine-tuning and evaluation
├── data_preprocessing.ipynb   # Shared transforms, fixed splits and dataloaders
├── export_onnx.py             # Exports the trained PyTorch model to ONNX
├── inference.py               # Lightweight ONNX preprocessing and inference
├── main.py                    # FastAPI application
├── schema.py                  # API response models
├── frontend/                  # Static HTML, CSS and JavaScript frontend
├── tests/test_api.py          # Automated API tests
├── requirements.txt           # Training, notebooks and local development
├── requirements-api.txt       # Lightweight production API dependencies
└── requirements-dev.txt       # Test dependencies
```

The dataset and trained model files are intentionally excluded from Git. The deployment model is available as an asset on the [`model-v1` GitHub release](https://github.com/arush296/EuroSAT-Land-Cover-Classification/releases/tag/model-v1).

## Local setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/arush296/EuroSAT-Land-Cover-Classification.git
cd EuroSAT-Land-Cover-Classification
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

Download EuroSAT into the expected local directory:

```bash
python -c "from torchvision.datasets import EuroSAT; EuroSAT(root='data', download=True)"
```

The two model notebooks call `data_preprocessing.ipynb` themselves, so they can be run independently after the dataset has been downloaded.

## Run the API locally

Download the ONNX model from the GitHub release:

```bash
curl -L --fail \
  "https://github.com/arush296/EuroSAT-Land-Cover-Classification/releases/download/model-v1/resnet18_eurosat.onnx" \
  -o resnet18_eurosat.onnx
```

Start FastAPI:

```bash
uvicorn main:app --reload
```

Then open:

- API documentation: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>
- Model information: <http://127.0.0.1:8000/model-info>

Example prediction request:

```bash
curl -X POST \
  -F "file=@path/to/satellite-image.jpg" \
  "http://127.0.0.1:8000/predict?top_k=3"
```

Example response:

```json
{
  "filename": "satellite-image.jpg",
  "predicted_class": "Pasture",
  "score": 0.9999,
  "top_predictions": [
    {"class_name": "Pasture", "score": 0.9999},
    {"class_name": "PermanentCrop", "score": 0.0001},
    {"class_name": "Industrial", "score": 0.0}
  ]
}
```

### API endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Reports whether the API and model are ready |
| `GET` | `/model-info` | Returns the architecture, input size and classes |
| `POST` | `/predict?top_k=3` | Validates an uploaded image and returns ranked predictions |

Uploads are limited to 10 MB and 25 million pixels. JPEG, PNG, WebP, TIFF, and BMP images are supported.

## Run the frontend locally

Keep the API running, then open a second terminal:

```bash
python -m http.server 3000 --directory frontend
```

Open <http://127.0.0.1:3000>. The frontend previews the chosen image, calls the FastAPI backend, and displays the top three class probabilities. It also retries the health check while a sleeping Render service wakes up.

## Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/test_api.py -q
```

The tests use a deterministic fake inference session, so the API contract and validation behavior can be tested without loading the real 43 MB model.

## Export a new ONNX model

After retraining and saving `resnet_fully_fine_tuned.pth`, run:

```bash
python export_onnx.py
```

This creates `resnet18_eurosat.onnx`. Both formats are ignored by Git; publish deployment models as release assets instead of committing them to the repository.

## Deployment

### Render backend

The production service uses only `requirements-api.txt`, avoiding the memory cost of importing PyTorch on Render's free tier.

Build command:

```bash
pip install -r requirements-api.txt && curl -L --fail "https://github.com/arush296/EuroSAT-Land-Cover-Classification/releases/download/model-v1/resnet18_eurosat.onnx" -o resnet18_eurosat.onnx
```

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Relevant environment variables:

```text
EUROSAT_CHECKPOINT=resnet18_eurosat.onnx
EUROSAT_DEVICE=cpu
ORT_INTRA_OP_THREADS=1
WEB_CONCURRENCY=1
OMP_NUM_THREADS=1
MALLOC_ARENA_MAX=2
FRONTEND_ORIGINS=https://eurosat-classifier.vercel.app
```

### Vercel frontend

Import the same repository into Vercel and use:

```text
Framework preset: Other
Root directory: frontend
Build command: empty
Output directory: .
```

The backend address is configured in `frontend/config.js`.

## Current limitations and next steps

- Predictions assume images resemble EuroSAT RGB satellite patches; this is not a general-purpose scene classifier.
- Confidence scores have not yet been calibrated.
- The random split does not measure geographic generalization between regions.
- Render's free backend can sleep after inactivity, so the first request may take longer.

Planned extensions include confidence calibration, data-efficiency experiments, and geographically separated evaluation.
