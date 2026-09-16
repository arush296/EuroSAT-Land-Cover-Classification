## Project Status

This project is actively under development.

### Completed

- EuroSAT data-loading and preprocessing pipeline
- Train/validation/test split
- Custom CNN implementation
- Dropout and no-dropout comparison
- Custom CNN best test accuracy: 92.85%
- Pretrained ResNet18 full fine-tuning: ~96.41% test accuracy, best validation accuracy around 97%
- ResNet18 fixed feature extraction (frozen backbone, only final FC layer trained): ~81.41% test accuracy
- Transfer learning comparison and analysis: full fine-tuning substantially outperformed frozen features by ~15 percentage points in test accuracy

### In Progress

- Per-class evaluation and confusion matrix
- Misclassification analysis

### Planned

- Data augmentation
- Data-efficiency experiments
- Confidence calibration
- Geographic generalization
