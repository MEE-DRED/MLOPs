# Model Registry

This directory stores the trained model versions and their associated metadata for the Diabetic Retinopathy Detection MLOps project.

## Model Architecture

- **Base model:** MobileNetV2 (ImageNet pretrained)
- **Input:** 224 × 224 × 3 RGB images
- **Output:** Five diabetic retinopathy severity classes
- **Classification head:** Global Average Pooling, Dropout and Dense Softmax

## Model Versions

| Version | Status | Description |
|---|---|---|
| v1 | Accepted | Initial trained model |
| v2 | Accepted | Subsequent trained model |
| v3 | Rejected | Did not meet model acceptance criteria |
| v4 | Accepted | Current model |
| v5 | Rejected | Failed the accuracy-drop acceptance threshold |

## Current Model

The current accepted model is **Version 4**.

- Model file: `model_v4.keras`
- Test accuracy: 59.50%
- Test macro-F1: 49.78%
- Test weighted-F1: 60.09%
- Macro ROC-AUC: 84.11%
- Quadratic weighted kappa: 0.7393

The current model pointer is maintained in `current_model.json`.

## Directory Structure

```text
models/
├── model_v1.keras
├── model_v2.keras
├── model_v3.keras
├── model_v4.keras
├── model_v5.keras
├── current_model.json
├── README.md
└── metadata/
    ├── [version 1 metadata]
    ├── [version 2 metadata]
    ├── [version 3 metadata]
    ├── [version 4 metadata]
    └── [version 5 metadata]
```

## Metadata

Each model version has an associated metadata file containing its:

- Architecture and input configuration
- Training dataset and split information
- Hyperparameters and training configuration
- Validation and test evaluation metrics
- Acceptance status and validation checks
- Model version and parent version

## Model Selection and Acceptance

Models are evaluated against predefined acceptance criteria before they can become the current model. These include model integrity, minimum evaluation sample count, macro-F1 performance, acceptable accuracy drop and class-level recall stability.

A rejected candidate does not replace the current accepted model.

**Important:** Version 4 remains the current accepted model. Version 5 was rejected, so it does not replace version 4.

