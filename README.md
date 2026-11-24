# Diabetic Retinopathy Detection Using Deep Learning and MLOps



| Project information | Status / link |
|---|---|
| Video demonstration | To be added |
| Cloud deployment | Pending |
| GitHub repository | To be added |

## 1. Project Overview

This project explores the classification of diabetic retinopathy (DR) severity from retinal fundus photographs using deep learning and an MLOps workflow.

It extends an earlier Introduction to Machine Learning project, which used tabular health indicators, into medical image classification using **RetinaMNIST-224**.

The model uses **MobileNetV2 transfer learning**, with ImageNet-pretrained weights, to classify retinal images into five diabetic retinopathy severity grades.

Beyond model development, the project is designed around an MLOps lifecycle that includes data validation, preprocessing, training, evaluation, model versioning, prediction, bulk labelled-data upload, retraining, model acceptance, monitoring, and deployment.

### Project objectives

1. Acquire and validate the RetinaMNIST-224 dataset.
2. Explore the dataset through statistical analysis and visualisations.
3. Develop and evaluate a MobileNetV2-based image classifier.
4. Implement single-image prediction through an API and dashboard.
5. Support bulk upload of labelled images for potential model retraining.
6. Implement candidate-model evaluation and an acceptance policy.
7. Track model versions and preserve the current accepted model when a candidate is rejected.
8. Support monitoring, containerisation, load testing, and cloud deployment.

## 2. Dataset

The project uses **RetinaMNIST-224**, from the **MedMNIST v2** collection. It contains 224 × 224 RGB retinal fundus images labelled across five ordered diabetic retinopathy severity grades.

### Dataset distribution

| Split | Images | No DR | Mild | Moderate | Severe | Proliferative DR |
|---|---:|---:|---:|---:|---:|---:|
| Training | 1,080 | 486 | 128 | 206 | 194 | 66 |
| Validation | 120 | 54 | 12 | 28 | 20 | 6 |
| Test | 400 | 174 | 46 | 92 | 68 | 20 |
| **Total** | **1,600** | **714** | **186** | **326** | **282** | **92** |

The original training, validation, and test partitions are preserved. The validation set supports model selection, early stopping, and candidate acceptance. The held-out test set is used for final performance reporting.

### Dataset validation

The recorded notebook run passed the dataset validation checks:

- All images had the expected 224 × 224 × 3 dimensions and `uint8` data type.
- No blank images were detected.
- No duplicates were found within individual splits.
- No identical images were detected across the training, validation, and test splits.
- No blocking issues were reported by the dataset validation process.

The dataset is not included in this repository. Refer to [`data/README.md`](data/README.md) for acquisition and placement instructions.

Dataset source: [MedMNIST](https://medmnist.com/)

## 3. Classification Labels

The model predicts one of five ordered classes:

| Label | Class | Description |
|---:|---|---|
| 0 | No DR | No diabetic retinopathy |
| 1 | Mild | Mild non-proliferative diabetic retinopathy |
| 2 | Moderate | Moderate non-proliferative diabetic retinopathy |
| 3 | Severe | Severe non-proliferative diabetic retinopathy |
| 4 | Proliferative DR | Proliferative diabetic retinopathy |

The classes are ordinal. A prediction that is one grade away from the true label is different from a prediction several grades away. Consequently, the evaluation includes both conventional classification metrics and ordinal performance measures.

## 4. Exploratory Data Analysis

The notebook includes class-distribution analysis, sample-image visualisations, image-feature exploration, augmentation examples, and training/evaluation visualisations.

Four image-derived characteristics were examined:

- **Brightness:** average pixel intensity.
- **Contrast:** variation in pixel intensity.
- **Sharpness:** variance of the Laplacian, used as a basic indicator of edge detail and focus.
- **Retina fraction:** proportion of the image occupied by the detected retinal field rather than the surrounding dark area.

The recorded analysis found the following Spearman associations with the ordered DR grade:

| Image characteristic | Spearman association |
|---|---:|
| Brightness | -0.323 |
| Contrast | -0.104 |
| Sharpness | -0.019 |
| Retina fraction | -0.431 |

These measurements describe image characteristics, not direct clinical measurements of disease. Their association with labels raises a potential shortcut-learning concern: the model may learn acquisition-related differences, such as exposure or image framing, that happen to correlate with the dataset labels.

A balanced logistic-regression baseline using the four image statistics achieved **36.50% test accuracy** and **0.2603 macro-F1**. This is weaker than the deep-learning model, but it shows that simple image characteristics contain some label-associated information.

## 5. Model Architecture and Training

### Architecture

The project uses MobileNetV2 with its original classification head removed and a custom five-class classification head.

```text
Input image (224 × 224 × 3)
        |
Data augmentation (training only)
        |
Rescaling to [-1, 1]
        |
MobileNetV2 backbone
(ImageNet pretrained weights)
        |
Global Average Pooling
        |
Dropout
        |
Dense layer (5 softmax outputs)
        |
Predicted DR class
```

The recorded architecture contains **2,264,389 total parameters**.

### Preprocessing

The preprocessing workflow validates image inputs, converts accepted images to RGB, preserves aspect ratio while fitting images to a square, and resizes them to 224 × 224 pixels.

The model includes its rescaling layer. Therefore, prediction and retraining inputs are supplied in the 0–255 pixel range and scaled within the model.

The application preprocessing path accepts JPEG and PNG images. Invalid, unreadable, empty, or unsupported images are rejected. Retraining labels must be integers from 0 to 4.

### Training strategy

The notebook implements:

- ImageNet-pretrained MobileNetV2 transfer learning.
- A frozen-backbone classification-head training stage.
- Fine-tuning of the top 40 backbone layers.
- Adam optimisation and sparse categorical cross-entropy.
- Balanced class weights to address class imbalance.
- Training-only image augmentation.
- Dropout and L2 regularisation.
- Early stopping and learning-rate reduction.
- Validation macro-F1 for model and training-stage selection.
- A six-trial random hyperparameter search.

The selected configuration in the recorded run used:

| Parameter | Selected value |
|---|---:|
| Dropout | 0.2 |
| Fine-tuning learning rate | 0.0001 |
| Batch size | 32 |
| Random seed | 42 |
| Selected training stage | Stage 2 |
| Best validation macro-F1 | Approximately 0.6086 |

## 6. Model Evaluation

### Accepted model: Version 4

The model registry identifies **model version 4** as the current accepted model. The notebook reloads the registered model and verifies that it reproduces the saved test accuracy.

The recorded test results for version 4 are:

| Metric | Test result |
|---|---:|
| Accuracy | **59.50%** |
| Loss | 1.2445 |
| Macro precision | 0.5354 |
| Macro recall | 0.4963 |
| Macro-F1 | **0.4978** |
| Weighted-F1 | 0.6009 |
| Macro ROC-AUC (one-vs-rest) | **0.8411** |
| Quadratic weighted kappa | **0.7393** |
| Within-one-grade accuracy | **89.75%** |

### Baseline comparison

| Model | Test accuracy | Macro-F1 |
|---|---:|---:|
| Majority-class baseline | 43.50% | — |
| Image-statistics logistic-regression baseline | 36.50% | 0.2603 |
| MobileNetV2 (accepted version 4) | **59.50%** | **0.4978** |

The accepted image-classification model outperformed the simple baselines on test accuracy and macro-F1.

The within-one-grade accuracy of 89.75% indicates that most predictions were either correct or no more than one severity grade away from the true label. However, the model's performance is not uniform across classes.

**Proliferative DR had a recall of 25%**, based on only 20 test images. This is a significant limitation and should not be interpreted as reliable performance for detecting severe disease.

### Retraining candidate: Version 5

The notebook also demonstrates candidate retraining and model acceptance.

The latest candidate was registered as **version 5**, but it was rejected. Its separate evaluation output reported:

| Metric | Version 5 candidate |
|---|---:|
| Test accuracy | 54.00% |
| Macro-F1 | 0.4716 |
| Registry status | Rejected |

The candidate failed the validation accuracy-drop requirement: its validation accuracy decreased by 0.025, exceeding the policy's maximum permitted drop of 0.02.

Although it passed the other recorded integrity, minimum-sample, macro-F1, and class-recall checks, it did not meet the full acceptance policy.

**Version 4 therefore remains the current accepted and served model.** Version 5 is retained as a rejected candidate rather than replacing the accepted model.

### Evaluation interpretation

The results show that:

- The accepted model performed better than the simple baselines.
- Ordinal evaluation provides additional information about the distance between predicted and actual grades.
- Performance is uneven across severity classes.
- Proliferative DR recall is low and based on a small test sample.
- The dataset's image characteristics may introduce shortcut-learning or confounding risks.

The model is an experimental result and is not suitable for clinical decision-making.

## 7. Model Saving and Version Registry

Models are saved in Keras `.keras` format. The notebook performs a save-and-reload integrity check, which passed in the recorded run. The saved model file was approximately 23 MB.

The registry maintains model versions, metadata, and a pointer to the current accepted model.

| Version | Status | Registry outcome |
|---:|---|---|
| 1 | Accepted | Previously accepted |
| 2 | Accepted | Previously accepted |
| 3 | Rejected | Not served |
| 4 | **Accepted** | **Current served model** |
| 5 | Rejected | Latest candidate; version 4 remains current |

Model metadata records information such as training configuration, dataset provenance, evaluation metrics, acceptance decisions, and parent-version information.

This allows rejected candidates to be retained for audit purposes without replacing a better-performing accepted model.

## 8. Prediction Demonstration

The notebook loads the current model through the registry and uses the shared preprocessing and prediction workflow to classify an individual retinal image.

The prediction output includes:

- Predicted class number and class name.
- Confidence score for the predicted class.
- Probability values for all five classes.
- Model version used for prediction.

The recorded demonstration loaded **model version 4**. One example image was classified as **No DR**, with a confidence score of approximately **0.8924**.

The notebook also demonstrates rejection of an invalid image input.

These outputs are experimental model predictions, not medical diagnoses.

## 9. MLOps Application

The repository contains application code intended to support the wider model lifecycle.

The main components include:

| Component | Purpose |
|---|---|
| Data acquisition | Obtain or locate the RetinaMNIST dataset |
| Preprocessing | Validate and prepare images consistently |
| Model training | Train the MobileNetV2 classifier |
| Evaluation | Calculate classification and ordinal metrics |
| Prediction | Run inference on individual images |
| Model registry | Store model versions, metadata, and the current-model pointer |
| API | Provide application endpoints |
| Database | Store application and upload records |
| Bulk upload | Accept multiple labelled images |
| Retraining | Fine-tune a candidate model |
| Monitoring | Track service and model activity |
| Dashboard | Provide a user-facing application interface |

### API

The FastAPI application is intended to expose the following routes. Confirm the route definitions in the source code when running the application.

| Endpoint | Intended purpose |
|---|---|
| `GET /health` | Check API health, model availability, version, and uptime |
| `POST /predict` | Predict the class of one uploaded image |
| `POST /upload-retraining-data` | Upload multiple labelled images |
| `GET /retraining-data` | View uploaded-data counts and eligibility |
| `POST /retrain` | Trigger a retraining job |
| `GET /retrain/status` | Check retraining job status |
| `GET /monitoring` | View monitoring information |
| `GET /models` | View model version history |
| `POST /models/rollback` | Roll back to a previous accepted model |

Start the API from the project root:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

When the API is running, its interactive documentation should be available at:

`http://localhost:8000/docs`

Example single-image prediction:

```bash
curl -F "file=@sample.png" http://localhost:8000/predict
```

The exact upload field names and request formats should be checked against the API implementation.

### Streamlit dashboard

Start the dashboard in a separate terminal:

```bash
streamlit run app/dashboard.py
```

The dashboard is intended to provide access to model status, prediction, dataset visualisations, bulk upload, retraining controls, and monitoring information.

The dashboard expects the API to be available. Configure `DR_API_URL` if the API is running at a different address.

## 10. Bulk Upload and Retraining Workflow

The application is designed to support retraining using newly uploaded labelled retinal images.

The intended workflow is:

1. A user uploads multiple retinal images and provides a label for each.
2. The API validates the image files and labels.
3. Accepted images and upload metadata are stored.
4. The retraining process loads the current accepted model.
5. A candidate is fine-tuned using the uploaded images together with a replay sample from the original training data.
6. The current model and candidate are evaluated against the original validation split.
7. The model acceptance policy determines whether the candidate qualifies for promotion.
8. An accepted candidate becomes the new current version. A rejected candidate is retained for audit, while the previous accepted model remains in service.

The notebook demonstrates model versioning and candidate acceptance/rejection behaviour. The complete end-to-end application workflow should be tested through the running API and dashboard before it is described as fully verified.

## 11. Monitoring

The project includes monitoring functionality intended to track service and model activity.

Monitoring indicators include:

- Service uptime and model availability.
- Current model version.
- Prediction activity and failures.
- Request latency.
- Predicted-class distribution.
- Upload activity.
- Retraining events.
- Changes in predicted-label proportions and image-statistic distributions.

These indicators can help identify unusual activity or potential changes in incoming data. They do not, by themselves, establish model degradation or clinical performance.

## 12. Repository Structure

The project is organised around the notebook, application, model lifecycle code, and deployment configuration.

```text
Diabetic-Retinopathy-MLOps/
├── README.md
├── requirements.txt
├── requirements-docker.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── notebook/
│   ├── diabetic_retinopathy_image_classification.ipynb
│   └── colab_run_app_and_tests.ipynb
├── src/
│   ├── data_acquisition.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── prediction.py
│   ├── retrain.py
│   ├── model_registry.py
│   ├── database.py
│   ├── api.py
│   └── monitoring.py
├── app/
│   └── dashboard.py
├── data/
│   └── README.md
├── models/
│   ├── model_vN.keras
│   ├── metadata/
│   └── current_model.json
├── outputs/
│   ├── figures/
│   ├── reports/
│   └── tables/
├── scripts/
├── tests/
│   └── load/
└── docs/
    ├── architecture.md
    ├── deployment.md
    ├── demonstration_guide.md
    ├── rubric_compliance.md
    ├── submission_checklist.md
    └── test_status.md
```

This is a guide to the intended project organisation. The final repository should reflect the files actually committed.

The dataset, trained models, and generated outputs may be stored outside GitHub when their size makes committing them impractical. If so, provide clear instructions for obtaining them and make sure the application can locate the required files.

Do not commit credentials, API keys, private patient images, or unnecessary runtime database files.

## 13. Installation and Local Execution

The model-development notebook was run successfully in Google Colab. Local execution is optional and may require substantial disk space, memory, and compatible TensorFlow dependencies.

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the test suite, if available:

```bash
python -m unittest discover -s tests -t .
```

Ensure that the required dataset and model artefacts are available before starting services that depend on them.

## 14. Google Colab

Google Colab was used for the recorded model-development and training run.

The recorded environment included:

| Component | Version / configuration |
|---|---|
| Python | 3.13.16 |
| NumPy | 2.4.6 |
| Pandas | 2.2.3 |
| TensorFlow | 2.21.0 |
| Keras | 3.13.2 |
| Accelerator | GPU detected |

The notebook uses Google Drive for persistent storage under:

```text
MyDrive/DR_MLOps/
```

### Running the notebook

1. Open `notebook/diabetic_retinopathy_image_classification.ipynb` in Google Colab.
2. Mount Google Drive when prompted.
3. Confirm that the dataset is available in the configured location or allow the notebook's acquisition process to obtain it.
4. Select a GPU runtime where available.
5. Run the notebook cells in order.
6. Review the dataset validation report, training history, evaluation metrics, and model integrity checks.
7. Confirm the model registry state and the current accepted model.
8. Confirm that required models, metadata, figures, and reports are saved to persistent storage.

A Colab runtime reset clears in-memory variables and files stored only in the temporary runtime. Save artefacts to Google Drive when they need to persist between sessions.

Training time depends on the available runtime and accelerator.

## 15. Docker and Cloud Deployment

The project includes Docker and deployment configuration intended to support containerised execution and cloud hosting.

**Deployment status: pending verification.**

The target deployment described in the project documentation is a Hugging Face Docker Space. Refer to [`docs/deployment.md`](docs/deployment.md) for the deployment procedure.

| Item | Status |
|---|---|
| Docker configuration | Included in the project |
| Docker image build | Requires verification |
| Container startup | Requires verification |
| Multi-container deployment | Requires verification |
| Cloud deployment | Pending |
| Public deployment URL | To be added |

A configuration file being present does not establish that the image builds, the application starts, or the cloud deployment is live. These steps must be executed and tested before being reported as completed.

## 16. Load and Flood-Request Testing

The project includes load-testing resources intended to evaluate application performance under different container configurations.

The assignment requires request simulation with different numbers of Docker containers.

**Verified load-test measurements are not yet available.** Results should be added after running the tests against the containerised application.

| Scenario | Containers | Requests | Concurrency | Successful | Failed | Throughput (req/s) | Mean latency (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| To be measured | 1 | — | — | — | — | — | — | — | — |
| To be measured | 2 | — | — | — | — | — | — | — | — |
| To be measured | 3 | — | — | — | — | — | — | — | — |

For a meaningful comparison, use the same request mix, concurrency, and test duration for each scenario. Record the test environment and retain the generated Locust report or equivalent evidence.

## 17. Demonstration

The video demonstration will be added after recording and publishing.

The demonstration should cover:

- The project and its objectives.
- The running application and dashboard.
- A single-image prediction.
- Bulk upload of labelled images.
- The retraining process and its outcome.
- Model versioning and the acceptance/rejection behaviour.
- Monitoring and deployment, once verified.

Add genuine screenshots from the running application to the repository, preferably under `docs/images/screenshots/`.

Refer to [`docs/demonstration_guide.md`](docs/demonstration_guide.md) for the recording checklist.

## 18. Limitations and Ethical Considerations

This project has several important limitations:

- The dataset is relatively small, with 1,080 training images and 120 validation images.
- The test set contains only 20 Proliferative DR images, making the corresponding class metrics uncertain.
- The recorded results do not provide repeated-seed confidence intervals.
- Image brightness and retinal framing are associated with labels and may contribute to shortcut learning.
- Resizing images to 224 × 224 may remove fine details relevant to image interpretation.
- Images from different cameras, clinical settings, populations, and acquisition conditions may produce domain shift.
- Grad-CAM visualisations indicate model attention but do not prove that the model identifies clinically meaningful lesions.
- The model has not undergone external or clinical validation.
- Retraining with a small number of new images may produce unstable results.
- Validation-based acceptance estimates are subject to uncertainty.
- Real patient images should not be uploaded unless the necessary consent, privacy, security, and governance requirements have been addressed.

## 19. Medical Disclaimer

This project is an educational and experimental machine-learning project. It is **not a medical device**, has **not been clinically validated**, and must **not** be used to diagnose, screen, treat, or make decisions about anyone's health.

The model may produce incorrect predictions, including for severe disease. Its outputs must not replace professional eye examinations or clinical judgement.

Always consult a qualified eye-care professional.

## 20. References

1. Yang, J. et al. (2023). *MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification*. Scientific Data, 10, 41.
2. Liu, R. et al. (2022). *DeepDRiD: Diabetic Retinopathy – Grading and Image Quality Estimation Challenge*. Patterns, 3(6), 100512.
3. Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
4. Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. Proceedings of the IEEE International Conference on Computer Vision.

Dataset information and licence terms: [https://medmnist.com/](https://medmnist.com/)

