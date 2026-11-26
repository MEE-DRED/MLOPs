# Diabetic Retinopathy Detection Using Deep Learning and MLOps

| Project information | Status / link |
|---|---|
| GitHub repository | https://github.com/MEE-DRED/MLOPs.git |
| Local Docker deployment | Verified |
| Dashboard | `http://localhost:8501` |
| API | `http://localhost:8000` |
| Cloud deployment | https://dr-mlops-app.nicesmoke-10bb86fd.southafricanorth.azurecontainerapps.io 

| Video demonstration: https://youtu.be/PhL55tYvOU0

## 1. Project Overview

This project explores the classification of diabetic retinopathy (DR) severity from retinal fundus photographs using deep learning and an MLOps workflow.

It extends an earlier Introduction to Machine Learning project, which used tabular health indicators, into medical image classification using **RetinaMNIST-224** from MedMNIST v2.

The model uses **MobileNetV2 transfer learning**, with ImageNet-pretrained weights, to classify retinal images into five diabetic retinopathy severity grades.

Beyond model development, the project implements an MLOps lifecycle covering data validation, preprocessing, training, evaluation, model versioning, prediction, bulk labelled-data upload, retraining, model acceptance, monitoring, and containerised deployment.

### Project objectives

1. Acquire and validate the RetinaMNIST-224 dataset.
2. Explore the dataset through statistical analysis and visualisations.
3. Develop and evaluate a MobileNetV2-based image classifier.
4. Implement single-image prediction through an API and dashboard.
5. Support bulk upload of labelled images for potential model retraining.
6. Implement candidate-model evaluation and an acceptance policy.
7. Track model versions and preserve the current accepted model when a candidate is rejected.
8. Implement monitoring, containerisation, and load testing.
9. Demonstrate the application through a running Docker deployment.

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

Dataset source: [MedMNIST](https://medmnist.com/).

## 3. Classification Labels

The model predicts one of five ordered classes:

| Label | Class | Description |
|---:|---|---|
| 0 | No DR | No diabetic retinopathy |
| 1 | Mild | Mild non-proliferative diabetic retinopathy |
| 2 | Moderate | Moderate non-proliferative diabetic retinopathy |
| 3 | Severe | Severe non-proliferative diabetic retinopathy |
| 4 | Proliferative DR | Proliferative diabetic retinopathy |

The classes are ordinal. A prediction that is one grade away from the true label is different from a prediction several grades away. Consequently, the evaluation includes conventional classification metrics and ordinal performance measures.

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

The model registry identifies **model version 4** as the current accepted model. The original recorded evaluation for this version reported the following results:

| Metric | Recorded result |
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

These are the historical evaluation results recorded for version 4. They should not be confused with the later notebook evaluation.

### Baseline comparison

| Model | Test accuracy | Macro-F1 |
|---|---:|---:|
| Majority-class baseline | 43.50% | — |
| Image-statistics logistic-regression baseline | 36.50% | 0.2603 |
| MobileNetV2 (historical version 4 evaluation) | **59.50%** | **0.4978** |

The recorded version 4 results outperformed the simple baselines on test accuracy and macro-F1.

### Subsequent Evaluation: Model Version 5

A later evaluation in the notebook resulted in a new candidate, registered as **model version 5**. However, the candidate did not meet the model acceptance criteria and was therefore rejected.

| Metric | Candidate evaluation |
|---|---:|
| Test accuracy | 54.00% |
| Macro-F1 | 0.4716 |
| Registry status | Rejected |

The candidate failed the validation accuracy-drop requirement: its validation accuracy decreased by 0.025, exceeding the policy's maximum permitted drop of 0.02.

It passed the other recorded integrity, minimum-sample, macro-F1, and class-recall checks. However, it did not satisfy the complete acceptance policy and therefore did not replace version 4.

The later notebook classification report recorded the following class-specific results:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No DR | 0.827 | 0.661 | 0.735 | 174 |
| Mild | 0.269 | 0.696 | 0.388 | 46 |
| Moderate | 0.420 | 0.370 | 0.393 | 92 |
| Severe | 0.659 | 0.426 | 0.518 | 68 |
| Proliferative DR | 0.353 | 0.300 | 0.324 | 20 |

The later evaluation recorded 54.00% test accuracy and 0.4716 macro-F1. Proliferative DR recall was 30%, based on only 20 test images.

### Evaluation interpretation

The results show that:

- The historical accepted model outperformed the simple baselines.
- Ordinal evaluation provides additional information about the distance between predicted and actual

## 7. Model Saving and Version Registry

The trained model is saved in Keras `.keras` format. After saving, it is loaded again to check that it works correctly and can still make predictions.

The project uses a model registry to keep track of saved model versions and their evaluation results. It also identifies which version is currently being used by the application.

| Version | Status |
|---|---|
| v1 | Accepted |
| v2 | Accepted |
| v3 | Rejected |
| v4 | Accepted – Current model |
| v5 | Rejected |

Version 4 remains the model served by the application. Version 5 was registered after a later notebook evaluation but did not meet the acceptance criteria, so it was not promoted.

The registry also stores information about each model, including its architecture, training configuration, evaluation metrics, dataset details, and acceptance decision. This makes it easier to track previous experiments and see why a model was accepted or rejected.

## 8. Prediction Demonstration

The application loads the current model through the registry and uses the shared preprocessing and prediction workflow to classify an individual retinal image.

The prediction output includes:

- Predicted class number and class name.
- Confidence score for the predicted class.
- Probability values for all five classes.
- Model version used for prediction.

### Verified application prediction

A test image named `0002.jpg` was submitted through the dashboard and the API.

| Output | Result |
|---|---|
| Image | `0002.jpg` |
| Predicted class | 1 — Mild |
| Confidence | 76.0% |
| Model version | 4 |
| Dashboard result | Successful |
| API result | HTTP 200; same predicted class |

The dashboard and API returned consistent prediction results.

The probability values are model softmax scores. They are not calibrated medical risk estimates.

The application also correctly rejected an unreadable test file with HTTP **422 Unprocessable Entity**.

These outputs are experimental model predictions, not medical diagnoses.

## 9. MLOps Application

The project includes a FastAPI backend and a Streamlit dashboard to support model prediction, data uploads, model versioning, retraining, and monitoring.

### Main Components

| Component | Purpose |
|---|---|
| Data acquisition | Obtain the RetinaMNIST dataset |
| Preprocessing | Validate and prepare retinal images |
| Model training | Train the MobileNetV2 classifier |
| Evaluation | Calculate classification and ordinal metrics |
| Prediction | Classify individual retinal images |
| Model registry | Track model versions and evaluation results |
| API | Handle prediction and other application requests |
| Database | Store application records and upload information |
| Bulk upload | Accept multiple labelled images |
| Retraining | Fine-tune and evaluate candidate models |
| Monitoring | Track service, prediction, upload, and retraining activity |
| Dashboard | Provide a user interface for the application |

### API

The FastAPI application exposes the following endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check API health, model availability, version, and uptime |
| `POST /predict` | Predict the class of an uploaded image |
| `POST /upload-retraining-data` | Upload labelled images |
| `GET /retraining-data` | View uploaded-data counts and retraining eligibility |
| `POST /retrain` | Trigger a retraining job |
| `GET /retrain/status` | Check retraining job status |
| `GET /monitoring` | View monitoring information |
| `GET /models` | View model version history |
| `POST /models/rollback` | Roll back to a previous accepted model |

The `/retrain`, `/upload-retraining-data`, and `/models/rollback` endpoints require API-key authentication.

The API was tested through HTTP requests during local Docker and cloud deployment verification.

To start the API outside Docker:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

The `/docs` route returned HTTP 404 during testing and is not available as an interactive API documentation page.

Example single-image prediction:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample.png"
```

### Streamlit Dashboard

To start the dashboard outside Docker:

```bash
streamlit run app/dashboard.py
```

The dashboard provides:

- Overview of API and model status.
- Single-image prediction.
- Dataset insights and image characteristics.
- Bulk upload of labelled images.
- Retraining status and candidate acceptance information.
- Monitoring information.

The dashboard communicates with the API. Set `DR_API_URL` if the API is running at a different address.

10. Bulk Upload and Retraining Workflow
The application supports bulk uploading labelled retinal images and includes a retraining workflow for fine-tuning the current model.
Retraining Workflow
The intended workflow is:
1. A user uploads retinal images and provides a label for each image.
2. The application checks that the images are valid and the labels are within the supported classes.
3. Accepted images and their metadata are stored.
4. When retraining is triggered, the application loads the current accepted model.
5. The candidate model is fine-tuned using the uploaded images together with a replay sample from the original training data.
6. The current model and candidate are evaluated using the original validation data.
7. The acceptance policy determines whether the candidate can replace the current model.
8. If accepted, the candidate becomes the new current model. If rejected, the existing accepted model remains in service.
Verified Bulk Upload Test
A bulk upload test was performed through the deployed dashboard using five labelled retinal images, covering all five classes.
Result	Count
Total images submitted	5
Accepted images	5
Rejected images	0
Images stored after upload	5The application successfully accepted all five images.

The upload process also includes checks for invalid labels, unreadable images, duplicate files, and images matching those in the original dataset's training, validation, or test splits. 

### Retraining

The application includes a manual retraining option and a configurable automatic trigger. The retraining process uses newly uploaded labelled images together with samples from the original training dataset.

Before a candidate model can replace the current model, it is evaluated against the existing model using the validation data. The acceptance policy determines whether the candidate is promoted or rejected.

The notebook demonstrates this process through model evaluation and version registration. In the subsequent evaluation, candidate version 5 did not meet the acceptance criteria and was rejected, leaving version 4 as the current model.

## 11. Monitoring

The application includes monitoring functionality for service and model activity.

Monitoring indicators include:

- Service uptime and model availability.
- Current model version.
- Prediction activity and failures.
- Request latency.
- Predicted-class distribution.
- Upload activity.
- Retraining events.
- Changes in predicted-label proportions and image-statistic distributions.

During testing, the dashboard displayed the API status, current model version, prediction activity, upload counts, and retraining information. These indicators provide an overview of application activity and can help identify unusual changes in incoming data.

## 12. Repository Structure

The project is organised around the notebook, application, model lifecycle code, deployment configuration, and test evidence.

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
│   ├── tables/
│   └── screenshots/
├── scripts/
├── tests/
│   └── load/
└── docs/
    ├── architecture.md
    └── test_status.md
```

The dataset is not committed to the repository. Refer to `data/README.md` for acquisition instructions.

The `outputs/reports/` directory contains the generated flood-test JSON reports. The `outputs/screenshots/` directory contains dashboard demonstration screenshots.

Do not commit credentials, API keys, private patient images, or unnecessary runtime database files.

## 13. Installation and Local Execution

The model-development notebook was run in Google Colab. Local execution may require substantial disk space, memory, and compatible TensorFlow dependencies.

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

Run the test suite:

Run the test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v --ignore=tests/test_submission_zip.py -k "not real_repo_has_everything_the_dockerfile_copies"
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

## 15. Docker Deployment

The project was successfully built and deployed locally using Docker Desktop and Docker Compose.

### Verified deployment results

| Item | Result |
|---|---|
| Docker image build | Successful |
| Docker Compose configuration | Valid |
| API container startup | Successful |
| Dashboard container startup | Successful |
| API health check | Healthy |
| Model loaded by API | Version 4 |
| Dashboard access | Successful |
| API health endpoint | Successful |

The application was accessed at:

- Dashboard: `http://localhost:8501`
- API: `http://localhost:8000`
- API health: `http://localhost:8000/health`

The health endpoint confirmed that the API was online, model loading was successful, and version 4 was being served.

To build and start the application:

```bash
docker compose build
docker compose up -d
```

To inspect container status:

```bash
docker compose ps
```

To inspect logs:

```bash
docker compose logs
```

### Cloud Deployment

The application was also deployed to Microsoft Azure using Azure Container Apps. The Docker image was published to Docker Hub and used for the deployment.

The deployed dashboard is publicly accessible at:

https://dr-mlops-app.nicesmoke-10bb86fd.southafricanorth.azurecontainerapps.io

The Azure deployment was tested successfully. The dashboard and API health status were confirmed, and model version 4 loaded and served predictions.

The local Docker deployment and Azure deployment were both verified. The Azure load tests and replica comparisons are documented in Section 16.

## 16. Load and Flood-Request Testing

Flood-request tests were executed against the running Docker API using `scripts/flood_test.py`.

The tests covered the prediction endpoint and the health endpoint.

### Results

| Scenario | Endpoint | Requests | Concurrency | Successful | Failed | Throughput (req/s) | Mean latency (ms) | P95 (ms) | P99 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prediction test 1 | POST `/predict` | 20 | 2 | 20 | 0 | 1.86 | 1,070.4 | 2,961.5 | 3,590.7 |
| Prediction test 2 | POST `/predict` | 50 | 5 | 50 | 0 | 2.76 | 1,760.4 | 4,731.6 | 5,487.8 |
| Health test | GET `/health` | 500 | 25 | 500 | 0 | 147.54 | 162.2 | 349.3 | 408.2 |
| Prediction test 3 | POST `/predict` | 100 | 10 | 100 | 0 | 3.09 | 3,173.9 | 8,698.6 | 9,506.0 |

**Overall result: 670 successful requests and zero failed requests.**

All prediction requests returned HTTP 200. All health requests returned HTTP 200.

### Interpretation

The API completed all tested requests without failures.

The health endpoint handled 500 requests at concurrency 25, achieving 147.54 requests per second with a mean latency of 162.2 ms.

Prediction throughput increased from 1.86 to 3.09 requests per second as the tested concurrency increased from 2 to 10. However, prediction latency also increased. At concurrency 10, mean latency was approximately 3.17 seconds and P95 latency was approximately 8.70 seconds.

These results demonstrate successful request handling under the tested conditions. They do not establish performance under unlimited traffic, and they do not demonstrate the benefits of running multiple API containers.

The four generated JSON reports are retained in `outputs/reports/`.

### Azure Dashboard Load Tests

Additional load tests were conducted against the publicly deployed Azure dashboard using Locust. Each test ran for 60 seconds.

| Concurrent users | Requests | Failed requests | Average response (ms) | P95 (ms) | Throughput (req/s) |
|---:|---:|---:|---:|---:|---:|
| 5 | 140 | 0 | 107 | 260 | 2.41 |
| 10 | 273 | 0 | 112 | 270 | 4.64 |
| 20 | 539 | 0 | 105 | 130 | 9.10 |
| 30 | 827 | 0 | 104 | 170 | 14.07 |
| 50 | 1,372 | 0 | 106 | 140 | 23.14 |

All five tests completed without failed requests.

These tests targeted the dashboard homepage. They measured the application's response to concurrent dashboard requests, not the performance of image prediction.

### Replica Comparison

The Azure deployment was also tested with one, two, and three replicas under the same load of 50 concurrent users for 60 seconds.

| Metric | 1 replica | 2 replicas | 3 replicas |
|---|---:|---:|---:|
| Requests | 1,355 | 1,351 | 1,385 |
| Failed requests | 0 | 0 | 0 |
| Average response (ms) | 106 | 107 | 106 |
| Median response (ms) | 94 | 94 | 96 |
| P95 (ms) | 170 | 120 | 140 |
| Maximum response (ms) | 550 | 930 | 530 |
| Throughput (req/s) | 23.20 | 22.89 | 23.47 |

The results were similar across all three configurations. Three replicas achieved slightly higher throughput, but the difference was small. Increasing the replica count did not produce a substantial performance improvement under this test.

These tests measured dashboard homepage requests rather than image prediction requests.

### Scaling Evaluation

The Azure deployment was tested using one, two, and three replicas under the same load of 50 concurrent users. All three configurations handled the requests without failures.

The results showed similar performance across the configurations, with three replicas recording slightly higher throughput. However, the improvement was not substantial under the tested conditions.

## 17. Demonstration

Dashboard screenshots are retained under `outputs/screenshots/`.

The captured sections include:

- Overview and API/model status.
- Single-image prediction.
- Dataset insights.
- Image characteristics.
- Bulk upload.
- Retraining.
- Monitoring.

The prediction demonstration uses `0002.jpg`, classified as Mild with 76.0% confidence by model version 4.

The notebook demonstrates candidate model evaluation and rejection. Candidate version 5 was rejected because it did not meet the acceptance criteria, leaving version 4 as the current accepted model. The Azure dashboard also provides the retraining interface, although a live retraining run was not completed during cloud testing.

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

Dataset information and licence terms: [MedMNIST](https://medmnist.com/).