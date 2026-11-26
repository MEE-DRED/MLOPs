# Test and Verification Status

This document records the tests executed, their results, and the components verified in their intended runtime environments. It distinguishes automated sandbox tests from live Docker application tests.

## 1. Model Evaluation

**Current accepted model: Version 4**

The project uses RetinaMNIST-224 (MedMNIST v2), containing 1,600 images across five diabetic retinopathy severity classes.

| Dataset split | Images |
|---|---:|
| Training | 1,080 |
| Validation | 120 |
| Test | 400 |

Model versions 1, 2, and 4 passed their recorded acceptance checks. Versions 3 and 5 were rejected during the notebook model-development workflow. Version 4 is the current accepted model.

### Version 4 Test Results

The recorded version 4 evaluation results are:

| Metric | Result |
|---|---:|
| Test accuracy | 59.50% |
| Test loss | 1.2445 |
| Macro precision | 53.54% |
| Macro recall | 49.63% |
| Macro-F1 | 49.78% |
| Weighted-F1 | 60.09% |
| Macro ROC-AUC | 84.11% |
| Quadratic weighted kappa | 0.7393 |
| Within-one-grade accuracy | 89.75% |
| Majority-class baseline accuracy | 43.50% |

The recorded acceptance checks for version 4 covered data integrity, minimum evaluation samples, macro-F1 comparison, permitted accuracy drop, and class-level recall.

Version 5 was rejected because its validation accuracy dropped by 0.025, exceeding the configured maximum drop of 0.02. Its separate test evaluation reported 54.00% accuracy and 0.4716 macro-F1. Version 4 therefore remained the current accepted model.

The version 4 metadata records GPU availability during training (`/physical_device:GPU:0`). This does not establish that every application or notebook component was executed on a GPU.

**Model evaluation limitation:** These results are experimental evaluation results on RetinaMNIST-224. They do not establish clinical validity or suitability for medical diagnosis.

## 2. Automated Tests Executed in the Sandbox

Automated tests were executed to check the project's data handling, preprocessing, model evaluation, registry, prediction, database, upload, retraining, monitoring, dashboard, and service logic.

### Test Command

The following command was used for the main test run on Windows:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v --ignore=tests/test_submission_zip.py -k "not real_repo_has_everything_the_dockerfile_copies"
```

### Test Results

| Result | Count |
|---|---:|
| Passed | 196 |
| Skipped | 3 |
| Deselected | 1 |
| Subtests passed | 18 |
| Warnings | 16 |
| Failed | 0 |

The test run completed in 94.38 seconds.

The three skipped tests required the original RetinaMNIST dataset, which was not available in the test environment. One platform-specific executable-permission test was excluded on Windows.

### Submission ZIP Tests

The submission ZIP tests were executed separately:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_submission_zip.py -v
```

**Result: 7 passed.**

### Test Environment

The tests were executed in the local Python environment. Some tests use simulated components or test doubles to verify application logic without requiring a complete live application runtime.

The automated tests complement the live Docker and Azure tests documented in the following sections. They do not replace testing with the actual application dependencies or clinical validation.

## 3. Notebook Testing

The model-development notebook was used for dataset preparation, model training, evaluation, and model registry operations.

The notebook evaluation recorded results for the validation and test datasets. It also included a comparison against an image-statistics-only baseline.

The notebook demonstrated model evaluation and acceptance decisions. During the recorded evaluation, candidate version 5 was rejected because its validation accuracy dropped by 0.025, exceeding the configured maximum drop of 0.02. Version 4 remained the current accepted model.

### Model Loading Correction

A Keras model-loading issue was identified during development:

```text
AttributeError: module 'keras._tf_keras.keras' has no attribute 'saving'
```

The affected source loading paths were corrected to use `keras.models.load_model`, including `Predictor.load_path` and `KerasTrainer.load`. The notebook loading code was also updated.

The source guard test checks against reintroducing the problematic `keras.saving` namespace.

The corrected application loading and prediction paths were subsequently tested in Docker. The API successfully loaded model version 4 and completed real prediction requests.

**Remaining limitation:** Successful application model loading and inference do not establish that every TensorFlow-dependent notebook cell, training experiment, or retraining configuration was executed end to end.

## 4. Docker Deployment and Live Application Verification

The application was built and tested locally using Docker Desktop and Docker Compose. It was also deployed to Microsoft Azure using Azure Container Apps.

### Local Docker Deployment Checks

| Check | Result |
|---|---|
| Docker engine available | Passed |
| Docker Compose configuration validation | Passed |
| Docker image build | Passed |
| API container startup | Passed |
| Dashboard container startup | Passed |
| API health check | Passed |
| Model loaded by API | Passed; version 4 |
| Dashboard opened in browser | Passed |
| API health endpoint opened in browser | Passed |

The local application was accessed at:

- Dashboard: `http://localhost:8501`
- API: `http://localhost:8000`
- Health endpoint: `http://localhost:8000/health`

The health endpoint returned a successful response confirming:

- API status: `ok`
- API: `up`
- Model loaded: `true`
- Model version: `4`
- Model error: `null`
- Retraining running: `false`

### Azure Cloud Deployment

The Docker image was published to Docker Hub and deployed using Azure Container Apps.

The public dashboard is accessible at:

https://dr-mlops-app.nicesmoke-10bb86fd.southafricanorth.azurecontainerapps.io/

The Azure application was opened successfully. The dashboard displayed the API as online, and model version 4 loaded successfully.

The deployment was also tested using public dashboard load tests and replica comparisons, documented in Section 8.

**Deployment scope:** Both local Docker deployment and Azure cloud deployment were completed and verified. The application successfully served predictions using model version 4.


## 5. Live Prediction API Tests

The prediction endpoint was tested through actual HTTP requests against the running Docker API.

### Valid image test

A retinal image named `0002.jpg` was submitted to:

```text
POST http://localhost:8000/predict
```

The API expects the multipart upload field to be named `file`.

The response returned:

| Output | Result |
|---|---|
| HTTP status | 200 |
| Predicted class | 1 |
| Predicted class name | Mild |
| Confidence | Approximately 76.04% |
| Model version | 4 |

The dashboard independently processed the same image and displayed class 1 — Mild, with 76.0% confidence.

The dashboard and API prediction results were consistent.

### Invalid image test

A text file containing non-image content was submitted to the prediction endpoint.

The API returned:

- HTTP status: **422**
- Error: invalid image / unreadable image.

This is the expected validation behaviour. The API remained healthy after the invalid request, and model version 4 remained loaded.

## 6. Bulk Upload and Retraining Tests

### Bulk Upload

The dashboard bulk-upload workflow was tested using five labelled retinal images, with one image representing each of the five supported classes.

| Check | Result |
|---|---:|
| Images submitted | 5 |
| Images accepted | 5 |
| Images rejected | 0 |
| Classes represented | 5 |

All five images were accepted and stored successfully.

The application also checks uploaded images for invalid labels, unreadable files, duplicates, and matches with the original dataset.

### Retraining

The application includes a manual retraining workflow and a configurable automatic trigger. However, a complete live retraining run was not achieved during cloud testing.

The deployed environment required at least 20 valid labelled images across two or more classes. It also needed the original RetinaMNIST dataset for replay sampling and validation. These requirements were not met during the cloud test.

The notebook separately demonstrates candidate model evaluation and rejection. Candidate version 5 was rejected because it did not meet the configured acceptance criteria, leaving version 4 as the current accepted model.

### Persistence Test

The Azure application was stopped and started to check whether uploaded data persisted.

The application recovered successfully, but the five previously uploaded images and session monitoring counters were no longer available after the restart.

This indicates that uploaded application data was not persistent across the Stop–Start operation in the tested deployment.

## 7. Monitoring Verification

The dashboard Monitoring page was opened during application testing.

The dashboard displayed service and model information, including:

- API availability.
- Current model version.
- Service uptime.
- Prediction activity.
- Successful and failed prediction counts.
- Uploaded image counts.
- Retraining status and events.

During cloud testing, five uploaded images were visible before the Stop–Start test. After the application was started again, the uploaded images and session monitoring counters were no longer available.

Monitoring indicators provide information about application activity. They should not be interpreted as independent measures of clinical model performance.

## 8. Load and Flood-Request Testing

Load and flood-request tests were conducted against the local Docker API and the deployed Azure application to evaluate request handling, response times, throughput, and performance under different loads.

### 8.1 Local Docker API Flood Tests

The local tests were executed using:

```bash
python -m scripts.flood_test
```

The tests used the retinal image `prediction_test_images/0002.jpg` for prediction requests.

| Scenario | Endpoint | Requests | Concurrency | Successful | Failed | Success rate | Throughput (req/s) | Mean latency (ms) | P95 (ms) | P99 (ms) | Max (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prediction test 1 | POST `/predict` | 20 | 2 | 20 | 0 | 100% | 1.86 | 1,070.4 | 2,961.5 | 3,590.7 | 3,748.0 |
| Prediction test 2 | POST `/predict` | 50 | 5 | 50 | 0 | 100% | 2.76 | 1,760.4 | 4,731.6 | 5,487.8 | 5,689.1 |
| Health test | GET `/health` | 500 | 25 | 500 | 0 | 100% | 147.54 | 162.2 | 349.3 | 408.2 | 442.3 |
| Prediction test 3 | POST `/predict` | 100 | 10 | 100 | 0 | 100% | 3.09 | 3,173.9 | 8,698.6 | 9,506.0 | 9,734.8 |
| **Total** | | **670** | | **670** | **0** | **100%** | | | | | |

All prediction requests returned HTTP 200, and all health requests returned HTTP 200.

### 8.2 Azure Dashboard Load Tests

The deployed Azure application was tested using Locust at different concurrency levels. Each test ran for 60 seconds.

| Concurrent users | Requests | Failed requests | Average response (ms) | P95 (ms) | Throughput (req/s) |
|---:|---:|---:|---:|---:|---:|
| 5 | 140 | 0 | 107 | 260 | 2.41 |
| 10 | 273 | 0 | 112 | 270 | 4.64 |
| 20 | 539 | 0 | 105 | 130 | 9.10 |
| 30 | 827 | 0 | 104 | 170 | 14.07 |
| 50 | 1,372 | 0 | 106 | 140 | 23.14 |

All tests completed without failed requests. Average response times remained relatively consistent across the tested concurrency levels.

**Note:** These Azure tests measured dashboard homepage requests, not image prediction requests.

### 8.3 Azure Replica Comparison

The Azure deployment was further tested with one, two, and three replicas under the same load of 50 concurrent users for 60 seconds.

| Metric | 1 replica | 2 replicas | 3 replicas |
|---|---:|---:|---:|
| Requests | 1,355 | 1,351 | 1,385 |
| Failed requests | 0 | 0 | 0 |
| Average response (ms) | 106 | 107 | 106 |
| Median response (ms) | 94 | 94 | 96 |
| P95 (ms) | 170 | 120 | 140 |
| Maximum response (ms) | 550 | 930 | 530 |
| Throughput (req/s) | 23.20 | 22.89 | 23.47 |

All three configurations completed their tests without failures. The results showed similar performance, with three replicas recording slightly higher throughput. However, increasing the replica count did not produce a substantial performance improvement.

### 8.4 Generated Test Reports

The local flood-test script saved the following JSON reports in `outputs/reports/`:

| Report | Test |
|---|---|
| `flood_test_predict_20261124_175919.json` | 20 prediction requests, concurrency 2 |
| `flood_test_predict_20261124_180031.json` | 50 prediction requests, concurrency 5 |
| `flood_test_health_20261124_181458.json` | 500 health requests, concurrency 25 |
| `flood_test_predict_20261124_181616.json` | 100 prediction requests, concurrency 10 |

### 8.5 Overall Interpretation

The local Docker API handled all 670 tested requests without failures, including actual image prediction and health-check requests.

The Azure dashboard also handled all tested homepage requests without failures at concurrency levels ranging from 5 to 50 users. The replica comparison showed broadly similar performance across one, two, and three replicas.

The results demonstrate successful request handling under the specific test conditions. They do not establish performance under unlimited traffic or different hardware configurations.

## 9. Dashboard Screenshot Evidence

Screenshots were captured from the running dashboard and saved in the project's `outputs/screenshots/` directory.

The captured dashboard sections include:

| Dashboard section | Evidence |
|---|---|
| Overview | API status, current model, activity metrics, and system workflow |
| Predict | Uploaded image, predicted class, confidence, model version, and probability scores |
| Dataset insights | Class distribution and retraining-image distribution |
| Image characteristics | Brightness, contrast, sharpness, retinal framing, and associated analysis |
| Bulk upload | Upload requirements, accepted image count, invalid-upload count, and class coverage |
| Retraining | Retraining workflow and candidate outcome |
| Monitoring | API status, model availability, activity, and monitoring indicators |

The screenshots provide visual evidence of the running application. They do not replace the underlying API responses or generated test reports.


## 10. Final Test Summary

| Test / verification | Result |
|---|---|
| Model evaluation for version 4 | Passed recorded acceptance checks |
| Automated pytest suite | 196 passed, 3 skipped, 1 deselected |
| Additional subtests | 18 passed |
| Submission ZIP tests | 7 passed |
| Notebook model evaluation | Completed; candidate version 5 rejected |
| Docker Compose configuration | Passed |
| Docker image build | Passed |
| Local Docker API and dashboard startup | Passed |
| Azure Container Apps deployment | Completed |
| API health check | Passed; model version 4 loaded |
| Valid image prediction | Passed; HTTP 200 |
| Invalid image rejection | Passed; HTTP 422 |
| Bulk upload | Passed; 5 labelled images accepted |
| Notebook candidate rejection | Demonstrated; version 5 rejected |
| Current-model preservation | Version 4 remained current |
| Local Docker flood tests | Passed; 670/670 requests successful |
| Azure dashboard load tests | Completed at 5, 10, 20, 30 and 50 concurrent users |
| Azure replica comparison | Completed for 1, 2 and 3 replicas |
| Dashboard screenshot capture | Completed, subject to repository file check |
| Video demonstration | https://youtu.be/PhL55tYvOU0 
| Clinical validation | Not performed |

The separate submission ZIP test suite completed with 7 passed.

The local Docker tests verified actual image prediction, invalid-image handling, API health, and request handling. The Azure tests verified the public dashboard, replica configurations, and dashboard response under the documented loads.

The cloud bulk-upload test accepted five labelled images. The notebook separately demonstrated candidate evaluation and rejection, with version 4 remaining the current accepted model.

The Stop–Start test confirmed that the Azure application recovered.

**Overall conclusion:** The project has a verified local Docker deployment and Azure cloud deployment, a working dashboard and API, successful prediction and bulk-upload tests, a notebook-based model acceptance and rejection demonstration, and completed local and Azure load tests. 

The system is an experimental academic demonstration, not a medical device. It has not been clinically validated and must not be used for medical decisions.
