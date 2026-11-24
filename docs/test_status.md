# Test and Verification Status

This document records the tests executed, their results, and the components that have not yet been verified in their intended runtime environments.

## 1. Model Evaluation

**Current accepted model: Version 4**

The project uses RetinaMNIST-224 (MedMNIST v2), containing 1,600 images across five diabetic retinopathy severity classes.

| Dataset split | Images |
|---|---:|
| Training | 1,080 |
| Validation | 120 |
| Test | 400 |

Model versions 1, 2, and 4 passed the acceptance checks. Versions 3 and 5 were rejected. Version 4 is the current accepted model, selected from version 2.

**Version 4 test results (n = 400)**

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

All five model acceptance checks passed for version 4: data integrity, minimum evaluation samples, macro-F1 comparison, permitted accuracy drop, and no class collapse.

Version 5 was rejected because its accuracy dropped by 0.025, exceeding the configured maximum drop of 0.02.

The version 4 metadata records GPU availability during training (`/physical_device:GPU:0`). This does not establish that every application or notebook component was executed on a GPU.

**Model evaluation limitation:** These results are experimental evaluation results on RetinaMNIST-224. They do not establish clinical validity or suitability for medical diagnosis.

## 2. Automated Tests Executed in the Sandbox

**Command:**

`python -m unittest discover -s tests -t .`

The test environment used Python 3.12. TensorFlow, FastAPI, Streamlit, network access, and GPU access were unavailable.

| Test file | Tests | Coverage |
|---|---:|---|
| `test_data_acquisition.py` | 14 | Dataset loading, corrupt and incomplete files, label ranges, leakage, blank images, data types, and Drive-copy logic. Includes 3 tests on the supplied dataset. |
| `test_preprocessing.py` | 16 | Image decoding, validation and rejection, padding versus cropping, label validation, and model-input contract. |
| `test_image_analysis.py` | 6 | Brightness, contrast, and sharpness calculations. |
| `test_evaluate.py` | 22 | Evaluation metrics, integrity checks, and model acceptance policy across 8 scenarios. |
| `test_model_registry.py` | 12 | Model versioning, rejected versions, rollback, atomic writes, and persistence across restarts. |
| `test_prediction.py` | 5 | Prediction output structure, missing-model handling, and invalid-image rejection before inference. |
| `test_database.py` | 11 | Database schema, persistence, duplicate handling, statistics, event lifecycle, and interrupted-job recovery. |
| `test_upload.py` | 10 | Bulk uploads, missing or invalid labels, invalid images, duplicates, dataset leakage, hostile filenames, and label parsing. |
| `test_retraining.py` | 24 | Retraining acceptance and rejection, crash handling, current-model initialization, test-set exclusion, stratified replay, single-flight locking, and automatic trigger thresholds, cooldown, and failure handling. |
| `test_monitoring.py` | 5 | Class-distribution shift, image-statistics drift, and minimum-sample checks. |
| `test_dashboard_helpers.py` | 13 | Dashboard helper functions and API client behaviour, including payload formatting, error mapping, API keys, and unreachable API handling. |
| `test_dashboard.py` | 19 | Dashboard pages tested with a stand-in Streamlit module and the real client, service, database, and registry. Covers prediction, insights, uploads, retraining, polling, rollback, monitoring, and auto-refresh. |
| `test_scripts.py` | 22 | Runtime preparation, Space bundle generation, Dockerfile COPY sources, flood-test and end-to-end test engines, and dashboard admin-password protection. |
| `test_submission_zip.py` | 7 | ZIP validation, required files, missing models, dangling pointers, credential exclusions, dataset and cache exclusions, layout, and draft mode. |
| `test_source_guards.py` | 2 | Source checks preventing use of the problematic `keras.saving` namespace. |
| `test_service.py` | 14 | Health checks, prediction logging, upload-to-retraining flow, model reload, rollback, automatic retraining triggers, and monitoring snapshots. |

**Mutation checks**

Mutation testing was also performed on selected components. The retraining code was deliberately modified to always accept candidates and to remove locking and cooldown protections. Dashboard behaviour was also modified to disable retraining safeguards, allow uploads without labels, and stop polling. The relevant tests failed under these changes and passed again after the original code was restored.

**Test environment limitation**

The sandbox did not contain the full application runtime. Dashboard tests used a stand-in Streamlit module, and some retraining tests used simulated training components. These tests verify application logic and expected behaviour but do not replace full integration testing with the actual dependencies.

## 3. Notebook Testing

The training notebook was executed using:

`python tools/run_notebook_cpu.py`

The runner successfully executed all 11 non-TensorFlow code cells against the supplied dataset. TensorFlow-dependent cells were skipped.

The notebook's application and testing sections (19–28) were also executed in the sandbox using a stub model. This was a plumbing check only; the stub-generated metrics are not valid model evaluation results.

A Keras model-loading issue was identified:

`AttributeError: module 'keras._tf_keras.keras' has no attribute 'saving'`

The source code was corrected to use `keras.models.load_model` in the relevant loading paths, including `Predictor.load_path` and `KerasTrainer.load`. The notebook loading code was also updated. The source guard test checks against reintroducing the problematic namespace.

**Not yet verified after the fix:** The corrected loading and application paths have not been confirmed through a complete real-TensorFlow execution in the intended environment.

## 4. Components Not Yet Verified in Their Full Runtime Environments

The following areas require testing with their actual dependencies rather than sandbox substitutes.

| Component | Verification limitation |
|---|---|
| TensorFlow model training and augmentation | Full execution of the TensorFlow model, training pipeline, augmentation, and `make_tf_dataset` paths has not been established by the sandbox tests. |
| `KerasTrainer` in `src/retrain.py` | Real fine-tuning, backbone unfreezing after model reload, and CPU runtime remain to be verified. |
| `Predictor.load_path` and `.keras` loading | The source fix is in place, but a real TensorFlow model-loading round trip after the fix remains unverified. |
| `app/dashboard.py` | Dashboard logic was tested using a stand-in Streamlit module. Actual Streamlit rendering, widget behaviour, `st.fragment` polling, and `st.rerun` remain unverified. |
| `src/api.py` | The service logic is covered by `test_service.py`, but real HTTP request handling remains unverified. Six API tests were skipped because FastAPI and HTTPX were unavailable. |
| TensorFlow-dependent notebook cells | Cells tagged `[needs-tensorflow]` and `[needs-model]` require execution with the actual TensorFlow environment and trained model. |

The service-level tests cover the logic behind the API endpoints. The outstanding API verification concerns the actual HTTP layer, not the underlying service functions.

## 5. Final Test Results

| Test execution | Result |
|---|---|
| `python -m unittest discover -s tests -t .` | 207 tests run, all passed; 6 skipped. |
| `python tools/run_notebook_cpu.py` | 11 non-TensorFlow cells executed successfully; TensorFlow-dependent cells skipped. |
| Notebook sections 19–28 in sandbox | Executed using a stub model; plumbing check only, not valid model evaluation. |

The six skipped tests in `tests/test_api.py` require FastAPI and HTTPX.

The automated test results demonstrate that the tested application logic, data handling, model registry, service behaviour, and supporting scripts passed their sandbox test cases. They should not be interpreted as proof that every component has passed full integration testing in its intended runtime environment.