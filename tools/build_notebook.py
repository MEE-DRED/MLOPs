"""Generates notebook/diabetic_retinopathy_image_classification.ipynb.

Run:  python tools/build_notebook.py
Cells whose first line is ``# [needs-tensorflow]`` or ``# [needs-model]`` are
skipped by tools/run_notebook_cpu.py (the sandbox has no TensorFlow).
"""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


# ======================================================================= 1-3
md('''
# Diabetic Retinopathy Detection Using Deep Learning and MLOps
### Notebook: data, model development, evaluation and model registration (RetinaMNIST-224, MobileNetV2)

> **Experimental academic project. Not a medical device. Predictions must never be used for diagnosis or any medical decision.**

## 1. Project introduction
This notebook is the *machine-learning half* of an end-to-end MLOps project. It takes retinal fundus images from the
RetinaMNIST-224 dataset, builds a MobileNetV2 transfer-learning classifier, evaluates it on a held-out test set and registers the
trained model as **version 1** in a small model registry. The same `src/` modules are then used by the FastAPI service, the
Streamlit dashboard and the retraining workflow, so the notebook and the deployed application share one preprocessing path.

This is **not** the model from my earlier Intro-to-ML project (tabular BRFSS diabetes survey data). That project predicted
self-reported diabetes from 21 health indicators; this one grades diabetic retinopathy from images. Only working practices were reused
(fixed seed, validation-based selection, test set used once, honest limitations).

## 2. Problem statement
Diabetic retinopathy (DR) is damage to the retina's blood vessels caused by diabetes. Grading its severity from fundus photographs
normally needs a trained specialist. The task here is **5-class severity grading** from a single image:
`0 No DR · 1 Mild · 2 Moderate · 3 Severe · 4 Proliferative DR`.

## 3. Objectives
1. Acquire and validate the dataset (shapes, labels, class balance, duplicates / leakage between splits).
2. Explore it with visualisations that mean something (class balance, image brightness / contrast / sharpness).
3. Build a MobileNetV2 transfer-learning model with regularisation, class weighting, two-stage fine-tuning and early stopping.
4. Tune a few hyper-parameters cheaply and select on the **validation** set only.
5. Evaluate once on the **test** set with accuracy, loss, precision, recall, F1, confusion matrix, ROC-AUC and quadratic weighted kappa.
6. Analyse errors honestly, including Grad-CAM (model attention, not a diagnosis).
7. Save the model + metadata as a registry version that the API can serve and retraining can start from.

### How to run (Google Colab)
1. **Runtime → Change runtime type → T4 GPU.**
2. Put `retinamnist_224.npz` in your Drive folder `MyDrive/DR_MLOps/data/` *or* let the notebook download it with the official `medmnist` package.
3. Put the project ZIP (`Diabetic-Retinopathy-MLOps.zip`) in `MyDrive/DR_MLOps/` *or* set `REPO_URL` below to your GitHub repo.
4. **Runtime → Run all.**

### What is kept and what is lost
| Location | Survives a runtime reset? | Holds |
|---|---|---|
| Google Drive `MyDrive/DR_MLOps/` (when mounted) | **Yes** | dataset copy, `models/` (versions + metadata), `outputs/` (figures, tables, reports), checkpoints, retraining uploads, database |
| Colab local disk `/content/...` (Drive **not** mounted) | **No - everything is lost** | the same folders, if Drive is unavailable |
| Python variables (trained model object, history) | **No** | re-load the saved model from `models/` |

If Drive cannot be mounted the notebook still runs using temporary storage, but you must download `models/` and `outputs/` before the session ends.
''')

md('''
## 4. Dataset description
**RetinaMNIST** is part of the **MedMNIST v2** collection (https://medmnist.com/). It is derived from the DeepDRiD challenge
fundus images and provides an ordinal 5-level DR grading task. This project uses the **224 x 224** version.

| Label | Class | | Split | Images (expected) |
|---|---|---|---|---|
| 0 | No Diabetic Retinopathy | | Train | 1,080 |
| 1 | Mild | | Validation | 120 |
| 2 | Moderate | | Test | 400 |
| 3 | Severe | | | |
| 4 | Proliferative DR | | | |

The expected sizes are *checked*, not assumed, in section 7. The original train/validation/test split is preserved exactly. The test
split is never used for training, tuning, early stopping, model selection or model acceptance.

**References (please verify before citing):** Yang et al., *MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image
classification*, Scientific Data 10, 41 (2023). Liu et al., *DeepDRiD: Diabetic Retinopathy - Grading and Image Quality Estimation Challenge*, Patterns 3(6), 2022.
''')

# ======================================================================= 5
md('## 5. Imports and environment setup')
code('''
# --- storage & code location ------------------------------------------------------
import os, sys, json, platform, zipfile, subprocess
from pathlib import Path

USE_DRIVE = True                                   # persist outputs on Google Drive
ALLOW_CPU = False                                  # True = run without a GPU anyway (a full run then takes hours)
DRIVE_PROJECT_DIR = "/content/drive/MyDrive/DR_MLOps"
REPO_URL = ""                                      # e.g. "https://github.com/<you>/Diabetic-Retinopathy-MLOps.git"
IN_COLAB = "google.colab" in sys.modules

drive_ok = False
if IN_COLAB and USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        Path(DRIVE_PROJECT_DIR).mkdir(parents=True, exist_ok=True)
        drive_ok = True
    except Exception as exc:                       # user declined / Drive unavailable
        print("Google Drive NOT available:", exc)

if drive_ok:
    PERSIST_DIR = Path(DRIVE_PROJECT_DIR)
elif IN_COLAB:
    PERSIST_DIR = Path("/content/dr_mlops_temporary")
    print("WARNING: using temporary Colab disk. Everything below is LOST when the runtime resets.")
else:
    PERSIST_DIR = Path(os.environ.get("DR_LOCAL_WORKDIR", "./dr_mlops_workdir")).resolve()
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# --- make the project's src/ package importable --------------------------------------
def find_repo_root():
    for c in (Path.cwd(), Path.cwd().parent):
        if (c / "src" / "config.py").exists():
            return c
    return None

repo = find_repo_root()
if repo is None:
    zip_path = PERSIST_DIR / "Diabetic-Retinopathy-MLOps.zip"
    code_dir = Path("/content/dr_code") if IN_COLAB else PERSIST_DIR / "code"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(code_dir)
    elif REPO_URL:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(code_dir)], check=True)
    else:
        raise RuntimeError("Project code not found. Put Diabetic-Retinopathy-MLOps.zip in "
                           f"{PERSIST_DIR} or set REPO_URL above.")
    repo = next(p.parent.parent for p in code_dir.rglob("src/config.py"))
os.chdir(repo)
sys.path.insert(0, str(repo))

# config.py reads these at import time, so set them BEFORE importing src.config
os.environ.update({
    "DR_DATA_DIR": str(PERSIST_DIR / "data"),
    "DR_MODELS_DIR": str(PERSIST_DIR / "models"),
    "DR_OUTPUTS_DIR": str(PERSIST_DIR / "outputs"),
    "DR_DB_PATH": str(PERSIST_DIR / "database" / "dr_mlops.sqlite3"),
})
print("Code   :", repo)
print("Storage:", PERSIST_DIR, "(Google Drive)" if drive_ok else "(NOT persistent across sessions in Colab)")
''')

code('''
import json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:                    # lets the data-only sections run without TensorFlow
    tf, HAS_TF = None, False

from src import config, visualization as viz
from src.data_acquisition import (ensure_dataset, load_retinamnist, validate_dataset,
                                  class_distribution_table, file_sha256)
from src.image_analysis import features_dataframe, FEATURE_EXPLANATIONS
from src.preprocessing import (load_image, validate_label, to_model_input, InvalidImageError,
                               InvalidLabelError, mobilenet_v2_reference_scaling)
from src.evaluate import (compute_metrics, text_report, predict_probabilities, misclassified_indices,
                          model_integrity_check, evaluate_acceptance)
from src.model_registry import ModelRegistry, build_metadata

FIG, TAB, REP = config.FIGURES_DIR, config.TABLES_DIR, config.REPORTS_DIR
for d in (FIG, TAB, REP):
    d.mkdir(parents=True, exist_ok=True)

env = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
if HAS_TF:
    env.update(tensorflow=tf.__version__, keras=tf.keras.__version__,
               gpus=[g.name for g in tf.config.list_physical_devices("GPU")])
    if not env["gpus"]:
        gpu_list = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout.strip()
        msg = ("TensorFlow sees NO GPU, so training would take hours (a CPU run of this notebook took about 3 hours).\\n"
               f"nvidia-smi says: {gpu_list or 'no NVIDIA GPU is attached to this runtime'}\\n"
               "- If no GPU is attached: Runtime > Change runtime type > T4 GPU, then Runtime > Run all.\\n"
               "- If a GPU IS listed but TensorFlow cannot use it: run  !pip install -q 'tensorflow[and-cuda]'  then "
               "Runtime > Restart session and Run all.\\n"
               "- To continue on CPU anyway, set ALLOW_CPU = True in the setup cell.")
        if not ALLOW_CPU:
            raise RuntimeError(msg)
        print("WARNING:", msg)
else:
    print("TensorFlow is not installed here: data sections will run, model sections need Colab.")
print(json.dumps(env, indent=2))
''')

# ======================================================================= 6-7
md('''
## 6. Dataset acquisition
`ensure_dataset` tries, in order: (1) the file already in the project data folder, (2) copies in Drive/Colab upload locations,
(3) the official `medmnist` package, (4) a direct download (URL not verified). It never silently substitutes another dataset.
If everything fails it raises an error telling you where to put the file by hand - no laptop download is required if you
use the `medmnist` route or already have the file in Drive.
''')
code('''
if IN_COLAB and not Path(config.DATASET_NPZ).exists():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "medmnist"], check=False)   # official downloader

DATASET_PATH = ensure_dataset(
    config.DATASET_NPZ,
    search_paths=[Path("/content/retinamnist_224.npz"),
                  Path("/content/drive/MyDrive/retinamnist_224.npz"),
                  PERSIST_DIR / "retinamnist_224.npz"])
print("Dataset file:", DATASET_PATH, f"({DATASET_PATH.stat().st_size / 1e6:.0f} MB)")
''')

md('''
## 7. Dataset validation
Checks performed: required keys, image shape (N,224,224,3), `uint8` dtype, one label per image, labels within 0-4,
fully black (unreadable-looking) images, duplicate images inside each split and **identical images shared between splits
(data leakage)**. Any blocking issue stops the notebook.
''')
code('''
data = load_retinamnist(DATASET_PATH)
report = validate_dataset(data)
print(json.dumps(report, indent=2))
assert report["ok"], f"Dataset problems: {report['issues']}"
print("\\nSHA-256 of dataset file:", file_sha256(DATASET_PATH))

(X_train, y_train), (X_val, y_val), (X_test, y_test) = data["train"], data["val"], data["test"]
dist = class_distribution_table(data)
counts = dist.pivot(index="class", columns="split", values="count").loc[list(config.CLASS_NAMES), ["train", "val", "test"]]
counts.loc["TOTAL"] = counts.sum()
display(counts)
''')

# ======================================================================= 8-10
md('''
## 8. Exploratory data analysis
A first look at the raw arrays: size in memory, dtype and value range. The full training split is about 160 MB as `uint8`, so it fits
comfortably in Colab RAM; images are converted to `float32` one **batch** at a time inside the `tf.data` pipeline.
''')
code('''
summary = pd.DataFrame({
    "images": {s: len(data[s][1]) for s in data},
    "shape": {s: str(data[s][0].shape[1:]) for s in data},
    "dtype": {s: str(data[s][0].dtype) for s in data},
    "pixel min": {s: int(data[s][0].min()) for s in data},
    "pixel max": {s: int(data[s][0].max()) for s in data},
    "memory (MB)": {s: round(data[s][0].nbytes / 1e6, 1) for s in data},
})
display(summary)
''')

md('''
## 9. Image visualisations
### 9a. Example images per class
Rows are DR grades 0-4. Differences between neighbouring grades (e.g. Mild vs Moderate) are small and subtle in the images, which already hints that those
classes will be hard to separate.
''')
code('''
fig = viz.plot_sample_grid(X_train, y_train, per_class=4, save_path=FIG / "sample_images.png")
plt.show()
''')

md('''
### 9b. Image-derived characteristics: brightness, contrast, sharpness
These are measured on the **retinal field only** (the black circular border is excluded):

* **Brightness** - average light level inside the retina (0-255).
* **Contrast** - spread of the light levels (higher = more difference between dark and bright areas).
* **Sharpness** - variance of the Laplacian, a standard focus measure (higher = crisper fine detail; also rises with noise).

They describe the *photographs*. They are **not** explanations of disease.
''')
code('''
feat = features_dataframe(data, config.CLASS_NAMES)
feat.to_csv(TAB / "image_features.csv", index=False)

fig = viz.plot_feature_distributions(feat, save_path=FIG / "image_features.png")
plt.show()

order = list(config.CLASS_NAMES)
display(feat.groupby("class")[["brightness", "contrast", "sharpness", "retina_fraction"]].median().loc[order].round(2))
print("Spearman correlation of each characteristic with DR grade (0-4):")
display(feat[["brightness", "contrast", "sharpness", "retina_fraction"]].corrwith(feat["label"], method="spearman").round(3).to_frame("rho"))
print("Median per split (splits should look alike):")
display(feat.groupby("split")[["brightness", "contrast", "sharpness"]].median().round(2))
''')

md('''
### What the three characteristics tell us (from the reference run on the supplied dataset; the cell above prints the exact numbers)
* **Brightness is the one characteristic that clearly differs by grade.** The median for *No DR* images is about 106, versus about 83-88 for the four DR grades,
  and brightness is moderately negatively correlated with grade (Spearman rho about -0.32). In this dataset, healthier eyes were photographed brighter.
  That could reflect biology (for example clouding of the eye's lens or smaller pupils, which are more common with long-standing diabetes) **or** simply differences in camera,
  exposure or clinic. The data cannot tell us which. The practical risk is **shortcut learning**: a network may use overall darkness as a hint for the label.
  That is one reason Grad-CAM (section 23) and a cautious reading of the results matter.
* **Contrast** is almost the same in every class (medians about 30-32) with only a very weak trend (rho about -0.10). It does not separate the grades.
* **Sharpness** varies enormously between images (from nearly 0 to over 300) but has no monotonic relationship with grade (rho about 0, not significant). It mostly says
  *how well focused each photograph is*. The proliferative class has a somewhat lower median, but with only 92 proliferative images in the whole dataset that is weak evidence.
* **Splits look alike.** Train, validation and test have almost identical medians, so there is no obvious distribution shift between the splits.
* **Retina area fraction reveals two different photographic framings - and they are strongly linked to the label.** About 0.72 of the image area is retina for 988 images and about 0.66 for 510 images
  (two clear peaks). The share of the *narrow* framing rises from roughly 13 % in *No DR* to roughly 34 % in *Mild*, 53 % in *Moderate*, 60 % in *Severe* and 63 % in *Proliferative DR*
  (Spearman rho about -0.43, **stronger than brightness**). The train split also has a somewhat larger narrow-framing share than the test split (about 37 % vs 29 %).
  So camera set-up or field of view is entangled with disease grade in this dataset. This is a **confounder**: a network can partly "cheat" by recognising the framing.

None of this is evidence that brightness, contrast or sharpness *cause* or *indicate* retinopathy.
''')
md('''
### 9c. How much label information do these simple statistics already carry? (shortcut baseline)
To quantify the confounding above, a deliberately simple classifier (balanced logistic regression) is trained on **only the four image statistics** - no pixels, no deep learning - and scored on the test set.
It is not a serious model. It is a **yardstick**: whatever the CNN achieves should be judged against it, and a CNN that barely beats it would deserve suspicion.
''')
code('''
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

stat_cols = ["brightness", "contrast", "sharpness", "retina_fraction"]
f_tr, f_te = feat[feat.split == "train"], feat[feat.split == "test"]
stats_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
stats_clf.fit(f_tr[stat_cols], f_tr["label"])
stats_baseline_test = compute_metrics(f_te["label"].values, stats_clf.predict_proba(f_te[stat_cols]))
display(pd.Series({k: stats_baseline_test[k] for k in ["accuracy", "f1_macro", "f1_weighted", "quadratic_weighted_kappa", "within_one_grade_accuracy"]}).round(3).to_frame("image-statistics-only baseline (test)"))
print("Accuracy of always predicting 'No DR':", round(stats_baseline_test["majority_class_baseline_accuracy"], 3))
''')
md('''
**Reading this:** these statistics alone do not beat the majority-class *accuracy*, but they do carry ordinal signal (quadratic weighted kappa clearly above zero) and a macro-F1 well above what always predicting *No DR* gives (which is about 0.12).
In other words, some of what a CNN might "learn" can come from the photograph's style rather than from lesions. Section 19 therefore puts this baseline next to the CNN, and section 23 (Grad-CAM) checks where the CNN looks.
''')


md('''
## 10. Class distribution
The classes are **imbalanced**: *No DR* is 45 % of the training set and *Proliferative DR* only 6 %. Consequences used later:
(1) accuracy alone is misleading, so **macro-F1** is the selection metric, (2) training uses **balanced class weights**, and (3) the test set holds only 20 proliferative images,
so any per-class number for that class is statistically noisy.
''')
code('''
fig = viz.plot_class_distribution(data, save_path=FIG / "class_distribution.png")
plt.show()
display(dist.pivot(index="class", columns="split", values="percent").loc[order, ["train", "val", "test"]].round(1))
print("Majority-class accuracy on test (what 'always predict No DR' scores):",
      round(float(np.bincount(y_test).max() / len(y_test)), 3))
''')

# ======================================================================= 11-13
md('''
## 11. Image preprocessing
One preprocessing path is used by training, the API and retraining (`src/preprocessing.py`):

1. **Validate** - only JPEG/PNG (checked from the file content), readable, not truncated, at least 32 px, below a size limit.
2. **Convert to RGB** and make square (pad with black, or optionally centre-crop - no distortion of aspect ratio).
3. **Resize to 224 x 224.**
4. **Pixel scaling happens exactly once, inside the model** (`Rescaling(1/127.5, offset=-1)`, the convention MobileNetV2 was trained with). Everything *outside* the model supplies raw float32 values in the 0-255 range.
   This avoids applying normalisation twice and guarantees training and serving see identical inputs.
5. **Labels** must be integers 0-4. Anything else is rejected before it can reach supervised retraining.

The dataset images are already 224 x 224, so steps 1-3 matter mainly for images uploaded later.
''')
code('''
import io
from PIL import Image

def synthetic_png(size, mode="RGB", fmt="PNG"):
    buf = io.BytesIO(); Image.new(mode, size, (90, 40, 20) if mode == "RGB" else 90).save(buf, format=fmt); return buf.getvalue()

ok = load_image(synthetic_png((640, 480)))
print("640x480 PNG ->", ok.shape, ok.dtype, "| top-left pixel (padding):", ok[0, 0])

print("\\nInvalid inputs are rejected with a clear message:")
bad = {"random text": b"not an image", "empty file": b"", "GIF (unsupported)": synthetic_png((64, 64), fmt="GIF"),
       "8x8 image (too small)": synthetic_png((8, 8)), "truncated PNG": synthetic_png((200, 200))[:-50]}
for name, blob in bad.items():
    try:
        load_image(blob); print(f"  {name:<24} ACCEPTED (unexpected!)")
    except InvalidImageError as exc:
        print(f"  {name:<24} rejected: {exc}")

print("\\nLabel validation:")
for raw in [0, "3", 4, 5, -1, "abc", 1.5, None]:
    try:
        print(f"  {raw!r:<6} -> {validate_label(raw)}")
    except InvalidLabelError as exc:
        print(f"  {raw!r:<6} rejected: {exc}")

batch = to_model_input(X_val[:2])
print("\\nModel input:", batch.shape, batch.dtype, "range", batch.min(), "-", batch.max(), "(raw 0-255; the model rescales internally)")
print("In-model scaling maps 0 / 127.5 / 255 to", mobilenet_v2_reference_scaling(np.array([0, 127.5, 255])))
''')

md('''
## 12. Data augmentation
Augmentation is built into the model as Keras layers that are **active only during training** (`training=True`). Validation, test and prediction
inputs are never altered. The transformations are deliberately mild so retinal structures are not distorted:
horizontal flip, rotation up to about 14 degrees, 10 % zoom, +/-10 % brightness and contrast. No vertical flips, shear or heavy warps.
Augmented images are variations of existing images - they are **not** new independent data and are never counted as such.
''')
code('''
# [needs-tensorflow]
from src.preprocessing import build_augmentation
aug = build_augmentation(config.SEED)
sample = X_train[np.where(y_train == 2)[0][0]].astype("float32")
batch = np.repeat(sample[None], 8, axis=0)
assert np.allclose(aug(batch, training=False).numpy(), batch), "augmentation must be inactive at inference"
out = np.clip(aug(batch, training=True).numpy(), 0, 255).astype("uint8")
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
axes[0, 0].imshow(sample.astype("uint8")); axes[0, 0].set_title("original")
for ax, img in zip(axes.ravel()[1:], out):
    ax.imshow(img)
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout(); plt.savefig(FIG / "augmentation_examples.png", dpi=130); plt.show()
''')

md('''
## 13. Train / validation / test separation
The dataset's official split is used unchanged. The leakage check in section 7 found no identical images shared between splits. Roles:

| Split | Used for |
|---|---|
| Train (1,080) | fitting weights (with augmentation and class weights) |
| Validation (120) | early stopping, hyper-parameter selection, **and model acceptance during retraining** |
| Test (400) | **one** final evaluation of the chosen model. Never used for training, tuning or acceptance |

A 120-image validation set is small, so validation scores are noisy and selecting on them can overfit slightly to it. This is a limitation, discussed in section 28.
''')
code('''
assert sum(report["cross_split_duplicates"].values()) == 0
for name, (x, y) in {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}.items():
    print(f"{name:<5} {len(y):>5} images | class counts {np.bincount(y, minlength=5).tolist()}")
''')

# ======================================================================= 14-15
md('''
## 14. MobileNetV2 architecture
```
Input 224x224x3 (raw 0-255)  ->  augmentation (training only)  ->  Rescaling to [-1, 1]
  ->  MobileNetV2 backbone (ImageNet weights, no top)  ->  GlobalAveragePooling  ->  Dropout  ->  Dense(5, softmax, L2)
```
**Why MobileNetV2?** It is small (about 2.2 M backbone parameters), fast on a free Colab GPU and on CPU for the deployed API, and its ImageNet features transfer well to
small image datasets. **Regularisation:** dropout and L2 on the output layer, frozen early layers, augmentation. If the ImageNet weights cannot be downloaded the code stops with an error;
it only continues with random weights if you explicitly allow it, and the metadata would then record that no pretrained weights were used.
''')
code('''
# [needs-tensorflow]
from src.model import build_model, set_backbone_trainable, compile_model
from src.train import (TrainConfig, set_seed, balanced_class_weights, make_datasets, run_stage1, run_stage2,
                       train_two_stage, search_hyperparameters, save_training_record)

set_seed(config.SEED)
BASE_CFG = TrainConfig()
print("Baseline training configuration:"); print(json.dumps(BASE_CFG.to_dict(), indent=2))

model, model_info = build_model(BASE_CFG.dropout, BASE_CFG.l2, weights="imagenet",
                                allow_weight_fallback=False, augment=True)
print(json.dumps(model_info, indent=2))
model.summary(show_trainable=False)
''')

md('''
## 15. Transfer learning
In **stage 1** the whole backbone is frozen: only the new classification head (pooling -> dropout -> dense) is trained, so the pretrained features are not destroyed by large random-head gradients.
In **stage 2** the top 40 backbone layers are unfrozen (BatchNorm layers stay frozen) and trained with a 10x smaller learning rate.
''')
code('''
# [needs-tensorflow]
stats = set_backbone_trainable(model, 0)
print("Stage-1 (frozen backbone):", stats)
stats = set_backbone_trainable(model, BASE_CFG.unfreeze_layers)
print("Stage-2 (top", BASE_CFG.unfreeze_layers, "layers unfrozen):", stats)
set_backbone_trainable(model, 0)   # back to stage-1 state
''')

# ======================================================================= 16-17
md('''
## 16. Initial model training (stage 1: classification head)
**Settings:** Adam, learning rate 1e-3, sparse categorical cross-entropy, batch size 32, **balanced class weights**, up to 15 epochs.
**Early stopping** and checkpointing watch *validation macro-F1* (not accuracy, because of the class imbalance); the best epoch is restored.
This first run uses the default configuration as a **baseline**; section 18 then compares alternatives.
''')
code('''
# [needs-tensorflow]
class_weights = balanced_class_weights(y_train)
print("Class weights:", {config.CLASS_NAMES[k]: round(v, 2) for k, v in class_weights.items()})

train_ds, val_ds = make_datasets((X_train, y_train), (X_val, y_val), BASE_CFG.batch_size, BASE_CFG.seed)
CKPT_DIR = PERSIST_DIR / "checkpoints" / "baseline"
t0 = time.time()
h1, best1, ckpt1, s1 = run_stage1(model, BASE_CFG, train_ds, val_ds, (X_val, y_val), class_weights, CKPT_DIR)
print(f"\\nStage 1 done in {time.time() - t0:.0f}s | epochs run: {len(h1['loss'])} | best validation macro-F1: {best1:.4f}")
''')

md('''
## 17. Fine-tuning (stage 2)
Top 40 backbone layers unfrozen, learning rate 1e-4, early stopping (patience 8). If fine-tuning never beats the stage-1 validation macro-F1, the stage-1 weights are restored,
so fine-tuning cannot leave the baseline worse than it was.
''')
code('''
# [needs-tensorflow]
t0 = time.time()
h2, best_ft, chosen_stage, s2 = run_stage2(model, BASE_CFG, train_ds, val_ds, (X_val, y_val), class_weights,
                                           CKPT_DIR, best1, ckpt1)
baseline_history = {k: h1.get(k, []) + h2.get(k, []) for k in set(h1) | set(h2)}
baseline_val_f1 = best_ft
print(f"Stage 2 done in {time.time() - t0:.0f}s | epochs run: {len(h2['loss'])} | "
      f"best validation macro-F1: {best_ft:.4f} | weights kept from stage {chosen_stage}")
viz.plot_history(baseline_history, save_path=FIG / "baseline_training_curves.png"); plt.show()
del model                                   # free GPU memory before the search
import gc; gc.collect(); tf.keras.backend.clear_session()
''')

# ======================================================================= 18
md('''
## 18. Hyper-parameter optimisation
A full grid search is unnecessary and too slow for a free GPU. Instead a **small random search** (6 trials, the baseline included as trial 0) over:

| Hyper-parameter | Values tried |
|---|---|
| dropout | 0.2, 0.5 |
| fine-tuning learning rate | 1e-4, 3e-5 |
| batch size | 16, 32 |

Each trial gets a **shortened** budget (5 head epochs + 6 fine-tuning epochs) so all candidates are compared fairly; they are ranked by **validation macro-F1**.
The winning configuration is then trained again with the full epoch budget to produce the final model. Caveat: with 120 validation images the ranking can be noisy,
so differences of a few hundredths should not be over-interpreted.
''')
code('''
# [needs-tensorflow]
GRID = {"dropout": [0.2, 0.5], "lr_finetune": [1e-4, 3e-5], "batch_size": [16, 32]}
N_TRIALS = 6
t0 = time.time()
search_df = search_hyperparameters(BASE_CFG, (X_train, y_train), (X_val, y_val), GRID, N_TRIALS,
                                   trial_epochs=(5, 6), work_dir=PERSIST_DIR / "checkpoints" / "search")
print(f"Search finished in {time.time() - t0:.0f}s")
display(search_df.drop(columns="config").round(4))
search_df.drop(columns="config").to_csv(TAB / "hyperparameter_search.csv", index=False)
BEST_CFG = TrainConfig(**search_df.loc[0, "config"])
print("Selected configuration:", json.dumps(BEST_CFG.to_dict(), indent=2))
''')

md('''
### Final model
The selected configuration is trained from the pretrained ImageNet weights with the full epoch budget (stage 1 + stage 2, early stopping).
The notebook always retrains rather than re-using the earlier baseline object, so the saved model is exactly reproducible from `BEST_CFG`.
''')
code('''
# [needs-tensorflow]
t0 = time.time()
final_model, final_history, final_info = train_two_stage(
    BEST_CFG, (X_train, y_train), (X_val, y_val), PERSIST_DIR / "checkpoints" / "final", verbose=2)
print(f"Final training took {time.time() - t0:.0f}s")
print(json.dumps({k: v for k, v in final_info.items() if k != "class_weights"}, indent=2, default=str))
viz.plot_history(final_history, save_path=FIG / "final_training_curves.png"); plt.show()
save_training_record(REP, BEST_CFG, final_history, final_info)
''')

# ======================================================================= 19-21
md('''
## 19. Model evaluation
All metrics come from the actual predictions of the final model. **Averaging:** *macro* = unweighted mean over the five classes (rare severe grades count as much as *No DR*),
*weighted* = mean weighted by class size (dominated by *No DR*). **Loss** is the multiclass cross-entropy of the predicted probabilities.
**Quadratic weighted kappa** and **within-one-grade accuracy** are included because DR grades are ordered: confusing grade 2 with 3 is a smaller error than confusing 0 with 4.

Validation metrics are shown for reference. The **test set is evaluated exactly once, here, after every selection decision has been made.**
''')
code('''
# [needs-tensorflow]
val_probs = predict_probabilities(final_model, X_val, BEST_CFG.batch_size)
val_metrics = compute_metrics(y_val, val_probs)

# ---- the single, final test-set evaluation ----
test_probs = predict_probabilities(final_model, X_test, BEST_CFG.batch_size)
test_metrics = compute_metrics(y_test, test_probs)

keys = ["accuracy", "loss", "precision_macro", "recall_macro", "f1_macro", "precision_weighted",
        "recall_weighted", "f1_weighted", "roc_auc_ovr_macro", "quadratic_weighted_kappa",
        "within_one_grade_accuracy", "majority_class_baseline_accuracy"]
summary_df = pd.DataFrame({"validation (n=%d)" % val_metrics["n_samples"]: {k: val_metrics[k] for k in keys},
                           "TEST (n=%d)" % test_metrics["n_samples"]: {k: test_metrics[k] for k in keys},
                           "image-stats-only baseline (TEST)": {k: stats_baseline_test.get(k) for k in keys}})
display(summary_df.round(4))
summary_df.to_csv(TAB / "metrics_summary.csv")
(REP / "metrics.json").write_text(json.dumps({"validation": val_metrics, "test": test_metrics}, indent=2))
np.savez_compressed(REP / "test_probabilities.npz", probs=test_probs, labels=y_test)
''')

md('## 20. Confusion matrix')
code('''
# [needs-tensorflow]
cm = np.array(test_metrics["confusion_matrix"])
viz.plot_confusion_matrix(cm, title="Test set - counts", save_path=FIG / "confusion_matrix_counts.png")
viz.plot_confusion_matrix(cm, normalize=True, title="Test set - row-normalised (recall)", save_path=FIG / "confusion_matrix_normalised.png")
plt.show()
''')

md('''
## 21. Classification report
Per-class precision, recall and F1 on the test set. Remember the support column: the *Proliferative DR* row rests on only 20 images.
''')
code('''
# [needs-tensorflow]
print(text_report(y_test, test_probs))
per_class = pd.DataFrame(test_metrics["per_class"]).T
per_class["roc_auc"] = pd.Series(test_metrics["roc_auc_per_class"])
display(per_class.round(3))
per_class.to_csv(TAB / "per_class_metrics_test.csv")
(REP / "classification_report_test.txt").write_text(text_report(y_test, test_probs))
''')

# ======================================================================= 22-23
md('''
## 22. Error analysis
Wrong predictions, most confidently wrong first. Questions to ask: are errors mostly between **neighbouring** grades (expected, because grading is subtle and partly subjective)?
Are the rare classes the ones being missed? Are misclassified images systematically darker or blurrier?
''')
code('''
# [needs-model]
y_pred = test_probs.argmax(1)
wrong = misclassified_indices(y_test, test_probs)
print(f"{len(wrong)} of {len(y_test)} test images misclassified ({100 * len(wrong) / len(y_test):.1f}%)")
step = np.abs(y_pred[wrong] - y_test[wrong])
print("Errors that are exactly one grade off:", f"{(step == 1).sum()} of {len(wrong)}")
print("Errors that are two or more grades off:", f"{(step >= 2).sum()} of {len(wrong)}")

pairs = pd.Series([f"{config.CLASS_NAMES[t]} -> {config.CLASS_NAMES[p]}" for t, p in zip(y_test[wrong], y_pred[wrong])]).value_counts()
print("\\nMost frequent confusions (true -> predicted):"); display(pairs.head(8).to_frame("count"))

ft = feat[feat.split == "test"].reset_index(drop=True)
ft["correct"] = (y_pred == y_test)
print("Median image characteristics, correct vs misclassified test images:")
display(ft.groupby("correct")[["brightness", "contrast", "sharpness"]].median().round(1))

viz.plot_misclassified(X_test, y_test, test_probs, wrong, n=12, save_path=FIG / "misclassified_examples.png"); plt.show()
''')
md('''
### Interpreting the errors
Read the output above before writing your own conclusion; typical explanations to check against it: (1) **adjacent grades look alike**, and even specialists disagree on them;
(2) **small classes** (Mild, Proliferative) give the network few examples to learn from; (3) **image quality** - very dark or blurred photographs hide the small lesions that distinguish grades;
(4) **dataset size** - 1,080 training images is tiny for deep learning. Do not hide poor per-class results: the report and README quote the test numbers as they came out.
''')

md('''
## 23. Grad-CAM visualisation
Grad-CAM shows which image regions most increased the model's score for its chosen class. **It shows where the model is looking, not where disease is.**
Highlighted areas are not validated lesion locations and must not be read as a clinical explanation. A heat-map concentrated on the dark border, the image edge or the optic disc,
rather than on the retina, would be a warning sign of shortcut learning (see the brightness finding in section 9b).
''')
code('''
# [needs-model]
from src.gradcam import grad_cam
rng = np.random.default_rng(config.SEED)
right = np.where(y_pred == y_test)[0]
picks = list(rng.choice(right, 3, replace=False)) + list(wrong[:3])
for i in picks:
    heat, cls, probs = grad_cam(final_model, X_test[i])
    status = "correct" if cls == y_test[i] else "WRONG"
    viz.plot_gradcam(X_test[i], heat, title=f"true: {config.CLASS_NAMES[y_test[i]]} | pred: {config.CLASS_NAMES[cls]} ({probs[cls]:.2f}) - {status}",
                     save_path=FIG / f"gradcam_test_{i}.png")
plt.show()
''')

# ======================================================================= 24-25
md('''
## 24. Model saving
The model is saved in Keras's native `.keras` format, **re-loaded**, compared with the in-memory model and integrity-checked before it is registered.
Registration goes through `ModelRegistry`: versions are numbered, never overwritten, and a rejected candidate can never become the served model.
If a current model already exists (for example you re-ran the notebook), the new model must pass the **acceptance policy** against it rather than silently replacing it.
''')
code('''
# [needs-model]
STAGING = PERSIST_DIR / "staging"; STAGING.mkdir(exist_ok=True)
candidate_path = STAGING / "model_candidate.keras"
final_model.save(str(candidate_path))

reloaded = tf.keras.models.load_model(str(candidate_path), compile=False)
a = predict_probabilities(final_model, X_val[:32]); b = predict_probabilities(reloaded, X_val[:32])
assert np.allclose(a, b, atol=1e-4), "re-loaded model does not reproduce the original predictions"
integrity_ok, integrity_msg = model_integrity_check(reloaded)
print("Round-trip check passed | integrity:", integrity_ok, integrity_msg, "| file size:", f"{candidate_path.stat().st_size / 1e6:.1f} MB")
''')

md('''
## 25. Model metadata
Every version stores: architecture, pretrained-weight source, input contract, class names, dataset provenance (including the file hash), the full training configuration, validation **and** test metrics
(clearly labelled), the acceptance decision, library versions and a medical disclaimer.
''')
code('''
# [needs-model]
registry = ModelRegistry(config.MODELS_DIR)
current_meta = registry.get_current_metadata()
acceptance = evaluate_acceptance(val_metrics, current_meta["metrics"]["validation"] if current_meta else None,
                                 integrity_ok, integrity_msg)
print("Acceptance decision:", json.dumps(acceptance, indent=2))

training_dataset = {
    "name": "RetinaMNIST-224 (MedMNIST v2)", "file": DATASET_PATH.name, "sha256": file_sha256(DATASET_PATH),
    "split_sizes": {s: len(data[s][1]) for s in data},
    "class_counts": {s: np.bincount(data[s][1], minlength=5).tolist() for s in data},
    "uploaded_images_used": 0,
}
metadata = build_metadata(
    architecture=final_info["architecture"], pretrained_weights=final_info["pretrained_weights"],
    training_dataset=training_dataset,
    training_config={**BEST_CFG.to_dict(), "class_weights": final_info["class_weights"],
                     "epochs_run": final_info["epochs_run"], "selected_stage": final_info["selected_stage"]},
    metrics={"validation": val_metrics, "test": test_metrics},
    acceptance=acceptance, parent_version=registry.current_version(), environment=env,
    notes="Initial notebook-trained model." if current_meta is None else "Re-trained from the notebook.")

status = "accepted" if acceptance["accepted"] else "rejected"
saved = registry.register_version(candidate_path, metadata, status, make_current=acceptance["accepted"],
                                  reason="trained in notebook")
print(f"\\nRegistered version {saved['version']} with status '{status}'. Current served version:", registry.current_version())
display(pd.DataFrame(registry.list_versions())[["version", "status", "created_at", "model_size_bytes"]])
''')

# ======================================================================= 26
md('''
## 26. Prediction demonstration
*Resuming after a runtime reset? You do not need to retrain: run the setup cells (sections 5-7) and then jump straight here and to section 28 - both read the saved model from the registry.*

Predictions use the **registry's current model** through `Predictor`, the same class the API uses. First, a few correctly classified
test images are exported as ordinary PNG files, so you can upload them in the dashboard demo. (Choosing images the model gets right is for a clean demo; the error analysis above
shows the cases it gets wrong. One misclassified example is exported too and is labelled `WRONG` so you can discuss it openly.)
''')
code('''
# [needs-model]
from PIL import Image
from src.prediction import Predictor
registry = ModelRegistry(config.MODELS_DIR)
predictor = Predictor.from_registry(registry)               # loads the registry's CURRENT model from disk
print("Loaded model version:", predictor.version)
y_pred = predict_probabilities(predictor.model, X_test).argmax(1)       # predictions of the SAVED model
wrong = np.where(y_pred != y_test)[0]
print(f"Saved model reproduces {100 * (y_pred == y_test).mean():.1f}% test accuracy ({len(wrong)} misclassified)")

demo_dir = PERSIST_DIR / "outputs" / "demo_samples"; demo_dir.mkdir(parents=True, exist_ok=True)
for c in range(5):
    idx = [i for i in np.where(y_test == c)[0] if y_pred[i] == c][:2]
    for i in idx:
        Image.fromarray(X_test[i]).save(demo_dir / f"true{c}_{config.CLASS_NAMES[c].replace(' ', '_')}_test{i}.png")
if len(wrong):
    i = int(wrong[0])
    Image.fromarray(X_test[i]).save(demo_dir / f"WRONG_true{y_test[i]}_pred{y_pred[i]}_test{i}.png")
print("Demo images:", sorted(p.name for p in demo_dir.iterdir()))

def predict_image(path_or_bytes):
    """Validate, preprocess and classify one image file. Returns the API-style result dict."""
    blob = Path(path_or_bytes).read_bytes() if isinstance(path_or_bytes, (str, Path)) else path_or_bytes
    return predictor.predict_bytes(blob)

sample_file = sorted(demo_dir.glob("true*"))[0]
result = predict_image(sample_file)
print(sample_file.name); print(json.dumps(result, indent=2))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].imshow(Image.open(sample_file)); ax[0].axis("off"); ax[0].set_title("input")
ax[1].barh(list(result["probabilities"]), list(result["probabilities"].values())); ax[1].set_xlim(0, 1)
ax[1].set_title(f"{result['predicted_class_name']} ({result['confidence']:.2f}) - experimental, not a diagnosis")
plt.tight_layout(); plt.show()

try:
    predict_image(b"definitely not an image")
except InvalidImageError as exc:
    print("Invalid upload correctly rejected:", exc)
''')

# ======================================================================= 27
md('''
## 27. Retraining workflow (implemented in `src/retrain.py`, `src/service.py`, `src/api.py`)
This notebook produced **version 1**. The deployed system retrains from it, never from a fresh network:

```
labelled images uploaded (API/dashboard) -> label validation -> files on disk + metadata in SQLite
   -> same preprocessing -> load CURRENT registry model -> fine-tune top layers at a small learning rate
   -> mix new images with a replay sample of the ORIGINAL training data (prevents forgetting / overfitting on a handful of images)
   -> evaluate candidate AND current model on the ORIGINAL validation split
   -> acceptance policy (section 24 / src/evaluate.py) -> register as new version (accepted or rejected)
   -> accepted: prediction service switches to it; rejected: current model keeps serving
```
Triggers: a manual **Retrain** button, and an optional automatic trigger that fires only when a configured number of *new valid labelled* images has accumulated, never while another job runs,
and subject to a cool-down. Images count as "new" until they have been part of one training attempt, so even a failed job cannot cause a retry loop. The test set is never used in any of this.

**Why there is no retraining cell here:** the only labelled retinal images available to this project are the dataset's own. Re-uploading any of them is *correctly rejected* by the upload check
(identical to a validation/test/training image), because duplicates are not new data and must not be passed off as such. Retraining is therefore demonstrated through the API/dashboard with genuinely new, correctly labelled fundus images.
The retraining logic itself is covered by automated tests that use a controllable fake trainer (`tests/test_retraining.py`); the real Keras fine-tuning step runs only where TensorFlow is installed.
''')

# ======================================================================= 28
md('''
## 28. Final conclusions
The cell below collects the **actual** numbers from this run. Add your own two or three sentences of interpretation underneath it after you have read them. Do not copy numbers from anywhere else.
''')
code('''
# [needs-model]
meta = registry.get_current_metadata()
t, cfg = meta["metrics"]["test"], meta["training_config"]
print(f"Current model: version {meta['version']} (status {meta['status']}, created {meta['created_at']})")
print(f"Configuration: dropout={cfg['dropout']}, lr_finetune={cfg['lr_finetune']}, batch={cfg['batch_size']}, selected stage={cfg['selected_stage']}")
print(f"TEST accuracy {t['accuracy']:.3f} vs majority-class baseline {t['majority_class_baseline_accuracy']:.3f}")
print(f"TEST macro-F1 {t['f1_macro']:.3f} | weighted-F1 {t['f1_weighted']:.3f} | macro ROC-AUC {t['roc_auc_ovr_macro']:.3f} | QWK {t['quadratic_weighted_kappa']:.3f}")
weakest = min(t["per_class"], key=lambda k: t["per_class"][k]["recall"])
print(f"Lowest-recall class: {weakest} (recall {t['per_class'][weakest]['recall']:.2f}, n={t['per_class'][weakest]['support']})")
print("\\nAll registered versions:")
display(pd.DataFrame(registry.list_versions())[["version", "status", "created_at", "parent_version"]])
print("\\nSaved under", PERSIST_DIR, ":")
for p in sorted(PERSIST_DIR.rglob("*")):
    if p.is_file() and "checkpoints" not in p.parts and p.suffix != ".npz":
        print("  ", p.relative_to(PERSIST_DIR))
''')
md('''
### Limitations and honest caveats
* **Small data:** 1,080 training images; 120 validation images make model selection noisy; the test set has only 20 *Proliferative DR* images, so that class's metrics are very uncertain.
* **Single run, single split:** no confidence intervals or repeated seeds. Small differences between configurations are not evidence of real superiority.
* **Possible shortcuts:** brightness differs between grades (section 9b); the model may partly exploit acquisition differences rather than lesions.
* **Downsampled images:** the 224 x 224 MedMNIST version loses fine detail (micro-aneurysms) that clinicians rely on.
* **Domain shift:** photographs uploaded to the app may be framed or cropped differently from the MedMNIST preparation, which can hurt accuracy.
* **Ordinal labels** are treated as nominal classes by the cross-entropy loss; QWK is reported to reflect the ordering.
* **No external or clinical validation.** This is an academic demonstration and must not be used for any medical decision.

### What to put in your write-up
Your own interpretation of: how the fine-tuned model compares with the majority-class baseline; which grades are confused and why; what the brightness finding implies for trusting the model; and how retraining on new data is protected against making the model worse.
''')

nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}, "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 4}

out = Path(__file__).resolve().parents[1] / "notebook" / "diabetic_retinopathy_image_classification.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "-", len(cells), "cells")
