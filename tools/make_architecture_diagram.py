"""Draws docs/images/architecture.png  (python tools/make_architecture_diagram.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 10.5))
ax.set_xlim(0, 16.3); ax.set_ylim(0, 10.5); ax.axis("off")
C = {"train": "#dbe9f6", "serve": "#e2f0d9", "retrain": "#fdebd0", "store": "#ececec"}


def box(x, y, w, h, text, kind, fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=C[kind], ec="#444", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None, wrap=True)


def arrow(x1, y1, x2, y2, label=None, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14, lw=1.3, color="#333", linestyle=ls))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.16, label, ha="center", fontsize=8, color="#333")


def band(y, h, title, color):
    ax.add_patch(FancyBboxPatch((0.15, y), 15.7, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc="none", ec=color, lw=2, linestyle="--"))
    ax.text(0.35, y + h - 0.28, title, fontsize=11, fontweight="bold", color=color, va="top")


ax.text(8, 10.2, "Diabetic Retinopathy Detection - MLOps architecture", ha="center", fontsize=15, fontweight="bold")

# ---- A. offline training
band(7.15, 2.6, "A. Offline model development (Google Colab, notebook + src/)", "#2b6cb0")
xs = [0.5, 3.0, 5.5, 8.0, 10.5, 13.0]
labels = ["RetinaMNIST-224\n(.npz, train/val/test)", "Data acquisition\n+ validation\n(src/data_acquisition)", "Preprocessing\n(src/preprocessing)\nin-model scaling +\ntrain-only augmentation",
          "Training (src/train)\nMobileNetV2 transfer learning\nstage 1 head, stage 2 fine-tune\nclass weights, early stopping", "Evaluation (src/evaluate)\nval for selection,\ntest used once", "Model registry\n(src/model_registry)\nmodel_vN.keras + metadata\ncurrent_model.json"]
for x, t in zip(xs, labels):
    box(x, 7.45, 2.2, 1.55, t, "train", fs=8.6)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(a + 2.2, 8.22, b, 8.22)

# ---- B. serving
band(3.55, 3.3, "B. Serving and monitoring (Docker / Hugging Face Space)", "#2f855a")
box(0.5, 4.0, 2.6, 2.2, "User\n(browser)", "serve", bold=True)
box(4.0, 4.0, 3.2, 2.2, "Streamlit dashboard\n(app/dashboard.py)\nPredict | Insights | Bulk upload\nRetraining | Monitoring", "serve", fs=9)
box(8.1, 4.0, 3.3, 2.2, "FastAPI service\n(src/api.py -> src/service.py)\n/health /predict /monitoring\n/upload-retraining-data /retrain\n/models /models/rollback", "serve", fs=8.6)
box(12.3, 5.25, 3.2, 0.95, "Predictor\n(current accepted model)", "serve", fs=8.8)
box(12.3, 4.0, 3.2, 0.95, "Monitoring + drift indicators\n(src/monitoring)", "serve", fs=8.6)
arrow(3.1, 5.1, 4.0, 5.1, style="<|-|>"); arrow(7.2, 5.1, 8.1, 5.1, "HTTP", style="<|-|>")
arrow(11.4, 5.7, 12.3, 5.7); arrow(11.4, 4.5, 12.3, 4.5)

# ---- C. retraining
band(0.15, 3.1, "C. Retraining workflow (manual button or automatic threshold; one job at a time)", "#c05621")
steps = ["Bulk upload\n+ labels (0-4)", "Label + image\nvalidation, dedup,\nleakage check", "Store: files on disk\n+ SQLite metadata\n(src/uploads, database)", "Fine-tune CURRENT\nmodel + replay sample\n(src/retrain)", "Evaluate candidate AND\ncurrent on ORIGINAL\nvalidation split", "Acceptance policy\n(src/evaluate)", "Accept -> new version\nReject -> keep current"]
xs2 = [0.4 + i * 2.2 for i in range(7)]
for x, t in zip(xs2, steps):
    box(x, 0.45, 2.0, 1.55, t, "retrain", fs=8.1)
for a, b in zip(xs2[:-1], xs2[1:]):
    arrow(a + 2.0, 1.22, b, 1.22)
ax.text(8, 2.45, "test split is never used for retraining, selection or acceptance", ha="center", fontsize=9, style="italic", color="#c05621")

# cross-links
arrow(14.1, 7.45, 14.1, 6.2, "", ls="--")                              # registry -> predictor
ax.text(14.25, 6.7, "serves the\ncurrent version", fontsize=8, va="center")
arrow(5.6, 3.55, 1.5, 2.0, "", ls="--")                               # dashboard upload -> retraining
ax.text(3.2, 3.05, "bulk upload / retrain button", fontsize=8)
# new version -> registry, routed along the right margin so it crosses no box
ax.plot([15.6, 15.78, 15.78], [1.22, 1.22, 8.22], color="#333", lw=1.3, ls="--")
arrow(15.78, 8.22, 15.2, 8.22, ls="--")
ax.text(15.9, 4.7, "accepted version registered; service reloads it", fontsize=8, rotation=90, va="center")
fig.savefig("docs/images/architecture.png", dpi=110, bbox_inches="tight", facecolor="white")
print("saved")
