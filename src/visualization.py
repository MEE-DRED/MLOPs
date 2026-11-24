"""Matplotlib figures used by the notebook (and later the dashboard)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import config


def _finish(fig, save_path):
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


def plot_class_distribution(data, class_names=config.CLASS_NAMES, save_path=None):
    splits = list(data)
    counts = {s: np.bincount(data[s][1], minlength=len(class_names)) for s in splits}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    x, w = np.arange(len(class_names)), 0.8 / len(splits)
    for i, s in enumerate(splits):
        pct = 100 * counts[s] / counts[s].sum()
        axes[0].bar(x + i * w, pct, w, label=f"{s} (n={counts[s].sum()})")
    axes[0].set_xticks(x + w * (len(splits) - 1) / 2)
    axes[0].set_xticklabels([f"{i}\n{n}" for i, n in enumerate(class_names)], fontsize=8)
    axes[0].set_ylabel("% of split"); axes[0].set_title("Class distribution per split"); axes[0].legend()
    axes[1].bar(splits, [counts[s].sum() for s in splits], color="#4c78a8")
    for i, s in enumerate(splits):
        axes[1].text(i, counts[s].sum(), str(counts[s].sum()), ha="center", va="bottom")
    axes[1].set_title("Images per split"); axes[1].set_ylabel("images")
    return _finish(fig, save_path)


def plot_sample_grid(images, labels, class_names=config.CLASS_NAMES, per_class=4, seed=config.SEED,
                     save_path=None):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(class_names), per_class, figsize=(2.4 * per_class, 2.4 * len(class_names)))
    for c, name in enumerate(class_names):
        idx = np.where(labels == c)[0]
        pick = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        for j in range(per_class):
            ax = axes[c, j]; ax.axis("off")
            if j < len(pick):
                ax.imshow(images[pick[j]])
                if j == 0:
                    ax.set_title(f"{c}: {name}", fontsize=9, loc="left")
    return _finish(fig, save_path)


def plot_feature_distributions(df, features=("brightness", "contrast", "sharpness"), save_path=None):
    """Histogram per feature (top) and box plot by class (bottom)."""
    class_order = [n for n in config.CLASS_NAMES if n in set(df["class"])]
    fig, axes = plt.subplots(2, len(features), figsize=(5 * len(features), 8))
    for j, f in enumerate(features):
        axes[0, j].hist(df[f], bins=40, color="#4c78a8", edgecolor="white")
        axes[0, j].set_title(f"{f.capitalize()} (all images)"); axes[0, j].set_xlabel(f)
        axes[1, j].boxplot([df.loc[df["class"] == c, f] for c in class_order], showfliers=False)
        axes[1, j].set_xticklabels([c.replace(" ", "\n") for c in class_order], fontsize=8)
        axes[1, j].set_title(f"{f.capitalize()} by DR grade")
    return _finish(fig, save_path)


def plot_confusion_matrix(cm, class_names=config.CLASS_NAMES, normalize=False, title="", save_path=None):
    cm = np.asarray(cm, dtype=float)
    shown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(shown, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right"); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title or ("Normalised" if normalize else "Counts"))
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{shown[i, j]:.2f}" if normalize else int(cm[i, j]), ha="center", va="center",
                    color="white" if shown[i, j] > shown.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _finish(fig, save_path)


def plot_history(history, save_path=None):
    """Loss, accuracy and validation macro-F1 curves with the stage boundary marked."""
    stage = history.get("stage", [])
    boundary = next((i for i, s in enumerate(stage) if s == 2), None)
    panels = [("Loss", "loss", "val_loss"), ("Accuracy", "accuracy", "val_accuracy"),
              ("Validation macro-F1", None, "val_macro_f1")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (title, tr, va) in zip(axes, panels):
        if tr and tr in history:
            ax.plot(history[tr], label="train")
        if va in history:
            ax.plot(history[va], label="validation")
        if boundary is not None:
            ax.axvline(boundary - 0.5, color="gray", ls="--", label="fine-tuning starts")
        ax.set_title(title); ax.set_xlabel("epoch"); ax.legend()
    return _finish(fig, save_path)


def plot_misclassified(images, y_true, probs, indices, class_names=config.CLASS_NAMES, n=12, save_path=None):
    indices = list(indices)[:n]
    cols = 4
    rows = max(1, int(np.ceil(len(indices) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), indices):
        pred = int(np.argmax(probs[i]))
        ax.imshow(images[i])
        ax.set_title(f"true: {class_names[int(y_true[i])]}\npred: {class_names[pred]} ({probs[i][pred]:.2f})",
                     fontsize=9, color="firebrick")
    return _finish(fig, save_path)


def plot_gradcam(image_uint8, heatmap, title="", save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
    axes[0].imshow(image_uint8); axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(image_uint8); axes[1].imshow(heatmap, cmap="jet", alpha=0.4)
    axes[1].set_title(title or "Grad-CAM (model attention)"); axes[1].axis("off")
    return _finish(fig, save_path)
