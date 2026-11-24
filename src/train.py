"""Two-stage training, hyper-parameter search and reproducibility helpers.

Stage 1: backbone frozen, train the classification head.
Stage 2: unfreeze the top backbone layers and fine-tune at a lower rate.

Model selection uses validation macro-F1 (the classes are imbalanced, so
accuracy alone would reward predicting "No DR"). The test split is never
touched in this module.
"""
from __future__ import annotations

import gc
import json
import os
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from . import config
from .evaluate import predict_probabilities
from .model import build_model, compile_model, set_backbone_trainable
from .preprocessing import make_tf_dataset


@dataclass(frozen=True)
class TrainConfig:
    seed: int = config.SEED
    dropout: float = 0.3
    l2: float = 1e-4
    batch_size: int = 32
    lr_head: float = 1e-3
    lr_finetune: float = 1e-4
    epochs_head: int = 15
    epochs_finetune: int = 25
    unfreeze_layers: int = 40
    patience_head: int = 5
    patience_finetune: int = 8
    use_class_weights: bool = True
    augment: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int = config.SEED) -> None:
    """Seed Python, NumPy and TensorFlow. (Full GPU determinism is not
    guaranteed; small run-to-run differences are expected.)"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def balanced_class_weights(labels: np.ndarray, num_classes: int = config.NUM_CLASSES) -> dict:
    classes = np.arange(num_classes)
    w = compute_class_weight("balanced", classes=classes, y=np.asarray(labels).reshape(-1))
    return {int(c): float(x) for c, x in zip(classes, w)}


class MacroF1Callback(keras.callbacks.Callback):
    """Adds ``val_macro_f1`` to the logs each epoch.

    Must come BEFORE EarlyStopping / ModelCheckpoint in the callback list so
    they can see the value.
    """

    def __init__(self, val_images, val_labels, batch_size=32):
        super().__init__()
        self.x, self.y, self.bs = val_images, np.asarray(val_labels).reshape(-1), batch_size
        self.values: list[float] = []          # own record: does not depend on Keras History internals

    def on_epoch_end(self, epoch, logs=None):
        probs = predict_probabilities(self.model, self.x, self.bs)
        logs = logs if logs is not None else {}
        value = float(f1_score(self.y, probs.argmax(1), average="macro",
                               labels=list(range(config.NUM_CLASSES)), zero_division=0))
        self.values.append(value)
        logs["val_macro_f1"] = value           # visible to EarlyStopping / ModelCheckpoint


def make_datasets(train, val, batch_size, seed=config.SEED):
    """``train``/``val`` are ``(uint8 images, int labels)`` tuples."""
    return (make_tf_dataset(*train, batch_size=batch_size, shuffle=True, seed=seed),
            make_tf_dataset(*val, batch_size=batch_size, shuffle=False))


def _fit_stage(model, stage, lr, epochs, patience, train_ds, val_ds, val, class_weight,
               ckpt_path, verbose, batch_size):
    compile_model(model, lr)
    f1_cb = MacroF1Callback(val[0], val[1], batch_size)
    callbacks = [
        f1_cb,                                                             # must be first
        keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_macro_f1", mode="max",
                                        save_best_only=True, save_weights_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight,
                     callbacks=callbacks, verbose=verbose).history
    if Path(ckpt_path).is_file():          # explicit restore of the best epoch (independent of Keras version)
        model.load_weights(str(ckpt_path))
    hist = {k: [float(v) for v in vals] for k, vals in hist.items()}
    hist["val_macro_f1"] = list(f1_cb.values)
    hist["stage"] = [stage] * len(hist["loss"])
    best = max(f1_cb.values) if f1_cb.values else float("-inf")
    return hist, best


def run_stage1(model, cfg: TrainConfig, train_ds, val_ds, val, class_weight, out_dir, verbose=2):
    """Frozen backbone, train the head. Returns (history, best_val_macro_f1, ckpt_path)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stats = set_backbone_trainable(model, 0)
    ckpt = out_dir / "stage1_best.weights.h5"
    hist, best = _fit_stage(model, 1, cfg.lr_head, cfg.epochs_head, cfg.patience_head, train_ds,
                            val_ds, val, class_weight, ckpt, verbose, cfg.batch_size)
    return hist, best, ckpt, stats


def run_stage2(model, cfg: TrainConfig, train_ds, val_ds, val, class_weight, out_dir,
               stage1_best, stage1_ckpt, verbose=2):
    """Fine-tune the top backbone layers.

    If fine-tuning never beats the stage-1 validation macro-F1, the stage-1
    weights are restored, so fine-tuning can never make the saved model worse
    on validation. Returns (history, best_f1, chosen_stage, stats).
    """
    out_dir = Path(out_dir)
    stats = set_backbone_trainable(model, cfg.unfreeze_layers)
    ckpt = out_dir / "stage2_best.weights.h5"
    hist, best = _fit_stage(model, 2, cfg.lr_finetune, cfg.epochs_finetune, cfg.patience_finetune,
                            train_ds, val_ds, val, class_weight, ckpt, verbose, cfg.batch_size)
    chosen = 2
    if not (best > stage1_best):
        model.load_weights(str(stage1_ckpt))
        chosen, best = 1, stage1_best
    return hist, best, chosen, stats


def train_two_stage(cfg: TrainConfig, train, val, out_dir, weights="imagenet",
                    allow_weight_fallback=False, verbose=2):
    """Full pipeline: build -> stage 1 -> stage 2. ``train``/``val`` = (images, labels).

    Returns (model, history, info). ``history`` concatenates both stages
    (key ``stage`` marks each epoch).
    """
    set_seed(cfg.seed)
    model, info = build_model(cfg.dropout, cfg.l2, weights, allow_weight_fallback, cfg.augment, seed=cfg.seed)
    cw = balanced_class_weights(train[1]) if cfg.use_class_weights else None
    train_ds, val_ds = make_datasets(train, val, cfg.batch_size, cfg.seed)
    h1, best1, ckpt1, s1 = run_stage1(model, cfg, train_ds, val_ds, val, cw, out_dir, verbose)
    h2, best, chosen, s2 = run_stage2(model, cfg, train_ds, val_ds, val, cw, out_dir, best1, ckpt1, verbose)
    history = {k: h1.get(k, []) + h2.get(k, []) for k in set(h1) | set(h2)}
    info.update({"class_weights": cw, "best_val_macro_f1": best, "stage1_best_val_macro_f1": best1,
                 "selected_stage": chosen, "stage1_params": s1, "stage2_params": s2,
                 "epochs_run": {"stage1": len(h1["loss"]), "stage2": len(h2["loss"])}})
    return model, history, info


def search_hyperparameters(base: TrainConfig, train, val, grid: dict, n_trials: int,
                           trial_epochs=(5, 6), work_dir="/tmp/dr_search", seed=config.SEED, verbose=0):
    """Small random search with shortened training; selects on validation macro-F1.

    ``grid`` maps TrainConfig field -> list of values. The base config is always
    trial 0 so the default is compared fairly (same shortened budget).
    Returns a pandas DataFrame sorted best-first; the ``config`` column holds dicts.
    """
    import itertools
    import pandas as pd
    keys = sorted(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    rng = random.Random(seed)
    rng.shuffle(combos)
    trials = [{}] + [c for c in combos if any(getattr(base, k) != v for k, v in c.items())][:max(n_trials - 1, 0)]

    rows = []
    for i, overrides in enumerate(trials):
        cfg = replace(base, epochs_head=trial_epochs[0], epochs_finetune=trial_epochs[1],
                      patience_head=trial_epochs[0], patience_finetune=trial_epochs[1], **overrides)
        model, hist, info = train_two_stage(cfg, train, val, Path(work_dir) / f"trial_{i}", verbose=verbose)
        probs = predict_probabilities(model, val[0], cfg.batch_size)
        f1 = f1_score(np.asarray(val[1]).reshape(-1), probs.argmax(1), average="macro", zero_division=0)
        acc = float((probs.argmax(1) == np.asarray(val[1]).reshape(-1)).mean())
        rows.append({"trial": i, "overrides": overrides or "(base config)", "val_macro_f1": float(f1),
                     "val_accuracy": acc, "selected_stage": info["selected_stage"],
                     "config": {**asdict(replace(base, **overrides))}})
        del model
        keras.backend.clear_session(); gc.collect()
    return pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)


def save_training_record(out_dir, cfg: TrainConfig, history: dict, info: dict) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "training_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "training_info.json").write_text(json.dumps(info, indent=2, default=str))
