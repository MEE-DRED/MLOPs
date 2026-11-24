"""Shared test fixtures (no TensorFlow required)."""
import io
import numpy as np
from PIL import Image

from src import config


def png_bytes(size=(300, 200), mode="RGB", color=(120, 60, 30), fmt="PNG"):
    img = Image.new(mode, size, color if mode == "RGB" else 128)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def fake_fundus(n=1, seed=0):
    """Random uint8 images with a dark border, shape (n,224,224,3)."""
    rng = np.random.default_rng(seed)
    x = rng.integers(40, 200, size=(n, 224, 224, 3)).astype(np.uint8)
    yy, xx = np.mgrid[:224, :224]
    x[:, ((yy - 112) ** 2 + (xx - 112) ** 2) > 105 ** 2] = 0
    return x


class StubModel:
    """predict() returns a fixed-ish softmax; lets us test without TensorFlow."""

    def __init__(self, favourite=2, bad=None):
        self.favourite, self.bad = favourite, bad
        self.calls = 0

    def predict(self, batch, verbose=0):
        self.calls += 1
        n = len(batch)
        if self.bad == "nan":
            return np.full((n, config.NUM_CLASSES), np.nan)
        if self.bad == "shape":
            return np.ones((n, 3)) / 3
        out = np.full((n, config.NUM_CLASSES), 0.05)
        out[:, self.favourite] = 0.8
        return out


# ----------------------------------------------------------- Phase 3 fixtures
import json
import tempfile
import threading
from pathlib import Path


def array_png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class FakeModel:
    def __init__(self, acc):
        self.acc = acc

    def predict(self, batch, verbose=0):
        out = np.full((len(batch), config.NUM_CLASSES), 0.1)
        out[:, 0] = 0.6
        return out


class FakeTrainer:
    """Models are small JSON files ``{"acc": x}``. ``predict_probs`` returns predictions that are right for
    the first ``acc`` fraction of the validation images, so metrics are fully controllable."""

    def __init__(self, val_labels, candidate_acc=0.9, fail=False, gate: "threading.Event | None" = None):
        self.val_labels = np.asarray(val_labels)
        self.candidate_acc, self.fail, self.gate = candidate_acc, fail, gate
        self.finetune_calls = []

    def load(self, path):
        return FakeModel(json.loads(Path(path).read_text())["acc"])

    def predict_probs(self, model, images):
        n = len(images)
        labels = self.val_labels[:n]
        n_right = int(round(model.acc * n))
        pred = np.where(np.arange(n) < n_right, labels, (labels + 1) % config.NUM_CLASSES)
        probs = np.full((n, config.NUM_CLASSES), 0.025)
        probs[np.arange(n), pred] = 0.9
        return probs

    def finetune(self, model, new, replay, val, settings):
        self.finetune_calls.append({"new": new, "replay": replay, "val": val, "settings": settings})
        if self.gate is not None:
            self.gate.wait(timeout=10)
        if self.fail:
            raise RuntimeError("simulated training crash")
        model.acc = self.candidate_acc
        return {"epochs_run": 3}

    def save(self, model, path):
        Path(path).write_text(json.dumps({"acc": model.acc}))


def make_reference(n_train=60, n_val=60, n_test=20):
    from src.reference import ArrayReference
    rng = np.random.default_rng(5)
    train = (fake_fundus(n_train, seed=101), (np.arange(n_train) % 5).astype(np.int64))
    val = (fake_fundus(n_val, seed=202), rng.permutation(np.arange(n_val) % 5).astype(np.int64))
    test = (fake_fundus(n_test, seed=303), (np.arange(n_test) % 5).astype(np.int64))
    return ArrayReference(train, val, test)


def make_env(current_acc=0.60, candidate_acc=0.90, **trainer_kw):
    """Temp dir with db, registry (v1 current), reference data, fake trainer, upload dir."""
    from src.database import Database
    from src.model_registry import ModelRegistry, build_metadata
    root = Path(tempfile.mkdtemp())
    ref = make_reference()
    trainer = FakeTrainer(ref.val()[1], candidate_acc=candidate_acc, **trainer_kw)
    reg = ModelRegistry(root / "models")
    f = root / "v1.keras"
    f.write_text(json.dumps({"acc": current_acc}))
    from src.evaluate import compute_metrics
    base = compute_metrics(ref.val()[1], trainer.predict_probs(FakeModel(current_acc), ref.val()[0]))
    reg.register_version(f, build_metadata(architecture="fake", training_dataset={}, training_config={},
                                           metrics={"validation": base}), "accepted", make_current=True)
    return {"root": root, "db": Database(root / "db.sqlite3"), "registry": reg, "reference": ref,
            "trainer": trainer, "upload_dir": root / "uploads"}


def upload_images(env, n, seed=900, labels=None, start_idx=0):
    """Ingest ``n`` distinct valid labelled images; returns the ingest report."""
    from src.uploads import ingest_batch
    imgs = fake_fundus(n, seed=seed)
    files = [(f"img{start_idx + i}.png", array_png(imgs[i])) for i in range(n)]
    labels = labels if labels is not None else [i % 3 for i in range(n)]
    return ingest_batch(env["db"], files, labels, upload_dir=env["upload_dir"],
                        hash_index=env["reference"].hash_index())
