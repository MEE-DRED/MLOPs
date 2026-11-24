"""Monitoring helpers: uptime and simple, honestly-labelled shift indicators.

Two DIFFERENT things are reported and must not be confused:

* ``class_distribution_shift``: how the mix of uploaded *labels* differs from the original
  training labels. This is a change in class proportions only. It says nothing about image content.
* ``feature_drift``: a two-sample Kolmogorov-Smirnov test comparing image statistics
  (brightness, contrast, sharpness, retina area) of recently seen images with the training images.
  This is a crude covariate-shift indicator, not proof of model degradation.
Neither is assessed below ``min_n`` samples.
"""
from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from . import config


class ServiceClock:
    """Process start time -> uptime. Resets when the service restarts (by design)."""

    def __init__(self):
        self.started_monotonic = time.monotonic()
        self.started_at = time.time()

    def uptime_seconds(self) -> float:
        return time.monotonic() - self.started_monotonic


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return (f"{d}d " if d else "") + f"{h:02d}:{m:02d}:{s:02d}"


def class_distribution_shift(observed_counts: Sequence[int], reference_counts: Sequence[int],
                             min_n: int = 30) -> dict:
    obs, ref = np.asarray(observed_counts, float), np.asarray(reference_counts, float)
    n = int(obs.sum())
    out = {"kind": "class_distribution_change (labels only, NOT image drift)", "n_observed": n,
           "assessable": n >= min_n and ref.sum() > 0, "min_n": min_n}
    if not out["assessable"]:
        out["note"] = f"Need at least {min_n} labelled uploads to compare class proportions."
        return out
    o, r = obs / obs.sum(), ref / ref.sum()
    out.update({"observed_share": o.round(4).tolist(), "reference_share": r.round(4).tolist(),
                "total_variation_distance": float(0.5 * np.abs(o - r).sum())})
    return out


def feature_drift(reference: dict, observed: dict, min_n: int = 30, alpha: float = 0.01) -> dict:
    from scipy.stats import ks_2samp

    n = min((len(v) for v in observed.values()), default=0)
    out = {"kind": "image_statistics_shift (KS test; indicator only)", "n_observed": int(n),
           "assessable": n >= min_n, "alpha": alpha, "features": {}}
    if not out["assessable"]:
        out["note"] = f"Need at least {min_n} recent images to compare image statistics."
        return out
    for name, ref_vals in reference.items():
        if name not in observed or len(observed[name]) == 0:
            continue
        res = ks_2samp(np.asarray(ref_vals), np.asarray(observed[name]))
        out["features"][name] = {"ks_statistic": float(res.statistic), "p_value": float(res.pvalue),
                                 "shift_flagged": bool(res.pvalue < alpha),
                                 "reference_median": float(np.median(ref_vals)),
                                 "observed_median": float(np.median(observed[name]))}
    out["any_flagged"] = any(f["shift_flagged"] for f in out["features"].values())
    out["note"] = ("A flag means the recent images look statistically different from the training images "
                   "(e.g. another camera). It does not mean accuracy has dropped.")
    return out
