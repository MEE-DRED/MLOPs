"""End-to-end verification against a RUNNING API with the REAL trained model.

    python -m scripts.e2e_check --url http://127.0.0.1:8000 --samples outputs/demo_samples
Sample files named like ``true3_Severe_test27.png`` give the expected class, so the check also reports how many
samples the live model classifies as expected (informational, not a pass/fail criterion).
By default it does NOT start retraining (that needs genuinely new labelled images); add --run-retrain to try it.
Exit code 0 = every check passed.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from app.api_client import ApiClient, ApiError, ApiUnavailable

KEYS = {"predicted_class", "predicted_class_name", "confidence", "probabilities", "model_version"}


def _expect(cond: bool, detail: str):
    return bool(cond), detail


def run_checks(client: ApiClient, samples: list, run_retrain: bool = False, retrain_timeout: int = 3600) -> list[dict]:
    """``samples``: list of (filename, bytes, expected_label or None). Returns [{name, ok, detail}]."""
    checks: list[dict] = []

    def check(name, fn):
        try:
            ok, detail = fn()
        except (ApiError, ApiUnavailable, Exception) as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        checks.append({"name": name, "ok": ok, "detail": detail})
        return ok

    state = {}
    def c_health():
        h = client.health(); state["health"] = h
        return _expect(h.get("model_loaded") and h.get("api") == "up", f"status={h['status']} model_version={h.get('model_version')} uptime={h['uptime_seconds']}s")
    if not check("GET /health: API up and model loaded", c_health):
        return checks

    before = client.monitoring()["predictions"]
    matched = 0
    def c_predict():
        nonlocal matched
        if not samples:
            return False, "no sample images supplied (--samples)"
        lines = []
        for name, data, expected in samples:
            r = client.predict(data, name)
            if not KEYS <= set(r) or abs(sum(r["probabilities"].values()) - 1) > 1e-3 or len(r["probabilities"]) != 5:
                return False, f"bad response structure for {name}: {sorted(r)}"
            if r["model_version"] != state["health"]["model_version"]:
                return False, f"prediction used v{r['model_version']} but /health says v{state['health']['model_version']}"
            if expected is not None:
                matched += int(r["predicted_class"] == expected)
            lines.append(f"{name}->{r['predicted_class_name']}({r['confidence']:.2f})")
        return True, "; ".join(lines[:4]) + (" ..." if len(lines) > 4 else "")
    check("POST /predict: valid images give a well-formed result", c_predict)

    def c_invalid():
        try:
            client.predict(b"this is not an image", "bad.png")
        except ApiError as exc:
            return _expect(exc.status == 422, f"HTTP {exc.status}: {exc.detail}")
        return False, "invalid file was accepted"
    check("POST /predict: invalid file rejected with 422", c_invalid)

    def c_monitor():
        after = client.monitoring()["predictions"]
        return _expect(after["total"] - before["total"] == len(samples) + 1 and after["failed"] - before["failed"] == 1,
                       f"+{after['total'] - before['total']} requests, +{after['failed'] - before['failed']} failed recorded")
    check("GET /monitoring: counts reflect the requests just made", c_monitor)

    def c_bad_label():
        if not samples:
            return False, "no samples"
        rep = client.upload_retraining_data([(samples[0][0], samples[0][1])], [9])
        return _expect(rep["n_accepted"] == 0 and rep["n_rejected"] == 1 and "invalid label" in rep["results"][0]["reason"],
                       rep["results"][0]["reason"])
    check("POST /upload-retraining-data: invalid label rejected", c_bad_label)

    def c_mismatch():
        try:                                   # bypass the client-side guard: this must be rejected by the SERVER
            client._request("POST", "/upload-retraining-data", files=[("files", ("a.png", b"x", "image/png")),
                                                                      ("files", ("b.png", b"y", "image/png"))],
                            data={"labels": "1"})
        except ApiError as exc:
            return _expect(exc.status == 422, f"HTTP {exc.status}: {exc.detail}")
        return False, "server accepted 2 files with 1 label"
    check("POST /upload-retraining-data: file/label count mismatch rejected with 422 by the server", c_mismatch)

    def c_data():
        d = client.retraining_data()
        return _expect({"eligible", "message", "n_accepted_images", "class_counts"} <= set(d), f"eligible={d['eligible']}: {d['message']}")
    check("GET /retraining-data: summary available", c_data)

    def c_refusal():
        d = client.retraining_data()
        if d["eligible"]:
            return True, "skipped (enough data uploaded to be eligible)"
        try:
            client.retrain()
        except ApiError as exc:
            return _expect(exc.status == 422 and isinstance(exc.detail, dict) and "message" in exc.detail, str(exc.detail.get("message") if isinstance(exc.detail, dict) else exc.detail))
        return False, "retraining started without enough data"
    check("POST /retrain: refused with a clear message when data is insufficient", c_refusal)

    def c_models():
        m = client.models()
        return _expect(m["current_version"] == state["health"]["model_version"] and m["versions"], f"current v{m['current_version']}, {len(m['versions'])} version(s) registered")
    check("GET /models: registry consistent with /health", c_models)

    if run_retrain:
        def c_retrain():
            if not client.retraining_data()["eligible"]:
                return False, "not eligible - upload at least the required number of NEW labelled images first"
            ev = client.retrain()["event_id"]; t0 = time.time()
            while time.time() - t0 < retrain_timeout:
                s = client.retrain_status(ev)
                if not s["running"]:
                    e = s["event"]; return _expect(e["status"] in ("accepted", "rejected"), f"{e['status']}: {e['message']}")
                time.sleep(5)
            return False, "timed out"
        check("POST /retrain: real retraining job runs to a decision", c_retrain)

    checks.append({"name": "Informational: live model matched the expected class on labelled samples",
                   "ok": True, "detail": f"{matched}/{sum(e is not None for _, _, e in samples)}"})
    return checks


def load_samples(folder: Path) -> list:
    out = []
    for p in sorted(Path(folder).glob("*.png")) + sorted(Path(folder).glob("*.jpg")):
        m = re.match(r"true(\d)_", p.name)
        out.append((p.name, p.read_bytes(), int(m.group(1)) if m else None))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--samples", default="outputs/demo_samples")
    ap.add_argument("--run-retrain", action="store_true")
    ap.add_argument("--out", default="outputs/reports/e2e_check.json")
    a = ap.parse_args()
    results = run_checks(ApiClient(a.url, a.api_key), load_samples(Path(a.samples)), a.run_retrain)
    for r in results:
        print(f"[{'PASS' if r['ok'] else 'FAIL'}] {r['name']}\n        {r['detail']}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed -> {a.out}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
