"""Flood-request simulation (the brief asks for its results in the README).

Fires many concurrent requests at the running API and reports success rate, throughput and latency percentiles.
Dependency-free apart from `requests` (and numpy). Locust alternative: tests/load/locustfile.py.

    python -m scripts.flood_test --url http://127.0.0.1:8000 --requests 200 --concurrency 10
    python -m scripts.flood_test --endpoint health --requests 1000 --concurrency 50

Results are written to outputs/reports/flood_test_<timestamp>.json. Nothing is invented: run it, then paste the printed table.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np


def run_flood(send: Callable[[], tuple], n_requests: int, concurrency: int) -> dict:
    """``send()`` performs ONE request and returns ``(status_code or None, seconds)``. Exceptions count as failures."""
    def one(_):
        t0 = time.perf_counter()
        try:
            status, _sec = send()
        except Exception as exc:                          # connection error, timeout ...
            return None, time.perf_counter() - t0, type(exc).__name__
        return status, time.perf_counter() - t0, None

    wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, range(n_requests)))
    return summarize(results, time.perf_counter() - wall0, concurrency)


def summarize(results: list, wall_seconds: float, concurrency: int) -> dict:
    n = len(results)
    ok = [r for r in results if r[0] is not None and 200 <= r[0] < 300]
    lat = np.array([r[1] for r in results]) * 1000
    ok_lat = np.array([r[1] for r in ok]) * 1000 if ok else np.array([0.0])
    return {
        "requests": n, "concurrency": concurrency, "succeeded": len(ok), "failed": n - len(ok),
        "success_rate_percent": round(100 * len(ok) / n, 2) if n else 0.0,
        "duration_seconds": round(wall_seconds, 3),
        "throughput_rps": round(n / wall_seconds, 2) if wall_seconds > 0 else None,
        "latency_ms_successful": {"mean": round(float(ok_lat.mean()), 1), "p50": round(float(np.percentile(ok_lat, 50)), 1),
                                  "p95": round(float(np.percentile(ok_lat, 95)), 1),
                                  "p99": round(float(np.percentile(ok_lat, 99)), 1), "max": round(float(ok_lat.max()), 1)},
        "status_codes": dict(Counter(str(r[0]) if r[0] is not None else (r[2] or "error") for r in results)),
    }


def markdown_table(res: dict, label: str) -> str:
    l = res["latency_ms_successful"]
    return ("| Scenario | Requests | Concurrency | Success | Throughput (req/s) | Mean ms | p95 ms | p99 ms | Max ms |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
            f"| {label} | {res['requests']} | {res['concurrency']} | {res['succeeded']}/{res['requests']} "
            f"({res['success_rate_percent']}%) | {res['throughput_rps']} | {l['mean']} | {l['p95']} | {l['p99']} | {l['max']} |")


def _synthetic_png() -> bytes:
    from PIL import Image
    yy, xx = np.mgrid[:224, :224]
    img = np.zeros((224, 224, 3), np.uint8)
    img[((yy - 112) ** 2 + (xx - 112) ** 2) < 100 ** 2] = (120, 60, 30)
    buf = io.BytesIO(); Image.fromarray(img).save(buf, "PNG"); return buf.getvalue()


def main() -> None:
    import requests
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--endpoint", choices=["predict", "health"], default="predict")
    ap.add_argument("--image", help="image file for /predict (default: a synthetic fundus-like PNG)")
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--out-dir", default="outputs/reports")
    a = ap.parse_args()

    data = Path(a.image).read_bytes() if a.image else _synthetic_png()
    base = a.url.rstrip("/")
    def send():
        if a.endpoint == "health":
            r = requests.get(base + "/health", timeout=60)
        else:
            r = requests.post(base + "/predict", files={"file": ("flood.png", data, "image/png")}, timeout=120)
        return r.status_code, 0.0

    print(f"Flooding {base}/{a.endpoint}: {a.requests} requests, concurrency {a.concurrency} ...")
    res = run_flood(send, a.requests, a.concurrency)
    res.update({"endpoint": a.endpoint, "url": base, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"flood_test_{a.endpoint}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)); print("\n" + markdown_table(res, f"POST /predict" if a.endpoint == "predict" else "GET /health"))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
