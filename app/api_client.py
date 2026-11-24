"""HTTP client for the FastAPI backend used by the Streamlit dashboard.

The dashboard never touches the database or model files directly: everything goes through the API.
A ``session`` can be injected (tests use an in-process stand-in); by default ``requests`` is used.
"""
from __future__ import annotations

import json
import mimetypes
from typing import Optional, Sequence


class ApiUnavailable(Exception):
    """The API could not be reached (connection refused, timeout, DNS ...)."""


class ApiError(Exception):
    """The API answered with an error status."""

    def __init__(self, status: int, detail):
        self.status, self.detail = status, detail
        super().__init__(f"HTTP {status}: {detail}")


class ApiClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "", session=None,
                 timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        if session is None:
            import requests
            session = requests.Session()
        self.session = session

    def _request(self, method: str, path: str, timeout: Optional[float] = None, **kwargs):
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        try:
            resp = self.session.request(method, self.base_url + path, headers=headers,
                                        timeout=timeout or self.timeout, **kwargs)
        except OSError as exc:                       # requests.RequestException subclasses OSError
            raise ApiUnavailable(f"{type(exc).__name__}: {exc}") from exc
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise ApiError(resp.status_code, detail)
        return resp.json()

    # ------------------------------------------------------------ endpoints
    def health(self) -> dict:
        return self._request("GET", "/health", timeout=5)

    def predict(self, data: bytes, filename: str = "image.png") -> dict:
        return self._request("POST", "/predict", timeout=120,
                             files={"file": (filename, data, _mime(filename))})

    def upload_retraining_data(self, files: Sequence[tuple[str, bytes]], labels: Sequence[int]) -> dict:
        if len(files) != len(labels):
            raise ValueError("every file needs exactly one label")
        return self._request("POST", "/upload-retraining-data", timeout=300,
                             files=[("files", (n, b, _mime(n))) for n, b in files],
                             data={"labels": json.dumps([int(x) for x in labels])})

    def retraining_data(self) -> dict:
        return self._request("GET", "/retraining-data")

    def retrain(self) -> dict:
        return self._request("POST", "/retrain", params={"wait": "false"})

    def retrain_status(self, event_id: Optional[int] = None) -> dict:
        return self._request("GET", "/retrain/status", params={"event_id": event_id} if event_id else None)

    def monitoring(self) -> dict:
        return self._request("GET", "/monitoring")

    def models(self) -> dict:
        return self._request("GET", "/models")

    def rollback(self) -> dict:
        return self._request("POST", "/models/rollback")


def _mime(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"
