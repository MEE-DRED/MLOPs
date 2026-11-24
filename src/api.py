"""FastAPI application: a thin HTTP layer over ``AppService``.

Run:  uvicorn src.api:app --host 0.0.0.0 --port 8000
State-changing endpoints require the ``X-API-Key`` header only if the
environment variable ``DR_API_KEY`` is set.

NOTE: written against FastAPI >= 0.110; NOT executed in the build sandbox (FastAPI not installed there).
"""
from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .model_registry import RegistryError
from .prediction import ModelNotLoadedError
from .preprocessing import InvalidImageError
from .retrain import RetrainingBusy, RetrainingNotPossible
from .service import AppService


def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if config.API_KEY and not (x_api_key and hmac.compare_digest(x_api_key, config.API_KEY)):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


def create_app(service: Optional[AppService] = None) -> FastAPI:
    state: dict = {"service": service}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["service"] is None:
            state["service"] = AppService()
        state["service"].startup()
        yield

    app = FastAPI(title="Diabetic Retinopathy Detection API", version="0.3.0", lifespan=lifespan,
                  description=config.DISCLAIMER)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def svc() -> AppService:
        return state["service"]

    @app.get("/health")
    def health():
        return svc().health()

    @app.post("/predict")
    def predict(file: UploadFile = File(...)):
        data = file.file.read()                     # sync endpoint -> runs in the threadpool, not the event loop
        try:
            return svc().predict(data)
        except InvalidImageError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid image: {exc}")
        except ModelNotLoadedError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    @app.post("/upload-retraining-data", dependencies=[Depends(require_key)])
    def upload_retraining_data(files: List[UploadFile] = File(...), labels: str = Form(...)):
        """``labels``: JSON list (``[0,3,1]``) or comma-separated (``0,3,1``), one per file, same order."""
        payload = [(f.filename or "unnamed", f.file.read()) for f in files]
        try:
            return svc().upload_retraining_data(payload, labels)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/retraining-data")
    def retraining_data():
        return svc().retraining_data_summary()

    @app.post("/retrain", dependencies=[Depends(require_key)], status_code=202)
    def retrain(wait: bool = Query(False, description="Block until finished (default: run in background)")):
        try:
            return svc().retrain(wait=wait)
        except RetrainingNotPossible as exc:
            raise HTTPException(status_code=422, detail={"message": str(exc), **exc.detail})
        except RetrainingBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.get("/retrain/status")
    def retrain_status(event_id: Optional[int] = None):
        return svc().retrain_status(event_id)

    @app.get("/monitoring")
    def monitoring():
        return svc().monitoring()

    @app.get("/models")
    def models():
        return svc().models()

    @app.post("/models/rollback", dependencies=[Depends(require_key)])
    def rollback():
        try:
            return svc().rollback()
        except (RegistryError, RetrainingBusy) as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    return app


app = create_app()
