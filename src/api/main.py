"""Servicio de inferencia FastAPI.

Al arrancar resuelve el alias del Model Registry  (models:/<MODEL_NAME>@<MODEL_ALIAS>)
contra el servidor de MLflow del clúster, obtiene número de versión y run_id, y carga el
modelo en memoria. Mientras no esté cargado, /ready responde 503 y Kubernetes no le envía
tráfico (readinessProbe). Cada respuesta incluye el nombre del pod que la atendió.
"""
from __future__ import annotations

import logging
import os
import socket
import threading
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ModelInfo, PredictRequest, PredictResponse, Prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("credit-api")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "credit-default-classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
POD_NAME = os.getenv("HOSTNAME", socket.gethostname())


class ModelState:
    def __init__(self) -> None:
        self.model = None
        self.version: str | None = None
        self.run_id: str | None = None
        self.features: list[str] | None = None
        self.error: str | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None


state = ModelState()


def load_model_from_registry(max_wait_s: int = 600) -> None:
    """Resuelve alias -> versión -> run y carga el modelo. Reintenta si MLflow aún no está listo."""
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(TRACKING_URI)
    deadline, delay = time.time() + max_wait_s, 2
    while time.time() < deadline:
        try:
            mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}")
            state.features = list(model.feature_names_in_)
            state.version, state.run_id, state.model, state.error = str(mv.version), mv.run_id, model, None
            log.info("Modelo cargado: %s@%s -> v%s (run %s)", MODEL_NAME, MODEL_ALIAS, mv.version, mv.run_id)
            return
        except Exception as exc:  # MLflow caído o alias inexistente: reintentar
            state.error = f"{type(exc).__name__}: {exc}"
            log.warning("No se pudo cargar el modelo (%s). Reintento en %ss", state.error, delay)
            time.sleep(delay)
            delay = min(delay * 2, 30)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.getenv("SKIP_MODEL_LOAD") != "1":
        threading.Thread(target=load_model_from_registry, daemon=True).start()
    yield


app = FastAPI(title="Credit Default API", version="1.0.0", lifespan=lifespan)
# CORS abierto para que una UI HTML/JS pueda consumir la API desde el navegador
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health() -> dict:
    """Liveness: el proceso está vivo (no depende del modelo)."""
    return {"status": "alive", "pod_name": POD_NAME}


@app.get("/ready")
def ready(response: Response) -> dict:
    """Readiness: solo 200 cuando el modelo está cargado."""
    if not state.ready:
        response.status_code = 503
    return {"ready": state.ready, "pod_name": POD_NAME, "error": state.error}


@app.get("/info", response_model=ModelInfo)
def info() -> ModelInfo:
    return ModelInfo(
        pod_name=POD_NAME, ready=state.ready, model_name=MODEL_NAME, model_alias=MODEL_ALIAS,
        model_version=state.version, run_id=state.run_id,
        model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}", tracking_uri=TRACKING_URI, error=state.error,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not state.ready:
        raise HTTPException(status_code=503, detail="Modelo aún no cargado")
    df = pd.DataFrame([r.model_dump() for r in req.records])[state.features]
    proba = state.model.predict_proba(df)[:, 1]
    preds = [
        Prediction(default_probability=round(float(p), 4), prediction=int(p >= THRESHOLD),
                   label="DEFAULT" if p >= THRESHOLD else "PAGA")
        for p in proba
    ]
    return PredictResponse(predictions=preds, pod_name=POD_NAME, model_name=MODEL_NAME,
                           model_alias=MODEL_ALIAS, model_version=state.version or "", run_id=state.run_id or "")
