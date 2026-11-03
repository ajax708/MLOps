# Servicio de inferencia (FastAPI + modelo del MLflow Model Registry)
# Python 3.11 = misma versión que el entorno donde se entrenó el modelo.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MODEL_NAME=credit-default-classifier \
    MODEL_ALIAS=champion

WORKDIR /app

# 1) dependencias primero: la capa se cachea mientras requirements no cambie
COPY requirements/api.txt requirements.txt
RUN pip install -r requirements.txt

# 2) código
COPY src/api ./api

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
