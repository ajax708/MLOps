# Servidor de MLflow (tracking + Model Registry + proxy de artefactos) para el clúster
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MLFLOW_DISABLE_AGENT_HINT=1
COPY requirements/mlflow-server.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && mkdir -p /mlflow

EXPOSE 5000
# Backend: SQLite en volumen persistente. Artefactos: servidos por el propio servidor
# (mlflow-artifacts:/), así los pods descargan el modelo por HTTP sin rutas locales.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", "--workers", "2", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--allowed-hosts", "mlflow,mlflow:*,mlflow.mlops.svc.cluster.local:*,localhost,localhost:*,127.0.0.1:*,10.*,172.*,192.168.*"]
