"""Constantes compartidas: rutas, nombres de variables y configuración de MLflow."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "credit_default.csv"
DATA_REF = ROOT / "data" / "reference"
DATA_PROD = ROOT / "data" / "production"
REPORTS = ROOT / "reports"

SEED = 42
TARGET = "default"

# Variables numéricas continuas -> prueba Kolmogorov-Smirnov
NUMERIC = [
    "LIMIT_BAL", "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
# Variables categóricas / ordinales de pocos niveles -> Chi-cuadrado + PSI
CATEGORICAL = [
    "SEX", "EDUCATION", "MARRIAGE",
    "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
]
FEATURES = NUMERIC + CATEGORICAL

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:30500")
EXPERIMENT = "credit-default-training"
MONITORING_EXPERIMENT = "credit-default-monitoring"
MODEL_NAME = "credit-default-classifier"
PROD_ALIAS = "champion"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
