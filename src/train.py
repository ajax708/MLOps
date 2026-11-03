"""Experimentación con MLflow: varias familias de modelos e hiperparámetros.

Cada configuración es un run del experimento `credit-default-training` con:
  parámetros, métricas en test (roc_auc = métrica principal), tags de trazabilidad
  (commit de git, hash de los datos), el dataset de entrada y el modelo con signature.

Uso:  python src/train.py            (MLFLOW_TRACKING_URI por defecto http://localhost:30500)
"""
from __future__ import annotations

import json
import subprocess
import warnings

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (CATEGORICAL, DATA_REF, EXPERIMENT, FEATURES, MLFLOW_URI, NUMERIC, SEED,
                    TARGET)

# Las 23 variables del dataset son enteras y sin nulos: el aviso de MLflow no aplica.
warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column.*")

# Al menos 5 configuraciones comparables (mismo split, misma semilla, mismas métricas)
CONFIGS = [
    ("logreg", {"C": 0.1, "max_iter": 1000}),
    ("logreg", {"C": 1.0, "max_iter": 1000}),
    ("random_forest", {"n_estimators": 100, "max_depth": 6, "min_samples_leaf": 5}),
    ("random_forest", {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 10}),
    ("random_forest", {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 20}),
    ("hist_gb", {"learning_rate": 0.1, "max_depth": 3, "max_iter": 200}),
    ("hist_gb", {"learning_rate": 0.05, "max_depth": 5, "max_iter": 300}),
    ("hist_gb", {"learning_rate": 0.03, "max_depth": None, "max_iter": 500, "l2_regularization": 1.0}),
]


def build_pipeline(family: str, params: dict) -> Pipeline:
    if family == "logreg":
        clf = LogisticRegression(random_state=SEED, class_weight="balanced", **params)
    elif family == "random_forest":
        clf = RandomForestClassifier(random_state=SEED, n_jobs=-1, class_weight="balanced", **params)
    elif family == "hist_gb":
        clf = HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced", **params)
    else:
        raise ValueError(family)
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("preprocess", pre), ("model", clf)])


def log_sklearn_model(pipe: Pipeline, sample: pd.DataFrame) -> None:
    """Guarda el modelo con el formato seguro skops (por defecto en MLflow 3).

    skops no ejecuta código arbitrario al cargar (a diferencia de pickle), pero exige declarar
    los tipos "no estándar" (p. ej. sklearn.tree._tree.Tree). Como el modelo lo entrenamos
    nosotros, se declaran explícitamente esos tipos y quedan registrados en el MLmodel.
    """
    import skops.io as sio
    trusted = sio.get_untrusted_types(data=sio.dumps(pipe))
    mlflow.set_tag("skops_trusted_types", ",".join(trusted) or "none")
    mlflow.sklearn.log_model(
        sk_model=pipe,
        name="model",
        signature=infer_signature(sample, pipe.predict_proba(sample)[:, 1]),
        input_example=sample,
        skops_trusted_types=trusted or None,
    )


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "uncommitted"


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    train = pd.read_csv(DATA_REF / "train.csv")
    test = pd.read_csv(DATA_REF / "test.csv")
    manifest = json.loads((DATA_REF / "manifest.json").read_text())
    X_tr, y_tr = train[FEATURES], train[TARGET]
    X_te, y_te = test[FEATURES], test[TARGET]

    train_ds = mlflow.data.from_pandas(train, name="train", targets=TARGET)
    test_ds = mlflow.data.from_pandas(test, name="test", targets=TARGET)

    for family, params in CONFIGS:
        run_name = f"{family}-" + "-".join(f"{k}{v}" for k, v in params.items())
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "model_family": family,
                "git_commit": git_commit(),
                "data.train_sha256": manifest["train_sha256"],
                "data.test_sha256": manifest["test_sha256"],
                "data.raw_sha256": manifest["raw_sha256"],
            })
            mlflow.log_input(train_ds, context="training")
            mlflow.log_input(test_ds, context="evaluation")
            mlflow.log_params({"model_family": family, "seed": SEED, "split": "60/20/20 stratified", **params})

            pipe = build_pipeline(family, params).fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            metrics = {
                "roc_auc": roc_auc_score(y_te, proba),
                "f1": f1_score(y_te, pred),
                "precision": precision_score(y_te, pred),
                "recall": recall_score(y_te, pred),
                "accuracy": accuracy_score(y_te, pred),
            }
            mlflow.log_metrics(metrics)

            log_sklearn_model(pipe, X_tr.head(5))
            print(f"{run_name:70s} roc_auc={metrics['roc_auc']:.4f} f1={metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
