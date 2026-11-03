"""Registro y versionado en el Model Registry de MLflow.

Política (champion / challenger):
  - Se registra el mejor run de cada familia de modelos -> varias versiones del mismo nombre.
  - La versión con mayor roc_auc en test recibe el alias  @champion  (la que sirve el clúster).
  - La segunda mejor recibe  @challenger  (candidata a reemplazo).
La API consume siempre  models:/credit-default-classifier@champion , nunca una ruta de archivo.
"""
from __future__ import annotations

import mlflow
from mlflow import MlflowClient

from common import EXPERIMENT, MLFLOW_URI, MODEL_NAME, PROD_ALIAS


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs([exp.experiment_id], order_by=["metrics.roc_auc DESC"])

    best_per_family = {}
    for r in runs:
        best_per_family.setdefault(r.data.tags["model_family"], r)
    ranked = sorted(best_per_family.values(), key=lambda r: r.data.metrics["roc_auc"])  # peor -> mejor

    versions = []
    for r in ranked:
        model_uri = f"runs:/{r.info.run_id}/model"
        mv = mlflow.register_model(model_uri, MODEL_NAME)
        client.set_model_version_tag(MODEL_NAME, mv.version, "roc_auc", f"{r.data.metrics['roc_auc']:.4f}")
        client.set_model_version_tag(MODEL_NAME, mv.version, "model_family", r.data.tags["model_family"])
        versions.append((mv, r))
        print(f"registrada v{mv.version}  run={r.info.run_id}  {r.info.run_name}  roc_auc={r.data.metrics['roc_auc']:.4f}")

    champion, champ_run = versions[-1]
    client.set_registered_model_alias(MODEL_NAME, PROD_ALIAS, champion.version)
    client.update_model_version(MODEL_NAME, champion.version,
                                description=f"Desplegado en Kubernetes. Run {champ_run.info.run_id} ({champ_run.info.run_name}).")
    if len(versions) > 1:
        client.set_registered_model_alias(MODEL_NAME, "challenger", versions[-2][0].version)
    print(f"\n@{PROD_ALIAS} -> v{champion.version} (run {champ_run.info.run_id})")


if __name__ == "__main__":
    main()
