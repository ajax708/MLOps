"""Continuous Training (CT) disparado por la alarma de concept drift: champion vs challenger.

1. Toma los lotes recientes YA etiquetados del escenario de crisis (lotes 6-10).
2. Entrena un challenger con la MISMA configuración que el champion (sklearn.clone) sobre esa
   VENTANA RECIENTE. Tras un cambio de concepto, la relación antigua X->y ya no es válida:
   mezclarla con el histórico diluye la señal nueva (se registra también esa variante para comparar).
3. Compara champion vs challenger sobre los lotes más recientes (11-12), no usados para entrenar.
4. Registra el challenger como nueva versión con alias @challenger.
5. Solo con --promote y si el challenger gana: mueve @champion a la nueva versión.
   Luego:  kubectl rollout restart deployment/credit-api -n mlops

Uso:  python src/retrain.py [--promote]
"""
from __future__ import annotations

import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from common import DATA_PROD, DATA_REF, EXPERIMENT, FEATURES, MLFLOW_URI, MODEL_NAME, PROD_ALIAS, TARGET
from drift.concept_drift import DRIFT_START, make_batches
from train import log_sklearn_model

RETRAIN_BATCHES = range(DRIFT_START, 11)   # lotes 6..10 (1-indexados)
EVAL_BATCHES = (11, 12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    champ_mv = client.get_model_version_by_alias(MODEL_NAME, PROD_ALIAS)
    champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{champ_mv.version}")

    train = pd.read_csv(DATA_REF / "train.csv")
    batches = make_batches(pd.read_csv(DATA_PROD / "pool.csv"))
    recent = pd.concat([batches[i - 1] for i in RETRAIN_BATCHES])
    evaluation = pd.concat([batches[i - 1] for i in EVAL_BATCHES])
    new_train = recent.reset_index(drop=True)
    history_plus_recent = pd.concat([train, recent], ignore_index=True)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"retrain-challenger-from-v{champ_mv.version}") as run:
        challenger = clone(champion).fit(new_train[FEATURES], new_train[TARGET])
        mixed = clone(champion).fit(history_plus_recent[FEATURES], history_plus_recent[TARGET])
        auc_champ = roc_auc_score(evaluation[TARGET], champion.predict_proba(evaluation[FEATURES])[:, 1])
        auc_chall = roc_auc_score(evaluation[TARGET], challenger.predict_proba(evaluation[FEATURES])[:, 1])
        auc_mixed = roc_auc_score(evaluation[TARGET], mixed.predict_proba(evaluation[FEATURES])[:, 1])
        mlflow.log_metric("history_plus_recent_roc_auc", auc_mixed)
        mlflow.set_tags({"model_family": client.get_run(champ_mv.run_id).data.tags.get("model_family", "?"),
                         "trigger": "concept_drift_alarm", "parent_model_version": champ_mv.version,
                         "parent_run_id": champ_mv.run_id})
        mlflow.log_params({"retrain_batches": f"{RETRAIN_BATCHES.start}-{RETRAIN_BATCHES.stop - 1}",
                           "eval_batches": str(EVAL_BATCHES), "train_rows": len(new_train)})
        mlflow.log_metrics({"recent_roc_auc": auc_chall, "champion_recent_roc_auc": auc_champ})
        log_sklearn_model(challenger, new_train[FEATURES].head(5))

    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
    client.set_model_version_tag(MODEL_NAME, mv.version, "trigger", "concept_drift_alarm")
    print(f"champion  v{champ_mv.version}: ROC-AUC lotes recientes = {auc_champ:.4f}")
    print(f"challenger v{mv.version}: ROC-AUC lotes recientes = {auc_chall:.4f}  (run {run.info.run_id}, ventana reciente)")
    print(f"(variante histórico+reciente, no registrada: {auc_mixed:.4f})")

    if auc_chall > auc_champ and args.promote:
        client.set_registered_model_alias(MODEL_NAME, PROD_ALIAS, mv.version)
        print(f"PROMOVIDO: @{PROD_ALIAS} -> v{mv.version}. Ejecuta: kubectl rollout restart deployment/credit-api -n mlops")
    elif auc_chall > auc_champ:
        print("El challenger gana. Ejecuta con --promote para moverlo a @champion.")
    else:
        print("El challenger NO supera al champion: se mantiene el modelo en producción.")


if __name__ == "__main__":
    main()
