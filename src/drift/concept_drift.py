"""Detección de CONCEPT DRIFT: la relación X -> y cambia aunque X se vea igual.

Escenario simulado ("crisis económica"):
  12 lotes secuenciales de 500 clientes tomados del pool de producción (nunca vistos).
  - Lotes 1-5 : relación original.
  - Lotes 6-12: la crisis golpea a clientes que venían pagando al día (PAY_1 <= 0) pero con
                ALTA UTILIZACIÓN del crédito (BILL_AMT1 / LIMIT_BAL > 0.5, ~30 % de la cartera):
                una fracción creciente (30 %, 50 %, 70 %, 80 %...) de sus etiquetas 0 pasa a 1.
                Antes la utilización alta con pagos al día era "riesgo bajo"; ahora es riesgo alto.
  Las variables de entrada NO se modifican -> el data drift sale VERDE y el modelo se
  equivoca cada vez más. Solo se detecta cuando llegan las etiquetas reales.

Criterio de alarma / reentrenamiento (CT):
  baseline   = ROC-AUC del modelo en test
  sigma      = desviación estándar del ROC-AUC en 300 lotes bootstrap de 500 filas del test
               (variación "normal" esperable por tamaño de lote, sin drift)
  umbral     = baseline - 3*sigma         (un lote sano cae por debajo con prob. ~0.1 %)
  ALARMA     = 3 lotes CONSECUTIVOS por debajo del umbral  -> reentrenar (challenger)
  (Una caída aislada se trata como ruido; exigir persistencia evita reentrenos innecesarios.)

Uso:
  python src/drift/concept_drift.py              # usa models:/credit-default-classifier@champion
  python src/drift/concept_drift.py --local      # entrena un modelo rápido local (sin MLflow)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (DATA_PROD, DATA_REF, FEATURES, MLFLOW_URI, MODEL_NAME,  # noqa: E402
                    MONITORING_EXPERIMENT, PROD_ALIAS, REPORTS, SEED, TARGET)
from drift.data_drift import detect_data_drift  # noqa: E402

N_BATCHES = 12
BATCH_SIZE = 500
DRIFT_START = 6                      # primer lote (1-indexado) con concepto cambiado
FLIP_RATES = [0.30, 0.50, 0.70, 0.80, 0.80, 0.80, 0.80]
UTILIZATION_CUT = 0.5
K_SIGMA = 3
CONSECUTIVE = 3


@dataclass
class Baseline:
    auc: float
    sigma: float

    @property
    def threshold(self) -> float:
        return self.auc - K_SIGMA * self.sigma


def make_batches(pool: pd.DataFrame, seed: int = SEED) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    shuffled = pool.sample(frac=1, random_state=seed).reset_index(drop=True)
    batches = []
    for i in range(N_BATCHES):
        b = shuffled.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].copy()
        if i + 1 >= DRIFT_START:
            rate = FLIP_RATES[min(i + 1 - DRIFT_START, len(FLIP_RATES) - 1)]
            utilization = b["BILL_AMT1"] / b["LIMIT_BAL"]
            segment = (b["PAY_1"] <= 0) & (utilization > UTILIZATION_CUT) & (b[TARGET] == 0)
            flip = segment & (rng.random(len(b)) < rate)
            b.loc[flip, TARGET] = 1
        batches.append(b)
    return batches


def compute_baseline(model, test: pd.DataFrame, n_boot: int = 300) -> Baseline:
    rng = np.random.default_rng(SEED)
    p = model.predict_proba(test[FEATURES])[:, 1]
    y = test[TARGET].to_numpy()
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(test), BATCH_SIZE, replace=True)
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return Baseline(auc=roc_auc_score(y, p), sigma=float(np.std(aucs)))


def monitor(model, batches: list[pd.DataFrame], baseline: Baseline, reference: pd.DataFrame,
            ref_scores: np.ndarray) -> pd.DataFrame:
    rows, streak, alarm_at = [], 0, None
    for i, b in enumerate(batches, start=1):
        proba = model.predict_proba(b[FEATURES])[:, 1]
        auc = roc_auc_score(b[TARGET], proba)
        below = auc < baseline.threshold
        streak = streak + 1 if below else 0
        if streak >= CONSECUTIVE and alarm_at is None:
            alarm_at = i
        rows.append({
            "batch": i,
            "concept_changed": i >= DRIFT_START,
            "roc_auc": round(auc, 4),
            "below_threshold": below,
            "streak": streak,
            "alarm": alarm_at is not None and i >= alarm_at,
            # señales disponibles SIN etiquetas (retraso de etiquetas):
            "data_drift": detect_data_drift(reference, b).status,
            "prediction_drift_p": round(float(ks_2samp(ref_scores, proba).pvalue), 4),
            "mean_score": round(float(proba.mean()), 4),
            "observed_default_rate": round(float(b[TARGET].mean()), 4),
        })
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, baseline: Baseline, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvspan(DRIFT_START - 0.5, N_BATCHES + 0.5, color="#f4b6b6", alpha=0.25, label="concepto cambiado (simulado)")
    ax.plot(df["batch"], df["roc_auc"], marker="o", color="#1f4e79", lw=2, label="ROC-AUC por lote")
    ax.axhline(baseline.auc, color="#2e7d32", ls="--", label=f"baseline test = {baseline.auc:.3f}")
    ax.axhline(baseline.threshold, color="#c62828", lw=2,
               label=f"umbral = baseline - {K_SIGMA}σ = {baseline.threshold:.3f}")
    alarms = df[df["alarm"]]
    if not alarms.empty:
        first = alarms.iloc[0]
        ax.annotate(f"ALARMA: {CONSECUTIVE} lotes seguidos\nbajo umbral -> reentrenar",
                    xy=(first["batch"], first["roc_auc"]), xytext=(first["batch"] - 4.5, baseline.threshold - 0.07),
                    arrowprops={"arrowstyle": "->", "color": "#c62828"}, color="#c62828", fontweight="bold")
    ax.set_xlabel("Lote (tiempo)")
    ax.set_ylabel("ROC-AUC (etiquetas reales)")
    ax.set_title("Concept drift: degradación del modelo @champion en lotes sucesivos")
    ax.set_xticks(df["batch"])
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def load_champion():
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(MLFLOW_URI)
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, PROD_ALIAS)
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}"), mv


def train_local(train: pd.DataFrame):
    from train import build_pipeline
    return build_pipeline("logreg", {"C": 1.0, "max_iter": 1000}).fit(train[FEATURES], train[TARGET])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="no usar MLflow (modelo rápido local)")
    args = ap.parse_args()

    train = pd.read_csv(DATA_REF / "train.csv")
    test = pd.read_csv(DATA_REF / "test.csv")
    pool = pd.read_csv(DATA_PROD / "pool.csv")
    model, mv = (train_local(train), None) if args.local else load_champion()

    baseline = compute_baseline(model, test)
    ref_scores = model.predict_proba(test[FEATURES])[:, 1]
    df = monitor(model, make_batches(pool), baseline, train, ref_scores)

    REPORTS.mkdir(exist_ok=True)
    png, csv = REPORTS / "concept_drift.png", REPORTS / "concept_drift.csv"
    plot(df, baseline, png)
    df.to_csv(csv, index=False)

    print(f"baseline ROC-AUC={baseline.auc:.4f}  sigma(lote {BATCH_SIZE})={baseline.sigma:.4f}  "
          f"umbral={baseline.threshold:.4f}  regla: {CONSECUTIVE} lotes consecutivos bajo umbral")
    print(df.to_string(index=False))
    alarm = df["alarm"].any()
    first = int(df.loc[df["alarm"], "batch"].min()) if alarm else None
    print(("\033[91m[ROJO - REENTRENAMIENTO REQUERIDO] alarma en lote %d\033[0m" % first) if alarm
          else "\033[92m[VERDE] modelo estable\033[0m")
    print(f"gráfica: {png}")

    if mv is not None:
        import mlflow
        mlflow.set_experiment(MONITORING_EXPERIMENT)
        with mlflow.start_run(run_name=f"concept-drift-monitor-v{mv.version}"):
            mlflow.set_tags({"monitored_model": f"{MODEL_NAME}@{PROD_ALIAS}", "model_version": mv.version,
                             "model_run_id": mv.run_id, "alarm": str(alarm), "alarm_batch": str(first)})
            mlflow.log_params({"batch_size": BATCH_SIZE, "drift_start": DRIFT_START, "k_sigma": K_SIGMA,
                               "consecutive": CONSECUTIVE})
            mlflow.log_metrics({"baseline_auc": baseline.auc, "sigma": baseline.sigma, "threshold": baseline.threshold})
            for _, r in df.iterrows():
                mlflow.log_metric("batch_roc_auc", float(r["roc_auc"]), step=int(r["batch"]))
                mlflow.log_metric("batch_mean_score", float(r["mean_score"]), step=int(r["batch"]))
            mlflow.log_artifact(str(png))
            mlflow.log_artifact(str(csv))


if __name__ == "__main__":
    main()
