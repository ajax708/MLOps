"""Detección de DATA DRIFT: cada variable del lote nuevo vs. el baseline de entrenamiento.

Regla de alerta (por variable):
    DRIFT  <=>  p_value < ALPHA_BONF   Y   PSI >= PSI_WARN
  - ALPHA = 0.05 (nivel de significancia visto en clase) corregido por Bonferroni:
    se hacen 23 pruebas por lote; sin corrección, con datos SANOS la probabilidad de al menos
    una falsa alarma sería 1-(0.95^23) ≈ 69 %. Con 0.05/23 ≈ 0.0022 por prueba, la tasa de
    falsa alarma por lote queda <= 5 %. (Se verifica empíricamente con `--calibrate`.)
  - PSI >= 0.10 exige además que el cambio tenga magnitud práctica.
Estado del lote:
    ROJO     si alguna variable tiene DRIFT
    VERDE    en otro caso
    (AMARILLO informativo: alguna variable con p < ALPHA_BONF pero PSI < 0.10)

Uso:
    python src/drift/data_drift.py                 # escenarios verde y rojo
    python src/drift/data_drift.py --calibrate     # tasa de falsa alarma con 200 lotes sanos
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CATEGORICAL, DATA_PROD, DATA_REF, FEATURES, NUMERIC, REPORTS, SEED  # noqa: E402
from drift.stats import chi2_test, ks_test, psi_numeric  # noqa: E402

ALPHA = 0.05
ALPHA_BONF = ALPHA / len(FEATURES)
PSI_WARN = 0.10
PSI_CRIT = 0.25
BATCH_SIZE = 2000


@dataclass
class DriftResult:
    status: str               # VERDE | AMARILLO | ROJO
    report: pd.DataFrame
    drifted: list[str]

    @property
    def passed(self) -> bool:
        return self.status != "ROJO"


def detect_data_drift(reference: pd.DataFrame, batch: pd.DataFrame) -> DriftResult:
    rows = []
    for col in FEATURES:
        if col in NUMERIC:
            stat, p = ks_test(reference[col], batch[col])
            psi, test = psi_numeric(reference[col], batch[col]), "KS"
        else:
            stat, p, psi = chi2_test(reference[col], batch[col])
            test = "Chi2"
        significant = p < ALPHA_BONF
        drift = significant and psi >= PSI_WARN
        level = "CRITICO" if drift and psi >= PSI_CRIT else ("DRIFT" if drift else ("signif." if significant else "ok"))
        rows.append({"feature": col, "type": "numérica" if col in NUMERIC else "categórica", "test": test,
                     "statistic": round(stat, 4), "p_value": p, "psi": round(psi, 4), "drift": drift, "level": level})
    rep = pd.DataFrame(rows)
    drifted = rep.loc[rep["drift"], "feature"].tolist()
    status = "ROJO" if drifted else ("AMARILLO" if (rep["level"] == "signif.").any() else "VERDE")
    return DriftResult(status, rep, drifted)


def inject_drift(batch: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Escenario 'inflación + cambio de cartera':
    - montos de facturación y pagos x1.8 (inflación), límite de crédito x1.5
    - cartera más joven (AGE - 8 años, mínimo 21)
    - mezcla educativa distinta: 40 % de los clientes pasa a EDUCATION=3 (secundaria)
    """
    rng = np.random.default_rng(seed)
    d = batch.copy()
    for c in [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]:
        d[c] = (d[c] * 1.8).round()
    d["LIMIT_BAL"] = (d["LIMIT_BAL"] * 1.5).round()
    d["AGE"] = (d["AGE"] - 8).clip(lower=21)
    idx = rng.random(len(d)) < 0.40
    d.loc[idx, "EDUCATION"] = 3
    return d


def print_result(title: str, res: DriftResult) -> None:
    color = {"VERDE": "\033[92m", "AMARILLO": "\033[93m", "ROJO": "\033[91m"}[res.status]
    tag = "PASS" if res.passed else "FAIL"
    print(f"\n=== {title} ===")
    with pd.option_context("display.width", 140, "display.max_rows", 50):
        print(res.report.assign(p_value=res.report["p_value"].map(lambda v: f"{v:.2e}")).to_string(index=False))
    print(f"{color}[{res.status} - {tag}]\033[0m  variables con drift: {res.drifted or 'ninguna'}"
          f"  (alpha Bonferroni = {ALPHA_BONF:.4f}, PSI >= {PSI_WARN})")


def calibrate(reference: pd.DataFrame, pool: pd.DataFrame, n: int = 200) -> float:
    """Tasa de falsa alarma: lotes del MISMO origen que el entrenamiento que salen en ROJO."""
    rng = np.random.default_rng(SEED)
    reds = sum(detect_data_drift(reference, pool.sample(BATCH_SIZE, random_state=int(rng.integers(1e9)))).status == "ROJO"
               for _ in range(n))
    rate = reds / n
    print(f"Falsa alarma con {n} lotes sanos de {BATCH_SIZE} filas: {reds}/{n} = {rate:.1%}")
    return rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--batch", type=Path, help="CSV externo a evaluar contra el baseline")
    args = ap.parse_args()

    reference = pd.read_csv(DATA_REF / "train.csv")
    pool = pd.read_csv(DATA_PROD / "pool.csv")
    REPORTS.mkdir(exist_ok=True)

    if args.calibrate:
        calibrate(reference, pool)
        return
    if args.batch:
        res = detect_data_drift(reference, pd.read_csv(args.batch))
        print_result(f"Lote {args.batch.name}", res)
        sys.exit(0 if res.passed else 1)

    clean = pool.sample(BATCH_SIZE, random_state=SEED)
    drifted = inject_drift(clean)
    r1, r2 = detect_data_drift(reference, clean), detect_data_drift(reference, drifted)
    print_result("Lote del mismo origen (esperado VERDE)", r1)
    print_result("Lote con drift inyectado (esperado ROJO)", r2)
    r1.report.to_csv(REPORTS / "data_drift_clean.csv", index=False)
    r2.report.to_csv(REPORTS / "data_drift_injected.csv", index=False)


if __name__ == "__main__":
    main()
