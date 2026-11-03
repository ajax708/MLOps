"""Pruebas estadísticas de drift.

- Numéricas  -> Kolmogorov-Smirnov de dos muestras (scipy.stats.ks_2samp).
  No paramétrica: no asume normalidad (los montos de factura/pago son muy asimétricos) y
  compara la distancia máxima entre las funciones de distribución acumuladas.
- Categóricas -> Chi-cuadrado de homogeneidad (scipy.stats.chi2_contingency).
  KS no aplica a categorías sin orden/con pocos niveles (muchos empates); Chi² compara
  directamente las frecuencias observadas de cada categoría contra las esperadas.
- PSI (Population Stability Index) para ambas: no es una prueba de hipótesis sino una medida
  de MAGNITUD del cambio. Con muestras grandes un p-value puede ser "significativo" para
  diferencias irrelevantes; el PSI evita alertar por eso.
    PSI < 0.10 estable | 0.10-0.25 cambio moderado | >= 0.25 cambio crítico (convención de riesgo crediticio)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

EPS = 1e-4


def psi_from_props(ref: np.ndarray, cur: np.ndarray) -> float:
    ref = np.clip(ref, EPS, None)
    cur = np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """PSI con bins definidos por los cuantiles del baseline."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    return psi_from_props(r, c)


def _category_counts(ref: pd.Series, cur: pd.Series, min_share: float = 0.005):
    """Agrupa categorías raras (<0.5 % en el baseline) para que Chi² sea válido (esperados >= 5)."""
    ref_share = ref.value_counts(normalize=True)
    keep = set(ref_share[ref_share >= min_share].index)
    r = ref.where(ref.isin(keep), "rare").astype(str)
    c = cur.where(cur.isin(keep), "rare").astype(str)
    cats = sorted(set(r) | set(c))
    rc = r.value_counts().reindex(cats, fill_value=0).to_numpy()
    cc = c.value_counts().reindex(cats, fill_value=0).to_numpy()
    return rc, cc


def ks_test(ref: pd.Series, cur: pd.Series) -> tuple[float, float]:
    res = ks_2samp(ref, cur)
    return float(res.statistic), float(res.pvalue)


def chi2_test(ref: pd.Series, cur: pd.Series) -> tuple[float, float, float]:
    rc, cc = _category_counts(ref, cur)
    table = np.vstack([rc, cc])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 0.0, 1.0, 0.0
    stat, p, _, _ = chi2_contingency(table)
    psi = psi_from_props(rc / rc.sum(), cc / cc.sum())
    return float(stat), float(p), psi
