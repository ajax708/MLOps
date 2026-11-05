"""Precalcula lo que la web muestra: cartera, curvas ROC, matrices por umbral e importancia de variables.
Uso: python web/tools/build_data.py <carpeta con logreg/hist_gb/random_forest.joblib>"""
import json, sys, warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")
src = sys.argv[1]
train = pd.read_csv("data/reference/train.csv")
test = pd.read_csv("data/reference/test.csv")
pool = pd.read_csv("data/production/pool.csv")
allc = pd.concat([train, test, pool], ignore_index=True)
y = test["default"].to_numpy()


def rate(g):
    return {"label": str(g[0]), "n": int(len(g[1])), "rate": round(float(g[1]["default"].mean()), 4)}


def by(series, order=None):
    gr = list(allc.groupby(series, observed=True))
    return [rate(x) for x in gr]


pay = pd.cut(allc["PAY_1"], [-3, 0, 1, 2, 9], labels=["Al día", "1 mes", "2 meses", "3 o más"])
age = pd.cut(allc["AGE"], [20, 29, 39, 49, 59, 80], labels=["21-29", "30-39", "40-49", "50-59", "60+"])
lim = pd.cut(allc["LIMIT_BAL"], [0, 50000, 100000, 200000, 300000, 2e6], labels=["Hasta 50 mil", "50-100 mil", "100-200 mil", "200-300 mil", "Más de 300 mil"])
use = pd.cut((allc["BILL_AMT1"] / allc["LIMIT_BAL"]).clip(0, 1.5), [-1, .1, .3, .6, .9, 2], labels=["Menos de 10 %", "10-30 %", "30-60 %", "60-90 %", "Más de 90 %"])
late = (allc[[f"PAY_{i}" for i in range(1, 7)]] > 0).sum(axis=1)
out = {
    "portfolio": {"n": int(len(allc)), "rate": round(float(allc["default"].mean()), 4),
                  "pay": by(pay), "age": by(age), "limit": by(lim), "use": by(use), "lateMonths": by(late),
                  "medianLimit": float(allc["LIMIT_BAL"].median()), "medianAge": float(allc["AGE"].median())},
    "models": {},
}
cols = [c for c in test.columns if c != "default"]
ref = {c: (float(allc[c].median()) if c in ("LIMIT_BAL", "AGE") or "AMT" in c else int(allc[c].mode()[0])) for c in cols}
out["reference"] = ref

groups = {"Historial de pagos": [f"PAY_{i}" for i in range(1, 7)], "Cupo": ["LIMIT_BAL"], "Facturas": [f"BILL_AMT{i}" for i in range(1, 7)],
          "Pagos realizados": [f"PAY_AMT{i}" for i in range(1, 7)], "Edad": ["AGE"], "Datos personales": ["SEX", "EDUCATION", "MARRIAGE"]}
rng = np.random.default_rng(42)
for n in ["random_forest", "hist_gb", "logreg"]:
    m = joblib.load(f"{src}/{n}.joblib")
    X = test[list(m.feature_names_in_)]
    p = m.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y, p))
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.unique(np.linspace(0, len(fpr) - 1, 90).astype(int))
    th = []
    for t in np.arange(.05, .96, .01):
        pred = p >= t
        th.append([round(float(t), 2), int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())])
    imp = {}
    for g, cs in groups.items():
        drops = []
        for _ in range(4):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            Xp[cs] = Xp[cs].to_numpy()[perm]
            drops.append(auc - roc_auc_score(y, m.predict_proba(Xp)[:, 1]))
        imp[g] = round(float(np.mean(drops)), 4)
    out["models"][n] = {"auc": round(auc, 4), "roc": [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in idx], "thresholds": th,
                        "importance": imp, "meanScore": round(float(p.mean()), 3)}
    print(n, round(auc, 4), imp, "score medio", round(float(p.mean()), 3))
out["testPositives"] = int(y.sum()); out["testN"] = int(len(y))
json.dump(out, open("web/data/insights.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
