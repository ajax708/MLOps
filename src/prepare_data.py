"""Limpieza y partición reproducible del dataset.

Partición estratificada por la variable objetivo con semilla fija (42):
  - 60 % train      -> data/reference/train.csv  (también es el BASELINE de drift)
  - 20 % test       -> data/reference/test.csv   (evaluación y selección de modelo)
  - 20 % producción -> data/production/pool.csv  (datos "no vistos" para simular lotes)
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from common import CATEGORICAL, DATA_PROD, DATA_RAW, DATA_REF, SEED, TARGET, file_sha256


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # EDUCATION: 0, 5 y 6 no están documentados en UCI -> se agrupan en 4 ("otros")
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    # MARRIAGE: 0 no documentado -> 3 ("otros")
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    for c in CATEGORICAL:
        df[c] = df[c].astype("int64")
    return df


def main() -> None:
    df = clean(pd.read_csv(DATA_RAW))
    train, rest = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=SEED)
    test, prod = train_test_split(rest, test_size=0.5, stratify=rest[TARGET], random_state=SEED)

    DATA_REF.mkdir(parents=True, exist_ok=True)
    DATA_PROD.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": DATA_REF / "train.csv",
        "test": DATA_REF / "test.csv",
        "production_pool": DATA_PROD / "pool.csv",
    }
    train.to_csv(paths["train"], index=False, lineterminator="\n")
    test.to_csv(paths["test"], index=False, lineterminator="\n")
    prod.to_csv(paths["production_pool"], index=False, lineterminator="\n")

    manifest = {
        "seed": SEED,
        "raw_sha256": file_sha256(DATA_RAW),
        **{f"{k}_rows": len(v) for k, v in zip(paths, (train, test, prod))},
        **{f"{k}_default_rate": round(float(v[TARGET].mean()), 4) for k, v in zip(paths, (train, test, prod))},
        **{f"{k}_sha256": file_sha256(p) for k, p in paths.items()},
    }
    (DATA_REF / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
