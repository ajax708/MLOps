import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common import DATA_PROD, DATA_REF, FEATURES, TARGET  # noqa: E402


@pytest.fixture(scope="session")
def reference():
    return pd.read_csv(DATA_REF / "train.csv")


@pytest.fixture(scope="session")
def test_df():
    return pd.read_csv(DATA_REF / "test.csv")


@pytest.fixture(scope="session")
def pool():
    return pd.read_csv(DATA_PROD / "pool.csv")


@pytest.fixture(scope="session")
def model(reference):
    from train import build_pipeline
    return build_pipeline("logreg", {"C": 1.0, "max_iter": 1000}).fit(reference[FEATURES], reference[TARGET])
