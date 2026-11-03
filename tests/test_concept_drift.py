"""Concept drift: sin cambio de concepto no hay alarma; con cambio, la alarma se dispara."""
from common import FEATURES
from drift import concept_drift as cd


def test_no_alarm_without_concept_change(model, test_df, reference, pool, monkeypatch):
    monkeypatch.setattr(cd, "DRIFT_START", 99)  # nunca cambia el concepto
    baseline = cd.compute_baseline(model, test_df)
    df = cd.monitor(model, cd.make_batches(pool), baseline, reference, model.predict_proba(test_df[FEATURES])[:, 1])
    assert not df["alarm"].any()


def test_alarm_with_concept_change(model, test_df, reference, pool):
    baseline = cd.compute_baseline(model, test_df)
    df = cd.monitor(model, cd.make_batches(pool), baseline, reference, model.predict_proba(test_df[FEATURES])[:, 1])
    assert df["alarm"].any()
    assert df.loc[df["alarm"], "batch"].min() >= cd.DRIFT_START
    # las entradas no cambian: el data drift NO lo detecta (por eso hace falta la etiqueta real)
    assert (df["data_drift"] != "ROJO").all()
