"""Data drift: VERDE con datos del mismo origen, ROJO con drift inyectado."""
from drift.data_drift import BATCH_SIZE, detect_data_drift, inject_drift


def test_same_source_batch_is_green(reference, pool):
    batch = pool.sample(BATCH_SIZE, random_state=1)
    res = detect_data_drift(reference, batch)
    assert res.status != "ROJO", res.report[res.report["drift"]]


def test_injected_drift_is_red(reference, pool):
    batch = inject_drift(pool.sample(BATCH_SIZE, random_state=1))
    res = detect_data_drift(reference, batch)
    assert res.status == "ROJO"
    assert {"LIMIT_BAL", "BILL_AMT1", "EDUCATION"} <= set(res.drifted)


def test_false_alarm_rate_is_low(reference, pool):
    reds = sum(detect_data_drift(reference, pool.sample(BATCH_SIZE, random_state=s)).status == "ROJO"
               for s in range(30))
    assert reds / 30 <= 0.10
