"""API: contrato de /health, /ready y /predict (modelo inyectado, sin MLflow)."""
import os

os.environ["SKIP_MODEL_LOAD"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from api import main  # noqa: E402
from api.schemas import CreditClient  # noqa: E402

EXAMPLE = CreditClient.model_config["json_schema_extra"]["example"]


def test_not_ready_without_model():
    main.state.model = None
    with TestClient(main.app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json={"records": [EXAMPLE]}).status_code == 503


def test_predict_returns_pod_and_version(model):
    main.state.model, main.state.features = model, list(model.feature_names_in_)
    main.state.version, main.state.run_id = "7", "abc123"
    with TestClient(main.app) as c:
        assert c.get("/ready").status_code == 200
        r = c.post("/predict", json={"records": [EXAMPLE, EXAMPLE]})
        assert r.status_code == 200
        body = r.json()
        assert len(body["predictions"]) == 2
        assert 0 <= body["predictions"][0]["default_probability"] <= 1
        assert body["pod_name"] and body["model_version"] == "7" and body["run_id"] == "abc123"


def test_invalid_input_rejected():
    with TestClient(main.app) as c:
        bad = {**EXAMPLE, "SEX": 9}
        assert c.post("/predict", json={"records": [bad]}).status_code == 422
