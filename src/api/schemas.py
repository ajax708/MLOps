from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PayStatus = Field(ge=-2, le=8, description="-2/-1/0 = pago al día, 1..8 = meses de retraso")


class CreditClient(BaseModel):
    """Una fila de entrada (23 variables del dataset UCI Default of Credit Card Clients)."""
    model_config = ConfigDict(json_schema_extra={"example": {
        "LIMIT_BAL": 120000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
        "PAY_1": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 45000, "BILL_AMT2": 43000, "BILL_AMT3": 41000,
        "BILL_AMT4": 39000, "BILL_AMT5": 37000, "BILL_AMT6": 35000,
        "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 1800,
        "PAY_AMT4": 1500, "PAY_AMT5": 1500, "PAY_AMT6": 1500,
    }})

    LIMIT_BAL: float = Field(gt=0, description="Límite de crédito (NT$)")
    SEX: Literal[1, 2]
    EDUCATION: Literal[1, 2, 3, 4]
    MARRIAGE: Literal[1, 2, 3]
    AGE: int = Field(ge=18, le=100)
    PAY_1: int = PayStatus
    PAY_2: int = PayStatus
    PAY_3: int = PayStatus
    PAY_4: int = PayStatus
    PAY_5: int = PayStatus
    PAY_6: int = PayStatus
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


class PredictRequest(BaseModel):
    records: list[CreditClient] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    default_probability: float
    prediction: int
    label: str


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    pod_name: str
    model_name: str
    model_alias: str
    model_version: str
    run_id: str


class ModelInfo(BaseModel):
    pod_name: str
    ready: bool
    model_name: str
    model_alias: str
    model_version: str | None = None
    run_id: str | None = None
    model_uri: str
    tracking_uri: str
    error: str | None = None
