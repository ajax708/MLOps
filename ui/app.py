"""UI Streamlit que consume la API desplegada en Kubernetes (Service credit-api).

Dentro del clúster API_URL = http://credit-api  (DNS del Service -> balanceo entre réplicas).
"""
from __future__ import annotations

import os
from collections import Counter

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:30080")

st.set_page_config(page_title="Riesgo de default", page_icon="💳", layout="wide")
st.title("💳 Riesgo de default de tarjeta de crédito")
st.caption(f"Consumiendo la API en Kubernetes: `{API_URL}`")


def call(method: str, path: str, **kw):
    # requests.post/get sin Session => conexión nueva por petición => kube-proxy puede elegir otro pod
    return requests.request(method, f"{API_URL}{path}", timeout=10, headers={"Connection": "close"}, **kw)


with st.sidebar:
    st.subheader("Modelo en producción")
    try:
        info = call("GET", "/info").json()
        st.metric("Versión del registry", f"v{info['model_version']}")
        st.write(f"**URI:** `{info['model_uri']}`")
        st.write(f"**Run:** `{info['run_id']}`")
        st.write(f"**Pod:** `{info['pod_name']}`")
    except Exception as exc:
        st.error(f"API no disponible: {exc}")

PAY_HELP = "-2/-1/0 = al día · 1..8 = meses de retraso"
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Perfil")
    limit_bal = st.number_input("Límite de crédito (NT$)", 10000, 1000000, 120000, step=10000)
    age = st.slider("Edad", 21, 79, 35)
    sex = st.selectbox("Sexo", [1, 2], format_func=lambda v: {1: "Hombre", 2: "Mujer"}[v], index=1)
    education = st.selectbox("Educación", [1, 2, 3, 4], index=1,
                             format_func=lambda v: {1: "Posgrado", 2: "Universidad", 3: "Secundaria", 4: "Otros"}[v])
    marriage = st.selectbox("Estado civil", [1, 2, 3],
                            format_func=lambda v: {1: "Casado/a", 2: "Soltero/a", 3: "Otros"}[v])
with c2:
    st.subheader("Historial de pagos")
    pays = [st.slider(f"PAY_{i} (mes -{i})", -2, 8, 0, help=PAY_HELP) for i in range(1, 7)]
with c3:
    st.subheader("Montos (últimos 6 meses)")
    bill = st.number_input("Factura mensual promedio", 0, 1000000, 40000, step=1000)
    payment = st.number_input("Pago mensual promedio", 0, 1000000, 2000, step=500)

record = {"LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education, "MARRIAGE": marriage, "AGE": age,
          **{f"PAY_{i}": p for i, p in enumerate(pays, 1)},
          **{f"BILL_AMT{i}": bill for i in range(1, 7)}, **{f"PAY_AMT{i}": payment for i in range(1, 7)}}

b1, b2 = st.columns(2)
if b1.button("🔮 Predecir", type="primary", use_container_width=True):
    r = call("POST", "/predict", json={"records": [record]})
    if r.ok:
        body = r.json()
        p = body["predictions"][0]
        k1, k2, k3 = st.columns(3)
        k1.metric("Probabilidad de default", f"{p['default_probability']:.1%}")
        k2.metric("Decisión", p["label"])
        k3.metric("Atendido por el pod", body["pod_name"])
        st.caption(f"Modelo {body['model_name']}@{body['model_alias']} · v{body['model_version']} · run {body['run_id']}")
    else:
        st.error(f"{r.status_code}: {r.text}")

if b2.button("⚖️ Enviar 30 peticiones (balanceo de carga)", use_container_width=True):
    pods = []
    for _ in range(30):
        try:
            pods.append(call("POST", "/predict", json={"records": [record]}).json()["pod_name"])
        except Exception:
            pods.append("ERROR")
    counts = pd.DataFrame(Counter(pods).items(), columns=["pod", "peticiones"]).set_index("pod")
    st.subheader(f"{len(counts)} pods distintos atendieron las 30 peticiones")
    st.bar_chart(counts)
    st.dataframe(counts)
