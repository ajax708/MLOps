# Credit Default — Proyecto final MLOps

Modelo de riesgo de default de tarjeta de crédito (UCI Taiwán, 30.000 clientes). Se lleva desde el experimento en **MLflow** hasta un servicio **FastAPI** en **Docker**, desplegado en **Kubernetes** con 3 réplicas, con pruebas automatizadas de **data drift** y **concept drift** y una UI en Streamlit que consume el servicio del clúster.

Decisiones y justificaciones: [docs/architecture.md](docs/architecture.md) · Evidencias: [docs/evidence/](docs/evidence/)

## Reparto de trabajo

| Componente | Responsable |
|---|---|
| Datos, entrenamiento y MLflow (tracking + registry) | Luis Bryan (trabajo individual) |
| API FastAPI y Docker | Luis Bryan |
| Kubernetes (manifiestos, demos) | Luis Bryan |
| Data drift y concept drift | Luis Bryan |
| UI Streamlit, CI y documentación | Luis Bryan |

## Estructura

```
data/raw/                    dataset original (CSV)
data/reference/              train.csv (baseline), test.csv, manifest.json (hashes)
data/production/pool.csv     datos no vistos para simular lotes de producción
src/prepare_data.py          limpieza + partición 60/20/20 con semilla 42
src/train.py                 8 runs en MLflow (3 familias de modelos)
src/register.py              Model Registry: versiones + alias @champion/@challenger
src/api/                     FastAPI: /predict /info /ready /health
src/drift/                   data_drift.py, concept_drift.py, stats.py
tests/                       pytest: API, data drift verde/rojo, concept drift
ui/                          Streamlit + Dockerfile
Dockerfile                   imagen del servicio de inferencia
docker/mlflow/Dockerfile     imagen del servidor MLflow
k8s/                         namespace, mlflow, api (3 réplicas), ui
scripts/                     deploy.ps1 y demo_1..5 (generan docs/evidence/)
.github/workflows/ci.yml     tests + build de imágenes
```

## Cómo ejecutarlo (Windows + Docker Desktop con Kubernetes activado)

```powershell
python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt
.\scripts\deploy.ps1          # imágenes -> MLflow en K8s -> train/register -> API x3 + UI
```

| Servicio | URL |
|---|---|
| MLflow UI | http://localhost:30500 |
| API (Swagger) | http://localhost:30080/docs |
| UI | http://localhost:30501 |

## Demostraciones

```powershell
.\scripts\demo_1_replicas.ps1         # 3 pods Running + modelo/versión servida
.\scripts\demo_2_load_balancing.ps1   # 30 peticiones -> pod_name rota entre réplicas
.\scripts\demo_3_self_healing.ps1     # delete pod con tráfico continuo -> 0 errores, pod nuevo
.\scripts\demo_4_scaling.ps1          # 3 -> 5 -> 3 réplicas
.\scripts\demo_5_drift.ps1            # data drift (verde/rojo), concept drift, pytest
```

### Evidencias obtenidas (docs/evidence/)

| Prueba | Resultado |
|---|---|
| [01_replicas.txt](docs/evidence/01_replicas.txt) | 3/3 pods Running; `/info` → `credit-default-classifier@champion` v3, run `54686821…` |
| [02_load_balancing.txt](docs/evidence/02_load_balancing.txt) | 30 peticiones repartidas entre los 3 pods |
| [03_self_healing.txt](docs/evidence/03_self_healing.txt) | pod eliminado → reemplazo automático; **150 peticiones OK, 0 fallidas** |
| [04_scaling.txt](docs/evidence/04_scaling.txt) | 3 → 5 réplicas (tráfico en los 5 pods) → 3 |
| [05_drift.txt](docs/evidence/05_drift.txt) / [05_concept_drift.png](docs/evidence/05_concept_drift.png) | data drift VERDE/ROJO, 0/200 falsas alarmas, alarma de concept drift en el lote 9, 8 tests OK |
| [06_retrain_challenger.txt](docs/evidence/06_retrain_challenger.txt) | challenger v4: AUC 0,808 frente a 0,611 del champion en los lotes recientes |

Petición manual:
```powershell
curl.exe -s -X POST http://localhost:30080/predict -H "Content-Type: application/json" -d "@scripts/sample_request.json"
```

## Promover un nuevo modelo (sin reconstruir la imagen)

```powershell
# mover el alias en el registry (UI o API) y reiniciar los pods de forma gradual:
kubectl rollout restart deployment/credit-api -n mlops
```
