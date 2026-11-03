# Documento de arquitectura — Credit Default MLOps

## 1. Problema

Clasificación binaria: predecir si un cliente de tarjeta de crédito caerá en **default** el mes siguiente.
Dataset público **UCI "Default of Credit Card Clients"** (Taiwán, 2005): 30.000 registros, 23 variables, 22,1 % positivos.

| Grupo | Variables | Tratamiento en drift |
|---|---|---|
| Numéricas continuas (14) | `LIMIT_BAL`, `AGE`, `BILL_AMT1-6`, `PAY_AMT1-6` | Kolmogorov-Smirnov + PSI |
| Categóricas / ordinales (9) | `SEX`, `EDUCATION`, `MARRIAGE`, `PAY_1-6` | Chi-cuadrado + PSI |

**Métrica principal:** ROC-AUC. Es independiente del umbral de decisión y robusta al desbalance de clases. También se registran F1, precisión, recall y accuracy.

## 2. Vista general

```
                 ┌──────────────── Docker Desktop Kubernetes · namespace "mlops" ────────────────┐
  train.py ────► │  mlflow (1 pod) ── PVC: mlflow.db + artifacts     Service NodePort :30500 (UI) │
  register.py    │        ▲  models:/credit-default-classifier@champion                           │
  concept_drift  │        │  (al arrancar cada pod)                                               │
                 │  credit-api  ×3 réplicas  (FastAPI)  ◄── Service credit-api NodePort :30080 ◄──┼── curl / scripts
                 │        ▲                                                                       │
                 │  credit-ui (Streamlit) ── http://credit-api ──┘   Service NodePort :30501 ◄────┼── navegador
                 └────────────────────────────────────────────────────────────────────────────────┘
```

Organizada en los tres pilares de MLOps vistos en clase:

| Pilar | Cómo se cumple |
|---|---|
| **Escalable** | Deployment de 3 réplicas detrás de un Service, `kubectl scale`, RollingUpdate con `maxUnavailable: 0` |
| **Auditable** | MLflow: cada predicción devuelve `model_version` + `run_id`; cada run guarda commit de git, hashes SHA-256 de los datos, parámetros, métricas, dataset y modelo con signature |
| **Fiable** | Probes de readiness y liveness, `requests = limits` de memoria, pruebas automáticas de data drift y concept drift con umbrales justificados, CI en GitHub Actions |

## 3. Fase 1 — MLflow

- **Reproducibilidad:** semilla fija `42`. Partición estratificada **60 / 20 / 20** (train / test / pool de producción) en [`src/prepare_data.py`](../src/prepare_data.py). Los hashes de cada archivo quedan en `data/reference/manifest.json`, y el CI vuelve a generar la partición y comprueba que los hashes coinciden.
- **Experimento `credit-default-training`:** 8 runs de 3 familias de modelos (Regresión Logística, Random Forest y HistGradientBoosting), con el mismo split y las mismas métricas. Cada run es un `Pipeline` (StandardScaler + OneHotEncoder → clasificador), de modo que el preprocesamiento viaja junto con el modelo.
- **Registro ([`src/register.py`](../src/register.py)):** se registra el mejor run de cada familia bajo el nombre estable **`credit-default-classifier`**, lo que da 3 versiones.
  - `@champion` = mayor ROC-AUC en test. **Es la versión que sirve el clúster.**
  - `@challenger` = segunda mejor, candidata a reemplazo cuando se reentrena.
- **Consumo por alias:** la API resuelve `models:/credit-default-classifier@champion` → número de versión → `run_id` mediante `MlflowClient.get_model_version_by_alias`. Nunca usa una ruta de archivo.

### Modelo desplegado

| Campo | Valor |
|---|---|
| Registered model | `credit-default-classifier` |
| Alias | `@champion` |
| Versión | **{{VERSION}}** |
| Run | **{{RUN_NAME}}** (`{{RUN_ID}}`) |
| ROC-AUC test | **{{AUC}}** |

Se puede verificar en vivo con `GET http://localhost:30080/info`: el `run_id` que devuelve el pod es el mismo que aparece en la UI de MLflow.

## 4. Fase 2 — Docker

- [`Dockerfile`](../Dockerfile) propio, con base `python:3.11-slim` (la misma versión de Python con la que se entrenó). `PYTHONUNBUFFERED=1` permite ver los logs con `kubectl logs`, y el contenedor corre con un usuario no root.
- Las dependencias van **fijadas** en [`requirements/api.txt`](../requirements/api.txt), incluido `mlflow==3.17.2`, con las mismas versiones de mlflow, scikit-learn, pandas y numpy que el entorno de entrenamiento.
- Se instala `requirements` **antes** de copiar el código para aprovechar la caché de capas. `.dockerignore` excluye `.venv`, `.git`, datos y tests.
- **Decisión: el modelo NO se copia dentro de la imagen.** En clase se hizo `COPY modelo.joblib`, pero el enunciado exige consumir el modelo por su referencia en el registry. La imagen contiene solo código, así que la misma imagen sirve cualquier versión: promover un modelo consiste en mover el alias y hacer `kubectl rollout restart`, sin reconstruir la imagen.
- MLflow también corre en un contenedor propio ([`docker/mlflow/Dockerfile`](../docker/mlflow/Dockerfile)): backend SQLite y `--artifacts-destination` (proxy de artefactos). Los pods descargan el modelo por HTTP (`mlflow-artifacts:/`) y no dependen de rutas locales de la máquina anfitriona.

## 5. Fase 3 — Kubernetes

| Recurso | Detalle |
|---|---|
| `credit-api` Deployment | 3 réplicas, RollingUpdate (`maxSurge 1`, `maxUnavailable 0`), `minReadySeconds 5` |
| readinessProbe `/ready` | responde 503 hasta que el modelo está en memoria, así que un pod nuevo **no recibe tráfico** mientras carga. Es la clave de que el servicio no deje de responder cuando se elimina un pod |
| livenessProbe `/health` | reinicia el contenedor si el proceso se cuelga |
| resources | memoria `requests = limits = 512Mi` ("la RAM no perdona": se evita el OOMKill y el desalojo); CPU 250m → 500m |
| `credit-api` Service | NodePort 30080; kube-proxy reparte las conexiones entre los pods listos |
| `mlflow` | 1 réplica (SQLite + PVC ReadWriteOnce no admiten varias escrituras concurrentes) |

**Sobre el balanceo:** kube-proxy balancea **por conexión TCP**, no por petición HTTP. Los scripts usan `-DisableKeepAlive` y la UI usa `Connection: close` para abrir una conexión nueva en cada petición. Con keep-alive, un mismo cliente se quedaría "pegado" a un pod.

**Punto único de fallo conocido:** si el pod de MLflow cae, las réplicas que ya están corriendo **siguen respondiendo**, porque el modelo está en memoria. Lo que no podría hacer es arrancar un pod nuevo: se quedaría *NotReady* y reintentando con backoff hasta que MLflow vuelva. En producción, la solución sería PostgreSQL + almacenamiento de objetos (S3/MinIO) y varias réplicas de MLflow.

## 6. Monitoreo de drift

### 6.1 Data drift — [`src/drift/data_drift.py`](../src/drift/data_drift.py)

| Tipo | Prueba | Por qué |
|---|---|---|
| Numérica | **Kolmogorov-Smirnov** (2 muestras) | No paramétrica: no asume normalidad, y los montos son muy asimétricos. Compara las distribuciones acumuladas completas |
| Categórica | **Chi-cuadrado** de homogeneidad | KS no tiene sentido sin un orden continuo. Chi² compara las frecuencias de cada categoría. Las categorías raras (< 0,5 %) se agrupan para cumplir esperados ≥ 5 |
| Ambas | **PSI** (magnitud) | El p-value dice si la diferencia es *real*, no si es *importante*. Con miles de filas, diferencias minúsculas salen significativas |

**Umbral explícito:** una variable tiene drift ⇔ `p < 0,05 / 23 ≈ 0,0022` **y** `PSI ≥ 0,10`.
- `0,05` es el nivel de significancia visto en clase. Se divide entre 23 (**Bonferroni**) porque se hacen 23 pruebas por lote: sin corrección, un lote sano tendría un 1 − 0,95²³ ≈ **69 %** de probabilidad de alguna falsa alarma.
- `PSI 0,10 / 0,25` es la convención de la industria de riesgo crediticio (estable / moderado / crítico).
- **Validación empírica:** con 200 lotes sanos de 2.000 filas del pool hubo **0 / 200 falsas alarmas** (`python src/drift/data_drift.py --calibrate`).

**Escenarios:**
- Lote del pool de producción → **VERDE** (las 23 variables con p > 0,17 y PSI < 0,01).
- Drift inyectado (inflación: montos ×1,8 y límite ×1,5; cartera 8 años más joven; 40 % pasa a EDUCATION=3) → **ROJO**, con 15 variables con drift.

### 6.2 Concept drift — [`src/drift/concept_drift.py`](../src/drift/concept_drift.py)

- **Simulación ("crisis económica"):** 12 lotes secuenciales de 500 clientes no vistos. A partir del lote 6, la crisis golpea a los clientes que **venían pagando al día pero con alta utilización del crédito** (`PAY_1 ≤ 0` y `BILL_AMT1/LIMIT_BAL > 0,5`, ≈ 30 % de la cartera): una fracción creciente (30 → 80 %) de sus etiquetas pasa de 0 a 1. Antes ese perfil era de riesgo bajo; ahora es de riesgo alto. Las entradas **no cambian**. Se eligió un segmento identificable, y no un volteo de etiquetas al azar, porque el ruido aleatorio no se puede aprender: reentrenar no ayudaría y el ejercicio de CT perdería sentido.
- **Criterio de reentrenamiento:**
  - `umbral = AUC_baseline − 3σ`, donde σ es la desviación estándar del AUC en 300 lotes bootstrap de 500 filas del test, es decir, la variación esperable solo por tamaño de lote. Un lote sano queda por debajo con una probabilidad ≈ 0,1 %.
  - **ALARMA = 3 lotes consecutivos bajo el umbral → reentrenar.** La persistencia filtra caídas aisladas por ruido.
- **Resultado:** gráfica `reports/concept_drift.png`, registrada también en el experimento `credit-default-monitoring` de MLflow (métrica `batch_roc_auc` por step). El data drift se mantiene en **VERDE** en todos los lotes. Eso es justamente lo que demuestra que se trata de concept drift.
- **Respuesta (matriz de clase):** data drift sin degradación → *vigilar*. Degradación con o sin data drift → **reentrenar (CT)** con [`src/retrain.py`](../src/retrain.py):
  - el challenger usa la misma configuración que el champion (`sklearn.clone`) y se entrena con la **ventana reciente** etiquetada (lotes 6–10);
  - se evalúa contra el champion en los lotes 11–12, que no se usaron para entrenar;
  - se registra como nueva versión con `@challenger`; con `--promote`, si gana, se mueve `@champion` y se ejecuta `kubectl rollout restart` (sin reconstruir la imagen).

| Modelo | ROC-AUC lotes 11–12 |
|---|---|
| champion v3 (entrenado antes de la crisis) | 0,611 |
| challenger: histórico + ventana reciente | 0,684 |
| **challenger: solo ventana reciente** | **0,808** |

Mezclar el histórico diluye la relación nueva: después de un cambio de concepto, los datos viejos describen un mundo que ya no existe.

### 6.3 Retraso de etiquetas

En crédito, la etiqueta "default" llega como mínimo un mes después, y en definiciones de mora de 90 días, meses después. Mientras tanto:
1. **Data drift sobre las entradas** (lo anterior): es una alerta temprana, aunque no confirma degradación.
2. **Prediction drift:** KS entre la distribución de scores de producción y la del test, más la tasa de default predicha. Se calcula en cada lote (`prediction_drift_p`, `mean_score`).
3. **Etiquetas proxy tempranas:** p. ej., el primer pago fallido a 30 días (`PAY_1 ≥ 1` del mes siguiente) llega mucho antes que el default formal.
4. **Muestras etiquetadas manualmente** (revisión de analistas) sobre un subconjunto pequeño.

**Limitación honesta:** en la simulación, las señales 1 y 2 se mantienen en verde, porque las entradas no cambian y el modelo predice igual. Un concept drift "puro" **solo** se detecta con etiquetas (reales o proxy). Por eso son importantes las proxies y por eso el monitoreo con etiquetas se ejecuta en cuanto llegan.
