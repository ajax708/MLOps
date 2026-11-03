# Despliegue completo de cero: imágenes -> MLflow en K8s -> entrenamiento/registro -> API + UI
# Uso: .\scripts\deploy.ps1            (-SkipTraining si el registry ya tiene @champion)
param([switch]$SkipTraining)
. "$PSScriptRoot\common.ps1"
Set-Location $Root

Section "1. Construcción de imágenes"
docker build -t mlflow-server:3.17.2 -f docker/mlflow/Dockerfile .
docker build -t credit-api:1.0 .
docker build -t credit-ui:1.0 -f ui/Dockerfile .

Section "2. MLflow en el clúster"
kubectl apply -f k8s/00-namespace.yaml -f k8s/10-mlflow.yaml
kubectl rollout status deployment/mlflow -n $Ns --timeout=180s

if (-not $SkipTraining) {
    Section "3. Datos, entrenamiento (>=5 runs) y registro en el Model Registry"
    & $Py src/prepare_data.py
    & $Py src/train.py
    & $Py src/register.py
}

Section "4. API (3 réplicas) + UI"
kubectl apply -f k8s/20-api.yaml -f k8s/30-ui.yaml
kubectl rollout restart deployment/credit-api -n $Ns   # vuelve a resolver @champion
kubectl rollout status deployment/credit-api -n $Ns --timeout=300s
kubectl rollout status deployment/credit-ui -n $Ns --timeout=180s
kubectl get all -n $Ns
Write-Host "`nMLflow UI : http://localhost:30500`nAPI docs  : $Api/docs`nUI        : http://localhost:30501" -ForegroundColor Green
