# Demo 5: pruebas de data drift y concept drift
. "$PSScriptRoot\common.ps1"
Set-Location $Root
$env:MLFLOW_TRACKING_URI = "http://localhost:30500"
$log = Join-Path $Evidence "05_drift.txt"
& {
    Section "Data drift: escenario verde y rojo"
    & $Py src/drift/data_drift.py
    Section "Data drift: calibración del umbral (falsas alarmas)"
    & $Py src/drift/data_drift.py --calibrate
    Section "Concept drift sobre el modelo @champion (se registra en MLflow)"
    & $Py src/drift/concept_drift.py
    Section "Tests automatizados"
    & $Py -m pytest -p no:warnings
} *>&1 | Tee-Object -FilePath $log
Copy-Item reports\concept_drift.png (Join-Path $Evidence "05_concept_drift.png") -Force
Write-Host "Evidencia: $log" -ForegroundColor Green
