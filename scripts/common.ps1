$ErrorActionPreference = "Continue"   # kubectl escribe avisos en stderr; no abortar por eso
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$Root = Split-Path -Parent $PSScriptRoot
$Ns = "mlops"
$Api = "http://localhost:30080"
$Evidence = Join-Path $Root "docs\evidence"
New-Item -ItemType Directory -Force $Evidence | Out-Null
$Body = Get-Content (Join-Path $PSScriptRoot "sample_request.json") -Raw
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
$Py = Join-Path $Root ".venv\Scripts\python.exe"

function Invoke-Predict {
    # -DisableKeepAlive: conexión TCP nueva en cada petición -> kube-proxy reparte entre pods
    Invoke-RestMethod -Uri "$Api/predict" -Method Post -Body $Body -ContentType "application/json" `
        -DisableKeepAlive -TimeoutSec 5 -ErrorAction Stop
}

function Section($title) {
    Write-Host ""
    Write-Host "==== $title  ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) ====" -ForegroundColor Cyan
}
