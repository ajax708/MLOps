# Demo 1: 3 o más réplicas en estado Running simultáneamente
. "$PSScriptRoot\common.ps1"
$log = Join-Path $Evidence "01_replicas.txt"
& {
    Section "Deployment y pods de la API"
    kubectl get deployment credit-api -n $Ns
    kubectl get pods -n $Ns -l app=credit-api -o wide
    Section "Modelo que sirve cada pod (/info)"
    Invoke-RestMethod "$Api/info" -DisableKeepAlive | Format-List
} *>&1 | Tee-Object -FilePath $log
Write-Host "Evidencia: $log" -ForegroundColor Green
