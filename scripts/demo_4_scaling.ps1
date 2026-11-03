# Demo 4: escalado 3 -> 5 -> 3 réplicas
param([int]$To = 5)
. "$PSScriptRoot\common.ps1"
$log = Join-Path $Evidence "04_scaling.txt"
& {
    Section "Estado inicial"
    kubectl get deployment credit-api -n $Ns
    Section "kubectl scale --replicas=$To"
    kubectl scale deployment credit-api -n $Ns --replicas=$To
    kubectl rollout status deployment/credit-api -n $Ns --timeout=300s
    kubectl get pods -n $Ns -l app=credit-api -o wide
    Section "Tráfico tras escalar (pods distintos)"
    $pods = 1..25 | ForEach-Object { (Invoke-Predict).pod_name }
    $pods | Group-Object | Select-Object Name, Count | Format-Table -AutoSize
    Section "Volviendo a 3 réplicas"
    kubectl scale deployment credit-api -n $Ns --replicas=3
    kubectl rollout status deployment/credit-api -n $Ns --timeout=120s
    Start-Sleep -Seconds 5
    kubectl get pods -n $Ns -l app=credit-api
} *>&1 | Tee-Object -FilePath $log
Write-Host "Evidencia: $log" -ForegroundColor Green
