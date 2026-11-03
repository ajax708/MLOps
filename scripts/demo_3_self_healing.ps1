# Demo 3: autorreparación. Se elimina un pod mientras se envían peticiones sin parar.
param([int]$Seconds = 40)
. "$PSScriptRoot\common.ps1"
$log = Join-Path $Evidence "03_self_healing.txt"
& {
    Section "Pods ANTES"
    kubectl get pods -n $Ns -l app=credit-api -o wide
    $victim = (kubectl get pods -n $Ns -l app=credit-api -o jsonpath="{.items[0].metadata.name}")
    Section "Eliminando pod $victim"
    kubectl delete pod $victim -n $Ns --wait=false

    $ok = 0; $fail = 0; $seen = @{}
    $end = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $end) {
        try { $r = Invoke-Predict; $ok++; $seen[$r.pod_name] = 1 } catch { $fail++ }
        Start-Sleep -Milliseconds 200
    }
    Section "Resultado durante $Seconds s"
    "Peticiones OK: $ok   Fallidas: $fail"
    "Pods que respondieron: $($seen.Keys -join ', ')"
    Section "Pods DESPUÉS (Kubernetes creó un reemplazo)"
    kubectl get pods -n $Ns -l app=credit-api -o wide
    kubectl get events -n $Ns --field-selector involvedObject.kind=Pod --sort-by=.lastTimestamp | Select-Object -Last 8
} *>&1 | Tee-Object -FilePath $log
Write-Host "Evidencia: $log" -ForegroundColor Green
