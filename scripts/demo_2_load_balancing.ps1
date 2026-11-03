# Demo 2: peticiones sucesivas al Service son atendidas por pods distintos
param([int]$N = 30)
. "$PSScriptRoot\common.ps1"
$log = Join-Path $Evidence "02_load_balancing.txt"
& {
    Section "$N peticiones POST /predict al Service credit-api (NodePort 30080)"
    $pods = @()
    1..$N | ForEach-Object {
        $r = Invoke-Predict
        $pods += $r.pod_name
        "{0,3}  pod={1}  prob={2}  modelo v{3}" -f $_, $r.pod_name, $r.predictions[0].default_probability, $r.model_version
    }
    Section "Distribución por pod"
    $pods | Group-Object | Sort-Object Name | Select-Object Name, Count | Format-Table -AutoSize
} *>&1 | Tee-Object -FilePath $log
Write-Host "Evidencia: $log" -ForegroundColor Green
