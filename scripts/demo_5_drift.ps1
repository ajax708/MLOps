# Demo 5: pruebas de data drift y concept drift. El resumen final sale de los resultados de esta ejecución.
. "$PSScriptRoot\common.ps1"
Set-Location $Root
$env:MLFLOW_TRACKING_URI = "http://localhost:30500"
$log = Join-Path $Evidence "05_drift.txt"

function Resumen([string]$Drift, [string]$Calib, [string]$Concept, [string]$Tests) {
    $linea = "REVISAR"
    $verde = $Drift -match 'VERDE - PASS'
    if ($verde -and $Drift -match 'ROJO - FAIL.*?variables con drift: \[([^\]]*)\]') {
        $vars = ($Matches[1] -split ',').Count
        if ($Calib -match ': (\d+)/(\d+) =') {
            $linea = "detecta el lote alterado ($vars variables), $($Matches[1])/$($Matches[2]) falsas alarmas"
        }
    }
    Write-Host "Data drift:     $linea"

    if ($Concept -match 'alarma en lote (\d+)') {
        Write-Host "Concept drift:  alarma en el lote $($Matches[1])"
        Write-Host "Acción:         reentrenar"
    } elseif ($Concept -match 'modelo estable') {
        Write-Host "Concept drift:  sin alarma"
        Write-Host "Acción:         seguir vigilando"
    } else {
        Write-Host "Concept drift:  REVISAR"
        Write-Host "Acción:         REVISAR"
    }

    if ($Tests -match '(\d+) passed' -and $Tests -notmatch '\d+ (failed|error)') {
        Write-Host "Tests:          $($Matches[1]) OK"
    } else {
        Write-Host "Tests:          REVISAR"
    }
}

& {
    Section "Data drift: ¿cambiaron los clientes que llegan?"
    Write-Host "Lote normal: debe dar VERDE. Lote alterado: debe dar ROJO."
    & $Py src/drift/data_drift.py 2>&1 | Tee-Object -Variable drift

    Section "Falsas alarmas"
    Write-Host "200 lotes sanos: no debe saltar ninguna alarma."
    & $Py src/drift/data_drift.py --calibrate 2>&1 | Tee-Object -Variable calib

    Section "Concept drift: ¿el modelo sigue acertando?"
    Write-Host "Desde el lote 6, los clientes cambian de comportamiento."
    & $Py src/drift/concept_drift.py 2>&1 | Tee-Object -Variable concept

    Section "Tests"
    & $Py -m pytest -p no:warnings 2>&1 | Tee-Object -Variable tests

    Section "Resumen"
    Resumen ($drift | Out-String) ($calib | Out-String) ($concept | Out-String) ($tests | Out-String)
} *>&1 | Tee-Object -FilePath $log
Copy-Item reports\concept_drift.png (Join-Path $Evidence "05_concept_drift.png") -Force
Write-Host "Evidencia: $log" -ForegroundColor Green
