# Monitor Grid Search Progress
Write-Host ">>> Monitoring SAGE Grid Search Progress..." -ForegroundColor Cyan

while($true) {
    Clear-Host
    Write-Host "=== Experiment Status ($(Get-Date -Format 'HH:mm:ss')) ===" -ForegroundColor Yellow
    
    # 1. Dashboard (NEW)
    Write-Host "`n--- Live Dashboard ---" -ForegroundColor Green
    $statusFile = "checkpoints/live_status.txt"
    if (Test-Path $statusFile) {
        Get-Content $statusFile
    } else {
        Write-Host "Dashboard not yet available..."
    }

    # 2. Checkpoints
    Write-Host "`n--- Latest Checkpoints ---" -ForegroundColor Green
    Get-ChildItem -Path checkpoints -Recurse -Filter *.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 5 | Format-Table Name, LastWriteTime, @{Name="Size(MB)"; Expression={$_.Length / 1MB -as [int]}}
    
    # 3. Results JSON
    Write-Host "`n--- Metric Results ---" -ForegroundColor Green
    if (Test-Path data/results/grid_search_results_full.json) {
        $results = Get-Content data/results/grid_search_results_full.json | ConvertFrom-Json
        $results | Format-Table n_personas, em_iters, silhouette, perplexity
    } else {
        Write-Host "Results JSON not generated yet."
    }

    Write-Host "`nPress Ctrl+C to stop monitoring."
    Start-Sleep -Seconds 5
}
