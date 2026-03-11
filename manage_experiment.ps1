param(
    [Parameter(Mandatory=$true)]
    [string]$ExperimentName,
    
    [Parameter(Mandatory=$false)]
    [int]$NPersonas = 8,
    
    [Parameter(Mandatory=$false)]
    [int]$EMIters = 50,
    
    [Parameter(Mandatory=$false)]
    [string]$Description = "run"
)

# 1. Generate formatted identifiers
$timestamp = Get-Date -Format "yyyyMMdd-HHmm"
$branchName = "exp/$ExperimentName-$timestamp"
$tagName = "v-$ExperimentName-P$NPersonas-IT$EMIters-$Description"

Write-Host ">>> Creating experiment branch: $branchName"
git checkout -b $branchName

# 2. Record parameters (optional but good for tracking)
$metadata = @{
    name = $ExperimentName
    n_personas = $NPersonas
    em_iters = $EMIters
    timestamp = $timestamp
    description = $Description
}
$metadata | ConvertTo-Json | Out-File "experiment_meta.json"

Write-Host ">>> Committing experiment metadata"
git add experiment_meta.json
git commit -m "Setup experiment: $ExperimentName (P=$NPersonas, IT=$EMIters)"

# 3. Create Tag
Write-Host ">>> Tagging experiment: $tagName"
git tag -a $tagName -m "Experiment $ExperimentName with P=$NPersonas and IT=$EMIters"

Write-Host ">>> Done! You can now run your training and then push with:"
Write-Host "    git push origin $branchName --tags"
