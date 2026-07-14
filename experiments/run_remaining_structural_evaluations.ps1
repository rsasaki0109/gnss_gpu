param(
    [int]$WaitForProcessId = 0
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Invoke-CheckedPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "python exited with code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Invoke-Summary {
    param([string]$Prefix)
    Invoke-CheckedPython `
        experiments/summarize_structural_ablation.py `
        "experiments/results/${Prefix}_internal_epochs.csv" `
        --runs-csv "experiments/results/${Prefix}_runs.csv" `
        --output "experiments/results/${Prefix}_ablation_summary.csv"
}

if ($WaitForProcessId -gt 0) {
    Wait-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue
}

$modePrefix = "pf_mode_full6_p2k_diagnostic"
if (-not (Test-Path "experiments/results/${modePrefix}_runs.csv") -or
    -not (Test-Path "experiments/results/${modePrefix}_internal_epochs.csv")) {
    throw "PF mode prerequisite did not produce its complete six-run artifacts"
}
Invoke-Summary $modePrefix

$stagedDopplerPrefix = "rbpf_doppler_gej_full6_p2k"
if (-not (Test-Path "experiments/results/${stagedDopplerPrefix}_runs.csv")) {
    Invoke-CheckedPython `
        experiments/exp_ppc_ctrbpf_fgo.py `
        --runs all `
        --methods rbpf `
        --n-particles 2000 `
        --pf-mode-policy off `
        --write-internal-diagnostics `
        --doppler-systems "G,E,J" `
        --pos-dir results/rbpf_doppler_gej_full6_p2k_pos `
        --results-prefix $stagedDopplerPrefix
}
Invoke-Summary $stagedDopplerPrefix

$allSystemPrefix = "rbpf_doppler_gejcr_full6_p2k"
if (-not (Test-Path "experiments/results/${allSystemPrefix}_runs.csv")) {
    Invoke-CheckedPython `
        experiments/exp_ppc_ctrbpf_fgo.py `
        --runs all `
        --methods rbpf `
        --n-particles 2000 `
        --pf-mode-policy off `
        --write-internal-diagnostics `
        --doppler-systems "G,E,J,C,R" `
        --pos-dir results/rbpf_doppler_gejcr_full6_p2k_pos `
        --results-prefix $allSystemPrefix
}
Invoke-Summary $allSystemPrefix
Invoke-CheckedPython `
    experiments/compare_structural_ablation.py `
    "experiments/results/${stagedDopplerPrefix}_ablation_summary.csv" `
    "experiments/results/${allSystemPrefix}_ablation_summary.csv" `
    --output experiments/results/rbpf_doppler_gej_vs_gejcr_comparison.csv

$ffbsiPrefix = "pf_ffbsi_full6_p2k_lag10_paths8"
if (-not (Test-Path "experiments/results/${ffbsiPrefix}_runs.csv")) {
    Invoke-CheckedPython `
        experiments/exp_ppc_ctrbpf_fgo.py `
        --runs all `
        --methods pf `
        --n-particles 2000 `
        --enable-pf-ffbsi-smoother `
        --pf-ffbsi-lag-epochs 10 `
        --pf-ffbsi-paths 8 `
        --pf-ffbsi-mode marginal `
        --pf-mode-policy diagnostic `
        --pos-dir results/pf_ffbsi_full6_p2k_pos `
        --results-prefix $ffbsiPrefix
}
Invoke-Summary $ffbsiPrefix
Invoke-CheckedPython `
    experiments/compare_structural_ablation.py `
    experiments/results/pf_mode_full6_p2k_diagnostic_ablation_summary.csv `
    "experiments/results/${ffbsiPrefix}_ablation_summary.csv" `
    --output experiments/results/pf_vs_ffbsi_full6_comparison.csv

Invoke-CheckedPython experiments/run_tcfgo_blocked_spans.py
Invoke-CheckedPython experiments/run_recurrence_full_runs.py

Write-Output "All queued structural evaluations completed."
