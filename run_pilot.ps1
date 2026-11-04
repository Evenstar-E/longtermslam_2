param(
    [ValidateSet('Test', 'Smoke', 'Evaluation', 'Sensitivity')]
    [string]$Mode = 'Test',
    [string]$OutputDirectory = '',
    [string]$ConfigPath = 'configs/pilot.json'
)
$ErrorActionPreference = 'Stop'
$pilotPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pilotPython)) {
    throw 'Project Python is missing. Follow README.md environment setup.'
}
$env:PYTHONUTF8 = '1'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.cache\matplotlib'
Push-Location -LiteralPath $PSScriptRoot
try {
    if ($Mode -eq 'Test') {
        & $pilotPython -m unittest discover -s tests -v
    } else {
        $pilotSplit = switch ($Mode) {
            'Smoke' { 'development' }
            'Evaluation' { 'evaluation' }
            'Sensitivity' { 'sensitivity' }
        }
        if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
            $OutputDirectory = Join-Path 'results' ($pilotSplit + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
        }
        $pilotArguments = @('-m', 'uav_pilot.run', '--config', $ConfigPath,
                            '--split', $pilotSplit, '--output', $OutputDirectory)
        if ($Mode -eq 'Smoke') { $pilotArguments += @('--max-tasks', '6') }
        & $pilotPython @pilotArguments
    }
    if ($LASTEXITCODE -ne 0) { throw "Pilot command failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
