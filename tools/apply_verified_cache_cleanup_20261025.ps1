$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot)).TrimEnd('\')
$receiptDir = Join-Path $workspace 'research/cleanup_2026-10-25'
$planPath = Join-Path $receiptDir 'plan.json'
$plan = Get-Content -LiteralPath $planPath -Raw -Encoding utf8 | ConvertFrom-Json
if ($plan.workspace -cne $workspace) { throw 'Plan workspace mismatch' }

function Get-CheckedTarget([string]$relative) {
    if ([IO.Path]::IsPathRooted($relative)) { throw "Absolute target prohibited: $relative" }
    $target = [IO.Path]::GetFullPath((Join-Path $workspace $relative))
    if (-not $target.StartsWith($workspace + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target outside workspace: $target"
    }
    $cursor = $target
    while ($cursor -ne $workspace) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Reparse point prohibited: $cursor"
            }
        }
        $cursor = Split-Path -Parent $cursor
    }
    return $target
}

# Complete preflight before the first removal. No recursive deletions are used.
$targets = @()
foreach ($entry in $plan.files.PSObject.Properties) {
    $target = Get-CheckedTarget $entry.Name
    $item = Get-Item -LiteralPath $target -Force
    if ($item.PSIsContainer -or $item.Length -ne $entry.Value.bytes) {
        throw "Target changed since audit: $target"
    }
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -cne $entry.Value.sha256) {
        throw "Target hash changed since audit: $target"
    }
    $targets += [pscustomobject]@{ path = $entry.Name; absolute = $target; bytes = $item.Length }
}
Write-Output "PREFLIGHT_PASS $($targets.Count) files"
$removed = [Collections.Generic.List[object]]::new()
$failed = [Collections.Generic.List[object]]::new()
foreach ($target in $targets) {
    try {
        Remove-Item -LiteralPath $target.absolute -Force
        $removed.Add([pscustomobject]@{path=$target.path; bytes=$target.bytes})
    } catch {
        $failed.Add([pscustomobject]@{path=$target.path; error=$_.Exception.Message})
    }
}
$emptyDirsRemoved = [Collections.Generic.List[string]]::new()
foreach ($relative in $plan.empty_dir_candidates) {
    $target = Get-CheckedTarget $relative
    if ((Test-Path -LiteralPath $target -PathType Container) -and
        @(Get-ChildItem -LiteralPath $target -Force).Count -eq 0) {
        Remove-Item -LiteralPath $target -Force
        $emptyDirsRemoved.Add($relative)
    }
}
$receipt = [ordered]@{
    plan_sha256 = (Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant()
    completed_local = (Get-Date).ToString('o')
    removed = $removed.ToArray()
    failed = $failed.ToArray()
    empty_directories_removed = $emptyDirsRemoved.ToArray()
    removed_logical_bytes = ($removed | Measure-Object -Property bytes -Sum).Sum
}
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $receiptDir 'deletion_receipt.json') -Encoding utf8
Write-Output "REMOVED $($removed.Count) files; FAILED $($failed.Count); EMPTY_DIRS $($emptyDirsRemoved.Count); LOGICAL_BYTES $($receipt.removed_logical_bytes)"
if ($failed.Count) { throw 'Some deletion targets failed; consult receipt' }
