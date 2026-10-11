param([string]$DataDirectory = $PSScriptRoot, [switch]$VerifyOnly)
$ErrorActionPreference = 'Stop'
$dataRoot = [IO.Path]::GetFullPath($DataDirectory)
function FileHash($file) {
    $reader = [IO.File]::OpenRead($file)
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($hasher.ComputeHash($reader))).Replace('-', '').ToLowerInvariant() }
    finally { $reader.Dispose(); $hasher.Dispose() }
}
function LocalFile($name, $root = $dataRoot) {
    if ([IO.Path]::GetFileName($name) -ne $name -or $name -notmatch '^[A-Za-z0-9._-]+$') {
        throw "Invalid package filename: $name"
    }
    return Join-Path $root $name
}
function CheckFile($entry, $root = $dataRoot) {
    $file = LocalFile $entry.name $root
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $($entry.name)" }
    if ((Get-Item -LiteralPath $file).Length -ne [long]$entry.bytes) { throw "Incomplete file: $($entry.name)" }
    if ((FileHash $file) -ne $entry.sha256) {
        throw "SHA-256 check failed: $($entry.name)"
    }
    return $file
}
try {
    $manifest = Get-Content -LiteralPath (Join-Path $dataRoot 'installer-payload.json') -Raw | ConvertFrom-Json
    if ($manifest.schema -ne 1 -or $manifest.parts.Count -lt 1) { throw 'Invalid package manifest' }
    $setup = CheckFile $manifest.installer
    $payload = LocalFile $manifest.payload.name
    if (-not (Test-Path -LiteralPath $payload)) {
        # Explorer normally extracts ZIP contents into a child folder, leaving
        # the separately downloaded data volumes in its parent directory.
        $partsRoot = $dataRoot
        foreach ($candidate in @($dataRoot, [IO.Directory]::GetParent($dataRoot).FullName)) {
            if (-not $candidate) { continue }
            $missing = @($manifest.parts | Where-Object {
                -not (Test-Path -LiteralPath (LocalFile $_.name $candidate) -PathType Leaf)
            })
            if ($missing.Count -eq 0) { $partsRoot = $candidate; break }
        }
        $parts = @($manifest.parts | ForEach-Object { CheckFile $_ $partsRoot })
        $temporary = $payload + '.rebuilding'
        Write-Host 'Joining installation data. This can take a few minutes...'
        $writer = [IO.File]::Open($temporary, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
        try {
            foreach ($part in $parts) {
                $reader = [IO.File]::OpenRead($part)
                try { $reader.CopyTo($writer, 8388608) } finally { $reader.Dispose() }
            }
        } finally { $writer.Dispose() }
        if ((Get-Item -LiteralPath $temporary).Length -ne [long]$manifest.payload.bytes -or
            (FileHash $temporary) -ne $manifest.payload.sha256) {
            throw 'Joined installation data failed verification'
        }
        Move-Item -LiteralPath $temporary -Destination $payload
    }
    $null = CheckFile $manifest.payload
    Write-Host 'Installation data verified.'
    if (-not $VerifyOnly) {
        $installerProcess = Start-Process -FilePath $setup -PassThru -Wait
        exit $installerProcess.ExitCode
    }
} catch {
    Write-Error $_
    exit 1
}
