param(
    [ValidateSet("dev", "api")]
    [string]$Mode = "dev",
    [string]$ApiHost = "0.0.0.0",
    [int]$ApiPort = 8000,
    [int]$UiPort = 5173,
    [switch]$SkipFrontendInstall,
    [int]$FrontendInstallTimeout = 900
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Join-Path $repoRoot "ind-compliance-ai"
$venvDir = Join-Path $projectRoot ".venv"
$pythonExe = Join-Path $venvDir "Scripts\\python.exe"
$requirementsFile = Join-Path $projectRoot "requirements.txt"
$entrypoint = Join-Path $projectRoot "main.py"
$validator = Join-Path $projectRoot "scripts\\validate_environment.py"

if (-not (Test-Path $projectRoot)) {
    throw "Project directory not found: $projectRoot"
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install Python 3.12 first."
}

try {
    & py -3.12 -c "import sys; print(sys.executable)" | Out-Null
} catch {
    throw "Python 3.12 was not found. Install Python 3.12 to run ind-compliance-ai."
}

if (-not (Test-Path $pythonExe)) {
    Write-Host "Creating virtual environment with Python 3.12 ..."
    & py -3.12 -m venv $venvDir
}

Write-Host "Installing backend dependencies ..."
& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r $requirementsFile

Write-Host "Validating environment ..."
$validatorArgs = @($validator)
if ($Mode -eq "api") {
    $validatorArgs += "--skip-frontend"
}
& $pythonExe @validatorArgs

$argsList = @(
    $entrypoint,
    "--mode", $Mode,
    "--api-host", $ApiHost,
    "--api-port", "$ApiPort",
    "--ui-port", "$UiPort",
    "--frontend-install-timeout", "$FrontendInstallTimeout"
)

if ($SkipFrontendInstall) {
    $argsList += "--skip-frontend-install"
}

Write-Host "Starting ind-compliance-ai ..."
& $pythonExe @argsList
