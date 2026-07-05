# LocalML installer (Windows / PowerShell)
#
#   irm https://www.localml.tech/install.ps1 | iex
#
# Installs the `localml` command via pipx. LocalML is a Python app, so it needs
# an existing Python 3.10+ - this script does NOT install Python. If Python is
# missing (or too old) it stops and tells you where to get it.
$ErrorActionPreference = 'Stop'

function Info($m) { Write-Host $m -ForegroundColor DarkGray }
function Ok($m)   { Write-Host $m -ForegroundColor Green }
function Warn($m) { Write-Host $m -ForegroundColor Yellow }

# --- require Python 3.10+ (we do not install it) ---------------------------
$py = $null
foreach ($c in @('python', 'python3', 'py')) {
  if (Get-Command $c -ErrorAction SilentlyContinue) {
    & $c -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' 2>$null
    if ($LASTEXITCODE -eq 0) { $py = $c; break }
  }
}

if (-not $py) {
  Write-Host @'
LocalML needs Python 3.10 or newer, which wasn't found on your PATH.

Install Python first - https://www.python.org/downloads/
(tick "Add python.exe to PATH" in the installer), then re-run:

    irm https://www.localml.tech/install.ps1 | iex
'@ -ForegroundColor Red
  return
}
Info ("Using " + (& $py --version) + " (" + (Get-Command $py).Source + ")")

# --- ensure pipx ------------------------------------------------------------
& $py -m pipx --version *> $null
if ($LASTEXITCODE -ne 0) {
  Info 'Installing pipx...'
  & $py -m pip install --user pipx
  if ($LASTEXITCODE -ne 0) { Write-Host "Couldn't install pipx. Try:  $py -m pip install --user pipx" -ForegroundColor Red; return }
  & $py -m pipx ensurepath *> $null
}

# --- install LocalML (server only; the app installs the CPU/GPU stack on first
#     launch, so we don't pull torch here) --------------------------------------
Info 'Installing the LocalML server...'
& $py -m pipx install localml
if ($LASTEXITCODE -ne 0) { Write-Host 'Install failed. See the output above.' -ForegroundColor Red; return }

Write-Host ''
Ok 'LocalML is installed.'
Write-Host ''
Info 'Start it with:'
Write-Host '    localml' -ForegroundColor Green
Info 'then open http://localhost:11500 in your browser.'
Info 'On first launch, pick CPU or GPU to install the model runtime (PyTorch + transformers).'
Write-Host ''
Warn "If the 'localml' command isn't found, open a new terminal - pipx just updated your PATH."
