# InferML installer (Windows / PowerShell)
#
#   irm https://inferml.vercel.app/install.ps1 | iex
#
# Installs the `inferml` command via pipx. InferML is a Python app, so it needs
# an existing Python 3.10+ - this script does NOT install Python. If Python is
# missing (or too old) it stops and tells you where to get it.

function Install-InferML {
  # Native tools (pip / pipx) routinely print to stderr - pipx's version probe
  # emits "No module named pipx", pip emits PATH warnings. In Windows PowerShell
  # 5.1, if the caller's session has $ErrorActionPreference = 'Stop', that stderr
  # becomes a *terminating* error and aborts the install. Control flow here is
  # driven by explicit $LASTEXITCODE checks, so we run under 'Continue'. Setting
  # it inside this function keeps it local - the caller's session is untouched.
  $ErrorActionPreference = 'Continue'

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
InferML needs Python 3.10 or newer, which wasn't found on your PATH.

Install Python first - https://www.python.org/downloads/
(tick "Add python.exe to PATH" in the installer), then re-run:

    irm https://inferml.vercel.app/install.ps1 | iex
'@ -ForegroundColor Red
    return
  }
  Info ("Using " + (& $py --version) + " (" + (Get-Command $py).Source + ")")

  # --- ensure pipx ------------------------------------------------------------
  & $py -m pipx --version *> $null
  if ($LASTEXITCODE -ne 0) {
    Info 'Installing pipx...'
    # Stream pip's output (stdout + stderr) as plain text. Windows PowerShell
    # 5.1 otherwise renders native stderr - even benign progress lines - as red
    # NativeCommandError records; piping through Write-Host keeps it plain.
    & $py -m pip install --user pipx 2>&1 | ForEach-Object { Write-Host "$_" }
    if ($LASTEXITCODE -ne 0) { Write-Host "Couldn't install pipx. Try:  $py -m pip install --user pipx" -ForegroundColor Red; return }
    & $py -m pipx ensurepath *> $null
  }

  # --- install InferML (server only; the app installs the CPU/GPU stack on first
  #     launch, so we don't pull torch here) --------------------------------------
  Info 'Installing the InferML server...'
  # Stream pipx's output (stdout + stderr) as plain text - pipx prints
  # "creating virtual environment..." to stderr, which PowerShell 5.1 would
  # otherwise show as a red error even on a successful install.
  & $py -m pipx install inferml 2>&1 | ForEach-Object { Write-Host "$_" }
  if ($LASTEXITCODE -ne 0) { Write-Host 'Install failed. See the output above.' -ForegroundColor Red; return }

  Write-Host ''
  Ok 'InferML is installed.'
  Write-Host ''
  Info 'Start it with:'
  Write-Host '    inferml' -ForegroundColor Green
  Info 'then open http://localhost:11500 in your browser.'
  Info 'On first launch, pick CPU or GPU to install the model runtime (PyTorch + transformers).'
  Write-Host ''
  Warn "If the 'inferml' command isn't found, open a new terminal - pipx just updated your PATH."
}

Install-InferML
