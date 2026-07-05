#!/bin/sh
# LocalML installer (macOS / Linux)
#
#   curl -fsSL https://www.localml.tech/install.sh | sh
#
# Installs the `localml` command via pipx. LocalML is a Python app, so it needs
# an existing Python 3.10+ - this script does NOT install Python. If Python is
# missing (or too old) it stops and tells you where to get it.
set -eu

# Colors only when writing to a terminal (piped-to-sh keeps stdout a tty).
if [ -t 1 ]; then
  RED="$(printf '\033[31m')"; GRN="$(printf '\033[32m')"; YLW="$(printf '\033[33m')"
  DIM="$(printf '\033[2m')"; RST="$(printf '\033[0m')"
else
  RED=""; GRN=""; YLW=""; DIM=""; RST=""
fi
info() { printf '%s\n' "${DIM}$1${RST}"; }
ok()   { printf '%s\n' "${GRN}$1${RST}"; }
warn() { printf '%s\n' "${YLW}$1${RST}"; }
die()  { printf '%s\n' "${RED}$1${RST}" >&2; exit 1; }

# --- require Python 3.10+ (we do not install it) ---------------------------
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' 2>/dev/null; then
      PY="$c"; break
    fi
  fi
done

if [ -z "$PY" ]; then
  die "LocalML needs Python 3.10 or newer, which wasn't found on your PATH.

Install Python first - https://www.python.org/downloads/
(or via your package manager: brew install python  /  sudo apt install python3)

then re-run:  curl -fsSL https://www.localml.tech/install.sh | sh"
fi
info "Using $("$PY" --version 2>&1) ($(command -v "$PY"))"

# --- ensure pipx ------------------------------------------------------------
if ! "$PY" -m pipx --version >/dev/null 2>&1; then
  info "Installing pipx..."
  "$PY" -m pip install --user pipx >/dev/null 2>&1 \
    || die "Couldn't install pipx. Try manually:  $PY -m pip install --user pipx"
  "$PY" -m pipx ensurepath >/dev/null 2>&1 || true
fi

# --- install LocalML (server only; the app installs the CPU/GPU stack on first
#     launch, so we don't pull torch here) --------------------------------------
info "Installing the LocalML server..."
"$PY" -m pipx install localml \
  || die "Install failed. See the output above."

printf '\n'
ok "LocalML is installed."
printf '\n'
info "Start it with:"
printf '    %slocalml%s\n' "${GRN}" "${RST}"
info "then open http://localhost:11500 in your browser."
info "On first launch, pick CPU or GPU to install the model runtime (PyTorch + transformers)."
printf '\n'
warn "If the 'localml' command isn't found, open a new terminal - pipx just updated your PATH."
