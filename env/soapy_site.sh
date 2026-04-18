#!/usr/bin/env bash
# Source this to add Homebrew's SoapySDR Python site-packages to PYTHONPATH for current shell/venv.
# Usage: source env/soapy_site.sh

# NOTE:
# - This file is meant to be *sourced*, so we avoid `set -euo pipefail` (it mutates the caller shell).
# - Homebrew's SoapySDR Python bindings are commonly installed as SoapySDR.py + _SoapySDR*.so
#   (not as a SoapySDR/ package with __init__.py).

_soapy_err() { echo "soapy_site.sh: $*" >&2; }
_soapy_fail() { _soapy_err "$*"; return 1; }

# Locate Homebrew prefix
BREW_PREFIX="$(command -v brew >/dev/null 2>&1 && brew --prefix 2>/dev/null || true)"
[ -n "$BREW_PREFIX" ] || _soapy_fail "Homebrew not found. Install Homebrew to use this helper."

# Use the *current* python (venv-aware). Fall back to python3.
PY_BIN="$(command -v python >/dev/null 2>&1 && echo python || echo python3)"

# Determine Python major.minor
PYVER="$($PY_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
[ -n "$PYVER" ] || _soapy_fail "Could not determine Python version from '$PY_BIN'."

# Candidate site-packages paths
CAND=(
  "${BREW_PREFIX}/lib/python${PYVER}/site-packages"
  "/opt/homebrew/lib/python${PYVER}/site-packages"
  "/usr/local/lib/python${PYVER}/site-packages"
)

FOUND=""
for d in "${CAND[@]}"; do
  if [ -d "$d" ] && { [ -f "$d/SoapySDR.py" ] || ls "$d"/_SoapySDR*.so >/dev/null 2>&1; }; then
    FOUND="$d"
    break
  fi
done

if [ -z "$FOUND" ]; then
  _soapy_err "Could not locate SoapySDR Python bindings under Homebrew for Python ${PYVER}."
  _soapy_err "Looked in:"
  for d in "${CAND[@]}"; do _soapy_err "  - $d"; done
  _soapy_err ""
  _soapy_err "If you installed SoapySDR via Homebrew, the bindings may exist for a different Python version."
  _soapy_err "Try:"
  _soapy_err "  - python -V   (inside your venv)"
  _soapy_err "  - ls \"${BREW_PREFIX}/lib/python*/site-packages\" | grep -i soapy"
  _soapy_err "  - Or install bindings in the venv: pip install SoapySDR"
  return 1
fi

# Prepend to PYTHONPATH safely (only adds ':' when PYTHONPATH is already set)
export PYTHONPATH="${FOUND}${PYTHONPATH:+:$PYTHONPATH}"
echo "Added SoapySDR site-packages to PYTHONPATH: ${FOUND}"
