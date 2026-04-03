#!/usr/bin/env bash
# Repeatable PySCF install on WSL2 (Ubuntu) or native Linux.
#
# Prerequisites (one-time on the distro):
#   - WSL2 + Ubuntu (or another Debian/Ubuntu-based WSL distro)
#   - sudo for apt
#
# From Windows: open Ubuntu in WSL, then:
#   cd /mnt/d/Yaozheng/QuantumChemistry/LearningPlan/learning_materials   # adjust drive/path
#   chmod +x install_pyscf_wsl.sh
#
# With conda (recommended, matches README qc_chem):
#   conda activate qc_chem
#   ./install_pyscf_wsl.sh              # latest pyscf from PyPI
#   ./install_pyscf_wsl.sh 2.6.2        # pin version (e.g. same as qiskit-nature stack)
#
# Without conda (venv in this directory):
#   CREATE_VENV=1 ./install_pyscf_wsl.sh 2.6.2
#
# Environment:
#   PYTHON=python3.11  — interpreter to use (default: python3)
#   CREATE_VENV=1      — create & use ./.venv_wsl_pyscf before pip
#   SKIP_APT=1         — skip apt install (deps already present)

set -euo pipefail

usage() {
  echo "Usage: $0 [PYSCF_VERSION]" >&2
  echo "  PYSCF_VERSION  optional, e.g. 2.6.2  (passed to pip as pyscf==VERSION)" >&2
  exit 1
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage
[[ $# -gt 1 ]] && usage

PYSCF_VERSION="${1:-}"
# Prefer `python` when present (typical after `conda activate`); else `python3`.
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  PY=python3
fi

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "error: interpreter not found: $PY (set PYTHON=...)" >&2
  exit 1
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "error: this script is for Linux/WSL only (uname: $(uname -s))" >&2
  exit 1
fi

install_apt_deps() {
  if [[ "${SKIP_APT:-0}" == "1" ]]; then
    echo "[skip] SKIP_APT=1 — not running apt."
    return 0
  fi
  echo "[apt] Installing build dependencies (needs sudo)..."
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    python3-dev \
    libopenblas-dev \
    git
}

create_venv_if_requested() {
  if [[ "${CREATE_VENV:-0}" != "1" ]]; then
    return 0
  fi
  local root
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local vdir="$root/.venv_wsl_pyscf"
  echo "[venv] Creating $vdir"
  "$PY" -m venv "$vdir"
  # shellcheck source=/dev/null
  source "$vdir/bin/activate"
  PY="python"
  echo "[venv] Using: $(command -v python)"
}

verify_import() {
  echo "[test] import pyscf ..."
  "$PY" -c "import pyscf; print('pyscf', pyscf.__version__)"
}

main() {
  install_apt_deps
  create_venv_if_requested

  echo "[pip] Upgrading pip (user env: $($PY -c 'import sys; print(sys.executable)'))"
  "$PY" -m pip install -U pip setuptools wheel

  if [[ -n "$PYSCF_VERSION" ]]; then
    echo "[pip] Installing pyscf==${PYSCF_VERSION} ..."
    "$PY" -m pip install "pyscf==${PYSCF_VERSION}"
  else
    echo "[pip] Installing pyscf (latest on PyPI) ..."
    "$PY" -m pip install pyscf
  fi

  verify_import
  echo "[done] PySCF is installed for: $($PY -c 'import sys; print(sys.executable)')"
}

main
