#!/usr/bin/env python3
"""
Regenerate Phase-0 baseline figures for H2 VQE with fixed random seed and
documented hyperparameters (see docs/PHASE0_H2_BASELINE.md).

Usage:
    python -m quantum_chemistry.tutorials.plot_h2_baseline_figures

Outputs (repo root relative):
    docs/figures/h2_vqe_convergence.png
    docs/figures/h2_pes.png  (multi-point only if PySCF is available)
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

# Reproducibility (future-proof if stochastic layers are added)
BASELINE_SEED = 42
np.random.seed(BASELINE_SEED)
os.environ.setdefault("PL_GLOBAL_SEED", str(BASELINE_SEED))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_h2_module():
    """Load 01_complete_h2_vqe.py (numeric prefix is not a valid import name)."""
    path = Path(__file__).resolve().parent / "01_complete_h2_vqe.py"
    spec = importlib.util.spec_from_file_location("h2_complete_vqe", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    root = _repo_root()
    out_dir = root / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root))
    h2 = _load_h2_module()

    integrals = h2.get_h2_integrals(bond_length=0.74)
    results = h2.run_h2_vqe(integrals, verbose=False)
    conv_path = out_dir / "h2_vqe_convergence.png"
    h2.plot_energy_convergence(results, save_path=str(conv_path))
    print(f"Wrote {conv_path}")

    pes_path = out_dir / "h2_pes.png"
    pes = h2.scan_potential_energy_surface(verbose=False)
    h2.plot_potential_energy_surface(pes, save_path=str(pes_path))
    print(f"Wrote {pes_path}")

    # Optional: symlink-style copy to legacy paths at repo root (if desired)
    for name in ("h2_vqe_convergence.png", "h2_pes.png"):
        src = out_dir / name
        dst = root / name
        if src.exists():
            dst.write_bytes(src.read_bytes())
            print(f"Copied to {dst}")

    print(f"BASELINE_SEED={BASELINE_SEED} PySCF={integrals.get('from_pyscf', False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
