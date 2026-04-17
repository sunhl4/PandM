#!/usr/bin/env python3
"""
Reproducible H2 VQE micro-benchmark for Phase 4 / external reporting.

- Fixed seed (see BASELINE_SEED in docs/PHASE0_H2_BASELINE.md)
- Writes JSON summary next to optional parity check vs FCI reference

Usage:
    python benchmarks/reproducible_h2_vqe_benchmark.py --out benchmarks/h2_vqe_benchmark_result.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

BASELINE_SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_h2():
    path = REPO_ROOT / "quantum_chemistry" / "tutorials" / "01_complete_h2_vqe.py"
    spec = importlib.util.spec_from_file_location("h2_complete_vqe", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    np.random.seed(BASELINE_SEED)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bond-length",
        type=float,
        default=0.74,
        help="H-H bond length in Angstrom",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "h2_vqe_benchmark_result.json",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    h2 = _load_h2()

    t0 = time.perf_counter()
    integrals = h2.get_h2_integrals(bond_length=args.bond_length)
    results = h2.run_h2_vqe(integrals, verbose=False)
    elapsed = time.perf_counter() - t0

    err_vs_exact = float(abs(results["vqe_energy"] - results["exact_energy"]))
    err_vs_fci = float(abs(results["vqe_energy"] - integrals["fci_energy"]))
    # If qubit-Hamiltonian ED disagrees with PySCF FCI, the JW/Hamiltonian build may need review.
    ed_vs_fci = float(abs(results["exact_energy"] - integrals["fci_energy"]))

    payload = {
        "schema": "qml.h2_vqe_benchmark.v1",
        "seed": BASELINE_SEED,
        "bond_length_A": args.bond_length,
        "basis_note": "STO-3G (from tutorial get_h2_integrals)",
        "pyscf_integrals": bool(integrals.get("from_pyscf", False)),
        "energies_hartree": {
            "vqe": results["vqe_energy"],
            "exact_diagonalization": results["exact_energy"],
            "fci_reference": integrals["fci_energy"],
            "hf": integrals["hf_energy"],
        },
        "errors_hartree": {
            "vqe_minus_exact": err_vs_exact,
            "vqe_minus_fci": err_vs_fci,
            "exact_ed_minus_fci": ed_vs_fci,
        },
        "checks": {
            "ed_matches_fci_within_1e-3": ed_vs_fci < 1e-3,
            "vqe_matches_fci_within_1e-2": err_vs_fci < 1e-2,
            "note": (
                "If exact_ed_minus_fci is large, compare tutorial Hamiltonian to PySCF; "
                "install pyscf for consistent integrals."
            ),
        },
        "wall_time_seconds": elapsed,
        "n_iterations": int(results.get("n_iterations", 0)),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("Wrote", args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
