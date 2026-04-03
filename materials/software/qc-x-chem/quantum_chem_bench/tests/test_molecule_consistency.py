"""
HF energy consistency: HFSolver vs MoleculeBuilder must agree for the same MolSpec.

Guarantees that classical ``hf`` and integral paths use the same PySCF mean-field.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import quantum_chem_bench.classical_solvers  # noqa: F401
import quantum_chem_bench.quantum_solvers  # noqa: F401

from quantum_chem_bench.core.interfaces import MolSpec
from quantum_chem_bench.core.registry import registry
from quantum_chem_bench.molecule.builder import MoleculeBuilder


@pytest.mark.requires_pyscf
def test_hf_solver_matches_molecule_builder_energy() -> None:
    spec = MolSpec(
        geometry="H 0 0 0; H 0 0 0.735",
        basis="sto-3g",
        n_active_electrons=(1, 1),
        n_active_orbitals=2,
    )
    hf = registry.build("hf", category="solver").solve(spec)
    integrals = MoleculeBuilder(verbose=0).build(spec)
    assert abs(hf.energy - integrals.hf_energy) < 1e-10


@pytest.mark.requires_pyscf
def test_hf_matches_full_space_molspec() -> None:
    """When active space is omitted, full orbital space; HF still matches builder."""
    spec = MolSpec(geometry="H 0 0 0; H 0 0 0.735", basis="sto-3g")
    hf = registry.build("hf", category="solver").solve(spec)
    integrals = MoleculeBuilder(verbose=0).build(spec)
    assert abs(hf.energy - integrals.hf_energy) < 1e-10
