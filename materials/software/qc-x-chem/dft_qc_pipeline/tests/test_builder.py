"""Hamiltonian → FermionicOp (requires qiskit-nature)."""

from __future__ import annotations

import numpy as np
import pytest

from dft_qc_pipeline.core.interfaces import EmbeddedHamiltonian

pytest.importorskip("qiskit_nature")


def test_hamiltonian_to_fermionic_op_h2_like() -> None:
    from dft_qc_pipeline.hamiltonian.builder import hamiltonian_to_fermionic_op

    norb = 2
    h1e = np.array([[1.0, 0.1], [0.1, 1.0]], dtype=float)
    h2e = np.zeros((norb, norb, norb, norb), dtype=float)
    emb = EmbeddedHamiltonian(
        h1e=h1e,
        h2e=h2e,
        nelec=(1, 1),
        norb=norb,
        e_core=0.5,
        region_name="test",
    )
    fop, e_core = hamiltonian_to_fermionic_op(emb)
    assert e_core == 0.5
    assert getattr(fop, "num_spin_orbitals", None) == 2 * norb
