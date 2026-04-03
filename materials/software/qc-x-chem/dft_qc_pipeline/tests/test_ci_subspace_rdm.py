"""Subspace 1-RDM from CI coefficients (Slater basis)."""

from __future__ import annotations

import numpy as np

from dft_qc_pipeline.quantum_solvers.ci_subspace_rdm import (
    subspace_1rdm_spatial,
    _alpha_adag_a_matrix,
)


def test_single_determinant_double_occ_orb0() -> None:
    norb = 2
    ia = 1  # alpha at orbital 0
    ib = 1  # beta at orbital 0
    c = np.array([1.0])
    g = subspace_1rdm_spatial(c, [ia], [ib], norb)
    assert np.allclose(g, np.diag([2.0, 0.0]))


def test_equal_superposition_two_double_occupancies() -> None:
    """|Ψ⟩ = (|00 doubly occ⟩ + |11 doubly occ⟩)/√2 on 2 spatial orbitals."""
    norb = 2
    ia0, ib0 = 1, 1  # both spins on orb 0
    ia1, ib1 = 2, 2  # both spins on orb 1
    c = np.array([1.0, 1.0]) / np.sqrt(2.0)
    g = subspace_1rdm_spatial(c, [ia0, ia1], [ib0, ib1], norb)
    assert np.allclose(np.diag(g), [1.0, 1.0])
    assert np.isclose(np.trace(g), 2.0)


def test_alpha_single_excitation_off_diagonal() -> None:
    """Two determinants: same beta string, alpha differs by one swap → γ has off-diagonal."""
    norb = 2
    ib = 0  # empty beta (unphysical for 2e but tests alpha block in isolation)
    ia0, ia1 = 1, 2  # alpha at 0 vs alpha at 1
    c = np.array([1.0, 1.0]) / np.sqrt(2.0)
    g = subspace_1rdm_spatial(c, [ia0, ia1], [ib, ib], norb)
    assert np.allclose(g, g.T)
    assert max(abs(g[0, 1]), abs(g[1, 0])) > 1e-8


def test_alpha_adag_a_diagonal() -> None:
    m = _alpha_adag_a_matrix(0b101, 0b101, 3)
    assert np.allclose(np.diag(m), [1.0, 0.0, 1.0])
