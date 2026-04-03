"""Orbital localizer (scheme ``none`` avoids PySCF)."""

from __future__ import annotations

import numpy as np

from dft_qc_pipeline.hamiltonian.localizer import localize_orbitals


def test_localize_none_returns_same_coefficients() -> None:
    class FakeMF:
        mo_coeff = np.arange(12, dtype=float).reshape(4, 3)
        mo_occ = np.array([2.0, 2.0, 0.0])

    out = localize_orbitals(FakeMF(), "none")
    assert np.array_equal(out, FakeMF.mo_coeff)
