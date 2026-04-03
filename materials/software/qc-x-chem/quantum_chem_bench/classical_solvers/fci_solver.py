"""
FCI solver — exact diagonalization via PySCF.

Registered as ``"fci"`` in the solver registry.

FCI is exponentially expensive but serves as the exact reference for
benchmarking all other methods.
"""

from __future__ import annotations

import numpy as np

from quantum_chem_bench.core.interfaces import BaseSolver, MethodResult, MolSpec
from quantum_chem_bench.core.registry import registry
from quantum_chem_bench.molecule.builder import MoleculeBuilder


@registry.register("fci", category="solver")
class FCISolver(BaseSolver):
    """
    Full Configuration Interaction (FCI) — exact diagonalization.

    Uses PySCF's FCI solver in the active space defined by ``MolSpec``.
    For large active spaces this may be very slow or run out of memory;
    use with caution beyond ~16 spatial orbitals.
    """

    def __init__(self, nroots: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.nroots = nroots

    def solve(self, mol_spec: MolSpec) -> MethodResult:
        try:
            from pyscf import fci as pyscf_fci
        except ImportError as exc:
            raise ImportError("PySCF required: pip install pyscf") from exc

        t0 = self._start_timer()

        # Build active-space integrals
        builder = MoleculeBuilder(verbose=0)
        integrals = builder.build(mol_spec)

        # Run FCI
        cisolver = pyscf_fci.FCI(integrals.mol, integrals.mf.mo_coeff)
        cisolver.nroots = self.nroots

        e_fci, fcivec = cisolver.kernel(
            integrals.h1e,
            integrals.h2e,
            integrals.norb,
            integrals.nelec,
            ecore=integrals.e_core,
        )

        # Ground state energy
        if self.nroots == 1:
            energy = float(e_fci)
        else:
            energy = float(e_fci[0])

        e_corr = energy - integrals.hf_energy

        # Optionally compute 1-RDM
        rdm1 = None
        try:
            if self.nroots == 1:
                rdm1 = cisolver.make_rdm1(fcivec, integrals.norb, integrals.nelec)
        except Exception:  # noqa: BLE001
            pass

        extra: dict = {"hf_energy": integrals.hf_energy}
        if rdm1 is not None:
            extra["rdm1_trace"] = float(np.trace(rdm1))

        return MethodResult(
            method_name="FCI",
            energy=energy,
            corr_energy=float(e_corr),
            converged=True,
            n_qubits=None,
            wall_time=self._elapsed(t0),
            extra=extra,
        )
