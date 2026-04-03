"""
CISD solver (PySCF).

Registered as ``"cisd"`` in the solver registry.
"""

from __future__ import annotations

from quantum_chem_bench.core.interfaces import BaseSolver, MethodResult, MolSpec
from quantum_chem_bench.core.registry import registry


@registry.register("cisd", category="solver")
class CISDSolver(BaseSolver):
    """
    Configuration Interaction Singles and Doubles (CISD).

    Truncated CI: includes all single and double excitations from HF.
    Variational but not size-consistent.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def solve(self, mol_spec: MolSpec) -> MethodResult:
        try:
            from pyscf import gto, scf, ci
        except ImportError as exc:
            raise ImportError("PySCF required: pip install pyscf") from exc

        t0 = self._start_timer()

        mol = gto.Mole()
        mol.atom = mol_spec.geometry
        mol.basis = mol_spec.basis
        mol.charge = mol_spec.charge
        mol.spin = mol_spec.spin
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol) if mol_spec.spin == 0 else scf.ROHF(mol)
        mf.kernel()

        myci = ci.CISD(mf)
        myci.kernel()

        energy = float(myci.e_tot)
        e_corr = float(myci.e_corr)
        return MethodResult(
            method_name="CISD",
            energy=energy,
            corr_energy=e_corr,
            converged=bool(myci.converged),
            n_qubits=None,
            wall_time=self._elapsed(t0),
            extra={"e_corr": e_corr},
        )
