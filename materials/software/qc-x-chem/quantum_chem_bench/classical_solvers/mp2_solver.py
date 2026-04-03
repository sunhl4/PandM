"""
MP2 solver (PySCF).

Registered as ``"mp2"`` in the solver registry.
"""

from __future__ import annotations

from quantum_chem_bench.core.interfaces import BaseSolver, MethodResult, MolSpec
from quantum_chem_bench.core.registry import registry


@registry.register("mp2", category="solver")
class MP2Solver(BaseSolver):
    """
    Møller-Plesset second-order perturbation theory (MP2).

    Runs RHF first, then MP2 on top.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def solve(self, mol_spec: MolSpec) -> MethodResult:
        try:
            from pyscf import gto, scf, mp
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

        mp2 = mp.MP2(mf)
        e_corr, _ = mp2.kernel()

        energy = float(mf.e_tot + e_corr)
        return MethodResult(
            method_name="MP2",
            energy=energy,
            corr_energy=float(e_corr),
            converged=True,
            n_qubits=None,
            wall_time=self._elapsed(t0),
            extra={"e_corr": float(e_corr)},
        )
