"""
Hartree-Fock solver (PySCF RHF/ROHF).

Registered as ``"hf"`` in the solver registry.
"""

from __future__ import annotations

from quantum_chem_bench.core.interfaces import BaseSolver, MethodResult, MolSpec
from quantum_chem_bench.core.registry import registry


@registry.register("hf", category="solver")
class HFSolver(BaseSolver):
    """
    Hartree-Fock (RHF for closed-shell, ROHF for open-shell).

    As the mean-field reference, its correlation energy is defined as 0.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def solve(self, mol_spec: MolSpec) -> MethodResult:
        try:
            from pyscf import gto, scf
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
        if getattr(mol_spec, "density_fit", False):
            mf = mf.density_fit(auxbasis=mol_spec.auxbasis)
        mf.kernel()

        energy = float(mf.e_tot)
        return MethodResult(
            method_name="HF",
            energy=energy,
            corr_energy=0.0,
            converged=bool(mf.converged),
            n_qubits=None,
            wall_time=self._elapsed(t0),
            extra={"mo_energy": mf.mo_energy.tolist()},
        )
