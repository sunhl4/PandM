"""
CCSD and CCSD(T) solvers (PySCF).

Registered as ``"ccsd"`` and ``"ccsd_t"`` in the solver registry.
"""

from __future__ import annotations

from quantum_chem_bench.core.interfaces import BaseSolver, MethodResult, MolSpec
from quantum_chem_bench.core.registry import registry


@registry.register("ccsd", category="solver")
class CCSDSolver(BaseSolver):
    """
    Coupled Cluster Singles and Doubles (CCSD).

    Size-consistent and size-extensive. Gold standard for weakly correlated
    systems at moderate cost O(N^6).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def solve(self, mol_spec: MolSpec) -> MethodResult:
        try:
            from pyscf import gto, scf, cc
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

        mycc = cc.CCSD(mf)
        mycc.kernel()

        energy = float(mycc.e_tot)
        e_corr = float(mycc.e_corr)
        return MethodResult(
            method_name="CCSD",
            energy=energy,
            corr_energy=e_corr,
            converged=bool(mycc.converged),
            n_qubits=None,
            wall_time=self._elapsed(t0),
            extra={"e_corr": e_corr, "t1_norm": float((mycc.t1**2).sum()**0.5)},
        )


@registry.register("ccsd_t", category="solver")
class CCSDTSolver(BaseSolver):
    """
    CCSD with perturbative triples correction CCSD(T).

    Often called the "gold standard" of quantum chemistry. O(N^7) cost.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def solve(self, mol_spec: MolSpec) -> MethodResult:
        try:
            from pyscf import gto, scf, cc
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

        mycc = cc.CCSD(mf)
        mycc.kernel()

        # Perturbative triples
        e_t = mycc.ccsd_t()

        energy = float(mycc.e_tot + e_t)
        e_corr = float(mycc.e_corr + e_t)
        return MethodResult(
            method_name="CCSD(T)",
            energy=energy,
            corr_energy=e_corr,
            converged=bool(mycc.converged),
            n_qubits=None,
            wall_time=self._elapsed(t0),
            extra={
                "e_ccsd": float(mycc.e_corr),
                "e_triples": float(e_t),
            },
        )
