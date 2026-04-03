"""
Abstract base classes for the quantum_chem_bench platform.

Every pluggable solver (classical or quantum) implements BaseSolver.
Data flows through three shared dataclasses:

    MolSpec  →  BaseSolver.solve()  →  MethodResult
    list[MethodResult]              →  BenchResult
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Shared data containers
# ---------------------------------------------------------------------------

@dataclass
class MolSpec:
    """
    Complete specification of a molecular problem to be solved.

    Parameters
    ----------
    geometry : str
        PySCF-style atom string, e.g. ``"H 0 0 0; H 0 0 0.735"``.
    basis : str
        Basis set name, e.g. ``"sto-3g"``, ``"6-31g"``, ``"cc-pVDZ"``.
    charge : int
        Total charge of the molecule.
    spin : int
        Spin multiplicity 2S (0 = singlet, 1 = doublet, 2 = triplet …).
    n_active_electrons : (int, int) or None
        (n_alpha, n_beta) in the active space.  If None the full space is used.
    n_active_orbitals : int or None
        Number of spatial orbitals in the active space.  If None full space.
    mapper_type : str
        Qubit mapper to use for quantum solvers: ``"jw"``, ``"parity"``, ``"bk"``.
    z2symmetry_reduction : bool
        Whether to apply Z2 symmetry reduction in qubit mapping.
    density_fit : bool
        If True, PySCF mean-field uses density fitting (RI/J) for larger bases.
    auxbasis : str or None
        Auxiliary basis for density fitting; None lets PySCF choose.
    """
    geometry: str
    basis: str
    charge: int = 0
    spin: int = 0
    n_active_electrons: tuple[int, int] | None = None
    n_active_orbitals: int | None = None
    mapper_type: str = "parity"
    z2symmetry_reduction: bool = True
    density_fit: bool = False
    auxbasis: str | None = None


@dataclass
class MethodResult:
    """
    Output from any solver (classical or quantum).

    Parameters
    ----------
    method_name : str
        Human-readable identifier, e.g. ``"CCSD"``, ``"VQE-UCCSD"``.
    energy : float
        Total electronic energy in Hartree.
    corr_energy : float
        Correlation energy relative to HF (0.0 for HF itself).
    converged : bool
        Whether the solver converged to its criterion.
    n_qubits : int or None
        Number of qubits used (None for classical methods).
    wall_time : float
        Wall-clock time in seconds.
    extra : dict
        Solver-specific extras (circuit depth, optimizer iterations, etc.).
    """
    method_name: str
    energy: float
    corr_energy: float
    converged: bool
    n_qubits: int | None
    wall_time: float
    extra: dict = field(default_factory=dict)


@dataclass
class BenchResult:
    """
    Aggregated results for one benchmark run (one molecule, multiple solvers).

    Parameters
    ----------
    mol_spec : MolSpec
        The molecule specification used for this run.
    results : dict[str, MethodResult]
        Mapping from method name to MethodResult.
    hf_energy : float
        Hartree-Fock reference energy (used to compute correlation energies).
    fci_energy : float or None
        FCI reference energy if available (used for accuracy assessment).
    """
    mol_spec: MolSpec
    results: dict[str, MethodResult] = field(default_factory=dict)
    hf_energy: float = 0.0
    fci_energy: float | None = None

    def add(self, result: MethodResult) -> None:
        """Add a MethodResult to the collection."""
        self.results[result.method_name] = result

    def summary_table(self) -> list[dict]:
        """Return list of dicts suitable for pandas DataFrame construction."""
        rows = []
        for name, r in self.results.items():
            error_vs_fci = None
            if self.fci_energy is not None:
                error_vs_fci = (r.energy - self.fci_energy) * 1000  # mHa
            rows.append({
                "Method": name,
                "Energy (Ha)": r.energy,
                "Corr. Energy (Ha)": r.corr_energy,
                "Error vs FCI (mHa)": error_vs_fci,
                "Converged": r.converged,
                "N Qubits": r.n_qubits,
                "Wall Time (s)": round(r.wall_time, 3),
            })
        return rows


# ---------------------------------------------------------------------------
# Molecule integral container (passed between molecule/ and solvers)
# ---------------------------------------------------------------------------

@dataclass
class MolIntegrals:
    """
    One-electron and two-electron integrals in active-space MO basis.

    Produced by ``molecule.builder.MoleculeBuilder`` and consumed by
    ``molecule.hamiltonian.HamiltonianBuilder`` and classical solvers.
    """
    h1e: np.ndarray               # shape (norb, norb)
    h2e: np.ndarray               # shape (norb, norb, norb, norb), chemist notation
    nelec: tuple[int, int]        # (n_alpha, n_beta)
    norb: int
    e_core: float                 # nuclear repulsion + frozen-core energy
    hf_energy: float              # full HF energy (for reference)
    mo_coeff: np.ndarray          # active MO coefficient matrix, shape (nao, norb)
    mol: Any                      # PySCF Mole object
    mf: Any                       # PySCF mean-field object


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseSolver(ABC):
    """
    Pluggable solver interface.

    All classical and quantum solvers must implement this interface.
    The ``solve`` method receives a ``MolSpec`` and returns a ``MethodResult``.

    Usage (with registry)::

        from quantum_chem_bench.core.registry import registry

        @registry.register("my_method", category="solver")
        class MySolver(BaseSolver):
            def solve(self, mol_spec: MolSpec) -> MethodResult:
                t0 = self._start_timer()
                energy = ...
                return MethodResult(
                    method_name="my_method",
                    energy=energy,
                    corr_energy=0.0,
                    converged=True,
                    n_qubits=None,
                    wall_time=self._elapsed(t0),
                )
    """

    def __init__(self, **kwargs: Any) -> None:
        self._cfg: dict[str, Any] = kwargs

    @abstractmethod
    def solve(self, mol_spec: MolSpec) -> MethodResult:
        """
        Solve the electronic structure problem defined by ``mol_spec``.

        Parameters
        ----------
        mol_spec : MolSpec
            Full molecular and active-space specification.

        Returns
        -------
        MethodResult
        """

    # ------------------------------------------------------------------
    # Convenience helpers for timing
    # ------------------------------------------------------------------

    @staticmethod
    def _start_timer() -> float:
        return time.perf_counter()

    @staticmethod
    def _elapsed(t0: float) -> float:
        return time.perf_counter() - t0
