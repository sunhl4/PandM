"""
Abstract base classes for every pluggable layer in the DFT + quantum-embedding pipeline.

Hierarchy:
  ClassicalBackend  – runs HF/DFT and exposes integrals
  EmbeddingMethod   – builds an effective fragment Hamiltonian
  FragmentRegion    – (data class, lives in hamiltonian/fragment_region.py; imported here)
  QuantumSolver     – solves a qubit Hamiltonian and returns energy + 1-RDM
  QubitMapperWrapper – wraps Qiskit Nature mappers behind a uniform interface
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

# ---------------------------------------------------------------------------
# Shared data containers
# ---------------------------------------------------------------------------

@dataclass
class BackendResult:
    """Output produced by any ClassicalBackend."""
    mol: Any                     # PySCF Mole (or equivalent)
    energy_hf: float             # HF / DFT energy
    mo_coeff: np.ndarray         # MO coefficient matrix, shape (nao, nmo)
    mo_occ: np.ndarray           # Occupation numbers, shape (nmo,)
    mo_energy: np.ndarray        # Orbital energies, shape (nmo,)
    ovlp: np.ndarray             # AO overlap matrix, shape (nao, nao)
    h1e_ao: np.ndarray           # 1-electron integrals in AO basis
    h2e_ao: np.ndarray | None    # 2-electron integrals in AO basis (may be None for large systems)
    nelec: tuple[int, int]       # (n_alpha, n_beta)
    mf: Any                      # The underlying mean-field object (for further use)
    scf_converged: bool = True   # False if SCF did not meet conv_tol
    extra: dict = field(default_factory=dict)


@dataclass
class EmbeddedHamiltonian:
    """Fragment Hamiltonian ready to be mapped to qubits."""
    h1e: np.ndarray              # 1e integrals in fragment MO basis, shape (norb, norb)
    h2e: np.ndarray              # 2e integrals (chemist notation), shape (norb,)*4
    nelec: tuple[int, int]       # (n_alpha, n_beta) in fragment
    norb: int                    # number of spatial orbitals
    e_core: float = 0.0          # core/environment energy correction
    region_name: str = "fragment"
    extra: dict = field(default_factory=dict)


@dataclass
class SolverResult:
    """Output from any QuantumSolver."""
    energy: float                # Total fragment energy (including e_core)
    rdm1: np.ndarray | None      # 1-RDM in fragment MO basis, shape (norb, norb) or (2, norb, norb)
    rdm2: np.ndarray | None      # 2-RDM, shape (norb,)*4 or None
    converged: bool = True
    extra: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Aggregated results for all fragments."""
    total_energy: float
    fragment_results: dict[str, SolverResult]
    backend_result: BackendResult
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class ClassicalBackend(ABC):
    """
    Layer 1 – runs a classical quantum-chemistry calculation and exposes
    the mean-field orbitals and integrals needed downstream.
    """

    @abstractmethod
    def run(self, geometry: str, basis: str, **kwargs) -> BackendResult:
        """
        Execute HF/DFT and return a BackendResult.

        Parameters
        ----------
        geometry : str
            Atom specification in PySCF-style string, e.g. ``"H 0 0 0; H 0 0 0.735"``
        basis : str
            Basis set name, e.g. ``"sto-3g"``, ``"def2-SVP"``
        **kwargs :
            Backend-specific options (charge, spin, xc functional, etc.)
        """


class EmbeddingMethod(ABC):
    """
    Layer 2 – extracts an effective fragment Hamiltonian from a BackendResult.

    For one-shot methods (AVAS, simple CAS) ``update_from_rdm`` always returns
    ``True`` (converged).  For self-consistent methods (DMET) it updates the
    chemical potential and returns ``False`` until convergence.
    """

    @abstractmethod
    def embed(
        self,
        backend_result: BackendResult,
        region: "FragmentRegion",  # noqa: F821 – forward ref resolved at runtime
    ) -> EmbeddedHamiltonian:
        """Build and return the effective fragment Hamiltonian."""

    def update_from_rdm(self, rdm1: np.ndarray) -> bool:
        """
        Self-consistency update step (DMET, etc.).

        Parameters
        ----------
        rdm1 : np.ndarray
            1-RDM from the quantum solver in the fragment orbital basis.

        Returns
        -------
        bool
            ``True`` if converged (or one-shot method), ``False`` otherwise.
        """
        return True  # default: one-shot, always converged


class QuantumSolver(ABC):
    """
    Layer 5 – solves a qubit Hamiltonian (``SparsePauliOp``) and returns
    energy and optionally reduced density matrices.
    """

    @abstractmethod
    def solve(
        self,
        hamiltonian,            # qiskit.quantum_info.SparsePauliOp
        num_particles: tuple[int, int],
        num_spatial_orbitals: int,
    ) -> SolverResult:
        """
        Parameters
        ----------
        hamiltonian : SparsePauliOp
            The qubit Hamiltonian.
        num_particles : (int, int)
            (n_alpha, n_beta) in the active space.
        num_spatial_orbitals : int
            Number of spatial (alpha) orbitals in the active space.

        Returns
        -------
        SolverResult
        """


class QubitMapperWrapper(ABC):
    """
    Layer 4b – uniform interface around Qiskit Nature QubitMapper variants.
    """

    @abstractmethod
    def map(self, fermionic_op) -> Any:  # returns SparsePauliOp
        """Map a FermionicOp to a SparsePauliOp."""


class SupportsEmbeddedMap(Protocol):
    """Duck type for objects returned by ``hamiltonian.mappers.build_mapper``."""

    def map(
        self, emb_H: EmbeddedHamiltonian
    ) -> tuple[Any, tuple[int, int], int]:
        """Return ``(qubit_hamiltonian, num_particles, num_spatial_orbitals)``."""
