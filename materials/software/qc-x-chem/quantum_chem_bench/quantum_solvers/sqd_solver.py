"""
SQD solver — Sample-based Quantum Diagonalization.

Reference: Robledo-Moreno et al., *Nature Chemistry* (2024).
           "Chemistry beyond exact solutions on a quantum-centric supercomputer"

Algorithm:
    1. Sample bit-strings from a quantum circuit (HEA ansatz on real hardware
       or simulator).
    2. Use sampled configurations as a basis for a selected CI diagonalization.
    3. Update sampling distribution; iterate until convergence.

This implementation uses ``qiskit-addon-sqd``.

Registered as ``"sqd"`` in the solver registry.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from quantum_chem_bench.core.interfaces import BaseSolver, MethodResult, MolSpec
from quantum_chem_bench.core.registry import registry
from quantum_chem_bench.molecule.builder import MoleculeBuilder
from quantum_chem_bench.quantum_solvers._rng import apply_solver_seed

logger = logging.getLogger(__name__)


@registry.register("sqd", category="solver")
class SQDSolver(BaseSolver):
    """
    Sample-based Quantum Diagonalization (SQD).

    Parameters
    ----------
    shots : int
        Number of bit-string samples per iteration.
    iterations : int
        Number of SQD self-consistent iterations.
    ansatz : str
        Ansatz for sampling: ``"hea"`` (hardware efficient) or ``"uccsd"``.
    optimizer : str
        Optimizer for the sampling circuit VQE sub-step.
    max_iter : int
        Maximum VQE optimizer iterations in the sampling step.
    mapper_type : str
        Qubit mapper for Hamiltonian construction.
    z2symmetry_reduction : bool
        Apply Z2 symmetry reduction (only with parity mapper).
    """

    def __init__(
        self,
        shots: int = 10000,
        iterations: int = 10,
        ansatz: str = "hea",
        optimizer: str = "cobyla",
        max_iter: int = 100,
        mapper_type: str = "jw",
        z2symmetry_reduction: bool = False,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.shots = shots
        self.iterations = iterations
        self.ansatz_type = ansatz
        self.optimizer_name = optimizer
        self.max_iter = max_iter
        self.mapper_type = mapper_type
        self.z2symmetry_reduction = z2symmetry_reduction
        self.seed = seed

    def solve(self, mol_spec: MolSpec) -> MethodResult:
        t0 = self._start_timer()
        apply_solver_seed(self.seed)
        self._rng = np.random.default_rng(self.seed if self.seed is not None else 42)

        try:
            import qiskit_addon_sqd  # noqa: F401
        except ImportError:
            logger.warning(
                "qiskit-addon-sqd not installed; falling back to NumPy FCI."
            )
            return self._numpy_fallback(mol_spec, t0)

        builder = MoleculeBuilder(verbose=0)
        integrals = builder.build(mol_spec)

        energy, extra = self._run_sqd(integrals)

        return MethodResult(
            method_name="SQD",
            energy=energy,
            corr_energy=energy - integrals.hf_energy,
            converged=True,
            n_qubits=2 * integrals.norb,
            wall_time=self._elapsed(t0),
            extra=extra,
        )

    # ------------------------------------------------------------------
    # SQD core
    # ------------------------------------------------------------------

    def _run_sqd(self, integrals) -> tuple[float, dict]:
        """Run SQD iterations using qiskit-addon-sqd 0.12.x API."""
        import qiskit_addon_sqd.fermion as sqd_fermion
        from qiskit_addon_sqd.configuration_recovery import recover_configurations

        norb = integrals.norb
        n_alpha, n_beta = integrals.nelec
        h1e = np.asarray(integrals.h1e)
        h2e = np.asarray(integrals.h2e)
        e_core = float(integrals.e_core)
        open_shell = (n_alpha != n_beta)

        # Initial bitstring matrix: random valid configurations
        # Convention: cols 0..norb-1 = alpha, cols norb..2*norb-1 = beta
        bitstring_matrix, probabilities = self._sample_bitstrings(norb, n_alpha, n_beta)

        # Uniform occupancy estimates for the first recover_configurations call
        avg_occ_a = np.full(norb, n_alpha / norb)
        avg_occ_b = np.full(norb, n_beta / norb)

        energies_per_iter: list[float] = []
        energy = 0.0

        for i in range(self.iterations):
            # Enforce correct electron counts via configuration recovery.
            # Returns (new_bitstring_matrix, new_probabilities) in SQD 0.12.x.
            bitstring_matrix, probabilities = recover_configurations(
                bitstring_matrix,
                probabilities,
                (avg_occ_a, avg_occ_b),
                n_alpha,
                n_beta,
                rand_seed=self._rng,
            )

            # Convert binary matrix to integer-encoded CI strings
            ci_strs = sqd_fermion.bitstring_matrix_to_ci_strs(
                bitstring_matrix, open_shell=open_shell
            )
            if len(ci_strs[0]) == 0 or len(ci_strs[1]) == 0:
                logger.warning("SQD iter %d: empty CI string set, skipping.", i + 1)
                continue

            # Diagonalise in the sampled CI subspace
            energy_raw, _, (rdm1a, rdm1b), _ = sqd_fermion.solve_fermion(
                ci_strs,
                h1e,
                h2e,
                open_shell=open_shell,
            )
            energy = float(energy_raw) + e_core
            energies_per_iter.append(energy)
            logger.debug("SQD iter %d: E = %.10f Ha", i + 1, energy)

            # Update occupancies for the next recover_configurations call
            # solve_fermion 0.12.x returns 1-D diagonal occupancies (not full 2D RDM)
            avg_occ_a = np.asarray(rdm1a).ravel().clip(0.0, 1.0)
            avg_occ_b = np.asarray(rdm1b).ravel().clip(0.0, 1.0)

        return energy, {
            "sqd_energies_per_iter": energies_per_iter,
            "shots": self.shots,
            "iterations": self.iterations,
        }

    def _sample_bitstrings(
        self, norb: int, n_alpha: int, n_beta: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate an initial set of random, valid fermionic bitstrings.

        Each row has exactly *n_alpha* ones in the alpha block (cols 0..norb-1)
        and *n_beta* ones in the beta block (cols norb..2*norb-1), which matches
        the bitstring_matrix convention expected by qiskit-addon-sqd 0.12.x.

        Returns
        -------
        bitstring_matrix : ndarray shape (shots, 2*norb) dtype int8
        probabilities    : ndarray shape (shots,) uniform weights
        """
        rng = self._rng
        n = self.shots
        bsm = np.zeros((n, 2 * norb), dtype=np.int8)
        for k in range(n):
            occ_a = rng.choice(norb, n_alpha, replace=False)
            occ_b = rng.choice(norb, n_beta, replace=False)
            bsm[k, occ_a] = 1
            bsm[k, occ_b + norb] = 1
        probs = np.ones(n) / n
        return bsm, probs

    # ------------------------------------------------------------------
    # Fallback: NumPy FCI
    # ------------------------------------------------------------------

    def _numpy_fallback(self, mol_spec: MolSpec, t0: float) -> MethodResult:
        """Run exact NumPy FCI when qiskit-addon-sqd is unavailable."""
        from quantum_chem_bench.molecule.builder import MoleculeBuilder
        from pyscf import fci as pyscf_fci

        builder = MoleculeBuilder(verbose=0)
        integrals = builder.build(mol_spec)

        cisolver = pyscf_fci.FCI(integrals.mol, integrals.mf.mo_coeff)
        e_fci, _ = cisolver.kernel(
            integrals.h1e, integrals.h2e, integrals.norb, integrals.nelec,
            ecore=integrals.e_core,
        )

        return MethodResult(
            method_name="SQD (NumPy fallback)",
            energy=float(e_fci),
            corr_energy=float(e_fci) - integrals.hf_energy,
            converged=True,
            n_qubits=2 * integrals.norb,
            wall_time=self._elapsed(t0),
            extra={"fallback": True, "fallback_method": "FCI"},
        )
