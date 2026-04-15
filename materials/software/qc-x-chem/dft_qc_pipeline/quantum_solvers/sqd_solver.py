"""
SQD (Sample-based Quantum Diagonalization) solver.

Wraps the ``qiskit-addon-sqd`` library to perform:
1. Prepare a parametrized ansatz circuit and sample bitstrings.
2. Recover valid fermionic configurations from the bitstring histogram.
3. Project the Hamiltonian onto the sampled subspace and diagonalize classically.
4. Iterate until convergence.

Reference: Robledo-Moreno et al., Nature Chemistry (2024);
           IBM qiskit-addon-sqd documentation.

Registered as ``"sqd"`` in the ``"solver"`` category.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..core.interfaces import QuantumSolver, SolverResult
from ..core.registry import registry
from ._rng import apply_solver_seed

logger = logging.getLogger(__name__)


@registry.register("sqd", category="solver")
class SQDSolver(QuantumSolver):
    """
    Sample-based Quantum Diagonalization solver.

    Parameters
    ----------
    sqd_iterations : int
        Number of SQD outer iterations (sample → recover → diagonalize).
    sqd_shots : int
        Number of circuit shots per iteration.
    shots : int or None
        Alias for ``sqd_shots`` (for API symmetry with VQESolver).
    ansatz : str
        Ansatz used for sampling: ``"uccsd"`` or ``"hea"``.
    optimizer : str
        Classical optimizer name for pre-optimizing the ansatz.
    max_iter : int
        Max optimizer iterations for the pre-optimization step.
    n_batches : int
        Number of bitstring batches per SQD iteration.
    seed : int or None
        Optional RNG seed for pre-optimization fallback and global Qiskit/NumPy seeding.
    """

    def __init__(
        self,
        sqd_iterations: int = 10,
        sqd_shots: int = 10_000,
        shots: int | None = None,         # alias
        ansatz: str = "hea",
        optimizer: str = "cobyla",
        max_iter: int = 100,
        n_batches: int = 10,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.sqd_iterations = sqd_iterations
        self.sqd_shots = shots if shots is not None else sqd_shots
        self.ansatz_type = ansatz.lower()
        self.optimizer_name = optimizer.lower()
        self.max_iter = max_iter
        self.n_batches = n_batches
        self.seed = seed

    def solve(
        self,
        hamiltonian,
        num_particles: tuple[int, int],
        num_spatial_orbitals: int,
    ) -> SolverResult:
        apply_solver_seed(self.seed)
        # Try importing SQD addon
        try:
            import qiskit_addon_sqd  # noqa: F401
            _sqd_available = True
        except ImportError:
            _sqd_available = False

        if _sqd_available:
            return self._solve_with_sqd(
                hamiltonian, num_particles, num_spatial_orbitals
            )
        else:
            logger.warning(
                "qiskit-addon-sqd not installed. "
                "Falling back to NumPy exact diagonalization. "
                "Install with: pip install qiskit-addon-sqd"
            )
            from .numpy_solver import NumPySolver
            return NumPySolver().solve(hamiltonian, num_particles, num_spatial_orbitals)

    def _solve_with_sqd(
        self, hamiltonian, num_particles, num_spatial_orbitals
    ) -> SolverResult:
        """
        SQD using qiskit-addon-sqd 0.12.x API (solve_fermion with integrals).

        Requires that ``self._emb_H`` (EmbeddedHamiltonian) is set by the
        Pipeline before calling solve().  Without it we fall back to NumPy.
        """
        import qiskit_addon_sqd.fermion as sqd_fermion
        from qiskit_addon_sqd.configuration_recovery import recover_configurations

        emb_H = getattr(self, "_emb_H", None)
        if emb_H is None:
            logger.warning(
                "[SQDSolver] EmbeddedHamiltonian not available "
                "(set solver._emb_H before calling solve). "
                "Falling back to NumPy exact diagonalization."
            )
            from .numpy_solver import NumPySolver
            return NumPySolver().solve(hamiltonian, num_particles, num_spatial_orbitals)

        norb = int(emb_H.norb)
        n_alpha, n_beta = int(num_particles[0]), int(num_particles[1])
        h1e = np.asarray(emb_H.h1e, dtype=float)
        h2e = np.asarray(emb_H.h2e, dtype=float)
        e_core = float(emb_H.e_core)
        open_shell = (n_alpha != n_beta)

        rng = np.random.default_rng(self.seed if self.seed is not None else 42)

        # --- initial random bitstrings (cols 0..norb-1 = alpha, norb..2*norb-1 = beta) ---
        n_samples = self.sqd_shots
        bsm = np.zeros((n_samples, 2 * norb), dtype=np.int8)
        for k in range(n_samples):
            occ_a = rng.choice(norb, n_alpha, replace=False)
            occ_b = rng.choice(norb, n_beta, replace=False)
            bsm[k, occ_a] = 1
            bsm[k, occ_b + norb] = 1
        probs = np.ones(n_samples) / n_samples

        avg_occ_a = np.full(norb, n_alpha / norb)
        avg_occ_b = np.full(norb, n_beta / norb)

        best_energy = float("inf")
        best_rdm1: np.ndarray | None = None
        energy_history: list[float] = []

        for iteration in range(self.sqd_iterations):
            # Returns (new_bitstring_matrix, new_probabilities) in SQD 0.12.x.
            bsm, probs = recover_configurations(
                bsm, probs,
                (avg_occ_a, avg_occ_b),
                n_alpha, n_beta,
                rand_seed=rng,
            )

            ci_strs = sqd_fermion.bitstring_matrix_to_ci_strs(bsm, open_shell=open_shell)
            if len(ci_strs[0]) == 0 or len(ci_strs[1]) == 0:
                logger.warning("[SQDSolver] iter %d: empty CI strings, skipping.", iteration + 1)
                continue

            energy_raw, _, (rdm1a, rdm1b), _ = sqd_fermion.solve_fermion(
                ci_strs, h1e, h2e, open_shell=open_shell,
            )
            energy = float(energy_raw) + e_core
            energy_history.append(energy)
            logger.info("[SQDSolver] iter %d  E=%.10f Ha", iteration + 1, energy)

            if energy < best_energy:
                best_energy = energy
                # solve_fermion 0.12.x returns 1-D diagonal occupancies; build 2-D RDM
                occ_a = np.asarray(rdm1a).ravel()
                occ_b = np.asarray(rdm1b).ravel()
                best_rdm1 = np.diag(occ_a + occ_b)  # (norb, norb) diagonal RDM

            avg_occ_a = np.asarray(rdm1a).ravel().clip(0.0, 1.0)
            avg_occ_b = np.asarray(rdm1b).ravel().clip(0.0, 1.0)

        if best_energy == float("inf"):
            logger.error("[SQDSolver] No valid subspace found; returning e_core.")
            best_energy = e_core

        return SolverResult(
            energy=best_energy,
            rdm1=best_rdm1,
            rdm2=None,
            converged=True,
            extra={"energy_history": energy_history},
        )

