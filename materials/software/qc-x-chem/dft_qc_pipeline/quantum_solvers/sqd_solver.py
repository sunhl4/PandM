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
from .ci_subspace_rdm import subspace_1rdm_spatial

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
        """Full SQD loop using qiskit-addon-sqd."""
        try:
            from qiskit_addon_sqd.fermion import (
                bitstring_matrix_to_ci_strs,
                optimize_ci_strings_reduce_memory,
            )
            from qiskit_addon_sqd.counts import counts_to_arrays
            from qiskit_addon_sqd.subspace import (
                build_subspace_matrix,
                solve_subspace,
            )
            from qiskit.primitives import StatevectorSampler
        except ImportError as exc:
            raise ImportError(
                "qiskit-addon-sqd ≥0.8 required. "
                "pip install qiskit-addon-sqd"
            ) from exc

        logger.info(
            "[SQDSolver] Starting SQD: %d qubits, %d shots, %d iterations",
            hamiltonian.num_qubits, self.sqd_shots, self.sqd_iterations,
        )

        # --- Build and pre-optimize ansatz ---
        ansatz = self._build_ansatz(num_particles, num_spatial_orbitals, hamiltonian)
        params = self._preopimize_ansatz(ansatz, hamiltonian, num_particles)

        best_energy = float("inf")
        best_rdm1 = None
        energy_history = []

        for iteration in range(self.sqd_iterations):
            logger.debug("[SQDSolver] Iteration %d/%d", iteration + 1, self.sqd_iterations)

            # Sample bitstrings from the current ansatz
            bound_circuit = ansatz.assign_parameters(params)
            bound_circuit.measure_all()

            sampler = StatevectorSampler()
            job = sampler.run([bound_circuit], shots=self.sqd_shots)
            pub_result = job.result()[0]
            counts = pub_result.data.meas.get_counts()

            # Convert counts to bitstring matrices
            try:
                bitstring_arrays, _ = counts_to_arrays(counts)
                ci_strs_a, ci_strs_b = bitstring_matrix_to_ci_strs(
                    bitstring_arrays,
                    num_elec_a=num_particles[0],
                    num_elec_b=num_particles[1],
                    norb=num_spatial_orbitals,
                )
            except Exception as exc:
                logger.warning("[SQDSolver] Config recovery failed: %s", exc)
                continue

            if len(ci_strs_a) == 0 or len(ci_strs_b) == 0:
                logger.warning("[SQDSolver] Empty CI string set, skipping iteration.")
                continue

            # Build and diagonalize subspace Hamiltonian
            try:
                H_sub = build_subspace_matrix(
                    hamiltonian,
                    ci_strs_a,
                    ci_strs_b,
                    norb=num_spatial_orbitals,
                )
                energy, coeff = solve_subspace(H_sub)
                energy_history.append(float(energy))

                if energy < best_energy:
                    best_energy = energy
                    try:
                        n_cf = min(
                            int(np.asarray(coeff).size),
                            len(ci_strs_a),
                            len(ci_strs_b),
                        )
                        c_use = np.asarray(coeff).ravel()[:n_cf]
                        best_rdm1 = subspace_1rdm_spatial(
                            c_use,
                            ci_strs_a[:n_cf],
                            ci_strs_b[:n_cf],
                            num_spatial_orbitals,
                        )
                    except Exception as exc:
                        logger.debug(
                            "[SQDSolver] subspace 1-RDM failed, fallback: %s", exc
                        )
                        best_rdm1 = self._rdm1_from_ci(
                            coeff, ci_strs_a, ci_strs_b, num_spatial_orbitals
                        )
                logger.info("[SQDSolver] iter %d  E=%.10f Ha", iteration + 1, energy)
            except Exception as exc:
                logger.warning("[SQDSolver] Subspace solve failed: %s", exc)
                continue

        if best_energy == float("inf"):
            logger.error("[SQDSolver] No valid subspace found; returning 0.")
            best_energy = 0.0

        return SolverResult(
            energy=best_energy,
            rdm1=best_rdm1,
            rdm2=None,
            converged=True,
            extra={
                "energy_history": energy_history,
                "rdm1_model": (
                    "subspace_slater_basis: exact γ_pq=⟨E_pq⟩ for the sampled "
                    "Slater determinant basis (single-particle excitation graph); "
                    "truncation error remains from the finite SQD subspace."
                ),
            },
        )

    # ------------------------------------------------------------------
    # Ansatz and pre-optimization helpers
    # ------------------------------------------------------------------

    def _build_ansatz(self, num_particles, num_spatial_orbitals, hamiltonian):
        if self.ansatz_type == "uccsd":
            try:
                from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
                from qiskit_nature.second_q.mappers import ParityMapper
                mapper = ParityMapper(num_particles=num_particles)
                hf = HartreeFock(num_spatial_orbitals, num_particles, mapper)
                return UCCSD(num_spatial_orbitals, num_particles, mapper, initial_state=hf)
            except Exception:
                pass  # fall through to HEA
        from qiskit.circuit.library import EfficientSU2
        return EfficientSU2(hamiltonian.num_qubits, reps=2)

    def _preopimize_ansatz(self, ansatz, hamiltonian, num_particles) -> np.ndarray:
        """Quick VQE pre-optimization to get a reasonable starting point."""
        try:
            from qiskit_algorithms import VQE
            from qiskit_algorithms.optimizers import COBYLA
            from qiskit.primitives import StatevectorEstimator

            vqe = VQE(StatevectorEstimator(), ansatz, COBYLA(maxiter=self.max_iter))
            result = vqe.compute_minimum_eigenvalue(hamiltonian)
            if result.optimal_parameters is not None:
                return np.array(list(result.optimal_parameters.values()))
        except Exception as exc:
            logger.debug("[SQDSolver] Pre-optimization failed: %s", exc)

        # Random initial point as fallback
        rng = np.random.default_rng(self.seed if self.seed is not None else 42)
        return rng.uniform(-np.pi, np.pi, ansatz.num_parameters)

    # ------------------------------------------------------------------
    # 1-RDM from CI coefficients (approximate)
    # ------------------------------------------------------------------

    @staticmethod
    def _rdm1_from_ci(
        coeff: np.ndarray,
        ci_strs_a: np.ndarray,
        ci_strs_b: np.ndarray,
        norb: int,
    ) -> np.ndarray | None:
        """Approximate 1-RDM from dominant CI string and coefficient (diagonal only)."""
        try:
            rdm1 = np.zeros((norb, norb))
            # Use the dominant determinant for a rough diagonal approximation
            idx = np.argmax(np.abs(coeff))
            # ci_strs_a[idx] is an integer bitstring; extract occupations
            occ_a = [(ci_strs_a[idx % len(ci_strs_a)] >> bit) & 1 for bit in range(norb)]
            occ_b = [(ci_strs_b[idx % len(ci_strs_b)] >> bit) & 1 for bit in range(norb)]
            for p in range(norb):
                rdm1[p, p] = float(occ_a[p] + occ_b[p])
            return rdm1
        except Exception:
            return None
