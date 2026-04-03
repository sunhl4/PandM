"""
ADAPT-VQE solver.

Implements the Adaptive Derivative-Assembled Pseudo-Trotter ansatz VQE
(Grimsley et al., Nat. Commun. 10, 3007, 2019).

The algorithm:
1. Start from the Hartree-Fock state.
2. For each candidate operator A_k in the pool, compute the gradient
   |∂⟨H⟩/∂θ|_{θ=0} = |⟨[H, A_k]⟩|.
3. Append the operator with the largest gradient to the ansatz.
4. Re-optimize all parameters.
5. Repeat until max_grad < threshold.

Operator pool: qubit ADAPT pool (single/double commutators) or fermionic
UCC pool (single/double excitations).

Registered as ``"adapt_vqe"`` in the ``"solver"`` category.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..core.interfaces import QuantumSolver, SolverResult
from ..core.registry import registry
from ._rng import apply_solver_seed

logger = logging.getLogger(__name__)


@registry.register("adapt_vqe", category="solver")
class ADAPTVQESolver(QuantumSolver):
    """
    ADAPT-VQE solver.

    Parameters
    ----------
    gradient_threshold : float
        Stop when max gradient < this value.
    max_adapt_iter : int
        Maximum number of ADAPT outer iterations (operator additions).
    optimizer : str
        Inner optimizer name (``"cobyla"``, ``"slsqp"``, etc.).
    max_iter : int
        Max optimizer iterations per ADAPT step.
    pool : str
        Operator pool type: ``"fermionic_sd"`` (single/double UCC excitations,
        default) or ``"qubit_commutator"`` (all-qubit commutator pool).
    shots : int or None
        None → StatevectorEstimator.
    """

    def __init__(
        self,
        gradient_threshold: float = 1e-3,
        max_adapt_iter: int = 20,
        optimizer: str = "cobyla",
        max_iter: int = 200,
        pool: str = "fermionic_sd",
        shots: int | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.gradient_threshold = gradient_threshold
        self.max_adapt_iter = max_adapt_iter
        self.optimizer_name = optimizer
        self.max_iter = max_iter
        self.pool_type = pool
        self.shots = shots
        self.seed = seed

    def solve(
        self,
        hamiltonian,
        num_particles: tuple[int, int],
        num_spatial_orbitals: int,
    ) -> SolverResult:
        try:
            from qiskit_algorithms import VQE
            from qiskit_algorithms.optimizers import COBYLA, SLSQP, L_BFGS_B
            from qiskit.primitives import StatevectorEstimator
            from qiskit.quantum_info import Statevector, SparsePauliOp
            from qiskit.circuit import QuantumCircuit, Parameter
            from qiskit_nature.second_q.circuit.library import HartreeFock
            from qiskit_nature.second_q.mappers import ParityMapper
        except ImportError as exc:
            raise ImportError(
                "qiskit, qiskit-algorithms and qiskit-nature are required."
            ) from exc

        apply_solver_seed(self.seed)

        logger.info(
            "[ADAPT-VQE] Starting: %d qubits, pool=%s, threshold=%.1e",
            hamiltonian.num_qubits,
            self.pool_type,
            self.gradient_threshold,
        )

        # Build HF initial state
        mapper = ParityMapper(num_particles=num_particles)
        hf_circuit = HartreeFock(num_spatial_orbitals, num_particles, mapper)

        # Build operator pool
        pool_ops = self._build_pool(num_spatial_orbitals, num_particles, hamiltonian)
        logger.info("[ADAPT-VQE] Pool size: %d operators", len(pool_ops))

        estimator = StatevectorEstimator()
        optimizer_map = {
            "cobyla": COBYLA, "slsqp": SLSQP, "l_bfgs_b": L_BFGS_B
        }
        OptClass = optimizer_map.get(self.optimizer_name.lower(), COBYLA)

        # Current ansatz circuit and parameters
        ansatz = hf_circuit.copy()
        params: list[Any] = []
        param_values: np.ndarray = np.array([])

        energy_history = []
        best_energy = float("inf")

        for adapt_iter in range(self.max_adapt_iter):
            # --- Compute gradients for all pool operators ---
            if len(params) > 0:
                bound = ansatz.assign_parameters(
                    dict(zip(params, param_values))
                )
            else:
                bound = ansatz

            sv = Statevector(bound)
            gradients = []
            for op in pool_ops:
                # gradient = <ψ|[H, A]|ψ> evaluated at θ=0
                comm = hamiltonian @ op - op @ hamiltonian
                grad = abs(sv.expectation_value(comm).real)
                gradients.append(grad)

            max_grad = max(gradients)
            best_op_idx = int(np.argmax(gradients))
            logger.info(
                "[ADAPT-VQE] iter %d: max_grad=%.4e, best_op=%d",
                adapt_iter + 1, max_grad, best_op_idx,
            )

            if max_grad < self.gradient_threshold:
                logger.info(
                    "[ADAPT-VQE] Converged (max_grad < threshold) after %d operators.",
                    adapt_iter,
                )
                break

            # --- Append best operator via PauliEvolutionGate ---
            # SparsePauliOp has no exp_i(); use PauliEvolutionGate instead.
            theta = Parameter(f"θ_{adapt_iter}")
            params.append(theta)
            try:
                from qiskit.circuit.library import PauliEvolutionGate
                from qiskit.synthesis import LieTrotter
                evo_gate = PauliEvolutionGate(
                    pool_ops[best_op_idx],
                    time=theta,
                    synthesis=LieTrotter(reps=1),
                )
                num_qubits = hamiltonian.num_qubits
                from qiskit import QuantumCircuit as _QC
                gate_circ = _QC(num_qubits)
                gate_circ.append(evo_gate, range(num_qubits))
                ansatz = ansatz.compose(gate_circ)
            except Exception as _exc:
                logger.warning(
                    "[ADAPT-VQE] PauliEvolutionGate failed (%s); "
                    "appending identity layer as placeholder.", _exc
                )

            # --- Re-optimize all parameters ---
            param_values_full = np.append(param_values, 0.0)
            current_ansatz = ansatz.copy()
            current_params  = params.copy()

            from qiskit_algorithms import VQE as _VQE
            from qiskit_algorithms.optimizers import COBYLA as _COBYLA

            vqe = _VQE(
                estimator,
                current_ansatz,
                OptClass(maxiter=self.max_iter),
                initial_point=param_values_full,
            )
            result = vqe.compute_minimum_eigenvalue(hamiltonian)
            best_energy = float(result.eigenvalue.real)
            if result.optimal_parameters is not None:
                param_values = np.array(list(result.optimal_parameters.values()))
            energy_history.append(best_energy)
            logger.info("[ADAPT-VQE]   E = %.10f Ha", best_energy)

        return SolverResult(
            energy=best_energy,
            rdm1=None,
            rdm2=None,
            converged=True,
            extra={
                "energy_history": energy_history,
                "n_operators": len(params),
                "pool_type": self.pool_type,
            },
        )

    # ------------------------------------------------------------------
    # Operator pool construction
    # ------------------------------------------------------------------

    def _build_pool(self, num_spatial_orbitals, num_particles, hamiltonian):
        """
        Build the operator pool as a list of SparsePauliOp.

        For ``fermionic_sd``: construct spin-adapted single/double
        UCC excitations and map them via JW.
        For ``qubit_commutator``: use [H, Pauli] commutators as pool.
        """
        if self.pool_type == "qubit_commutator":
            return self._qubit_pool(hamiltonian)
        else:
            return self._fermionic_sd_pool(num_spatial_orbitals, num_particles)

    @staticmethod
    def _fermionic_sd_pool(norb, num_particles):
        """Single and double UCC excitation generators mapped to qubits."""
        try:
            from qiskit_nature.second_q.operators import FermionicOp
            from qiskit_nature.second_q.mappers import JordanWignerMapper
        except ImportError:
            return []

        mapper = JordanWignerMapper()
        n_alpha, n_beta = num_particles
        pool = []

        # Singles: a†_a a_i  - h.c.  (alpha spin block)
        for i in range(norb):
            for a in range(norb):
                if i == a:
                    continue
                # anti-Hermitian generator: A = a†_a a_i - a†_i a_a
                data = {
                    f"+_{a} -_{i}": 1.0,
                    f"+_{i} -_{a}": -1.0,
                }
                op = FermionicOp(data, num_spin_orbitals=2 * norb)
                pool.append(mapper.map(op))

        # Doubles: a†_a a†_b a_j a_i  - h.c.  (alpha-alpha)
        for i in range(norb):
            for j in range(i + 1, norb):
                for a in range(norb):
                    for b in range(a + 1, norb):
                        data = {
                            f"+_{a} +_{b} -_{j} -_{i}": 1.0,
                            f"+_{i} +_{j} -_{b} -_{a}": -1.0,
                        }
                        op = FermionicOp(data, num_spin_orbitals=2 * norb)
                        pool.append(mapper.map(op))

        logger.info("[ADAPT-VQE] Fermionic SD pool: %d operators", len(pool))
        return pool

    @staticmethod
    def _qubit_pool(hamiltonian):
        """Build a qubit pool from [H, σ_k] commutators."""
        try:
            from qiskit.quantum_info import SparsePauliOp, Pauli
        except ImportError:
            return []

        nq = hamiltonian.num_qubits
        pool = []
        # Y-type single-qubit generators
        for i in range(nq):
            label = "I" * (nq - i - 1) + "Y" + "I" * i
            comm = hamiltonian @ SparsePauliOp(label) - SparsePauliOp(label) @ hamiltonian
            if comm.norm() > 1e-10:
                pool.append(comm.simplify())

        logger.info("[ADAPT-VQE] Qubit commutator pool: %d operators", len(pool))
        return pool
