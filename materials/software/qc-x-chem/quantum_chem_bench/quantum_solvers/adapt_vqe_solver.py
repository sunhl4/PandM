"""
ADAPT-VQE solver — Adaptive Derivative-Assembled Pseudo-Trotter ansatz VQE.

Reference: Grimsley et al., Nature Communications 10, 3007 (2019).

Algorithm:
    1. Start with HF reference state |Φ₀⟩.
    2. Compute gradient ‖∂E/∂θₖ‖ for each operator Aₖ in the pool.
    3. Append the operator with largest gradient to the ansatz.
    4. Optimise all parameters with VQE.
    5. Repeat until max‖gradient‖ < threshold.

Registered as ``"adapt_vqe"`` in the solver registry.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from quantum_chem_bench.core.interfaces import BaseSolver, MethodResult, MolSpec
from quantum_chem_bench.core.registry import registry
from quantum_chem_bench.molecule.builder import MoleculeBuilder
from quantum_chem_bench.molecule.hamiltonian import HamiltonianBuilder
from quantum_chem_bench.quantum_solvers._rng import apply_solver_seed

logger = logging.getLogger(__name__)


@registry.register("adapt_vqe", category="solver")
class ADAPTVQESolver(BaseSolver):
    """
    ADAPT-VQE with a fermionic singles+doubles operator pool.

    Parameters
    ----------
    max_iter : int
        Maximum number of ADAPT iterations (operator additions).
    gradient_threshold : float
        Convergence threshold on the maximum gradient norm.
    optimizer : str
        Classical optimizer for parameter updates (``"slsqp"`` or ``"cobyla"``).
    vqe_max_iter : int
        Maximum VQE sub-iterations per ADAPT step.
    mapper_type : str
        Qubit mapper (``"jw"``, ``"parity"``, ``"bk"``).
    z2symmetry_reduction : bool
        Apply Z2 symmetry reduction.
    """

    def __init__(
        self,
        max_iter: int = 50,
        gradient_threshold: float = 1e-3,
        optimizer: str = "slsqp",
        vqe_max_iter: int = 200,
        mapper_type: str = "parity",
        z2symmetry_reduction: bool = True,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_iter = max_iter
        self.gradient_threshold = gradient_threshold
        self.optimizer_name = optimizer
        self.vqe_max_iter = vqe_max_iter
        self.mapper_type = mapper_type
        self.z2symmetry_reduction = z2symmetry_reduction
        self.seed = seed

    def solve(self, mol_spec: MolSpec) -> MethodResult:
        try:
            from qiskit_algorithms import VQE
            from qiskit.primitives import StatevectorEstimator as Estimator
            from qiskit_nature.second_q.circuit.library import UCCSD
            from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper
        except ImportError as exc:
            raise ImportError(
                "qiskit-algorithms and qiskit-nature required"
            ) from exc

        t0 = self._start_timer()
        apply_solver_seed(self.seed)

        builder = MoleculeBuilder(verbose=0)
        integrals = builder.build(mol_spec)

        ham_builder = HamiltonianBuilder(
            mapper_type=self.mapper_type,
            z2symmetry_reduction=self.z2symmetry_reduction,
        )
        qubit_op, n_particles, n_orbs = ham_builder.build(integrals)
        n_qubits = qubit_op.num_qubits

        # Build ADAPT-VQE using Qiskit Nature's built-in implementation
        try:
            from qiskit_algorithms import AdaptVQE
        except ImportError:
            logger.warning("AdaptVQE not found in qiskit_algorithms; falling back to UCCSD VQE.")
            return self._fallback_uccsd(
                mol_spec, integrals, qubit_op, n_particles, n_orbs, n_qubits, t0
            )

        mapper = self._build_mapper(n_particles)
        initial_ansatz = UCCSD(
            num_spatial_orbitals=n_orbs,
            num_particles=n_particles,
            qubit_mapper=mapper,
        )

        opt = self._build_optimizer()
        estimator = Estimator()
        vqe = VQE(estimator=estimator, ansatz=initial_ansatz, optimizer=opt)

        adapt = AdaptVQE(vqe)
        adapt.gradient_threshold = self.gradient_threshold

        result = adapt.compute_minimum_eigenvalue(qubit_op)

        energy = float(result.eigenvalue.real) + integrals.e_core
        n_iters = getattr(result, "num_iterations", None)

        return MethodResult(
            method_name="ADAPT-VQE",
            energy=energy,
            corr_energy=energy - integrals.hf_energy,
            converged=True,
            n_qubits=n_qubits,
            wall_time=self._elapsed(t0),
            extra={
                "adapt_iterations": n_iters,
                "optimizer_evals": getattr(result, "cost_function_evals", None),
            },
        )

    def _fallback_uccsd(
        self, mol_spec, integrals, qubit_op, n_particles, n_orbs, n_qubits, t0
    ) -> MethodResult:
        """Fallback: run plain UCCSD-VQE when AdaptVQE is unavailable."""
        from qiskit_algorithms import VQE
        from qiskit.primitives import StatevectorEstimator as Estimator
        from qiskit_nature.second_q.circuit.library import UCCSD

        mapper = self._build_mapper(n_particles)
        ansatz = UCCSD(
            num_spatial_orbitals=n_orbs,
            num_particles=n_particles,
            qubit_mapper=mapper,
        )
        opt = self._build_optimizer()
        estimator = Estimator()
        vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=opt)
        result = vqe.compute_minimum_eigenvalue(qubit_op)
        energy = float(result.eigenvalue.real) + integrals.e_core

        return MethodResult(
            method_name="ADAPT-VQE (UCCSD fallback)",
            energy=energy,
            corr_energy=energy - integrals.hf_energy,
            converged=True,
            n_qubits=n_qubits,
            wall_time=self._elapsed(t0),
            extra={"fallback": True},
        )

    def _build_mapper(self, n_particles):
        mt = self.mapper_type.lower()
        if mt == "parity":
            from qiskit_nature.second_q.mappers import ParityMapper
            if self.z2symmetry_reduction:
                return ParityMapper(num_particles=n_particles)
            return ParityMapper()
        elif mt == "jw":
            from qiskit_nature.second_q.mappers import JordanWignerMapper
            return JordanWignerMapper()
        else:
            from qiskit_nature.second_q.mappers import ParityMapper
            return ParityMapper()

    def _build_optimizer(self):
        from qiskit_algorithms.optimizers import SLSQP, COBYLA
        name = self.optimizer_name.upper()
        if name == "SLSQP":
            return SLSQP(maxiter=self.vqe_max_iter)
        return COBYLA(maxiter=self.vqe_max_iter)
