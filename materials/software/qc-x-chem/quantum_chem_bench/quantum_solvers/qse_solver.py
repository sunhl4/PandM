"""
Quantum Subspace Expansion (QSE) solver.

Reference: McClean et al., *Physical Review A* 95, 042308 (2017);
           also discussed in McArdle et al., Rev. Mod. Phys. 92, 015003 (2020).

Algorithm:
    1. Prepare a reference state |Ψ₀⟩ (typically from VQE).
    2. Generate an expansion basis {Mₖ|Ψ₀⟩} where Mₖ are Pauli/fermionic
       operators (singles + doubles from the reference).
    3. Build effective Hamiltonian H̃ₖₗ = ⟨Ψ₀|MₖHMₗ|Ψ₀⟩ and overlap
       Sₖₗ = ⟨Ψ₀|MₖMₗ|Ψ₀⟩.
    4. Solve the generalised eigenvalue problem H̃c = ESc for ground energy.

Particularly useful for excited states and post-processing VQE results.

Registered as ``"qse"`` in the solver registry.
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


@registry.register("qse", category="solver")
class QSESolver(BaseSolver):
    """
    Quantum Subspace Expansion (QSE).

    Uses statevector simulation for exact expectation values.
    Solves the generalised eigenvalue problem in the expanded subspace.

    Parameters
    ----------
    expansion_order : int
        Order of operator pool: 1 = singles only, 2 = singles+doubles.
    mapper_type : str
        Qubit mapper.
    z2symmetry_reduction : bool
        Apply Z2 symmetry reduction.
    vqe_optimizer : str
        Optimizer for VQE reference state preparation.
    vqe_max_iter : int
        Max iterations for VQE reference preparation.
    """

    def __init__(
        self,
        expansion_order: int = 2,
        mapper_type: str = "parity",
        z2symmetry_reduction: bool = True,
        vqe_optimizer: str = "slsqp",
        vqe_max_iter: int = 300,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.expansion_order = expansion_order
        self.mapper_type = mapper_type
        self.z2symmetry_reduction = z2symmetry_reduction
        self.vqe_optimizer = vqe_optimizer
        self.vqe_max_iter = vqe_max_iter
        self.seed = seed

    def solve(self, mol_spec: MolSpec) -> MethodResult:
        try:
            from qiskit_algorithms import VQE
            from qiskit.primitives import StatevectorEstimator as Estimator
            from qiskit_nature.second_q.circuit.library import UCCSD
            from qiskit.quantum_info import Statevector, SparsePauliOp
            import scipy.linalg as la
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

        # Step 1: Prepare reference state via VQE (UCCSD)
        mapper = self._build_mapper(n_particles)
        ansatz = UCCSD(
            num_spatial_orbitals=n_orbs,
            num_particles=n_particles,
            qubit_mapper=mapper,
        )
        opt = self._build_optimizer()
        estimator = Estimator()
        vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=opt)
        vqe_result = vqe.compute_minimum_eigenvalue(qubit_op)

        # Bind optimal parameters
        bound_circuit = ansatz.assign_parameters(vqe_result.optimal_parameters)
        ref_state = Statevector(bound_circuit)

        # Step 2: Build expansion operators (Pauli X, Y strings for S+D)
        expansion_ops = self._build_expansion_ops(qubit_op, n_qubits)

        if not expansion_ops:
            # No expansion — return VQE energy
            energy = float(vqe_result.eigenvalue.real) + integrals.e_core
            return MethodResult(
                method_name="QSE",
                energy=energy,
                corr_energy=energy - integrals.hf_energy,
                converged=True,
                n_qubits=n_qubits,
                wall_time=self._elapsed(t0),
                extra={"expansion_dim": 0, "note": "no expansion operators; returning VQE energy"},
            )

        # Step 3: Build H and S matrices
        dim = len(expansion_ops)
        H_mat = np.zeros((dim, dim), dtype=complex)
        S_mat = np.zeros((dim, dim), dtype=complex)

        H_mat_full = qubit_op.to_matrix()

        for i, Mi in enumerate(expansion_ops):
            Mi_mat = Mi.to_matrix()
            psi_i = Mi_mat @ ref_state.data
            for j, Mj in enumerate(expansion_ops):
                Mj_mat = Mj.to_matrix()
                psi_j = Mj_mat @ ref_state.data
                H_mat[i, j] = psi_i.conj() @ H_mat_full @ psi_j
                S_mat[i, j] = psi_i.conj() @ psi_j

        # Step 4: Solve generalised eigenvalue problem
        try:
            evals, _ = la.eigh(H_mat, S_mat)
            energy = float(evals[0].real) + integrals.e_core
        except la.LinAlgError:
            logger.warning("QSE generalised eigenproblem failed; using VQE energy.")
            energy = float(vqe_result.eigenvalue.real) + integrals.e_core

        return MethodResult(
            method_name="QSE",
            energy=energy,
            corr_energy=energy - integrals.hf_energy,
            converged=True,
            n_qubits=n_qubits,
            wall_time=self._elapsed(t0),
            extra={
                "expansion_dim": dim,
                "vqe_energy": float(vqe_result.eigenvalue.real) + integrals.e_core,
            },
        )

    def _build_expansion_ops(self, qubit_op, n_qubits: int) -> list:
        """Build a minimal set of Pauli operators for subspace expansion."""
        from qiskit.quantum_info import SparsePauliOp

        ops = []
        # Identity (always included)
        ops.append(SparsePauliOp.from_list([("I" * n_qubits, 1.0)]))

        if self.expansion_order >= 1 and n_qubits <= 20:
            # Single-qubit X excitations (proxy for singles)
            for i in range(min(n_qubits, 8)):
                label = "I" * i + "X" + "I" * (n_qubits - i - 1)
                ops.append(SparsePauliOp.from_list([(label, 1.0)]))

        if self.expansion_order >= 2 and n_qubits <= 16:
            # Two-qubit XY excitations (proxy for doubles)
            for i in range(min(n_qubits - 1, 4)):
                for j in range(i + 1, min(n_qubits, i + 4)):
                    p = ["I"] * n_qubits
                    p[i] = "X"
                    p[j] = "Y"
                    ops.append(SparsePauliOp.from_list([("".join(p), 1.0)]))

        return ops

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
        name = self.vqe_optimizer.upper()
        if name == "SLSQP":
            return SLSQP(maxiter=self.vqe_max_iter)
        return COBYLA(maxiter=self.vqe_max_iter)
