"""
VQE solver — Variational Quantum Eigensolver with multiple ansätze.

Registered names:
  ``"vqe_uccsd"``   — UCCSD ansatz (chemistry-inspired, Qiskit Nature)
  ``"vqe_hea"``     — Hardware-Efficient Ansatz (generic, noise-resilient)
  ``"vqe_kupccgsd"``— k-UpCCGSD (Grimsley / Lee 2019, parameter-efficient)

All three share VQESolverBase and differ only in ansatz construction.
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

_DEFAULT_OPTIMIZER = "slsqp"
_DEFAULT_MAX_ITER = 300


class VQESolverBase(BaseSolver):
    """
    Shared infrastructure for all VQE variants.

    Subclasses set ``self.ansatz_type`` and override ``_build_ansatz``.
    """

    ansatz_type: str = "uccsd"

    def __init__(
        self,
        optimizer: str = _DEFAULT_OPTIMIZER,
        max_iter: int = _DEFAULT_MAX_ITER,
        shots: int | None = None,
        mapper_type: str = "parity",
        z2symmetry_reduction: bool = True,
        k: int = 1,
        reps: int = 1,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.optimizer_name = optimizer
        self.max_iter = max_iter
        self.shots = shots
        self.mapper_type = mapper_type
        self.z2symmetry_reduction = z2symmetry_reduction
        self.k = k
        self.reps = reps
        self.seed = seed

    def solve(self, mol_spec: MolSpec) -> MethodResult:
        try:
            from qiskit_algorithms import VQE
            from qiskit_algorithms.optimizers import SLSQP, COBYLA, L_BFGS_B, SPSA
            from qiskit.primitives import StatevectorEstimator as Estimator
        except ImportError as exc:
            raise ImportError(
                "qiskit-algorithms required: pip install qiskit-algorithms"
            ) from exc

        t0 = self._start_timer()
        apply_solver_seed(self.seed)

        # Build active-space integrals
        builder = MoleculeBuilder(verbose=0)
        integrals = builder.build(mol_spec)

        # Build qubit Hamiltonian
        ham_builder = HamiltonianBuilder(
            mapper_type=self.mapper_type,
            z2symmetry_reduction=self.z2symmetry_reduction,
        )
        qubit_op, n_particles, n_orbs = ham_builder.build(integrals)

        # Build ansatz
        ansatz = self._build_ansatz(n_particles, n_orbs, qubit_op.num_qubits)

        # Build optimizer
        opt = self._build_optimizer()

        # Estimator + VQE
        estimator = Estimator()
        vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=opt)

        result = vqe.compute_minimum_eigenvalue(qubit_op)

        energy = float(result.eigenvalue.real) + integrals.e_core
        n_qubits = qubit_op.num_qubits

        return MethodResult(
            method_name=f"VQE-{self.ansatz_type.upper()}",
            energy=energy,
            corr_energy=energy - integrals.hf_energy,
            converged=True,
            n_qubits=n_qubits,
            wall_time=self._elapsed(t0),
            extra={
                "optimizer_evals": getattr(result, "cost_function_evals", None),
                "ansatz_params": ansatz.num_parameters,
                "circuit_depth": ansatz.decompose().depth(),
            },
        )

    def _build_ansatz(
        self, n_particles: tuple[int, int], n_orbs: int, n_qubits: int
    ):
        raise NotImplementedError

    def _build_optimizer(self):
        from qiskit_algorithms.optimizers import SLSQP, COBYLA, L_BFGS_B, SPSA
        name = self.optimizer_name.upper()
        if name == "SLSQP":
            return SLSQP(maxiter=self.max_iter)
        elif name == "COBYLA":
            return COBYLA(maxiter=self.max_iter)
        elif name in ("L_BFGS_B", "LBFGSB"):
            return L_BFGS_B(maxiter=self.max_iter)
        elif name == "SPSA":
            return SPSA(maxiter=self.max_iter)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")


# ---------------------------------------------------------------------------
# UCCSD
# ---------------------------------------------------------------------------

@registry.register("vqe_uccsd", category="solver")
class VQEUCCSDSolver(VQESolverBase):
    """
    VQE with UCCSD (Unitary Coupled Cluster Singles and Doubles) ansatz.

    Chemistry-inspired; approaches FCI as the ansatz includes excitations
    up to doubles.  Exponentially many parameters but strong accuracy.
    """

    ansatz_type = "uccsd"

    def _build_ansatz(self, n_particles, n_orbs, n_qubits):
        try:
            from qiskit_nature.second_q.circuit.library import UCCSD
            from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper, BravyiKitaevMapper
        except ImportError as exc:
            raise ImportError("qiskit-nature required") from exc

        mapper = self._get_mapper(n_particles, n_orbs)
        return UCCSD(
            num_spatial_orbitals=n_orbs,
            num_particles=n_particles,
            qubit_mapper=mapper,
        )

    def _get_mapper(self, n_particles, n_orbs):
        mt = self.mapper_type.lower()
        if mt == "parity":
            from qiskit_nature.second_q.mappers import ParityMapper
            if self.z2symmetry_reduction:
                return ParityMapper(num_particles=n_particles)
            return ParityMapper()
        elif mt == "jw":
            from qiskit_nature.second_q.mappers import JordanWignerMapper
            return JordanWignerMapper()
        elif mt == "bk":
            from qiskit_nature.second_q.mappers import BravyiKitaevMapper
            return BravyiKitaevMapper()
        else:
            raise ValueError(f"Unknown mapper: {self.mapper_type}")


# ---------------------------------------------------------------------------
# Hardware-Efficient Ansatz (HEA)
# ---------------------------------------------------------------------------

@registry.register("vqe_hea", category="solver")
class VQEHEASolver(VQESolverBase):
    """
    VQE with a Hardware-Efficient Ansatz (alternating rotation + entanglement layers).

    Noise-resilient shallow circuits; less systematic than UCCSD.
    """

    ansatz_type = "hea"

    def _build_ansatz(self, n_particles, n_orbs, n_qubits):
        from qiskit.circuit.library import EfficientSU2
        return EfficientSU2(
            num_qubits=n_qubits,
            reps=self.reps,
            entanglement="linear",
        )


# ---------------------------------------------------------------------------
# k-UpCCGSD
# ---------------------------------------------------------------------------

@registry.register("vqe_kupccgsd", category="solver")
class VQEkUpCCGSDSolver(VQESolverBase):
    """
    VQE with k-UpCCGSD (k unitary pair Coupled-Cluster Generalised Singles
    and Doubles) ansatz.

    Reference: Lee et al., J. Chem. Theory Comput. 15, 311 (2019).
    More accurate than HEA while using fewer parameters than UCCSD.
    """

    ansatz_type = "kupccgsd"

    def _build_ansatz(self, n_particles, n_orbs, n_qubits):
        try:
            from qiskit_nature.second_q.circuit.library import UCC
            from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper
        except ImportError as exc:
            raise ImportError("qiskit-nature required") from exc

        mapper = self._get_mapper(n_particles, n_orbs)

        # k-UpCCGSD: k repetitions of generalised paired double excitations
        return UCC(
            num_spatial_orbitals=n_orbs,
            num_particles=n_particles,
            qubit_mapper=mapper,
            excitations="gsd",
            reps=self.k,
        )

    def _get_mapper(self, n_particles, n_orbs):
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


@registry.register("vqe_uccsd_stack", category="solver")
class VQEUCCSDStackSolver(VQEkUpCCGSDSolver):
    """
    Alias for ``vqe_kupccgsd``: same UCC(GSD) ansatz (``UCC(..., excitations='gsd')``).

    Name reflects the **stacked paired-GSD** parameterization; literature k-UpCCGSD
    may differ in operator pool details (see dft_qc_pipeline TECH_DOC).
    """

    ansatz_type = "uccsd_stack"
