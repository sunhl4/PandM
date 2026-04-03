"""
VQE solver with pluggable ansätze.

Supports three ansatz families:
* ``"uccsd"``     – Unitary CCSD (Qiskit Nature ``UCCSD``).
* ``"hea"``       – Hardware-Efficient Ansatz (Qiskit ``EfficientSU2``).
* ``"kupccgsd"``  – k-UpCCGSD (Qiskit Nature ``PUCCD``-based, approximated here
                   as a repeated UCCSD block; true k-UpCCGSD requires a
                   custom ansatz – see docstring for details).

The solver is backend-agnostic: it works with the Qiskit
``StatevectorEstimator`` (noiseless simulation, default) or any real/fake
``Backend`` via ``Estimator``.

Registered as ``"vqe"`` in the ``"solver"`` category.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..core.interfaces import QuantumSolver, SolverResult
from ..core.registry import registry
from ._rng import apply_solver_seed

logger = logging.getLogger(__name__)

# Available optimizers
_OPTIMIZER_MAP = {
    "cobyla":  ("qiskit_algorithms.optimizers", "COBYLA"),
    "slsqp":   ("qiskit_algorithms.optimizers", "SLSQP"),
    "l_bfgs_b":("qiskit_algorithms.optimizers", "L_BFGS_B"),
    "spsa":    ("qiskit_algorithms.optimizers", "SPSA"),
    "adam":    ("qiskit_algorithms.optimizers", "ADAM"),
}


def _get_optimizer(name: str, max_iter: int = 300):
    import importlib
    name = name.lower()
    if name not in _OPTIMIZER_MAP:
        raise ValueError(
            f"Unknown optimizer '{name}'. Choose: {list(_OPTIMIZER_MAP)}"
        )
    mod_name, cls_name = _OPTIMIZER_MAP[name]
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls(maxiter=max_iter)


@registry.register("vqe", category="solver")
class VQESolver(QuantumSolver):
    """
    VQE with selectable ansatz and optimizer.

    Parameters
    ----------
    ansatz : str
        One of ``"uccsd"``, ``"hea"``, ``"kupccgsd"``.
    optimizer : str
        One of ``"cobyla"``, ``"slsqp"``, ``"l_bfgs_b"``, ``"spsa"``, ``"adam"``.
    max_iter : int
        Maximum number of optimizer iterations.
    shots : int or None
        If None, uses ``StatevectorEstimator`` (exact). Otherwise uses
        ``AerSimulator`` with the given shot count.
    reps : int
        Number of repetition layers for HEA / k-UpCCGSD.
    initial_point : array-like or None
        Initial variational parameters; None → random.
    """

    def __init__(
        self,
        ansatz: str = "uccsd",
        optimizer: str = "cobyla",
        max_iter: int = 300,
        shots: int | None = None,
        reps: int = 2,
        initial_point: list[float] | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.ansatz_type = ansatz.lower()
        self.optimizer_name = optimizer.lower()
        self.max_iter = max_iter
        self.shots = shots
        self.reps = reps
        self.initial_point = initial_point
        self.seed = seed

    def solve(
        self,
        hamiltonian,
        num_particles: tuple[int, int],
        num_spatial_orbitals: int,
    ) -> SolverResult:
        try:
            from qiskit_algorithms import VQE as QiskitVQE
            from qiskit.primitives import StatevectorEstimator
        except ImportError as exc:
            raise ImportError(
                "qiskit-algorithms is required: pip install qiskit-algorithms"
            ) from exc

        apply_solver_seed(self.seed)

        optimizer = _get_optimizer(self.optimizer_name, self.max_iter)
        ansatz = self._build_ansatz(num_particles, num_spatial_orbitals, hamiltonian)

        # --- Estimator ---
        if self.shots is None:
            estimator = StatevectorEstimator()
        else:
            estimator = self._build_shot_estimator()

        logger.info(
            "[VQESolver] ansatz=%s, optimizer=%s, params=%d, qubits=%d",
            self.ansatz_type,
            self.optimizer_name,
            ansatz.num_parameters,
            hamiltonian.num_qubits,
        )

        vqe = QiskitVQE(estimator, ansatz, optimizer)
        result = vqe.compute_minimum_eigenvalue(hamiltonian)
        energy = float(result.eigenvalue.real)
        converged = result.optimizer_result is not None

        rdm1 = None
        if hasattr(result, "optimal_circuit") and result.optimal_circuit is not None:
            rdm1 = self._extract_rdm1(
                result.optimal_circuit,
                result.optimal_parameters,
                num_spatial_orbitals,
                num_particles,
            )

        logger.info(
            "[VQESolver] Energy = %.10f Ha  (converged=%s)",
            energy, converged,
        )
        return SolverResult(
            energy=energy,
            rdm1=rdm1,
            rdm2=None,
            converged=converged,
            extra={
                "optimizer_evals": getattr(result.optimizer_result, "nfev", None),
                "optimal_value": result.optimal_value,
            },
        )

    # ------------------------------------------------------------------
    # Ansatz construction
    # ------------------------------------------------------------------

    def _build_ansatz(self, num_particles, num_spatial_orbitals, hamiltonian):
        if self.ansatz_type == "uccsd":
            return self._build_uccsd(num_particles, num_spatial_orbitals)
        elif self.ansatz_type == "hea":
            return self._build_hea(hamiltonian.num_qubits)
        elif self.ansatz_type in ("kupccgsd", "k_upccgsd", "uccsd_stack"):
            if self.ansatz_type in ("kupccgsd", "k_upccgsd"):
                logger.warning(
                    "[VQESolver] ansatz %r is a stacked-UCCSD approximation, not full "
                    "k-UpCCGSD (Lee JCTC 2019); prefer ansatz='uccsd_stack' for clarity.",
                    self.ansatz_type,
                )
            return self._build_kupccgsd(num_particles, num_spatial_orbitals)
        else:
            raise ValueError(
                f"Unknown ansatz '{self.ansatz_type}'. "
                "Choose: uccsd, hea, uccsd_stack, kupccgsd (alias of uccsd_stack)."
            )

    def _build_uccsd(self, num_particles, num_spatial_orbitals):
        try:
            from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
            from qiskit_nature.second_q.mappers import ParityMapper
        except ImportError as exc:
            raise ImportError("qiskit-nature required for UCCSD ansatz.") from exc

        mapper = ParityMapper(num_particles=num_particles)
        hf_init = HartreeFock(num_spatial_orbitals, num_particles, mapper)
        ansatz = UCCSD(
            num_spatial_orbitals,
            num_particles,
            mapper,
            initial_state=hf_init,
            reps=1,
        )
        return ansatz

    def _build_hea(self, num_qubits: int):
        try:
            from qiskit.circuit.library import EfficientSU2
        except ImportError as exc:
            raise ImportError("qiskit required for HEA (EfficientSU2).") from exc

        return EfficientSU2(num_qubits, reps=self.reps)

    def _build_kupccgsd(self, num_particles, num_spatial_orbitals):
        """
        Approximate k-UpCCGSD: stack ``reps`` UCCSD layers with shared parameters.

        A full k-UpCCGSD (Lee et al., JCTC 2019) pairs GSD excitations in
        generalized form; here we use ``reps`` layers of standard UCCSD as a
        tractable approximation.  For the true k-UpCCGSD implementation see
        the Qiskit Nature ``PUCCD`` family or custom ``EvolvedOperatorAnsatz``.
        """
        try:
            from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
            from qiskit_nature.second_q.mappers import ParityMapper
        except ImportError as exc:
            raise ImportError("qiskit-nature required for k-UpCCGSD ansatz.") from exc

        mapper = ParityMapper(num_particles=num_particles)
        hf_init = HartreeFock(num_spatial_orbitals, num_particles, mapper)
        ansatz = UCCSD(
            num_spatial_orbitals,
            num_particles,
            mapper,
            initial_state=hf_init,
            reps=self.reps,
        )
        logger.info(
            "[VQESolver] k-UpCCGSD approximation: UCCSD with reps=%d", self.reps
        )
        return ansatz

    # ------------------------------------------------------------------
    # Shot-based estimator
    # ------------------------------------------------------------------

    def _build_shot_estimator(self):
        try:
            from qiskit_aer.primitives import Estimator as AerEstimator
            return AerEstimator(run_options={"shots": self.shots})
        except ImportError:
            logger.warning(
                "qiskit-aer not found; falling back to StatevectorEstimator "
                "(shots ignored)."
            )
            from qiskit.primitives import StatevectorEstimator
            return StatevectorEstimator()

    # ------------------------------------------------------------------
    # 1-RDM extraction
    # ------------------------------------------------------------------

    def _extract_rdm1(
        self,
        circuit,
        params,
        num_spatial_orbitals: int,
        num_particles: tuple[int, int],
    ) -> np.ndarray | None:
        """Extract 1-RDM from the optimal VQE circuit via statevector."""
        try:
            from qiskit.quantum_info import Statevector
            from qiskit_nature.second_q.operators import FermionicOp
            from qiskit_nature.second_q.mappers import JordanWignerMapper

            bound = circuit.assign_parameters(params)
            sv = Statevector(bound)
            mapper = JordanWignerMapper()
            norb = num_spatial_orbitals
            rdm1 = np.zeros((norb, norb))

            for p in range(norb):
                for q in range(norb):
                    for ss in (0, norb):
                        key = f"+_{p+ss} -_{q+ss}"
                        op = FermionicOp({key: 1.0}, num_spin_orbitals=2 * norb)
                        rdm1[p, q] += sv.expectation_value(mapper.map(op)).real
            return rdm1
        except Exception as exc:
            logger.warning("VQE 1-RDM extraction failed: %s", exc)
            return None
