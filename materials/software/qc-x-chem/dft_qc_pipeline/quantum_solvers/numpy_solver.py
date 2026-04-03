"""
NumPy exact-diagonalization solver (FCI reference).

Uses Qiskit Nature's ``NumPyMinimumEigensolver`` to diagonalize the qubit
Hamiltonian exactly.  Suitable only for small active spaces (≤ ~12 qubits).

Registered as ``"numpy"`` in the ``"solver"`` category.
"""

from __future__ import annotations

import logging

import numpy as np

from ..core.interfaces import QuantumSolver, SolverResult
from ..core.registry import registry

logger = logging.getLogger(__name__)


@registry.register("numpy", category="solver")
class NumPySolver(QuantumSolver):
    """
    Exact diagonalization via ``NumPyMinimumEigensolver``.

    This solver ignores all VQE/circuit parameters and directly computes
    the exact ground-state energy (within the mapped qubit space).

    Parameters
    ----------
    compute_rdm : bool
        If ``True``, attempt to extract the 1-RDM from the ground-state
        eigenvector using the statevector.
    """

    def __init__(self, compute_rdm: bool = True, **kwargs) -> None:
        self.compute_rdm = compute_rdm

    def solve(
        self,
        hamiltonian,
        num_particles: tuple[int, int],
        num_spatial_orbitals: int,
    ) -> SolverResult:
        try:
            from qiskit_algorithms import NumPyMinimumEigensolver
        except ImportError as exc:
            raise ImportError(
                "qiskit-algorithms required: pip install qiskit-algorithms"
            ) from exc

        logger.info(
            "[NumPySolver] Exact diag: %d qubits, %d Pauli terms",
            hamiltonian.num_qubits,
            len(hamiltonian),
        )

        solver = NumPyMinimumEigensolver()
        result = solver.compute_minimum_eigenvalue(hamiltonian)
        energy = float(result.eigenvalue.real)

        rdm1 = None
        if self.compute_rdm and result.eigenstate is not None:
            rdm1 = self._rdm1_from_statevector(
                result.eigenstate, num_spatial_orbitals, num_particles
            )

        logger.info("[NumPySolver] Energy = %.10f Ha", energy)
        return SolverResult(
            energy=energy,
            rdm1=rdm1,
            rdm2=None,
            converged=True,
            extra={"eigenvalue": result.eigenvalue},
        )

    # ------------------------------------------------------------------
    # 1-RDM extraction from a statevector
    # ------------------------------------------------------------------

    @staticmethod
    def _rdm1_from_statevector(
        eigenstate,
        num_spatial_orbitals: int,
        num_particles: tuple[int, int],
    ) -> np.ndarray | None:
        """
        Compute the 1-particle reduced density matrix from a statevector.

        Uses the Qiskit ``Statevector`` representation and computes
        ⟨ψ| a†_p a_q |ψ⟩ for the JW/Parity-mapped qubit state.

        Returns a (norb, norb) array (spin-summed, closed-shell approximation).
        """
        try:
            from qiskit.quantum_info import Statevector
            from qiskit_nature.second_q.operators import FermionicOp
            from qiskit_nature.second_q.mappers import JordanWignerMapper
        except ImportError:
            logger.warning("Could not import Qiskit for RDM extraction.")
            return None

        norb = num_spatial_orbitals
        rdm1 = np.zeros((norb, norb))

        try:
            if hasattr(eigenstate, "data"):
                sv = eigenstate.data
            else:
                sv = np.asarray(eigenstate)

            sv_qiskit = Statevector(sv)
            mapper = JordanWignerMapper()

            for p in range(norb):
                for q in range(norb):
                    # Spin-summed: alpha + beta contributions
                    for spin_shift in (0, norb):
                        key = f"+_{p + spin_shift} -_{q + spin_shift}"
                        op = FermionicOp({key: 1.0}, num_spin_orbitals=2 * norb)
                        qubit_op = mapper.map(op)
                        val = sv_qiskit.expectation_value(qubit_op).real
                        rdm1[p, q] += val
        except Exception as exc:
            logger.warning("RDM extraction failed: %s", exc)
            return None

        return rdm1
