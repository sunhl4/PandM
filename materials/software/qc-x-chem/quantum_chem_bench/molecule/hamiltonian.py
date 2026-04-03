"""
HamiltonianBuilder — maps MolIntegrals to a qubit Hamiltonian (SparsePauliOp).

Supports Jordan-Wigner (jw), Parity, and Bravyi-Kitaev (bk) mappings via
Qiskit Nature.

Usage::

    from quantum_chem_bench.molecule.hamiltonian import HamiltonianBuilder
    builder = HamiltonianBuilder(mapper_type="parity", z2symmetry_reduction=True)
    qubit_op, n_particles, n_orbs = builder.build(integrals)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from quantum_chem_bench.core.interfaces import MolIntegrals

logger = logging.getLogger(__name__)


class HamiltonianBuilder:
    """
    Convert MolIntegrals → SparsePauliOp via Qiskit Nature.

    Parameters
    ----------
    mapper_type : str
        One of ``"jw"``, ``"parity"``, ``"bk"``.
    z2symmetry_reduction : bool
        Apply Z2 symmetry tapering (parity mapper only; ignored otherwise).
    """

    def __init__(
        self,
        mapper_type: str = "parity",
        z2symmetry_reduction: bool = True,
    ) -> None:
        self.mapper_type = mapper_type.lower()
        self.z2symmetry_reduction = z2symmetry_reduction

    def build(
        self, integrals: MolIntegrals
    ) -> tuple[Any, tuple[int, int], int]:
        """
        Build the qubit Hamiltonian.

        Returns
        -------
        qubit_op : SparsePauliOp
        num_particles : (int, int)
        num_spatial_orbitals : int
        """
        try:
            from qiskit_nature.second_q.operators import FermionicOp
            from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
            from qiskit_nature.second_q.problems import ElectronicStructureProblem
            from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
        except ImportError as exc:
            raise ImportError(
                "qiskit-nature is required: pip install qiskit-nature"
            ) from exc

        norb = integrals.norb
        nelec = integrals.nelec
        h1e = integrals.h1e
        h2e = integrals.h2e
        e_core = integrals.e_core

        # Build FermionicOp from integrals
        fermionic_op = self._integrals_to_fermionic_op(h1e, h2e, norb, e_core)

        # Select mapper
        mapper = self._build_mapper(nelec, norb)

        # Map to qubit operator
        qubit_op = mapper.map(fermionic_op)

        n_qubits = qubit_op.num_qubits
        logger.debug(
            "Mapped to %d qubits via %s (terms=%d)",
            n_qubits, self.mapper_type, len(qubit_op),
        )
        return qubit_op, nelec, norb

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _integrals_to_fermionic_op(
        self,
        h1e: np.ndarray,
        h2e: np.ndarray,
        norb: int,
        e_core: float,
    ):
        """Convert 1e/2e integrals to a FermionicOp (Qiskit Nature v0.7+)."""
        from qiskit_nature.second_q.operators import FermionicOp

        terms: dict[str, complex] = {}

        # Nuclear repulsion / core energy as identity
        terms[""] = complex(e_core)

        # One-electron terms
        for p in range(norb):
            for q in range(norb):
                val = float(h1e[p, q])
                if abs(val) > 1e-15:
                    # Spin-up
                    terms[f"+_{p} -_{q}"] = terms.get(f"+_{p} -_{q}", 0.0) + val
                    # Spin-down  (offset by norb)
                    terms[f"+_{p+norb} -_{q+norb}"] = (
                        terms.get(f"+_{p+norb} -_{q+norb}", 0.0) + val
                    )

        # Two-electron terms: (1/2) sum_{pqrs} h2e[p,q,r,s] a†_p a†_r a_s a_q
        # h2e in chemist notation: (pq|rs) = <pr|qs>
        for p in range(norb):
            for q in range(norb):
                for r in range(norb):
                    for s in range(norb):
                        val = 0.5 * float(h2e[p, q, r, s])
                        if abs(val) < 1e-15:
                            continue
                        # All spin combinations: uu, dd, ud, du
                        for sp, sq, sr, ss in [
                            (p, q, r + norb, s + norb),
                            (p + norb, q + norb, r, s),
                            (p, q, r, s),
                            (p + norb, q + norb, r + norb, s + norb),
                        ]:
                            key = f"+_{sp} +_{sr} -_{ss} -_{sq}"
                            terms[key] = terms.get(key, 0.0) + val

        return FermionicOp(terms, num_spin_orbs=2 * norb)

    def _build_mapper(self, nelec: tuple[int, int], norb: int):
        """Instantiate the requested Qiskit Nature QubitMapper."""
        mt = self.mapper_type
        if mt == "jw":
            from qiskit_nature.second_q.mappers import JordanWignerMapper
            return JordanWignerMapper()
        elif mt == "parity":
            from qiskit_nature.second_q.mappers import ParityMapper
            if self.z2symmetry_reduction:
                return ParityMapper(num_particles=nelec)
            return ParityMapper()
        elif mt == "bk":
            from qiskit_nature.second_q.mappers import BravyiKitaevMapper
            return BravyiKitaevMapper()
        else:
            raise ValueError(
                f"Unknown mapper_type='{mt}'. Choose from 'jw', 'parity', 'bk'."
            )
