"""
Qubit mapper wrappers.

``build_mapper`` reads a ``MapperConfig`` and returns an object with a
``.map(emb_H)`` method that:

  1. Converts ``EmbeddedHamiltonian`` → ``FermionicOp`` via the builder.
  2. Maps to a ``SparsePauliOp`` via the chosen Qiskit Nature mapper.
  3. Returns ``(SparsePauliOp, num_particles, num_spatial_orbitals)``.

Supported mappers
-----------------
* ``"jw"``     – Jordan-Wigner
* ``"parity"`` – Parity mapping (can reduce 2 qubits with Z2 symmetry)
* ``"bk"``     – Bravyi-Kitaev
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..core.interfaces import EmbeddedHamiltonian

logger = logging.getLogger(__name__)


class _MapperWrapper:
    """Internal wrapper holding a Qiskit Nature mapper + z2 reduction flag."""

    def __init__(self, qiskit_mapper, e_core: float = 0.0) -> None:
        self._mapper = qiskit_mapper
        self._last_e_core = e_core  # updated each call to map()

    def map(self, emb_H: EmbeddedHamiltonian):
        """
        Map an EmbeddedHamiltonian to a qubit Hamiltonian.

        Returns
        -------
        tuple[SparsePauliOp, tuple[int,int], int]
            (qubit_op, num_particles, num_spatial_orbitals)
        """
        from .builder import hamiltonian_to_fermionic_op

        fermionic_op, e_core = hamiltonian_to_fermionic_op(emb_H)
        self._last_e_core = e_core

        qubit_op = self._mapper.map(fermionic_op)
        logger.info(
            "Mapped to %d-qubit operator (%d Pauli terms)",
            qubit_op.num_qubits,
            len(qubit_op),
        )
        return qubit_op, emb_H.nelec, emb_H.norb


def build_mapper(mapper_cfg) -> _MapperWrapper:
    """
    Build a mapper wrapper from a ``MapperConfig``.

    Parameters
    ----------
    mapper_cfg : MapperConfig
    """
    try:
        from qiskit_nature.second_q.mappers import (
            JordanWignerMapper,
            ParityMapper,
            BravyiKitaevMapper,
        )
    except ImportError as exc:
        raise ImportError(
            "qiskit-nature is required: pip install qiskit-nature"
        ) from exc

    mtype = mapper_cfg.type.lower()

    if mtype == "jw":
        mapper = JordanWignerMapper()
    elif mtype == "parity":
        mapper = ParityMapper()
        # Z2 symmetry reduction is applied later via the ElectronicStructureProblem
        # or by passing num_particles to the mapper; store the flag for downstream use.
        if mapper_cfg.z2symmetry_reduction:
            logger.info(
                "Parity mapper: Z2 symmetry reduction will be applied when "
                "num_particles is available (passed at solve time)."
            )
    elif mtype == "bk":
        mapper = BravyiKitaevMapper()
    else:
        raise ValueError(
            f"Unknown mapper type '{mtype}'. Choose: jw, parity, bk."
        )

    logger.info("Mapper: %s", mtype.upper())
    return _MapperWrapper(mapper)
