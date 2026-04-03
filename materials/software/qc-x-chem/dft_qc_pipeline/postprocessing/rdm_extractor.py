"""
RDM (reduced density matrix) extractor.

Provides utilities to extract 1-RDM and 2-RDM from a ``SolverResult``
and transform them back to AO/MO bases for downstream DMET self-consistency
or force / property evaluation.
"""

from __future__ import annotations

import logging

import numpy as np

from ..core.interfaces import EmbeddedHamiltonian, SolverResult

logger = logging.getLogger(__name__)


def extract_rdm1(
    result: SolverResult,
    emb_H: EmbeddedHamiltonian | None = None,
) -> np.ndarray | None:
    """
    Return the 1-particle RDM in the active orbital basis.

    If ``result.rdm1`` is already populated (by the solver), return it
    directly.  Otherwise return ``None``.

    Parameters
    ----------
    result : SolverResult
    emb_H : EmbeddedHamiltonian, optional
        Not used currently; reserved for future basis-rotation.

    Returns
    -------
    np.ndarray of shape (norb, norb) or None
    """
    return result.rdm1


def rdm1_to_ao(
    rdm1_mo: np.ndarray,
    C_act: np.ndarray,
) -> np.ndarray:
    """
    Transform a 1-RDM from active MO basis to AO basis.

    Parameters
    ----------
    rdm1_mo : np.ndarray  shape (norb, norb)
    C_act : np.ndarray    shape (nao, norb)

    Returns
    -------
    np.ndarray  shape (nao, nao)
    """
    return C_act @ rdm1_mo @ C_act.T


def fragment_population(
    rdm1_mo: np.ndarray,
    C_act: np.ndarray,
    mol,
    atom_indices: list[int],
) -> float:
    """
    Compute the Mulliken electron population of the fragment in the active space.

    Parameters
    ----------
    rdm1_mo : np.ndarray  shape (norb, norb)
    C_act : np.ndarray    shape (nao, norb)
    mol : PySCF Mole
    atom_indices : list[int]

    Returns
    -------
    float  Total electron population on the selected atoms.
    """
    dm_ao = rdm1_to_ao(rdm1_mo, C_act)
    ovlp = mol.intor("int1e_ovlp")
    mulliken = (dm_ao @ ovlp)       # (nao, nao)

    slices = mol.aoslice_by_atom()
    pop = 0.0
    for iat in atom_indices:
        start, stop = slices[iat][2], slices[iat][3]
        pop += float(np.trace(mulliken[start:stop, start:stop]))
    return pop


def print_rdm1_summary(rdm1: np.ndarray, label: str = "1-RDM") -> None:
    """Print a brief summary of the 1-RDM."""
    if rdm1 is None:
        print(f"{label}: None")
        return
    np.set_printoptions(precision=4, suppress=True)
    print(f"\n{label}  (trace = {np.trace(rdm1):.4f})")
    print(rdm1)
