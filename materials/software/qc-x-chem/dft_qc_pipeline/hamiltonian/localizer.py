"""
Orbital localization utilities.

Wraps PySCF's localization modules behind a uniform interface::

    from dft_qc_pipeline.hamiltonian.localizer import localize_orbitals

    C_loc = localize_orbitals(mf, scheme="iao")   # (nao, nmo) localized MO coefficients

Supported schemes
-----------------
* ``"boys"``  – Foster-Boys (maximizes orbital self-repulsion).
* ``"pm"``    – Pipek-Mezey (maximizes atomic-population weights).
* ``"iao"``   – Intrinsic Atomic Orbitals (Knizia 2013); best for active-space selection.
* ``"none"``  – no localization; return canonical MOs unchanged.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def localize_orbitals(
    mf,
    scheme: str = "iao",
    mo_coeff: np.ndarray | None = None,
    occ_only: bool = True,
) -> np.ndarray:
    """
    Return localized MO coefficients.

    Parameters
    ----------
    mf : PySCF mean-field object
        Provides ``mol``, ``mo_coeff``, and ``mo_occ``.
    scheme : str
        One of ``"boys"``, ``"pm"``, ``"iao"``, ``"none"``.
    mo_coeff : ndarray, optional
        Override the MO coefficients from ``mf``.  Shape ``(nao, nmo)``.
    occ_only : bool
        If ``True`` (default), localize only the occupied block (
        ``mo_occ > 0``).  The virtual block is returned unchanged.

    Returns
    -------
    np.ndarray
        Localized MO coefficient matrix, shape ``(nao, nmo)``.
    """
    C = mf.mo_coeff if mo_coeff is None else mo_coeff
    occ_mask = mf.mo_occ > 0

    if scheme == "none":
        return C

    try:
        from pyscf import lo
    except ImportError as exc:
        raise ImportError("PySCF is required for orbital localization.") from exc

    C_loc = C.copy()
    C_occ = C[:, occ_mask]

    scheme_lower = scheme.lower()
    mol = mf.mol

    if scheme_lower == "boys":
        loc = lo.Boys(mol, C_occ)
        C_loc[:, occ_mask] = loc.kernel()
        logger.info("Boys localization: %d occupied MOs localized.", int(occ_mask.sum()))

    elif scheme_lower == "pm":
        loc = lo.PipekMezey(mol, C_occ)
        C_loc[:, occ_mask] = loc.kernel()
        logger.info("Pipek-Mezey localization: %d occupied MOs.", int(occ_mask.sum()))

    elif scheme_lower == "iao":
        # IAO localization (Knizia, JCTC 2013).
        # lo.iao.iao() returns a minimal-basis set of IAOs with shape (nao, n_minbas)
        # where n_minbas ≥ n_occ in general.  We project the occupied MOs onto the
        # IAO span so the result keeps the original (nao, n_occ) shape.
        S = mol.intor("int1e_ovlp")
        C_iao = lo.iao.iao(mol, C_occ)                # (nao, n_minbas)
        C_iao = lo.vec_lowdin(C_iao, S)               # orthonormalize IAOs
        # Express each occupied MO as a linear combination of IAOs, then reconstruct
        C_occ_loc = C_iao @ (C_iao.T @ S @ C_occ)     # (nao, n_occ)
        C_loc[:, occ_mask] = lo.vec_lowdin(C_occ_loc, S)
        logger.info("IAO localization applied to occupied block.")

    else:
        raise ValueError(
            f"Unknown localization scheme '{scheme}'. "
            "Choose one of: boys, pm, iao, none."
        )

    # Orthogonalize to preserve idempotency (IAO already handled inside its branch)
    if scheme_lower in ("boys", "pm"):
        C_loc[:, occ_mask] = lo.vec_lowdin(C_loc[:, occ_mask], mf.get_ovlp())

    return C_loc


def get_atom_orbital_indices(
    mol,
    C_loc: np.ndarray,
    atom_indices: list[int],
    mo_occ: np.ndarray,
    n_orbs: int,
) -> list[int]:
    """
    Return the *n_orbs* localized occupied MO indices most centred on *atom_indices*.

    Strategy: sort occupied MOs by their Mulliken population on the selected
    atoms (largest first) and return the top ``n_orbs``.

    Parameters
    ----------
    mol : PySCF Mole
    C_loc : np.ndarray  shape (nao, nmo)  localized coefficients
    atom_indices : list[int]   0-based atom indices
    mo_occ : np.ndarray   shape (nmo,)
    n_orbs : int   number of orbitals to select

    Returns
    -------
    list[int]
        Global MO column indices into ``C_loc`` / ``mf.mo_coeff`` (occupied
        subset only; at most ``n_orbs`` entries, capped by the number of
        occupied spatial orbitals).
    """
    occ_mask = mo_occ > 0
    occ_positions = np.where(occ_mask)[0]  # global MO column indices
    C_occ = C_loc[:, occ_mask]  # (nao, nocc)
    ovlp = mol.intor("int1e_ovlp")

    # Mulliken AO populations for each occupied MO
    # P_mu = (C_occ)_mu  *  (S @ C_occ)_mu   (element-wise)
    SC = ovlp @ C_occ
    pop_ao = (C_occ * SC)  # (nao, nocc)

    # Sum over AOs belonging to selected atoms
    atom_ao_slices = [
        slice(mol.aoslice_by_atom()[iat][2], mol.aoslice_by_atom()[iat][3])
        for iat in atom_indices
    ]
    atom_pop = np.zeros(C_occ.shape[1])
    for sl in atom_ao_slices:
        atom_pop += pop_ao[sl, :].sum(axis=0)

    # Sort descending; take top n_orbs → map back to global MO indices
    sorted_idx = np.argsort(-atom_pop)
    n_pick = min(int(n_orbs), len(sorted_idx))
    selected = occ_positions[sorted_idx[:n_pick]].tolist()
    logger.debug("Atom-projected orbital selection (global MO idx): %s", selected)
    return selected
