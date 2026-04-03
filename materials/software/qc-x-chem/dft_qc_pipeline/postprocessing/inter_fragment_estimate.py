"""
Classical inter-fragment Coulomb estimate from Mulliken atomic charges (PySCF).

This is **not** a full quantum DMET inter-fragment coupling; it is a
Hartree-level diagnostic: sum_{I in A, J in B, A<B} q_I q_J / R_IJ in Hartree
with Mulliken net charges q on atoms and distances in Bohr.

Reference style: textbook electrostatics in atomic units; DMET literature uses
more sophisticated bath / correlation terms — see ``energy_corrections`` notes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from ..core.config import RegionConfig

logger = logging.getLogger(__name__)


def mulliken_net_charges_per_atom(mol: Any, dm: np.ndarray) -> np.ndarray | None:
    """
    Mulliken **net** charge per atom: Z_A − (DS)_AA trace block.

    Returns
    -------
    ndarray, shape (natm,)
        Charge in elementary charge units; distances should be in Bohr for E in Ha.
    """
    try:
        natm = int(mol.natm)
        if natm < 1:
            return None
        ovlp = mol.intor_symmetric("int1e_ovlp")
        if isinstance(dm, tuple) and len(dm) == 2:
            dm = np.asarray(dm[0], dtype=float) + np.asarray(dm[1], dtype=float)
        elif dm.ndim == 3:
            dm = dm[0] + dm[1]
        ds = np.asarray(dm, dtype=float) @ ovlp
        slices = mol.aoslice_by_atom()
        pop = np.zeros(natm, dtype=float)
        for i in range(natm):
            a0, a1 = int(slices[i][2]), int(slices[i][3])
            pop[i] = float(np.trace(ds[a0:a1, a0:a1]))
        z = np.asarray(mol.atom_charges(), dtype=float)
        return z - pop
    except Exception as exc:
        logger.debug("Mulliken charges unavailable: %s", exc)
        return None


def inter_fragment_point_charge_ha(
    mol: Any,
    dm: np.ndarray,
    region_atom_indices: Sequence[Sequence[int]],
) -> float | None:
    """
    Classical Coulomb energy between region groups using Mulliken charges.

    E = Σ_{A<B} Σ_{i∈A} Σ_{j∈B} q_i q_j / R_ij  (Hartree, Bohr).
    """
    q = mulliken_net_charges_per_atom(mol, dm)
    if q is None:
        return None
    try:
        coords = np.asarray(mol.atom_coords(unit="Bohr"), dtype=float)
    except Exception:
        return None
    groups = [list(map(int, g)) for g in region_atom_indices if len(g) > 0]
    if len(groups) < 2:
        return None
    e = 0.0
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            for i in groups[a]:
                for j in groups[b]:
                    rij = float(np.linalg.norm(coords[i] - coords[j]))
                    if rij < 1e-12:
                        continue
                    e += float(q[i] * q[j] / rij)
    return e


def inter_fragment_point_charge_from_backend(
    backend_result: Any,
    regions: Sequence["RegionConfig"],
) -> float | None:
    """Use ``backend_result.mf`` / ``mol`` if available (PySCF path)."""
    mol = getattr(backend_result, "mol", None)
    mf = getattr(backend_result, "mf", None)
    if mol is None or mf is None:
        return None
    natm = getattr(mol, "natm", 0)
    if not natm or natm < 2:
        return None
    try:
        dm = mf.make_rdm1()
    except Exception as exc:
        logger.debug("make_rdm1 failed: %s", exc)
        return None
    atom_groups = [r.atom_indices for r in regions if r.atom_indices]
    if isinstance(dm, tuple):
        dm_use = dm
    else:
        dm_use = np.asarray(dm)
    return inter_fragment_point_charge_ha(mol, dm_use, atom_groups)
