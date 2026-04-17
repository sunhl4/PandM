from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from zno_pd_qml.lammps_dump import Box


def min_image_displacement(
    dr: np.ndarray,
    box: Optional[Box],
) -> np.ndarray:
    """
    Apply minimum-image convention to displacement vectors in periodic dimensions.

    Assumptions:
    - Orthorhombic box (typical LAMMPS dump BOX BOUNDS).
    - If box is None: no PBC applied.
    """
    if box is None:
        return dr
    dr = np.asarray(dr, dtype=float)
    L = box.lengths
    p = box.periodic
    out = dr.copy()
    for dim in range(3):
        if p[dim]:
            out[..., dim] -= np.round(out[..., dim] / L[dim]) * L[dim]
    return out


def pairwise_distances_min_image(
    A: np.ndarray,
    B: np.ndarray,
    box: Optional[Box],
    *,
    block: int = 512,
) -> np.ndarray:
    """
    Compute |A_i - B_j| with minimum-image in periodic dims.

    Returns:
      D: (len(A), len(B))

    Note:
      This is O(N*M) but uses blocking to avoid huge temporary arrays.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n, m = A.shape[0], B.shape[0]
    D = np.empty((n, m), dtype=float)
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        # (bi, 1, 3) - (1, m, 3) -> (bi, m, 3)
        dr = A[i0:i1, None, :] - B[None, :, :]
        dr = min_image_displacement(dr, box)
        D[i0:i1] = np.sqrt(np.sum(dr * dr, axis=-1))
    return D


def min_distance_to_set(
    A: np.ndarray,
    B: np.ndarray,
    box: Optional[Box],
    *,
    block: int = 512,
) -> np.ndarray:
    """For each a in A, return min_j |a - B_j| (with min-image in periodic dims)."""
    if A.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    if B.shape[0] == 0:
        return np.full((A.shape[0],), np.inf, dtype=float)
    D = pairwise_distances_min_image(A, B, box, block=block)
    return D.min(axis=1)


def inertia_tensor(positions: np.ndarray) -> np.ndarray:
    """
    Unweighted inertia tensor around the center of mass for a point cloud.
    """
    x = np.asarray(positions, dtype=float)
    if x.shape[0] == 0:
        return np.zeros((3, 3), dtype=float)
    c = x.mean(axis=0, keepdims=True)
    r = x - c
    xx, yy, zz = (r[:, 0] ** 2), (r[:, 1] ** 2), (r[:, 2] ** 2)
    xy, xz, yz = (r[:, 0] * r[:, 1]), (r[:, 0] * r[:, 2]), (r[:, 1] * r[:, 2])
    I = np.array(
        [
            [float((yy + zz).sum()), -float(xy.sum()), -float(xz.sum())],
            [-float(xy.sum()), float((xx + zz).sum()), -float(yz.sum())],
            [-float(xz.sum()), -float(yz.sum()), float((xx + yy).sum())],
        ],
        dtype=float,
    )
    return I


def radius_of_gyration(positions: np.ndarray) -> float:
    x = np.asarray(positions, dtype=float)
    if x.shape[0] == 0:
        return 0.0
    c = x.mean(axis=0, keepdims=True)
    rg2 = np.mean(np.sum((x - c) ** 2, axis=1))
    return float(np.sqrt(rg2))


def asphericity_from_inertia(positions: np.ndarray) -> float:
    """
    A simple asphericity metric based on inertia eigenvalues.

    Here we use:
      b = ((λ1-λ2)^2 + (λ2-λ3)^2 + (λ3-λ1)^2) / (2*(λ1+λ2+λ3)^2)
    where λ are eigenvalues of inertia tensor.
    """
    I = inertia_tensor(positions)
    w = np.sort(np.linalg.eigvalsh(I))
    s = float(w.sum())
    if s < 1e-12:
        return 0.0
    b = float(((w[2] - w[1]) ** 2 + (w[1] - w[0]) ** 2 + (w[2] - w[0]) ** 2) / (2.0 * s * s))
    return b


