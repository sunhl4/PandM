"""
Spatial one-particle reduced density matrix from a CI expansion in a Slater basis.

For orthonormal determinants |D_I⟩ built from spatial orbitals {φ_p} with
alpha/beta strings ``ia, ib`` (bit ``k`` = orbital ``k`` occupied for that spin),

    γ_pq = ⟨Ψ|E_pq|Ψ⟩,   E_pq = Σ_σ a†_{pσ} a_{qσ}

with |Ψ⟩ = Σ_I c_I |D_I⟩.  Expanding gives

    γ_pq = Σ_{IJ} c_I^* c_J ( δ(ib_I, ib_J) ⟨ia_I|a†_{pα}a_{qα}|ia_J⟩
                           + δ(ia_I, ia_J) ⟨ib_I|a†_{pβ}a_{qβ}|ib_J⟩ ).

Matrix elements of a†_p a_q between two alpha strings are standard Slater rules:
diagonal if ia==ja, or a single signed excitation if ia and ja differ by exactly
one spin-orbital swap.
"""

from __future__ import annotations

import numpy as np


def _lowest_bit_index(x: int) -> int:
    """Index of least-significant set bit (0-based)."""
    return (x & -x).bit_length() - 1


def _alpha_adag_a_matrix(ia: int, ja: int, norb: int) -> np.ndarray:
    """
    Matrix M with M[p,q] = ⟨ia|a†_{pα} a_{qα}|ja⟩ for spatial indices p,q.
    """
    out = np.zeros((norb, norb), dtype=np.float64)
    ia = int(ia)
    ja = int(ja)

    if ia == ja:
        for k in range(norb):
            if (ia >> k) & 1:
                out[k, k] = 1.0
        return out

    diff = ia ^ ja
    if diff.bit_count() != 2:
        return out

    holes = ja & ~ia
    parts = ia & ~ja
    if holes.bit_count() != 1 or parts.bit_count() != 1:
        return out

    q = _lowest_bit_index(holes)
    p = _lowest_bit_index(parts)

    lo, hi = (q, p) if q < p else (p, q)
    mask = ((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1)
    n_between = (ja & mask).bit_count()
    sign = -1.0 if (n_between % 2) else 1.0
    out[p, q] = sign
    return out


def subspace_1rdm_spatial(
    coeff: np.ndarray,
    ci_strs_a: np.ndarray,
    ci_strs_b: np.ndarray,
    norb: int,
) -> np.ndarray:
    """
    Full spatial 1-RDM (norb × norb) from CI coefficients in a Slater subspace.

    Parameters
    ----------
    coeff
        Complex or real CI vector (length = number of determinants).
    ci_strs_a, ci_strs_b
        Integer bitstrings for alpha/beta occupations (same length as coeff).
    norb
        Number of spatial orbitals represented in each string.
    """
    c = np.asarray(coeff, dtype=np.complex128).ravel()
    if c.size == 0:
        return np.zeros((norb, norb), dtype=np.float64)
    norm = np.linalg.norm(c)
    if norm < 1e-15:
        return np.zeros((norb, norb), dtype=np.float64)
    c = c / norm

    ia_arr = np.asarray(ci_strs_a, dtype=np.int64).ravel()
    ib_arr = np.asarray(ci_strs_b, dtype=np.int64).ravel()
    n_det = c.size
    if ia_arr.size < n_det or ib_arr.size < n_det:
        raise ValueError("CI string arrays must cover all determinant indices")

    gamma = np.zeros((norb, norb), dtype=np.float64)

    for i in range(n_det):
        for j in range(n_det):
            w = np.conj(c[i]) * c[j]
            if abs(w) < 1e-16:
                continue
            ia_i, ia_j = int(ia_arr[i]), int(ia_arr[j])
            ib_i, ib_j = int(ib_arr[i]), int(ib_arr[j])

            if ib_i == ib_j:
                gamma += np.real(w) * _alpha_adag_a_matrix(ia_i, ia_j, norb)
            if ia_i == ia_j:
                gamma += np.real(w) * _alpha_adag_a_matrix(ib_i, ib_j, norb)

    # Symmetrize against numerical drift (exact γ should be Hermitian)
    gamma = 0.5 * (gamma + gamma.T)
    return gamma
