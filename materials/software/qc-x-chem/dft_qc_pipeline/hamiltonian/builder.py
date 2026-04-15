"""
Fragment Hamiltonian builder.

Converts a set of active MO indices + PySCF integrals into a Qiskit Nature
``FermionicOp`` (which can then be mapped to qubits).

The effective one-body operator includes:
  * Kinetic + nuclear attraction in the active space.
  * Fock-matrix contribution from doubly-occupied (frozen) core orbitals.
  * The nuclear repulsion energy is included in ``e_core``.

Reference: Helgaker, Jørgensen, Olsen "Molecular Electronic Structure Theory"
Chapter 11 (2-electron integrals in MO basis).
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from ..core.interfaces import BackendResult, EmbeddedHamiltonian

logger = logging.getLogger(__name__)


def build_fragment_hamiltonian(
    backend_result: BackendResult,
    active_mo_indices: Sequence[int],
    nelec_frag: tuple[int, int],
    e_core_correction: float = 0.0,
    region_name: str = "fragment",
) -> EmbeddedHamiltonian:
    """
    Build the fragment Hamiltonian from PySCF integrals.

    The MO integrals are computed (or re-used from a cached AO set) for the
    given *active_mo_indices*.  Frozen (core) orbital contributions are
    folded into the 1e part.

    Parameters
    ----------
    backend_result : BackendResult
    active_mo_indices : sequence of int
        Column indices of the active MOs in ``backend_result.mo_coeff``.
    nelec_frag : (int, int)
        (n_alpha, n_beta) in the active space.
    e_core_correction : float
        Additional energy to add to ``EmbeddedHamiltonian.e_core``
        (e.g. from DMET chemical-potential shift or environment DFT energy).
    region_name : str

    Returns
    -------
    EmbeddedHamiltonian
    """
    try:
        from pyscf import ao2mo
    except ImportError as exc:
        raise ImportError("PySCF required for integral transformation.") from exc

    mol = backend_result.mol
    mf  = backend_result.mf
    C   = backend_result.mo_coeff           # (nao, nmo)
    nmo = C.shape[1]
    nao_full = mol.nao_nr()
    if nao_full > 50:
        logger.info(
            "build_fragment_hamiltonian '%s': nao=%d (>50); 2e integrals use ao2mo on "
            "active columns only (norb=%d), not full AO (see also backend.density_fit).",
            region_name,
            nao_full,
            len(active_mo_indices),
        )

    active_idx = list(active_mo_indices)
    norb = len(active_idx)
    C_act = C[:, active_idx]               # (nao, norb)

    # --- All occupied (for core contribution) ---
    occ_mask = backend_result.mo_occ > 0
    all_occ_idx = list(np.where(occ_mask)[0])
    core_idx = [i for i in all_occ_idx if i not in active_idx]

    # --- 1e integrals in MO active basis ---
    h1e_ao = backend_result.h1e_ao         # (nao, nao)
    h1e = C_act.T @ h1e_ao @ C_act         # (norb, norb)

    # --- 2e integrals in active MO basis (8-fold symmetry → full tensor) ---
    h2e = ao2mo.kernel(mol, C_act, compact=False)
    h2e = h2e.reshape(norb, norb, norb, norb)

    # --- Core energy + Fock contribution from frozen occupied MOs ---
    e_core = mol.energy_nuc() + e_core_correction
    h1e_eff = h1e.copy()

    if core_idx:
        C_core = C[:, core_idx]                    # (nao, n_core)
        dm_core = 2.0 * C_core @ C_core.T          # (nao, nao) closed-shell core density

        # Core energy contribution: Tr[h * dm_core] + 0.5 * J - 0.25 * K
        e_core += float(np.einsum("ij,ji->", h1e_ao, dm_core))

        # Get AO Fock matrix from mean-field (includes J+K for full density)
        # Then subtract the active part so only core→active coupling remains
        fock_ao = mf.get_fock()                    # (nao, nao)
        fock_core = C_act.T @ fock_ao @ C_act      # (norb, norb)  full Fock in act basis
        fock_active_only = h1e.copy()
        # Add core-induced effective 1e operator: v_ij = 2*(ij|kk) - (ik|kj) for k in core
        for k in core_idx:
            C_k = C[:, k]
            # Coulomb: 2 * <ij|kk>
            J = ao2mo.kernel(mol, [C_act, C_act, C_k[:, None], C_k[:, None]], compact=False)
            J = J.reshape(norb, norb)
            # Exchange: <ik|jk>
            K = ao2mo.kernel(mol, [C_act, C_k[:, None], C_act, C_k[:, None]], compact=False)
            K = K.reshape(norb, norb)
            h1e_eff += 2.0 * J - K
            e_core += 2.0 * float(np.einsum("ij,ji->", h1e_ao, np.outer(C_k, C_k)))

        logger.debug(
            "Core energy (nuc + core frozen): %.8f Ha, n_core=%d",
            e_core, len(core_idx),
        )

    logger.info(
        "Fragment Hamiltonian '%s': norb=%d, nelec=%s, e_core=%.8f Ha",
        region_name, norb, nelec_frag, e_core,
    )

    return EmbeddedHamiltonian(
        h1e=h1e_eff,
        h2e=h2e,
        nelec=nelec_frag,
        norb=norb,
        e_core=e_core,
        region_name=region_name,
    )


def hamiltonian_to_fermionic_op(emb_H: EmbeddedHamiltonian):
    """
    Convert an EmbeddedHamiltonian to a Qiskit Nature ``FermionicOp``.

    Returns
    -------
    tuple[FermionicOp, float]
        (fermionic_op, e_core)

    The ``e_core`` shift must be added to the solver's eigenvalue to get the
    total fragment energy.
    """
    try:
        from qiskit_nature.second_q.operators import FermionicOp
    except ImportError as exc:
        raise ImportError(
            "qiskit-nature is required: pip install qiskit-nature"
        ) from exc

    norb = emb_H.norb
    h1e  = emb_H.h1e
    h2e  = emb_H.h2e   # chemist notation: (pq|rs)

    data: dict[str, complex] = {}

    # --- 1e terms: h_pq  a†_p a_q ---
    for p in range(norb):
        for q in range(norb):
            for spin_shift in (0, norb):  # alpha block then beta block
                val = h1e[p, q]
                if abs(val) > 1e-14:
                    key = f"+_{p + spin_shift} -_{q + spin_shift}"
                    data[key] = data.get(key, 0.0) + val

    # --- 2e terms: 0.5 * (pq|rs) a†_p a†_r a_s a_q  (chemist → normal order) ---
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    val = 0.5 * h2e[p, q, r, s]
                    if abs(val) > 1e-14:
                        # spin-orbital indices: alpha=0..norb-1, beta=norb..2*norb-1
                        for sa in (0, norb):
                            for sb in (0, norb):
                                key = (
                                    f"+_{p+sa} +_{r+sb} "
                                    f"-_{s+sb} -_{q+sa}"
                                )
                                data[key] = data.get(key, 0.0) + val

    return FermionicOp(data, num_spin_orbitals=2 * norb), emb_H.e_core
