"""
Projector-based (Manby-Werner style) WF-in-DFT embedding.

Theory
------
The full system is partitioned into fragment A and environment B.
A level-shift projection operator μ·P_B is added to the fragment Hamiltonian
to enforce orthogonality between fragment and environment orbitals::

    h_A-in-B = h_full - v_B + μ * P_B

where v_B is the mean-field potential from B acting on A (Coulomb + XC
correction from subsystem DFT or HF), and P_B = S C_B C_B† S is the
projector onto the environment orbital space.

Implementation
--------------
**DFT note**: ``v_B`` is built from ``F_full - h_core`` (HF-like). For
Kohn–Sham DFT this omits explicit XC-kernel corrections to the embedding
potential; treat energies as approximate when ``method: dft``.

1. Run full-system DFT → canonical MOs.
2. Localize occupied MOs (Boys / PM / IAO).
3. Assign localized MOs to A or B based on atom_indices.
4. Build h_eff for fragment A including the level-shift term.
5. Treat virtual space with a threshold (include all virtuals for now).

Reference: Manby et al., JCTC 8, 2564 (2012);
           Bennie et al., JCTC 12, 2689 (2016).

Registered as ``"projector"`` in the ``"embedding"`` category.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.linalg as la

from ..core.interfaces import BackendResult, EmbeddedHamiltonian, EmbeddingMethod
from ..core.registry import registry
from ..hamiltonian.builder import build_fragment_hamiltonian
from ..hamiltonian.fragment_region import FragmentRegion
from ..hamiltonian.localizer import get_atom_orbital_indices, localize_orbitals

logger = logging.getLogger(__name__)


@registry.register("projector", category="embedding")
class ProjectorEmbedding(EmbeddingMethod):
    """
    Projection-based WF-in-DFT embedding.

    Parameters
    ----------
    mu : float
        Level-shift parameter (in Hartree).  Typical value: 1e6 (hard projection).
    max_iter : int
        Kept for API symmetry; one-shot.
    conv_tol : float
        Kept for API symmetry.
    """

    def __init__(
        self,
        mu: float = 1.0e6,
        max_iter: int = 1,
        conv_tol: float = 1e-6,
        **kwargs,
    ) -> None:
        self.mu = mu

    def embed(
        self,
        backend_result: BackendResult,
        region: FragmentRegion,
    ) -> EmbeddedHamiltonian:
        """
        Build the projected fragment Hamiltonian.
        """
        try:
            from pyscf import ao2mo
        except ImportError as exc:
            raise ImportError("PySCF required for ProjectorEmbedding.") from exc

        mol = backend_result.mol
        mf  = backend_result.mf
        ovlp = backend_result.ovlp
        h1e_ao = backend_result.h1e_ao

        if type(mf).__module__.startswith("pyscf.dft"):
            logger.warning(
                "[ProjectorEmbedding] region=%r: Kohn–Sham DFT backend detected. "
                "Embedding uses v_emb = F - h_core without explicit XC-kernel "
                "contributions to the embedding potential; treat energies as approximate.",
                region.name,
            )

        # Localize occupied MOs
        C_loc = localize_orbitals(mf, scheme=region.localization)
        mo_occ = backend_result.mo_occ
        occ_mask = mo_occ > 0

        # Identify fragment occupied orbitals (A) and environment (B)
        frag_occ_idx = get_atom_orbital_indices(
            mol=mol,
            C_loc=C_loc,
            atom_indices=region.atom_indices,
            mo_occ=mo_occ,
            n_orbs=region.norb,
        )
        all_occ_idx = list(np.where(occ_mask)[0])
        env_occ_idx = [i for i in all_occ_idx if i not in frag_occ_idx]

        # Also include virtual block for the active space
        virt_mask = mo_occ == 0
        virt_idx  = list(np.where(virt_mask)[0])
        # Use a few low-lying virtuals
        n_virt_use = max(1, region.norb - len(frag_occ_idx))
        virt_use   = virt_idx[:n_virt_use]

        active_idx = frag_occ_idx + virt_use
        norb = len(active_idx)
        C_act = C_loc[:, active_idx]            # (nao, norb)

        # --- Build level-shifted 1e operator ---
        # Environment projector in AO basis: P_B = S C_B C_B† S
        C_env = C_loc[:, env_occ_idx]           # (nao, n_env)
        P_B = ovlp @ C_env @ C_env.T @ ovlp    # (nao, nao)

        # Full Fock matrix from mean-field
        fock_ao = mf.get_fock()                 # (nao, nao)

        # Fragment-subsystem potential correction: Fock - hcore (= v_HXC or v_J+K)
        v_emb_ao = fock_ao - h1e_ao            # (nao, nao) embedding potential from full system

        # Effective 1e Hamiltonian: h_core + v_emb + μ * P_B
        h1e_eff_ao = h1e_ao + v_emb_ao + self.mu * P_B   # (nao, nao)
        h1e_eff    = C_act.T @ h1e_eff_ao @ C_act         # (norb, norb)

        # 2e integrals in fragment active space (bare, no DFT correction)
        h2e = ao2mo.kernel(mol, C_act, compact=False).reshape(norb, norb, norb, norb)

        # Core energy
        e_core = mol.energy_nuc()

        nelec_frag = (region.n_alpha, region.n_beta)
        logger.info(
            "[ProjectorEmbedding] '%s': norb=%d, nelec=%s, μ=%.1e",
            region.name, norb, nelec_frag, self.mu,
        )

        return EmbeddedHamiltonian(
            h1e=h1e_eff,
            h2e=h2e,
            nelec=nelec_frag,
            norb=norb,
            e_core=e_core,
            region_name=region.name,
            extra={"mu_projection": self.mu},
        )

    def update_from_rdm(self, rdm1: np.ndarray) -> bool:
        """One-shot."""
        return True
