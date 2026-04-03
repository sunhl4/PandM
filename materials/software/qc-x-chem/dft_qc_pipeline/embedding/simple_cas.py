"""
SimpleCAS embedding – baseline (no embedding).

This method bypasses any embedding machinery and directly selects an active
space by:

1. (Optionally) localizing the occupied MOs.
2. Using atom-projected population or explicit orbital indices from the region.
3. Building the fragment Hamiltonian with frozen-core folding.

Use this as a reference/baseline against DMET or projector-based methods.

Registered as ``"simple_cas"`` in the ``"embedding"`` category.
"""

from __future__ import annotations

import logging

import numpy as np

from ..core.interfaces import BackendResult, EmbeddedHamiltonian, EmbeddingMethod
from ..core.registry import registry
from ..hamiltonian.builder import build_fragment_hamiltonian
from ..hamiltonian.fragment_region import FragmentRegion
from ..hamiltonian.localizer import get_atom_orbital_indices, localize_orbitals

logger = logging.getLogger(__name__)


@registry.register("simple_cas", category="embedding")
class SimpleCASEmbedding(EmbeddingMethod):
    """
    Simplest possible active-space extraction: CAS on selected atoms/orbitals,
    no bath, no self-consistency.

    Parameters
    ----------
    max_iter : int
        Ignored (one-shot), kept for API symmetry.
    conv_tol : float
        Ignored.
    """

    def __init__(self, max_iter: int = 1, conv_tol: float = 1e-6, **kwargs) -> None:
        pass  # one-shot, no state needed

    def embed(
        self,
        backend_result: BackendResult,
        region: FragmentRegion,
    ) -> EmbeddedHamiltonian:
        """
        Extract a CAS Hamiltonian for the region.

        Steps
        -----
        1. Localize occupied MOs according to ``region.localization``.
        2. Select ``region.norb`` active MOs by atom population or explicit indices.
        3. Build ``EmbeddedHamiltonian`` with frozen-core folding.
        """
        mf = backend_result.mf

        # Step 1: Localize
        C_loc = localize_orbitals(mf, scheme=region.localization)

        # Step 2: Select active MOs
        if region.orbital_indices:
            active_idx = list(region.orbital_indices)
            logger.info(
                "[SimpleCAS] '%s': using explicit orbital indices %s",
                region.name, active_idx,
            )
        else:
            active_idx = get_atom_orbital_indices(
                mol        = backend_result.mol,
                C_loc      = C_loc,
                atom_indices = region.atom_indices,
                mo_occ     = backend_result.mo_occ,
                n_orbs     = region.norb,
            )
            logger.info(
                "[SimpleCAS] '%s': atom-projected active MOs: %s",
                region.name, active_idx,
            )

        # Step 3: Build Hamiltonian
        nelec_frag = (region.n_alpha, region.n_beta)
        # Temporarily patch mo_coeff with localized version for integral transform
        original_coeff = backend_result.mo_coeff
        backend_result.mo_coeff = C_loc
        try:
            emb_H = build_fragment_hamiltonian(
                backend_result  = backend_result,
                active_mo_indices = active_idx,
                nelec_frag      = nelec_frag,
                region_name     = region.name,
            )
        finally:
            backend_result.mo_coeff = original_coeff

        return emb_H

    def update_from_rdm(self, rdm1: np.ndarray) -> bool:
        """One-shot: always converged."""
        return True
