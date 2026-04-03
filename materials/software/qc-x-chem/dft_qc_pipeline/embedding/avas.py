"""
AVAS (Atomic Valence Active Space) embedding.

Uses PySCF's ``mcscf.avas`` module to automatically select an active space
based on atomic valence orbital projectors.  The selection is one-shot
(no self-consistency loop), making it ideal for catalytic active sites where
you know which atom types contribute the valence chemistry.

Reference: Sayfutyarova et al., JCTC 13, 4063 (2017).

Registered as ``"avas"`` in the ``"embedding"`` category.
"""

from __future__ import annotations

import logging

import numpy as np

from ..core.interfaces import BackendResult, EmbeddedHamiltonian, EmbeddingMethod
from ..core.registry import registry
from ..hamiltonian.builder import build_fragment_hamiltonian
from ..hamiltonian.fragment_region import FragmentRegion

logger = logging.getLogger(__name__)


@registry.register("avas", category="embedding")
class AVASEmbedding(EmbeddingMethod):
    """
    AVAS active-space selection.

    Automatically determines the active space from the overlap between
    occupied/virtual canonical MOs and a set of reference AOs
    (typically the valence d-orbitals of a metal, π system, etc.).

    Parameters
    ----------
    threshold : float
        AVAS threshold; orbitals with occupation ≥ threshold are included
        in the active space.
    canonicalize : bool
        If True (default), canonicalize the active space after selection.
    max_iter : int
        Kept for API symmetry; AVAS is one-shot.
    conv_tol : float
        Kept for API symmetry.
    """

    def __init__(
        self,
        threshold: float = 0.2,
        canonicalize: bool = True,
        max_iter: int = 1,
        conv_tol: float = 1e-6,
        ao_labels: list[str] | None = None,
        **kwargs,
    ) -> None:
        self.threshold = threshold
        self.canonicalize = canonicalize
        self.ao_labels = list(ao_labels) if ao_labels else None

    def embed(
        self,
        backend_result: BackendResult,
        region: FragmentRegion,
    ) -> EmbeddedHamiltonian:
        """
        Run AVAS to select the active space and build the fragment Hamiltonian.
        """
        try:
            from pyscf.mcscf import avas
        except ImportError as exc:
            raise ImportError(
                "PySCF ≥ 2.0 required for AVAS: pip install pyscf"
            ) from exc

        mol = backend_result.mol
        mf  = backend_result.mf

        # Build a list of reference AO labels (explicit list overrides heuristics)
        aolabels = (
            list(self.ao_labels)
            if self.ao_labels
            else self._get_ao_labels(mol, region)
        )
        logger.info(
            "[AVAS] '%s': AO labels for selection: %s", region.name, aolabels[:10]
        )

        # Run AVAS
        ncas, nelecas, mo_avas = avas.avas(
            mf,
            aolabels,
            threshold=self.threshold,
            canonicalize=self.canonicalize,
            verbose=mol.verbose,
        )
        logger.info(
            "[AVAS] '%s': selected ncas=%d, nelecas=%d",
            region.name, ncas, sum(nelecas) if hasattr(nelecas, '__iter__') else nelecas,
        )

        # Override region sizes if AVAS found different values
        norb_use = min(ncas, region.norb)
        nelec_int = nelecas if isinstance(nelecas, int) else int(np.sum(nelecas))
        nelec_frag = ((nelec_int + 1) // 2, nelec_int // 2)

        # Identify active MO indices in the AVAS-reordered mo_avas
        # AVAS returns the full MO set reordered; active orbitals are
        # in the first ncas columns of the occupied block by convention.
        occ_act_idx = list(range(norb_use))

        # Patch mo_coeff for integral transform
        original_coeff = backend_result.mo_coeff
        backend_result.mo_coeff = mo_avas
        try:
            emb_H = build_fragment_hamiltonian(
                backend_result   = backend_result,
                active_mo_indices = occ_act_idx,
                nelec_frag       = nelec_frag,
                region_name      = region.name,
            )
        finally:
            backend_result.mo_coeff = original_coeff

        return emb_H

    def update_from_rdm(self, rdm1: np.ndarray) -> bool:
        """One-shot: always converged."""
        return True

    @staticmethod
    def _get_ao_labels(mol, region: FragmentRegion) -> list[str]:
        """
        Build PySCF AO label strings for the selected atoms.

        Returns patterns like ``["Fe 3d", "Fe 4s"]`` that AVAS uses
        to project.  If atom_indices is empty, return all heavy-atom
        d-orbitals as a fallback.
        """
        if not region.atom_indices:
            # Fallback: all transition-metal-like atoms
            labels = []
            for i, atom in enumerate(mol.atom):
                sym = atom[0] if isinstance(atom, (list, tuple)) else atom.split()[0]
                if sym not in ("H", "C", "N", "O", "F", "Cl", "Br"):
                    labels.append(f"{sym} 3d")
            return labels or ["d"]

        labels = []
        for iat in region.atom_indices:
            sym = mol._atom[iat][0]
            # Include typical valence AOs
            labels.append(f"{sym} 3d")
            labels.append(f"{sym} 4s")
        return labels
