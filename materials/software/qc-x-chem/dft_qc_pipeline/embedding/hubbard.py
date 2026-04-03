"""
One-band Hubbard model as an ``EmbeddedHamiltonian`` (no PySCF integrals).

::

    H = -t \\sum_{\\langle i,j\\rangle,\\sigma} (c^\\dagger_{i\\sigma} c_{j\\sigma} + h.c.)
        + U \\sum_i n_{i\\uparrow} n_{i\\downarrow}

Open boundary: nearest-neighbour hopping along sites ``0..n_sites-1``.
Two-site ring (periodic) for ``n_sites==2`` adds the wrap ``(L-1)–0`` bond
identical to the existing ``0–1`` link (deduplicated).

On-site repulsion is encoded as **chemist** (ii|ii) = ``U`` for each spatial
site ``i``, consistent with ``hamiltonian.builder.hamiltonian_to_fermionic_op``.

Registered as ``"hubbard"`` in the ``"embedding"`` category.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..core.interfaces import BackendResult, EmbeddedHamiltonian, EmbeddingMethod
from ..core.registry import registry
from ..hamiltonian.fragment_region import FragmentRegion

logger = logging.getLogger(__name__)


@registry.register("hubbard", category="embedding")
class HubbardEmbedding(EmbeddingMethod):
    """
    Build a one-band Hubbard Hamiltonian in the site / spatial-orbital basis.

    Parameters
    ----------
    t : float
        Nearest-neighbour hopping (positive energy scale; off-diagonal ``-t``).
    U : float
        On-site Hubbard ``U`` (chemist ``(ii|ii)``).
    n_sites : int
        Number of spatial orbitals (sites). Must match ``region.norb``.
    periodic : bool
        If True and ``n_sites > 2``, add ``h_{0,L-1}`` hopping. For ``L==2`` the
        ring is already spanned by the single bond.
    """

    def __init__(
        self,
        t: float = 1.0,
        U: float = 4.0,
        n_sites: int = 2,
        periodic: bool = False,
        max_iter: int = 1,
        conv_tol: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        self.t = float(t)
        self.U = float(U)
        self.n_sites = int(n_sites)
        self.periodic = bool(periodic)

    def embed(
        self,
        backend_result: BackendResult,
        region: FragmentRegion,
    ) -> EmbeddedHamiltonian:
        L = self.n_sites
        if region.norb != L:
            raise ValueError(
                f"[Hubbard] region.norb={region.norb} must match embedding n_sites={L}"
            )
        na, nb = region.n_alpha, region.n_beta
        if na + nb != region.nelec:
            raise ValueError("[Hubbard] region.nelec inconsistent with n_alpha/n_beta")

        h1e = np.zeros((L, L), dtype=float)
        for i in range(L - 1):
            h1e[i, i + 1] = h1e[i + 1, i] = -self.t
        if self.periodic and L > 2:
            h1e[0, L - 1] = h1e[L - 1, 0] = -self.t

        h2e = np.zeros((L, L, L, L), dtype=float)
        for i in range(L):
            h2e[i, i, i, i] = self.U

        logger.info(
            "[Hubbard] '%s': L=%d, t=%.4f, U=%.4f, nelec=(%d,%d)",
            region.name,
            L,
            self.t,
            self.U,
            na,
            nb,
        )

        return EmbeddedHamiltonian(
            h1e=h1e,
            h2e=h2e,
            nelec=(na, nb),
            norb=L,
            e_core=0.0,
            region_name=region.name,
            extra={
                "model": "hubbard_1band",
                "t": self.t,
                "U": self.U,
                "periodic": self.periodic,
            },
        )

    def update_from_rdm(self, rdm1: np.ndarray) -> bool:
        return True
