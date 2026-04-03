"""
Minimal classical backend for models that do not need PySCF (e.g. Hubbard).

Returns an identity MO basis and zero integrals; the real physics lives in a
specialised ``EmbeddingMethod`` (see ``embedding/hubbard.py``).

Registered as ``"toy"`` in the ``"backend"`` category.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.interfaces import BackendResult, ClassicalBackend
from ..core.registry import registry


class _ToyMol:
    """Enough surface for logging / optional introspection."""

    __slots__ = ("norb",)

    def __init__(self, norb: int) -> None:
        self.norb = int(norb)

    def nao_nr(self) -> int:
        return self.norb

    def energy_nuc(self) -> float:
        return 0.0

    @property
    def atom(self) -> tuple:
        return ()


class _ToyMF:
    __slots__ = ("_n",)

    def __init__(self, n: int) -> None:
        self._n = n

    def make_rdm1(self):
        return np.zeros((self._n, self._n))

    def get_fock(self):
        return np.zeros((self._n, self._n))

    def get_ovlp(self):
        return np.eye(self._n)


@registry.register("toy", category="backend")
class ToyBackend(ClassicalBackend):
    """
    Placeholder backend: identity MOs, zero HF energy, no AO integrals.

    Parameters
    ----------
    norb : int
        Number of spatial orbitals (must match Hubbard / model embedding).
    """

    def __init__(self, norb: int = 2, **kwargs: Any) -> None:
        self.norb = int(norb)

    def run(self, geometry: str = "", basis: str = "", **kwargs: Any) -> BackendResult:
        n = int(kwargs.get("norb", self.norb))
        C = np.eye(n)
        occ = np.zeros(n)
        ovlp = np.eye(n)
        h1e_ao = np.zeros((n, n))
        mol = _ToyMol(n)
        mf = _ToyMF(n)
        return BackendResult(
            mol=mol,
            energy_hf=0.0,
            mo_coeff=C,
            mo_occ=occ,
            mo_energy=np.zeros(n),
            ovlp=ovlp,
            h1e_ao=h1e_ao,
            h2e_ao=None,
            nelec=(0, 0),
            mf=mf,
            scf_converged=True,
            extra={"backend": "toy"},
        )
