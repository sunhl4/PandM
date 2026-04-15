"""
PySCF classical backend.

Wraps PySCF HF or DFT into the ClassicalBackend interface, producing a
BackendResult that carries all integrals and orbital information needed by
downstream embedding and Hamiltonian-building steps.

Registered as ``"pyscf"`` in the ``"backend"`` category.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..core.interfaces import BackendResult, ClassicalBackend
from ..core.registry import registry

logger = logging.getLogger(__name__)


@registry.register("pyscf", category="backend")
class PySCFBackend(ClassicalBackend):
    """
    Run PySCF HF or DFT on a molecular geometry.

    Parameters
    ----------
    method : str
        ``"hf"`` for Hartree-Fock or ``"dft"`` for Kohn-Sham DFT.
    xc : str
        Exchange-correlation functional (ignored for HF), e.g. ``"pbe"``,
        ``"b3lyp"``, ``"lda,vwn"``.
    charge : int
        Total charge of the molecule.
    spin : int
        2S (number of unpaired electrons), 0 for closed-shell.
    verbose : int
        PySCF verbosity level (0 = silent, 3 = normal, 5 = debug).
    conv_tol : float
        SCF convergence threshold.
    max_cycle : int
        Maximum number of SCF cycles.
    density_fit : bool
        If True, use PySCF density fitting (RI/J) for the Coulomb part of J/K,
        which speeds up SCF for larger bases. Approximate vs exact ERIs.
    auxbasis : str or None
        Auxiliary basis for density fitting (e.g. ``"weigend"``). If None,
        PySCF chooses a default compatible with ``basis``.
    """

    def __init__(
        self,
        method: str = "hf",
        xc: str = "pbe",
        charge: int = 0,
        spin: int = 0,
        verbose: int = 0,
        conv_tol: float = 1e-9,
        max_cycle: int = 100,
        density_fit: bool = False,
        auxbasis: str | None = None,
        level_shift: float = 0.0,
        init_guess: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.method = method.lower()
        self.xc = xc
        self.charge = charge
        self.spin = spin
        self.verbose = verbose
        self.conv_tol = conv_tol
        self.max_cycle = max_cycle
        self.density_fit = bool(density_fit)
        self.auxbasis = auxbasis
        self.level_shift = float(level_shift)
        self.init_guess = init_guess  # e.g. "atom", "minao", None → PySCF default

    def run(self, geometry: str, basis: str, **kwargs) -> BackendResult:
        """
        Execute HF or DFT via PySCF.

        Parameters
        ----------
        geometry : str
            Atom specification, e.g. ``"H 0 0 0; H 0 0 0.735"``.
        basis : str
            Basis set, e.g. ``"sto-3g"``, ``"def2-svp"``.
        **kwargs :
            Override ``charge``, ``spin``, ``method``, ``xc`` at call-time.
        """
        try:
            from pyscf import gto, scf, dft
        except ImportError as exc:
            raise ImportError(
                "PySCF is required for PySCFBackend. "
                "Install with: pip install pyscf"
            ) from exc

        # Allow per-call overrides
        method  = kwargs.get("method",  self.method).lower()
        xc      = kwargs.get("xc",      self.xc)
        charge  = kwargs.get("charge",  self.charge)
        spin    = kwargs.get("spin",    self.spin)
        density_fit = bool(kwargs.get("density_fit", self.density_fit))
        auxbasis = kwargs.get("auxbasis", self.auxbasis)

        # --- Build Mole ---
        mol = gto.Mole()
        mol.atom   = geometry
        mol.basis  = basis
        mol.charge = charge
        mol.spin   = spin
        mol.verbose = self.verbose
        mol.build()

        # --- Mean-field ---
        if method == "hf":
            mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
        elif method == "dft":
            mf = dft.RKS(mol) if spin == 0 else dft.ROKS(mol)
            mf.xc = xc
        else:
            raise ValueError(f"Unsupported method '{method}'. Choose 'hf' or 'dft'.")

        if density_fit:
            df_kw: dict[str, Any] = {}
            if auxbasis:
                df_kw["auxbasis"] = auxbasis
            mf = mf.density_fit(**df_kw)
            logger.info(
                "Using density fitting for SCF (auxbasis=%s)",
                auxbasis or "default",
            )

        mf.conv_tol  = self.conv_tol
        mf.max_cycle = self.max_cycle
        if self.level_shift:
            mf.level_shift = self.level_shift
            logger.info("SCF level_shift=%.3f applied.", self.level_shift)

        if self.init_guess:
            dm0 = mf.init_guess_by_atom() if self.init_guess == "atom" else None
            e_hf = mf.kernel(dm0)
        else:
            e_hf = mf.kernel()

        if not mf.converged:
            logger.warning("SCF did not converge for geometry: %s", geometry[:60])

        # --- Integrals in AO basis ---
        ovlp    = mol.intor("int1e_ovlp")
        h1e_ao  = mol.intor("int1e_nuc") + mol.intor("int1e_kin")

        # 2e integrals: only compute for small systems (nao ≤ 50)
        nao = mol.nao_nr()
        if nao <= 50:
            h2e_ao = mol.intor("int2e")  # shape (nao, nao, nao, nao)
        else:
            h2e_ao = None
            logger.info(
                "Skipping full 2e AO integral storage (nao=%d > 50). "
                "Active-space transforms still use ao2mo.kernel on subsets; "
                "for faster SCF on large systems set backend.density_fit: true.",
                nao,
            )

        # --- Electron count ---
        mo_occ = mf.mo_occ
        n_alpha = int(np.sum(mo_occ > 0))  # RHF: all occupied MOs
        n_beta  = n_alpha if spin == 0 else int(np.sum(mo_occ == 1))
        if spin != 0:
            n_alpha = int(np.sum(mo_occ >= 1))
            n_beta  = int(np.sum(mo_occ == 2))

        n_from_mo = n_alpha + n_beta
        if mol.nelectron != n_from_mo:
            logger.warning(
                "Electron count mismatch: mol.nelectron=%d vs n_alpha+n_beta=%d "
                "(check spin/charge and reference type).",
                mol.nelectron,
                n_from_mo,
            )

        logger.info(
            "PySCF %s/%s  E=%.10f Ha  nalpha=%d  nbeta=%d  SCF_converged=%s",
            method.upper(),
            basis,
            e_hf,
            n_alpha,
            n_beta,
            mf.converged,
        )

        return BackendResult(
            mol        = mol,
            energy_hf  = e_hf,
            mo_coeff   = mf.mo_coeff,
            mo_occ     = mf.mo_occ,
            mo_energy  = mf.mo_energy,
            ovlp       = ovlp,
            h1e_ao     = h1e_ao,
            h2e_ao     = h2e_ao,
            nelec      = (n_alpha, n_beta),
            mf         = mf,
            scf_converged=bool(mf.converged),
        )
