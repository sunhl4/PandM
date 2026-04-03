"""
MoleculeBuilder — converts a MolSpec into active-space MO integrals.

The builder runs a PySCF RHF/ROHF calculation, optionally freezes core
orbitals, and extracts one- and two-electron integrals in the active MO
basis together with the frozen-core energy correction.

Usage::

    from quantum_chem_bench.molecule.builder import MoleculeBuilder
    from quantum_chem_bench.core.interfaces import MolSpec

    spec = MolSpec(geometry="H 0 0 0; H 0 0 0.735", basis="sto-3g")
    integrals = MoleculeBuilder().build(spec)
    print(integrals.hf_energy, integrals.norb, integrals.nelec)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from quantum_chem_bench.core.interfaces import MolIntegrals, MolSpec

logger = logging.getLogger(__name__)


class MoleculeBuilder:
    """
    Build MolIntegrals from a MolSpec using PySCF.

    Parameters
    ----------
    verbose : int
        PySCF verbosity level (0 = silent).
    """

    def __init__(self, verbose: int = 0) -> None:
        self.verbose = verbose

    def build(self, spec: MolSpec) -> MolIntegrals:
        """
        Run HF and extract active-space integrals.

        Parameters
        ----------
        spec : MolSpec

        Returns
        -------
        MolIntegrals
        """
        try:
            from pyscf import gto, scf, ao2mo
        except ImportError as exc:
            raise ImportError("PySCF is required: pip install pyscf") from exc

        # Build PySCF molecule
        mol = gto.Mole()
        mol.atom = spec.geometry
        mol.basis = spec.basis
        mol.charge = spec.charge
        mol.spin = spec.spin
        mol.verbose = self.verbose
        mol.build()

        # Run HF / ROHF (optional density fitting for larger systems)
        if spec.spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.ROHF(mol)
        if getattr(spec, "density_fit", False):
            mf = mf.density_fit(auxbasis=spec.auxbasis)
        mf.kernel()
        if not mf.converged:
            logger.warning("HF did not converge for %s", spec.geometry[:40])

        hf_energy = float(mf.e_tot)
        mo_coeff = mf.mo_coeff
        nmo = mo_coeff.shape[1]

        # Determine active space
        nae = spec.n_active_electrons
        nao = spec.n_active_orbitals

        if nae is None or nao is None:
            # Use full orbital space (minus frozen core via CASSCF helper)
            n_alpha, n_beta = mol.nelec
            nae = (n_alpha, n_beta)
            nao = nmo
            logger.debug(
                "Active space: full (%d spatial orbitals, %d+%d electrons)",
                nao, n_alpha, n_beta,
            )
        else:
            n_alpha, n_beta = int(nae[0]), int(nae[1])
            nao = int(nao)
            logger.debug(
                "Active space: %d orbitals, %d+%d electrons", nao, n_alpha, n_beta
            )

        # Build active-space MO integrals
        # Determine the active orbital slice: HOMO-centred
        n_elec_total = mol.nelectron
        n_frozen = (n_elec_total - n_alpha - n_beta) // 2
        active_mo_idx = list(range(n_frozen, n_frozen + nao))
        active_mo_coeff = mo_coeff[:, active_mo_idx]

        h1e_ao = mf.get_hcore()
        h1e = active_mo_coeff.T @ h1e_ao @ active_mo_coeff

        # Two-electron integrals (chemist notation)
        h2e = ao2mo.kernel(mol, active_mo_coeff, compact=False)
        h2e = h2e.reshape(nao, nao, nao, nao)

        # Core energy: nuclear repulsion + frozen-core contribution
        e_core = mol.energy_nuc()
        if n_frozen > 0:
            frozen_mo = mo_coeff[:, :n_frozen]
            dm_core = 2.0 * frozen_mo @ frozen_mo.T
            h1e_full = mf.get_hcore()
            veff_core = mf.get_veff(mol, dm_core)
            e_core += 0.5 * float(np.einsum("ij,ji->", h1e_full + h1e_full + veff_core, dm_core))
            # Subtract frozen contribution from h1e
            h1e = h1e + active_mo_coeff.T @ veff_core @ active_mo_coeff

        return MolIntegrals(
            h1e=h1e,
            h2e=h2e,
            nelec=(n_alpha, n_beta),
            norb=nao,
            e_core=e_core,
            hf_energy=hf_energy,
            mo_coeff=active_mo_coeff,
            mol=mol,
            mf=mf,
        )
