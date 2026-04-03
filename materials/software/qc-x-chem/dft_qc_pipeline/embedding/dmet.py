"""
DMET (Density Matrix Embedding Theory) embedding.

Algorithm overview
------------------
1. Run a low-level (mean-field) calculation on the full system → 1-RDM ``D``.
2. For each fragment:
   a. Perform Schmidt decomposition of the **occupied** mean-field space
      partitioned as fragment + bath.
   b. Build an impurity+bath Hamiltonian H_imp using the integrals projected
      onto this small space.
   c. Solve H_imp with a high-level quantum solver (VQE, SQD, …).
   d. Extract the fragment 1-RDM from the solution.
3. Match the fragment 1-RDM (quantum) to the mean-field prediction by tuning
   a local chemical-potential shift μ on the fragment.
4. Repeat until |ΔRDM| < conv_tol.

Reference: Knizia & Chan, PRL 109, 186404 (2012); Wouters et al., JCTC 12, 2706 (2016).

Registered as ``"dmet"`` in the ``"embedding"`` category.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.linalg as la

from ..core.interfaces import BackendResult, EmbeddedHamiltonian, EmbeddingMethod
from ..core.registry import registry
from ..hamiltonian.fragment_region import FragmentRegion

logger = logging.getLogger(__name__)


@registry.register("dmet", category="embedding")
class DMETEmbedding(EmbeddingMethod):
    """
    DMET with Schmidt-decomposition bath construction and 1-RDM self-consistency.

    Parameters
    ----------
    max_iter : int
        Maximum DMET self-consistency iterations.
    conv_tol : float
        Convergence criterion: max|ΔRDM1| < conv_tol.
    mu_init : float
        Initial chemical-potential shift on the fragment.
    mu_step : float
        Step size for bisection/gradient update of μ.
    bath_threshold : float
        Singular-value threshold for bath orbital selection.
        Bath orbitals with SV < bath_threshold are discarded.
    mu_update : str
        How to update μ after each fragment solve: ``gradient`` (default),
        ``damped`` (gradient with tanh damping), ``bisection`` (regula falsi
        on (μ, pop_error)), or ``bisection_bracket`` (find smallest μ-interval
        in recent history with opposite-sign population error, then take the
        midpoint — more stable than pure gradient when μ oscillates).
    mu_max_abs : float or None
        If set, clamp ``|μ|`` to this value after each update (stability).
    """

    def __init__(
        self,
        max_iter: int = 20,
        conv_tol: float = 1e-5,
        mu_init: float = 0.0,
        mu_step: float = 0.05,
        bath_threshold: float = 1e-6,
        mu_update: str = "gradient",
        mu_max_abs: float | None = None,
        **kwargs,
    ) -> None:
        self.max_iter = max_iter
        self.conv_tol = conv_tol
        self.mu = mu_init
        self.mu_step = mu_step
        self.bath_threshold = bath_threshold
        self.mu_update = (mu_update or "gradient").lower()
        self.mu_max_abs = float(mu_max_abs) if mu_max_abs is not None else None

        # State carried between embed() and update_from_rdm()
        self._rdm1_mf_frag: np.ndarray | None = None   # MF 1-RDM on fragment
        self._iteration: int = 0
        self._last_rdm1_solver: np.ndarray | None = None
        self._mu_err_hist: list[tuple[float, float]] = []
        # μ·(N_MF − N_solver) at last update (Ha); → 0 when DMET self-consistency holds
        self._last_mu_times_deltaN_ha: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(
        self,
        backend_result: BackendResult,
        region: FragmentRegion,
    ) -> EmbeddedHamiltonian:
        """
        Build the DMET impurity+bath Hamiltonian.

        Steps
        -----
        1. Compute full-system 1-RDM in the MO basis.
        2. Partition occupied space → fragment AOs + bath (Schmidt).
        3. Add chemical-potential shift μ on fragment diagonal of h1e.
        4. Return EmbeddedHamiltonian in the impurity+bath basis.
        """
        self._iteration += 1
        logger.info(
            "[DMET] embed() iteration %d, μ=%.6f", self._iteration, self.mu
        )

        try:
            from pyscf import ao2mo
        except ImportError as exc:
            raise ImportError("PySCF required for DMET.") from exc

        mol = backend_result.mol
        mf  = backend_result.mf
        C   = backend_result.mo_coeff     # (nao, nmo)
        mo_occ = backend_result.mo_occ
        ovlp = backend_result.ovlp

        # --- Full-system 1-RDM in AO basis ---
        dm_ao = mf.make_rdm1()            # (nao, nao)

        # --- Identify fragment AOs ---
        frag_ao_idx = self._get_fragment_ao_indices(mol, region)
        nfrag_ao = len(frag_ao_idx)
        nao = mol.nao_nr()

        # --- Occupied MO block → Schmidt decomposition ---
        occ_idx = np.where(mo_occ > 0)[0]
        n_occ = len(occ_idx)
        C_occ = C[:, occ_idx]            # (nao, n_occ)

        # Fragment block of C_occ
        C_frag = C_occ[frag_ao_idx, :]   # (nfrag_ao, n_occ)
        U, sv, Vt = la.svd(C_frag, full_matrices=False)  # sv shape (min(nfrag, n_occ),)

        # Bath orbitals: complement of fragment in the occupied space
        bath_mask = sv > self.bath_threshold
        n_bath = int(bath_mask.sum())
        bath_vecs = Vt[bath_mask, :].T    # (n_occ, n_bath) → rotation of occ MOs

        # Build embedding basis: fragment AOs + bath MOs
        # In the standard DMET formulation we work in a combined basis
        # of nfrag_ao fragment "site" orbitals + n_bath bath orbitals.
        # Here we use the localized representation directly for simplicity.
        norb_emb = min(nfrag_ao + n_bath, region.norb)
        logger.info(
            "[DMET] '%s': nfrag_ao=%d, n_bath=%d → embedding norb=%d",
            region.name, nfrag_ao, n_bath, norb_emb,
        )

        # --- Rotate occupied MOs to get bath orbitals ---
        # C_bath[nao, n_bath]: the bath orbitals as AO combinations
        C_bath = C_occ @ bath_vecs        # (nao, n_bath)

        # Embedding coefficient matrix: [fragment-IAO block | bath]
        # For the fragment block we use atom-projected occupied MOs (IAO spirit)
        from ..hamiltonian.localizer import get_atom_orbital_indices, localize_orbitals
        C_loc = localize_orbitals(mf, scheme=region.localization)
        frag_occ_idx = get_atom_orbital_indices(
            mol=mol,
            C_loc=C_loc,
            atom_indices=region.atom_indices,
            mo_occ=mo_occ,
            n_orbs=min(nfrag_ao, norb_emb),
        )
        C_frag_orbs = C_loc[:, frag_occ_idx]  # (nao, n_frag_orbs)

        # Combine fragment + bath, orthogonalize
        n_frag_orbs = C_frag_orbs.shape[1]
        n_bath_use  = norb_emb - n_frag_orbs
        C_emb = np.hstack([
            C_frag_orbs,
            C_bath[:, :n_bath_use],
        ])                                     # (nao, norb_emb)

        # Löwdin orthogonalization
        S_emb = C_emb.T @ ovlp @ C_emb
        evals, evecs = la.eigh(S_emb)
        evals = np.where(evals > 1e-12, evals, 1e-12)
        C_emb = C_emb @ evecs @ np.diag(evals ** -0.5)

        # --- 1e and 2e integrals in embedding basis ---
        h1e_ao = backend_result.h1e_ao    # (nao, nao)
        h1e_emb = C_emb.T @ h1e_ao @ C_emb   # (norb_emb, norb_emb)

        # Add chemical-potential shift μ on the fragment-site diagonal
        frag_in_emb = list(range(n_frag_orbs))   # first n_frag_orbs columns
        for fi in frag_in_emb:
            h1e_emb[fi, fi] -= self.mu

        # 2e integrals
        h2e_emb = ao2mo.kernel(mol, C_emb, compact=False).reshape(
            norb_emb, norb_emb, norb_emb, norb_emb
        )

        # Store MF 1-RDM on fragment for self-consistency comparison
        dm_emb = C_emb.T @ ovlp @ dm_ao @ ovlp @ C_emb  # projected dm in embedding basis
        self._rdm1_mf_frag = dm_emb[:n_frag_orbs, :n_frag_orbs]

        # Core energy
        e_core = mol.energy_nuc()

        # nelec in embedding space
        nelec_emb = self._count_emb_electrons(dm_ao, ovlp, C_emb, norb_emb)
        n_alpha_emb = (nelec_emb + 1) // 2
        n_beta_emb  = nelec_emb // 2

        return EmbeddedHamiltonian(
            h1e        = h1e_emb,
            h2e        = h2e_emb,
            nelec      = (n_alpha_emb, n_beta_emb),
            norb       = norb_emb,
            e_core     = e_core,
            region_name = region.name,
            extra       = {
                "n_frag_orbs": n_frag_orbs,
                "n_bath": n_bath_use,
                "mu": self.mu,
                "dmet_mu_times_deltaN_ha": float(self._last_mu_times_deltaN_ha),
            },
        )

    def update_from_rdm(self, rdm1: np.ndarray) -> bool:
        """
        Update the chemical potential and check convergence.

        The DMET self-consistency condition is::

            rdm1_solver[frag, frag]  ≈  rdm1_mf[frag, frag]

        A simple gradient step on μ is used to enforce this condition.

        Returns
        -------
        bool
            True if converged.
        """
        if self._rdm1_mf_frag is None:
            return True   # embed() not called yet

        self._last_rdm1_solver = rdm1

        # Fragment block of solver 1-RDM (first n_frag_orbs)
        n_frag = self._rdm1_mf_frag.shape[0]
        if rdm1.ndim == 3:
            # unrestricted: sum alpha + beta
            rdm1_frag = (rdm1[0, :n_frag, :n_frag] + rdm1[1, :n_frag, :n_frag])
        else:
            rdm1_frag = rdm1[:n_frag, :n_frag]

        # Population difference on fragment
        pop_solver = np.trace(rdm1_frag)
        pop_mf     = np.trace(self._rdm1_mf_frag)
        delta_pop  = pop_mf - pop_solver
        delta_rdm  = np.max(np.abs(rdm1_frag - self._rdm1_mf_frag))

        err = float(delta_pop)  # want → 0
        mu_before = float(self.mu)
        self._last_mu_times_deltaN_ha = mu_before * err

        logger.info(
            "[DMET] iter %d: Δpop=%.4f, max|ΔRDM|=%.2e, μ=%.6f → μ'=%.6f",
            self._iteration,
            delta_pop,
            delta_rdm,
            self.mu,
            self.mu + self.mu_step * delta_pop,
        )

        if delta_rdm < self.conv_tol:
            logger.info("[DMET] Converged after %d iterations.", self._iteration)
            return True

        # Chemical-potential update (see mu_update in __init__)
        if self.mu_update == "damped":
            self.mu += self.mu_step * float(np.tanh(err))
        elif self.mu_update == "bisection":
            self._mu_err_hist.append((float(self.mu), err))
            self._mu_err_hist = self._mu_err_hist[-12:]
            if len(self._mu_err_hist) >= 2:
                (mu0, e0), (mu1, e1) = self._mu_err_hist[-2], self._mu_err_hist[-1]
                denom = e1 - e0
                if abs(denom) > 1e-14:
                    self.mu = float(mu1 - e1 * (mu1 - mu0) / denom)
                else:
                    self.mu += self.mu_step * err
            else:
                self.mu += self.mu_step * err
        elif self.mu_update == "bisection_bracket":
            self._mu_err_hist.append((float(self.mu), err))
            self._mu_err_hist = self._mu_err_hist[-24:]
            best: tuple[float, float, float] | None = None
            hist = self._mu_err_hist
            for i in range(len(hist)):
                for j in range(i + 1, len(hist)):
                    mu_i, e_i = hist[i]
                    mu_j, e_j = hist[j]
                    if e_i * e_j <= 0 and abs(e_i) + abs(e_j) > 1e-30:
                        lo, hi = min(mu_i, mu_j), max(mu_i, mu_j)
                        span = hi - lo
                        # Skip degenerate pairs (same μ): not a bracket interval.
                        if span < 1e-12:
                            continue
                        if best is None or span < best[0]:
                            best = (span, lo, hi)
            if best is not None:
                _, lo, hi = best
                self.mu = 0.5 * (lo + hi)
                logger.debug(
                    "[DMET] bisection_bracket: new μ=%.6f from bracket [%.6f, %.6f]",
                    self.mu,
                    lo,
                    hi,
                )
            else:
                self.mu += self.mu_step * err
        else:
            # gradient (default)
            self.mu += self.mu_step * err

        if self.mu_max_abs is not None:
            cap = float(self.mu_max_abs)
            self.mu = max(-cap, min(cap, float(self.mu)))
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_fragment_ao_indices(mol, region: FragmentRegion) -> list[int]:
        """Return AO indices for the atoms listed in region.atom_indices."""
        slices = mol.aoslice_by_atom()
        ao_idx = []
        for iat in region.atom_indices:
            start = slices[iat][2]
            stop  = slices[iat][3]
            ao_idx.extend(range(start, stop))
        if not ao_idx:
            raise ValueError(
                f"[DMET] Region '{region.name}' has no atom_indices set. "
                "DMET requires atom-based fragment definition."
            )
        return ao_idx

    @staticmethod
    def _count_emb_electrons(
        dm_ao: np.ndarray,
        ovlp: np.ndarray,
        C_emb: np.ndarray,
        norb_emb: int,
    ) -> int:
        """Estimate the number of electrons in the embedding space."""
        dm_emb = C_emb.T @ ovlp @ dm_ao @ ovlp @ C_emb
        nelec_float = np.trace(dm_emb).real
        return max(2, int(round(nelec_float)))
