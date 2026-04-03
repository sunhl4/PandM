"""Unit tests for DMET chemical-potential update modes (no full PySCF pipeline)."""

from __future__ import annotations

import numpy as np

from dft_qc_pipeline.embedding.dmet import DMETEmbedding


def test_dmet_bisection_bracket_takes_midpoint_of_bracket() -> None:
    # conv_tol must be tighter than |ΔRDM| or update_from_rdm returns early (converged).
    emb = DMETEmbedding(mu_update="bisection_bracket", mu_step=0.1, conv_tol=1e-15)
    emb._rdm1_mf_frag = np.eye(2) * 0.5
    emb._iteration = 1

    # Two (mu, err) pairs with opposite sign → bracket → midpoint
    emb._mu_err_hist = [(0.0, 0.2), (0.5, -0.1)]
    emb.mu = 0.5
    rdm = np.eye(2) * 0.4
    emb.update_from_rdm(rdm)

    assert abs(emb.mu - 0.25) < 1e-9  # midpoint of [0, 0.5]


def test_dmet_gradient_fallback_when_no_bracket() -> None:
    emb = DMETEmbedding(mu_update="bisection_bracket", mu_step=0.1, conv_tol=1e-15)
    emb._rdm1_mf_frag = np.eye(2) * 0.5
    emb._iteration = 1
    emb.mu = 0.0
    emb._mu_err_hist = []
    rdm = np.eye(2) * 0.4
    mu_before = emb.mu
    emb.update_from_rdm(rdm)
    assert emb.mu != mu_before
