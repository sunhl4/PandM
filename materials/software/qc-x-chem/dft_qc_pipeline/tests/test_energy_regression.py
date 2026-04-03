"""
Reference energies for regression (PySCF + pipeline versions may shift slightly).

Values below were checked with PySCF 2.6.x; we use rtol/atol to tolerate minor
numeric differences across platforms.
"""

from __future__ import annotations

import numpy as np
import pytest

from dft_qc_pipeline import Pipeline, PipelineConfig
from dft_qc_pipeline.classical_backends.pyscf_backend import PySCFBackend


# H2, STO-3G, 0.735 Å, RHF total energy (Ha) — typical PySCF 2.x
REF_H2_RHF_STO3G_0735 = -1.117349


@pytest.mark.requires_pyscf
def test_h2_rhf_energy_regression() -> None:
    b = PySCFBackend(method="hf", verbose=0)
    r = b.run(
        geometry="H 0 0 0; H 0 0 0.735",
        basis="sto-3g",
        charge=0,
        spin=0,
    )
    assert r.scf_converged
    np.testing.assert_allclose(
        r.energy_hf,
        REF_H2_RHF_STO3G_0735,
        rtol=5e-5,
        atol=5e-5,
        err_msg="RHF energy drift — check PySCF version or geometry string",
    )


@pytest.mark.requires_pyscf
@pytest.mark.requires_qiskit_nature
def test_h2_simple_cas_numpy_fragment_energy_regression() -> None:
    """h2_vqe.yaml: full valence (2e,2o) NumPy FCI in STO-3G — energy band."""
    from pathlib import Path

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "h2_vqe.yaml"
    cfg = PipelineConfig.from_yaml(cfg_path)
    out = Pipeline(cfg).run()
    assert out.backend_result.scf_converged
    fr = out.fragment_results["H2_full"]
    assert np.isfinite(fr.energy)
    # FCI in (2e,2o) for minimal H2 is below RHF (~ -1.117) and above full-CI in larger bases
    assert -1.22 < fr.energy < -1.06, fr.energy


@pytest.mark.requires_pyscf
def test_pyscf_h2_density_fit_energy_close_to_reference() -> None:
    """DF SCF should match reference RHF energy within DF tolerance."""
    b = PySCFBackend(method="hf", verbose=0, density_fit=True)
    r = b.run(
        geometry="H 0 0 0; H 0 0 0.735",
        basis="sto-3g",
        charge=0,
        spin=0,
    )
    assert r.scf_converged
    np.testing.assert_allclose(
        r.energy_hf,
        REF_H2_RHF_STO3G_0735,
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.requires_pyscf
def test_merge_registry_builds_dmet_with_mu_update() -> None:
    """EmbeddingConfig.mu_update must reach DMETEmbedding without PySCF run."""
    from dft_qc_pipeline.core.config import EmbeddingConfig, merge_registry_kwargs
    from dft_qc_pipeline.core.registry import registry

    emb = EmbeddingConfig(type="dmet", mu_update="bisection", mu_step=0.04)
    typ, kw = merge_registry_kwargs(emb)
    dmet = registry.build({"type": typ, **kw}, category="embedding")
    assert dmet.mu_update == "bisection"
    assert dmet.mu_step == 0.04


@pytest.mark.requires_pyscf
def test_merge_registry_builds_dmet_with_bisection_bracket() -> None:
    """bisection_bracket must reach DMETEmbedding."""
    from dft_qc_pipeline.core.config import EmbeddingConfig, merge_registry_kwargs
    from dft_qc_pipeline.core.registry import registry

    emb = EmbeddingConfig(type="dmet", mu_update="bisection_bracket", mu_step=0.05)
    typ, kw = merge_registry_kwargs(emb)
    dmet = registry.build({"type": typ, **kw}, category="embedding")
    assert dmet.mu_update == "bisection_bracket"
