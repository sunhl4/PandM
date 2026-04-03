"""End-to-end tests with PySCF (skipped automatically if PySCF is absent)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dft_qc_pipeline import Pipeline, PipelineConfig
from dft_qc_pipeline.classical_backends.pyscf_backend import PySCFBackend


def _configs() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.requires_pyscf
def test_pyscf_backend_h2_rhf_energy_and_scf_flag() -> None:
    b = PySCFBackend(method="hf", verbose=0)
    r = b.run(
        geometry="H 0 0 0; H 0 0 0.735",
        basis="sto-3g",
        charge=0,
        spin=0,
    )
    assert r.scf_converged
    assert r.nelec == (1, 1)
    # STO-3G, ~0.735 Å — RHF total energy well within a loose chemical window
    assert -1.25 < r.energy_hf < -0.95
    assert r.mol.nelectron == 2


@pytest.mark.requires_pyscf
def test_pyscf_backend_h2_dft_pbe_energy() -> None:
    b = PySCFBackend(method="dft", xc="pbe", verbose=0)
    r = b.run(
        geometry="H 0 0 0; H 0 0 0.735",
        basis="sto-3g",
        charge=0,
        spin=0,
    )
    assert r.scf_converged
    assert -1.3 < r.energy_hf < -0.9


@pytest.mark.requires_pyscf
@pytest.mark.requires_qiskit_nature
def test_pipeline_h2_numpy_matches_yaml() -> None:
    cfg = PipelineConfig.from_yaml(_configs() / "h2_vqe.yaml")
    out = Pipeline(cfg).run()
    assert out.backend_result.scf_converged
    assert "H2_full" in out.fragment_results
    fr = out.fragment_results["H2_full"]
    assert np.isfinite(fr.energy)
    # Fragment energy includes e_core; should stay in a bounded window
    assert -2.5 < fr.energy < -0.5


@pytest.mark.requires_pyscf
@pytest.mark.requires_qiskit_nature
def test_pipeline_lih_numpy_from_yaml() -> None:
    p = _configs() / "lih_sto3g_numpy.yaml"
    if not p.is_file():
        pytest.skip("lih_sto3g_numpy.yaml missing")
    cfg = PipelineConfig.from_yaml(p)
    out = Pipeline(cfg).run()
    assert out.backend_result.scf_converged
    fr = out.fragment_results["LiH_valence"]
    assert np.isfinite(fr.energy)
    assert -10.0 < fr.energy < -5.0


@pytest.mark.requires_pyscf
@pytest.mark.requires_qiskit_nature
def test_pipeline_h2_dft_yaml_loads_and_runs() -> None:
    p = _configs() / "h2_sto3g_dft_pbe.yaml"
    if not p.is_file():
        pytest.skip("h2_sto3g_dft_pbe.yaml missing")
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.backend.method == "dft"
    out = Pipeline(cfg).run()
    assert out.backend_result.scf_converged
    assert np.isfinite(out.total_energy)
