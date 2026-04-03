"""Hubbard model path (toy backend + hubbard embedding) — no PySCF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dft_qc_pipeline import Pipeline, PipelineConfig

pytestmark = pytest.mark.requires_qiskit_nature


def _cfg() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "hubbard_2site_numpy.yaml"


def test_hubbard_yaml_pipeline_runs() -> None:
    cfg = PipelineConfig.from_yaml(_cfg())
    assert cfg.backend.type == "toy"
    assert cfg.embedding.type == "hubbard"
    out = Pipeline(cfg).run()
    assert np.isfinite(out.total_energy)
    assert "chain" in out.fragment_results
    ec = out.extra.get("energy_corrections")
    assert ec is not None
    assert ec.get("dmet_inter_fragment_ha") is None
    assert ec.get("dmet_correlation_potential_ha") is None
    assert ec.get("backend_reference_energy_ha") == 0.0
    assert ec.get("sum_fragment_energies_ha") is not None
    assert ec.get("delta_backend_minus_fragments_ha") is not None


def test_hubbard_two_site_half_filled_energy_window() -> None:
    """Loose regression window for t=1, U=4, 2 sites, 2e (1,1)."""
    out = Pipeline(PipelineConfig.from_yaml(_cfg())).run()
    e = out.fragment_results["chain"].energy
    assert -8.0 < e < 2.0


def test_parallel_regions_two_hubbard_fragments_smoke() -> None:
    """
    ``parallel_regions`` ThreadPoolExecutor path with toy backend (no PySCF).

    Two regions solve the same 2-site Hubbard instance; total energy is the sum.
    """
    cfg = PipelineConfig.from_dict(
        {
            "backend": {
                "type": "toy",
                "geometry": "hubbard",
                "basis": "none",
                "norb": 2,
            },
            "regions": [
                {
                    "name": "frag_a",
                    "atom_indices": [0, 1],
                    "nelec": 2,
                    "norb": 2,
                    "localization": "none",
                },
                {
                    "name": "frag_b",
                    "atom_indices": [0, 1],
                    "nelec": 2,
                    "norb": 2,
                    "localization": "none",
                },
            ],
            "embedding": {
                "type": "hubbard",
                "max_iter": 1,
                "conv_tol": 1.0e-8,
                "t": 1.0,
                "U": 4.0,
                "n_sites": 2,
                "periodic": False,
            },
            "mapper": {"type": "jw", "z2symmetry_reduction": False},
            "solver": {"type": "numpy", "compute_rdm": True},
            "parallel_regions": True,
            "max_parallel_workers": 2,
        }
    )
    out = Pipeline(cfg).run()
    assert "frag_a" in out.fragment_results
    assert "frag_b" in out.fragment_results
    ea = out.fragment_results["frag_a"].energy
    eb = out.fragment_results["frag_b"].energy
    assert np.isclose(out.total_energy, ea + eb)
    assert np.isclose(ea, eb)
