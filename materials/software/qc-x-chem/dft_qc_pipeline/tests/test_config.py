"""Configuration validation and registry kwargs merging."""

from __future__ import annotations

import pytest

from dft_qc_pipeline.core.config import (
    BackendConfig,
    EmbeddingConfig,
    PipelineConfig,
    RegionConfig,
    merge_registry_kwargs,
    validate_pipeline_config,
)


def test_validate_rejects_nelec_gt_2norb() -> None:
    cfg = PipelineConfig()
    cfg.regions[0].nelec = 9
    cfg.regions[0].norb = 2
    with pytest.raises(ValueError, match="exceeds 2\\*norb"):
        validate_pipeline_config(cfg)


def test_validate_rejects_empty_regions() -> None:
    cfg = PipelineConfig()
    cfg.regions = []
    with pytest.raises(ValueError, match="at least one region"):
        validate_pipeline_config(cfg)


def test_validate_mu_update() -> None:
    cfg = PipelineConfig()
    cfg.embedding.mu_update = "invalid"
    with pytest.raises(ValueError, match="mu_update"):
        validate_pipeline_config(cfg)


def test_validate_mu_update_bisection_bracket_ok() -> None:
    cfg = PipelineConfig()
    cfg.embedding.mu_update = "bisection_bracket"
    validate_pipeline_config(cfg)


def test_validate_parallel_regions_conflicts_benchmark() -> None:
    cfg = PipelineConfig()
    cfg.parallel_regions = True
    cfg.benchmark_mode = True
    with pytest.raises(ValueError, match="parallel_regions"):
        validate_pipeline_config(cfg)


def test_validate_rejects_duplicate_region_names() -> None:
    cfg = PipelineConfig()
    cfg.regions = [
        RegionConfig(name="dup", atom_indices=[0], nelec=2, norb=2),
        RegionConfig(name="dup", atom_indices=[1], nelec=2, norb=2),
    ]
    with pytest.raises(ValueError, match="duplicate region name"):
        validate_pipeline_config(cfg)


def test_merge_registry_kwargs_dataclass_overrides_extra() -> None:
    """Known dataclass fields must override keys duplicated in ``extra``."""
    b = BackendConfig(type="pyscf", charge=0, extra={"charge": 1, "tag": "x"})
    typ, kw = merge_registry_kwargs(b)
    assert typ == "pyscf"
    assert kw["charge"] == 0
    assert kw["tag"] == "x"


def test_merge_registry_kwargs_backend_density_fit() -> None:
    b = BackendConfig(
        type="pyscf",
        density_fit=True,
        auxbasis="weigend",
    )
    typ, kw = merge_registry_kwargs(b)
    assert typ == "pyscf"
    assert kw["density_fit"] is True
    assert kw["auxbasis"] == "weigend"


def test_merge_registry_kwargs_embedding_mu_max_abs() -> None:
    emb = EmbeddingConfig(type="dmet", mu_max_abs=2.5)
    typ, kw = merge_registry_kwargs(emb)
    assert typ == "dmet"
    assert kw["mu_max_abs"] == 2.5


def test_merge_registry_kwargs_passes_embedding_extra() -> None:
    emb = EmbeddingConfig(type="dmet", extra={"mu_step": 0.03, "bath_threshold": 1e-5})
    typ, kw = merge_registry_kwargs(emb)
    assert typ == "dmet"
    assert kw["mu_step"] == 0.03
    assert kw["bath_threshold"] == 1e-5
