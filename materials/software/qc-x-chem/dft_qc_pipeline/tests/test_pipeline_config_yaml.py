"""Load packaged YAML configs."""

from __future__ import annotations

from pathlib import Path

import pytest

from dft_qc_pipeline.core.config import PipelineConfig, validate_pipeline_config


def _configs_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    "name",
    [
        "h2_vqe.yaml",
        "h2_sto3g_dft_pbe.yaml",
        "lih_sto3g_numpy.yaml",
        "hubbard_2site_numpy.yaml",
        "n2_compare.yaml",
        "fen4_dmet_sqd.yaml",
    ],
)
def test_example_yaml_loads(name: str) -> None:
    p = _configs_dir() / name
    if not p.is_file():
        pytest.skip(f"missing {p}")
    cfg = PipelineConfig.from_yaml(p)
    validate_pipeline_config(cfg)
    assert cfg.backend.type in ("pyscf", "toy")
