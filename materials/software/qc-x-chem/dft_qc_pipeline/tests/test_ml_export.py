"""PES / result export helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dft_qc_pipeline.core.interfaces import BackendResult, PipelineResult, SolverResult
from dft_qc_pipeline.postprocessing.ml_export import (
    pipeline_result_to_record,
    write_pes_jsonl,
)


def test_pipeline_result_to_record_roundtrip_keys() -> None:
    br = BackendResult(
        mol=None,
        energy_hf=-1.0,
        mo_coeff=np.eye(2),
        mo_occ=np.zeros(2),
        mo_energy=np.zeros(2),
        ovlp=np.eye(2),
        h1e_ao=np.zeros((2, 2)),
        h2e_ao=None,
        nelec=(1, 1),
        mf=None,
    )
    sr = SolverResult(energy=-1.5, rdm1=None, rdm2=None)
    pr = PipelineResult(
        total_energy=-1.5,
        fragment_results={"f": sr},
        backend_result=br,
        extra={
            "total_energy_note": "test",
            "energy_corrections": {
                "backend_reference_energy_ha": -1.0,
                "sum_fragment_energies_ha": -1.5,
                "delta_backend_minus_fragments_ha": 0.5,
            },
        },
    )
    rec = pipeline_result_to_record(label="pt1", result=pr, metadata={"d": 0.74})
    assert rec["label"] == "pt1"
    assert rec["total_energy"] == -1.5
    assert "fragments" in rec
    assert rec.get("backend_reference_energy_ha") == -1.0
    assert rec.get("sum_fragment_energies_ha") == -1.5
    assert rec.get("delta_backend_minus_fragments_ha") == 0.5


def test_pipeline_result_to_record_computed_when_no_energy_corrections() -> None:
    """Without ``energy_corrections``, HF ref + fragment sums are derived from the result."""
    br = BackendResult(
        mol=None,
        energy_hf=-1.0,
        mo_coeff=np.eye(2),
        mo_occ=np.zeros(2),
        mo_energy=np.zeros(2),
        ovlp=np.eye(2),
        h1e_ao=np.zeros((2, 2)),
        h2e_ao=None,
        nelec=(1, 1),
        mf=None,
    )
    sr = SolverResult(energy=-1.5, rdm1=None, rdm2=None)
    pr = PipelineResult(
        total_energy=-1.5,
        fragment_results={"f": sr},
        backend_result=br,
        extra={},
    )
    rec = pipeline_result_to_record(label="x", result=pr)
    assert rec["backend_reference_energy_ha"] == -1.0
    assert rec["sum_fragment_energies_ha"] == -1.5
    assert rec["delta_backend_minus_fragments_ha"] == 0.5


def test_pipeline_result_to_record_computed_when_energy_corrections_placeholder_only() -> None:
    """Placeholder dict (e.g. notes only) does not carry numeric keys → use computed branch."""
    br = BackendResult(
        mol=None,
        energy_hf=-2.0,
        mo_coeff=np.eye(2),
        mo_occ=np.zeros(2),
        mo_energy=np.zeros(2),
        ovlp=np.eye(2),
        h1e_ao=np.zeros((2, 2)),
        h2e_ao=None,
        nelec=(1, 1),
        mf=None,
    )
    sr = SolverResult(energy=-0.5, rdm1=None, rdm2=None)
    pr = PipelineResult(
        total_energy=-0.5,
        fragment_results={"f": sr},
        backend_result=br,
        extra={"energy_corrections": {"notes": "DMET placeholders not yet filled"}},
    )
    rec = pipeline_result_to_record(label="y", result=pr)
    assert rec["backend_reference_energy_ha"] == -2.0
    assert rec["sum_fragment_energies_ha"] == -0.5
    assert rec["delta_backend_minus_fragments_ha"] == -1.5


def test_write_pes_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "out.jsonl"
    write_pes_jsonl(p, [{"a": 1}, {"b": 2}])
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["a"] == 1
