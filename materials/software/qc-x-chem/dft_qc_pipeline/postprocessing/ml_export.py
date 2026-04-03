"""
Export pipeline / PES data for external ML force-field workflows (CSV / JSONL).

This is a thin, dependency-free bridge — not a full NequIP/MACE dataset writer.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from ..core.interfaces import PipelineResult


def pipeline_result_to_record(
    *,
    label: str,
    result: PipelineResult,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten a single ``PipelineResult`` into a JSON-serialisable dict."""
    row: dict[str, Any] = {
        "label": label,
        "total_energy": float(result.total_energy),
        "backend_energy_hf": float(result.backend_result.energy_hf),
        "scf_converged": bool(result.backend_result.scf_converged),
        "fragments": {
            k: {
                "energy": float(v.energy),
                "converged": bool(v.converged),
            }
            for k, v in result.fragment_results.items()
        },
    }
    note = result.extra.get("total_energy_note")
    if note is not None:
        row["total_energy_note"] = note
    ec = result.extra.get("energy_corrections")
    if isinstance(ec, dict) and any(
        k in ec
        for k in (
            "backend_reference_energy_ha",
            "sum_fragment_energies_ha",
            "delta_backend_minus_fragments_ha",
        )
    ):
        row["backend_reference_energy_ha"] = ec.get("backend_reference_energy_ha")
        row["sum_fragment_energies_ha"] = ec.get("sum_fragment_energies_ha")
        row["delta_backend_minus_fragments_ha"] = ec.get("delta_backend_minus_fragments_ha")
        for k in ("dmet_correlation_potential_ha", "dmet_inter_fragment_ha"):
            if k in ec and ec.get(k) is not None:
                row[k] = ec.get(k)
    else:
        e_hf = float(result.backend_result.energy_hf)
        row["backend_reference_energy_ha"] = e_hf
        if result.fragment_results:
            sfrag = sum(float(v.energy) for v in result.fragment_results.values())
            row["sum_fragment_energies_ha"] = sfrag
            row["delta_backend_minus_fragments_ha"] = e_hf - sfrag
    if metadata:
        row["metadata"] = metadata
    return row


def write_pes_jsonl(
    path: str | Path,
    records: Iterable[dict[str, Any]],
) -> None:
    """Append one JSON object per line (easy streaming for training pipelines)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_pes_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    """Write a simple CSV; uses union of keys if ``fieldnames`` is omitted."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = fieldnames or sorted({k for r in rows for k in r.keys()})
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            flat = {k: _csv_val(r.get(k)) for k in keys}
            w.writerow(flat)


def _csv_val(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)
