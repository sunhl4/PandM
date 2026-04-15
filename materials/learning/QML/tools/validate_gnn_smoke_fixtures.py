#!/usr/bin/env python3
"""Load each CONTCAR in fixtures/gnn_smoke via ASE (no PyG). Run before GNN training."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    from ase.io import read

    csv_path = ROOT / "fixtures" / "gnn_smoke" / "adsorption.csv"
    if not csv_path.exists():
        print("Run: python tools/generate_phase2_smoke_fixtures.py")
        return 1
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    for row in rows:
        p = ROOT / row["path"]
        atoms = read(p, format="vasp")
        assert len(atoms) == 2
    print(f"OK: {len(rows)} structures loaded (ASE). Ready for PyG training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
