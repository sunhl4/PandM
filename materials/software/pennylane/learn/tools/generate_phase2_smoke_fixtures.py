#!/usr/bin/env python3
"""
Generate minimal fixtures for Phase 2 pipeline smoke tests:
  - fixtures/gnn_smoke/: VASP-style CONTCAR + adsorption.csv
  - fixtures/zno_smoke/dataset.npz: compatible with zno_pd_qml.train_qml

Run from repo root:
  python tools/generate_phase2_smoke_fixtures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_gnn_fixtures() -> None:
    from ase import Atoms
    from ase.io import write

    d = ROOT / "fixtures" / "gnn_smoke"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(6):
        # Simple dimer in a box — enough for neighbor graph + PBC
        z = 1.08 + 0.02 * i
        atoms = Atoms("CO", positions=[[0, 0, 0], [0, 0, z]], cell=[12, 12, 12], pbc=True)
        path = d / f"struct_{i:02d}_CONTCAR"
        write(path, atoms, format="vasp", direct=True)
        y = -0.5 - 0.01 * i
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "y": y,
                "group": f"grp{i % 2}",
            }
        )
    csv_path = d / "adsorption.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("path,y,group\n")
        for r in rows:
            f.write(f"{r['path']},{r['y']},{r['group']}\n")
    print("Wrote", csv_path)


def write_zno_npz() -> None:
    rng = np.random.default_rng(0)
    n = 12
    n_feat = 8
    X = rng.normal(size=(n, n_feat))
    # synthetic target
    y_cover = np.abs(X[:, 0]) * 0.1 + rng.normal(0, 0.01, size=n)
    y = y_cover.reshape(-1, 1)
    X_keys = np.array([f"f{i}" for i in range(n_feat)], dtype=object)
    y_keys = np.array(["y_cover_rate"], dtype=object)
    groups = np.array([f"m{i % 3}" for i in range(n)], dtype=object)
    meta = np.array(
        [{"o_model": f"m{i % 3}", "run": "smoke"} for i in range(n)],
        dtype=object,
    )
    out = ROOT / "fixtures" / "zno_smoke" / "dataset.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        X=X,
        y=y,
        X_keys=X_keys,
        y_keys=y_keys,
        groups=groups,
        meta=meta,
    )
    print("Wrote", out)


def main() -> int:
    write_gnn_fixtures()
    write_zno_npz()
    manifest = {
        "gnn_csv": "fixtures/gnn_smoke/adsorption.csv",
        "zno_npz": "fixtures/zno_smoke/dataset.npz",
    }
    (ROOT / "fixtures" / "phase2_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
