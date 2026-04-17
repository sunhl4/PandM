from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from zno_pd_qml.features import FeatureConfig, compute_frame_features
from zno_pd_qml.lammps_dump import iter_lammps_dump_frames
from zno_pd_qml.labeling import WindowLabelConfig, make_window_sample


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r") as f:
        return json.load(f)


def _as_float(x: Any, name: str) -> float:
    try:
        return float(x)
    except Exception as e:
        raise ValueError(f"Invalid float for {name}: {x}") from e


def _frames_to_windows(
    per_frame_rows: List[Dict[str, float]],
    *,
    dt_ps: float,
    window_ps: float,
    stride_ps: float,
    label_cfg: WindowLabelConfig,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    if len(per_frame_rows) < 2:
        return [], []
    window_frames = max(2, int(round(window_ps / dt_ps)))
    stride_frames = max(1, int(round(stride_ps / dt_ps)))
    Xs: List[Dict[str, float]] = []
    ys: List[Dict[str, float]] = []
    for i0 in range(0, len(per_frame_rows) - window_frames + 1, stride_frames):
        w = per_frame_rows[i0 : i0 + window_frames]
        X, y = make_window_sample(w, dt_ps=dt_ps, label_cfg=label_cfg)
        Xs.append(X)
        ys.append(y)
    return Xs, ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs_json",
        type=str,
        required=True,
        help="JSON list of runs (dump path + T/dPd/o_model + dt_ps [+ optional dump_every_steps])",
    )
    ap.add_argument("--type_map_json", type=str, required=True, help="JSON mapping elements to LAMMPS type ids")
    ap.add_argument("--out_npz", type=str, required=True)

    ap.add_argument("--window_ps", type=float, default=10.0)
    ap.add_argument("--stride_ps", type=float, default=5.0)

    ap.add_argument("--pd_surface_area_mode", type=str, default="nominal_sphere", choices=["nominal_sphere", "pd_rg_sphere"])
    ap.add_argument("--pd_shell_thickness_A", type=float, default=3.0)
    ap.add_argument("--pd_contact_cut_O_A", type=float, default=2.5)
    ap.add_argument("--pd_contact_cut_Zn_A", type=float, default=3.0)

    ap.add_argument("--z_ref_mode", type=str, default="auto_percentile", choices=["auto_percentile", "fixed"])
    ap.add_argument("--z_ref_fixed_A", type=float, default=0.0)
    ap.add_argument("--z_ref_percentile", type=float, default=10.0)
    ap.add_argument("--z_ref_exclude_pd_r_A", type=float, default=10.0)

    ap.add_argument("--coop_min_dcov", type=float, default=1e-5)
    ap.add_argument("--coop_min_dZn", type=float, default=1.0)
    ap.add_argument("--coop_min_dO", type=float, default=1.0)

    ap.add_argument("--max_frames_per_run", type=int, default=None)

    args = ap.parse_args()

    runs = _load_json(args.runs_json)
    type_map = _load_json(args.type_map_json)
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    feat_cfg = FeatureConfig(
        pd_surface_area_mode=str(args.pd_surface_area_mode),  # type: ignore[arg-type]
        pd_shell_thickness_A=float(args.pd_shell_thickness_A),
        pd_contact_cut_O_A=float(args.pd_contact_cut_O_A),
        pd_contact_cut_Zn_A=float(args.pd_contact_cut_Zn_A),
        z_ref_mode=str(args.z_ref_mode),  # type: ignore[arg-type]
        z_ref_fixed_A=float(args.z_ref_fixed_A),
        z_ref_percentile=float(args.z_ref_percentile),
        z_ref_exclude_pd_r_A=float(args.z_ref_exclude_pd_r_A),
    )
    label_cfg = WindowLabelConfig(
        coop_min_dcov=float(args.coop_min_dcov),
        coop_min_dZn=float(args.coop_min_dZn),
        coop_min_dO=float(args.coop_min_dO),
    )

    all_X: List[Dict[str, float]] = []
    all_y: List[Dict[str, float]] = []
    all_groups: List[str] = []
    all_meta: List[Dict[str, Any]] = []

    for run in runs:
        name = str(run.get("name", "run"))
        dump_path = str(run["dump_path"])
        T_K = int(run.get("T_K", -1))
        dPd_A = float(run.get("dPd_A")) if run.get("dPd_A") is not None else None
        o_model = str(run.get("o_model", "unknown"))
        # Optional: numeric oxygen richness/vacancy axis (recommended for regression/generalization)
        # Example: o_frac=1.0 for O-rich, 0.0 for O-vacancy rich.
        o_frac = float(run.get("o_frac")) if run.get("o_frac") is not None else None
        # Time axis:
        # - dt_ps is the *LAMMPS timestep* in ps (e.g., 0.4 fs -> 0.0004 ps)
        # - dump_every_steps is how many MD steps per dump frame (e.g., dump ... 1000 ...)
        # If you already pre-multiplied and set dt_ps as "ps per frame", set dump_every_steps=1 (default).
        dt_ps = _as_float(run["dt_ps"], "dt_ps")
        dump_every_steps = int(run.get("dump_every_steps", 1))
        if dump_every_steps < 1:
            raise ValueError(f"Invalid dump_every_steps={dump_every_steps} for run={name}. Must be >=1.")
        stride_frames = int(run.get("stride_frames", 1))

        per_frame_rows: List[Dict[str, float]] = []
        for fr in iter_lammps_dump_frames(dump_path, stride_frames=stride_frames, max_frames=args.max_frames_per_run):
            row = compute_frame_features(fr, type_map=type_map, dPd_A=dPd_A, cfg=feat_cfg)
            per_frame_rows.append(row)

        Xs, ys = _frames_to_windows(
            per_frame_rows,
            dt_ps=dt_ps * float(dump_every_steps) * float(stride_frames),
            window_ps=float(args.window_ps),
            stride_ps=float(args.stride_ps),
            label_cfg=label_cfg,
        )

        for i, (X, y) in enumerate(zip(Xs, ys)):
            # Add run-level condition variables as explicit features (numeric).
            X["T_K"] = float(T_K)
            X["dPd_A"] = float(dPd_A) if dPd_A is not None else float("nan")
            X["o_frac"] = float(o_frac) if o_frac is not None else float("nan")
            all_X.append(X)
            all_y.append(y)
            # Default group for out-of-distribution testing: by o_model
            all_groups.append(str(o_model))
            all_meta.append(
                {
                    "run_name": name,
                    "dump_path": dump_path,
                    "T_K": T_K,
                    "dPd_A": dPd_A if dPd_A is not None else np.nan,
                    "o_model": o_model,
                    "o_frac": o_frac if o_frac is not None else np.nan,
                    "window_idx": int(i),
                }
            )

    if len(all_X) == 0:
        raise SystemExit("No samples produced (check dump paths, dt_ps, window_ps/stride_ps).")

    # Convert dict rows -> arrays with fixed column order
    X_keys = sorted(all_X[0].keys())
    y_keys = sorted(all_y[0].keys())
    X = np.array([[row[k] for k in X_keys] for row in all_X], dtype=float)
    y = np.array([[row[k] for k in y_keys] for row in all_y], dtype=float)
    groups = np.array(all_groups, dtype=object)

    np.savez(
        out_npz,
        X=X,
        y=y,
        X_keys=np.array(X_keys, dtype=object),
        y_keys=np.array(y_keys, dtype=object),
        groups=groups,
        meta=np.array(all_meta, dtype=object),
    )
    print("Saved dataset:", out_npz)
    print("X shape:", X.shape, "y shape:", y.shape)
    print("X keys:", X_keys)
    print("y keys:", y_keys)


if __name__ == "__main__":
    main()


