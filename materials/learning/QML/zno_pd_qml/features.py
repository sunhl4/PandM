from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from zno_pd_qml.geometry import (
    asphericity_from_inertia,
    min_distance_to_set,
    radius_of_gyration,
)
from zno_pd_qml.lammps_dump import Box, Frame, load_type_map, type_mask


SurfaceAreaMode = Literal["nominal_sphere", "pd_rg_sphere"]


@dataclass(frozen=True)
class FeatureConfig:
    pd_surface_area_mode: SurfaceAreaMode = "nominal_sphere"
    pd_shell_thickness_A: float = 3.0
    pd_contact_cut_O_A: float = 2.5
    pd_contact_cut_Zn_A: float = 3.0
    z_ref_mode: Literal["auto_percentile", "fixed"] = "auto_percentile"
    z_ref_fixed_A: float = 0.0
    z_ref_percentile: float = 10.0
    z_ref_exclude_pd_r_A: float = 10.0


def pd_surface_area_A2(
    pd_pos: np.ndarray,
    *,
    dPd_A: Optional[float],
    mode: SurfaceAreaMode,
) -> float:
    """
    Compute Pd "surface area" used in your coverage definition.

    - nominal_sphere: use user-provided diameter dPd_A -> 4*pi*(d/2)^2 (stable, matches input grid)
    - pd_rg_sphere: use 4*pi*(sqrt(5/3)*Rg)^2 (sphere relation) as a rough shape-adaptive proxy
    """
    if mode == "nominal_sphere":
        if dPd_A is None:
            raise ValueError("dPd_A is required for pd_surface_area_mode=nominal_sphere")
        R = 0.5 * float(dPd_A)
        return float(4.0 * np.pi * R * R)

    if mode == "pd_rg_sphere":
        rg = radius_of_gyration(pd_pos)
        # For a uniform solid sphere: Rg^2 = 3/5 R^2  => R = sqrt(5/3)*Rg
        R = np.sqrt(5.0 / 3.0) * rg
        return float(4.0 * np.pi * R * R)

    raise ValueError(f"Unknown SurfaceAreaMode: {mode}")


def estimate_slab_z_ref(
    pos: np.ndarray,
    mask_slab_atoms: np.ndarray,
    *,
    pd_com_xy: Tuple[float, float],
    exclude_r_A: float,
    mode: Literal["auto_percentile", "fixed"],
    fixed_A: float,
    percentile: float,
) -> float:
    """
    Define a reference slab z (for 'climb' direction).

    We do a conservative, robust default:
    - consider Zn/O atoms (mask_slab_atoms)
    - exclude atoms too close (in xy) to Pd COM to avoid counting the climbing ones
    - use a low percentile (default 10%) as a stable reference plane

    This is NOT the only valid definition; it's a pragmatic default when you don't
    want to hand-pick top-layer atoms.
    """
    if mode == "fixed":
        return float(fixed_A)

    x = pos[mask_slab_atoms]
    if x.shape[0] == 0:
        return float(np.min(pos[:, 2]))

    dx = x[:, 0] - float(pd_com_xy[0])
    dy = x[:, 1] - float(pd_com_xy[1])
    rr = np.sqrt(dx * dx + dy * dy)
    x2 = x[rr > float(exclude_r_A)]
    if x2.shape[0] < 10:
        x2 = x
    return float(np.percentile(x2[:, 2], float(percentile)))


def compute_frame_features(
    frame: Frame,
    *,
    type_map: Dict[str, np.ndarray],
    dPd_A: Optional[float],
    cfg: FeatureConfig,
) -> Dict[str, float]:
    """
    Compute per-frame CVs/features used later to build time-window samples.

    No bond-order required; uses distance-based contacts.
    """
    tm = load_type_map(type_map)
    m_pd = type_mask(frame.types, tm.get("Pd", []))
    m_zn = type_mask(frame.types, tm.get("Zn", []))
    m_o = type_mask(frame.types, tm.get("O", []))

    pd = frame.pos[m_pd]
    zn = frame.pos[m_zn]
    oo = frame.pos[m_o]

    if pd.shape[0] == 0:
        raise ValueError("No Pd atoms found for this frame (check your type_map_json).")

    pd_com = pd.mean(axis=0)
    z_ref = estimate_slab_z_ref(
        frame.pos,
        (m_zn | m_o),
        pd_com_xy=(float(pd_com[0]), float(pd_com[1])),
        exclude_r_A=cfg.z_ref_exclude_pd_r_A,
        mode=cfg.z_ref_mode,
        fixed_A=cfg.z_ref_fixed_A,
        percentile=cfg.z_ref_percentile,
    )

    # Contact distances to Pd (min distance to any Pd atom)
    d_zn_pd = min_distance_to_set(zn, pd, frame.box) if zn.shape[0] else np.zeros((0,), dtype=float)
    d_o_pd = min_distance_to_set(oo, pd, frame.box) if oo.shape[0] else np.zeros((0,), dtype=float)

    # "On/near Pd surface" mask: within cutoff AND above slab reference plane
    # (you can refine later with more physics-based definitions)
    zn_on = (d_zn_pd <= float(cfg.pd_contact_cut_Zn_A)) & (zn[:, 2] > z_ref) if zn.shape[0] else np.zeros((0,), dtype=bool)
    o_on = (d_o_pd <= float(cfg.pd_contact_cut_O_A)) & (oo[:, 2] > z_ref) if oo.shape[0] else np.zeros((0,), dtype=bool)

    # Coverage denominator
    area = pd_surface_area_A2(pd, dPd_A=dPd_A, mode=cfg.pd_surface_area_mode)

    # Pd shape features
    rg = radius_of_gyration(pd)
    asp = asphericity_from_inertia(pd)

    out: Dict[str, float] = {
        "timestep": float(frame.timestep),
        "pd_rg": float(rg),
        "pd_asphericity": float(asp),
        "pd_com_z": float(pd_com[2]),
        "slab_z_ref": float(z_ref),
        "n_Zn_onPd": float(int(zn_on.sum())),
        "n_O_onPd": float(int(o_on.sum())),
        "cov_Zn_perA2": float(zn_on.sum() / area),
        "cov_O_perA2": float(o_on.sum() / area),
        "cov_ZnO_perA2": float((zn_on.sum() + o_on.sum()) / area),
        "n_Zn_total": float(int(zn.shape[0])),
        "n_O_total": float(int(oo.shape[0])),
        "mean_d_ZnPd": float(np.mean(d_zn_pd) if d_zn_pd.size else np.nan),
        "mean_d_OPd": float(np.mean(d_o_pd) if d_o_pd.size else np.nan),
        "min_d_ZnPd": float(np.min(d_zn_pd) if d_zn_pd.size else np.nan),
        "min_d_OPd": float(np.min(d_o_pd) if d_o_pd.size else np.nan),
    }
    return out


