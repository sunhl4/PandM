from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class WindowLabelConfig:
    """
    Label definitions for time-window samples.

    - y_cover_rate: slope of total coverage in the window
    - y_coop_event: whether Zn+O increase together "enough" within the window
    """

    coop_min_dcov: float = 1e-5
    coop_min_dZn: float = 1.0
    coop_min_dO: float = 1.0


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2 or b.size < 2:
        return 0.0
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return c


def _safe_slope(t: np.ndarray, y: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 2:
        return 0.0
    dt = t[-1] - t[0]
    if abs(dt) < 1e-12:
        return 0.0
    return float((y[-1] - y[0]) / dt)


def make_window_sample(
    rows: List[Dict[str, float]],
    *,
    dt_ps: float,
    label_cfg: WindowLabelConfig,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Aggregate per-frame features into one window-level sample (X) and labels (y).

    rows: consecutive frames
    """
    if len(rows) == 0:
        raise ValueError("Empty window.")

    # time axis in ps (relative)
    ts = np.array([r["timestep"] for r in rows], dtype=float)
    t_ps = (ts - ts[0]) * float(dt_ps)
    t_mid = float(0.5 * (t_ps[0] + t_ps[-1]))

    cov = np.array([r["cov_ZnO_perA2"] for r in rows], dtype=float)
    nZn = np.array([r["n_Zn_onPd"] for r in rows], dtype=float)
    nO = np.array([r["n_O_onPd"] for r in rows], dtype=float)
    dZn_series = np.diff(nZn)
    dO_series = np.diff(nO)
    joint_up_frac = float(np.mean((dZn_series > 0) & (dO_series > 0))) if dZn_series.size else 0.0

    # window features: mean levels + slopes of key CVs
    X = {
        "t_mid_ps": float(t_mid),
        "cov_start": float(cov[0]),
        "cov_mean": float(np.mean(cov)),
        "cov_end": float(cov[-1]),
        "cov_slope": _safe_slope(t_ps, cov),
        "cov_delta": float(cov[-1] - cov[0]),
        "nZn_mean": float(np.mean(nZn)),
        "nO_mean": float(np.mean(nO)),
        "nZn_slope": _safe_slope(t_ps, nZn),
        "nO_slope": _safe_slope(t_ps, nO),
        "nZn_delta": float(nZn[-1] - nZn[0]),
        "nO_delta": float(nO[-1] - nO[0]),
        "corr_nZn_nO": _safe_corr(nZn, nO),
        "corr_dZn_dO": _safe_corr(dZn_series, dO_series),
        "joint_up_frac": float(joint_up_frac),
        "pd_rg_mean": float(np.nanmean([r["pd_rg"] for r in rows])),
        "pd_rg_slope": _safe_slope(t_ps, np.array([r["pd_rg"] for r in rows], dtype=float)),
        "pd_asp_mean": float(np.nanmean([r["pd_asphericity"] for r in rows])),
        "pd_asp_slope": _safe_slope(t_ps, np.array([r["pd_asphericity"] for r in rows], dtype=float)),
        "slab_zref_mean": float(np.nanmean([r["slab_z_ref"] for r in rows])),
        "pd_comz_mean": float(np.nanmean([r["pd_com_z"] for r in rows])),
        "min_dZnpd_mean": float(np.nanmean([r["min_d_ZnPd"] for r in rows])),
        "min_dOpd_mean": float(np.nanmean([r["min_d_OPd"] for r in rows])),
    }

    # labels
    y_cover_rate = X["cov_slope"]
    dZn = float(X["nZn_delta"])
    dO = float(X["nO_delta"])
    dCov = float(X["cov_delta"])
    y_coop_event = float(
        (dCov >= label_cfg.coop_min_dcov) and (dZn >= label_cfg.coop_min_dZn) and (dO >= label_cfg.coop_min_dO)
    )
    # A soft score (continuous) can be more stable than a hard event threshold.
    # Ranges roughly [-1, 1], but we clamp for safety.
    y_coop_score = float(np.clip(0.5 * (X["corr_dZn_dO"] + X["joint_up_frac"]), -1.0, 1.0))
    y = {
        "y_cover_rate": float(y_cover_rate),
        "y_cover_final": float(cov[-1]),
        "y_coop_event": float(y_coop_event),
        "y_coop_score": float(y_coop_score),
    }
    return X, y


