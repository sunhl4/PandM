"""
3D angular shapes for s, p, d, f (real Cartesian factors on a directionally scaled sphere).
Run: python generate_orbital_3d_figures.py
Outputs to assets/orbitals/ (same folder as 2D figures for one place to look).
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets", "orbitals")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 180,
    }
)


def unit_sphere_mesh(n_theta: int = 72, n_phi: int = 96):
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    xs = np.sin(TH) * np.cos(PH)
    ys = np.sin(TH) * np.sin(PH)
    zs = np.cos(TH)
    return TH, PH, xs, ys, zs


def scale_radius(ang: np.ndarray, eps: float = 0.38, gain: float = 0.92, power: float = 0.5):
    """Inflate radius where |angular| is large; keep s-like nearly spherical."""
    a = np.abs(ang)
    a = a / (np.max(a) + 1e-14)
    return eps + gain * (a**power)


def quad_mean(a: np.ndarray) -> np.ndarray:
    return 0.25 * (a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:])


def draw_lower_left_axis_inset(ax, elev: float, azim: float) -> None:
    """
    在子图 **左下角**（axes 坐标）嵌入小 3D 坐标架，与主图相同的 elev/azim，
    便于对照屏幕上的取向；红 x、绿 y、蓝 z（右手系）。
    """
    ax_in = ax.inset_axes(
        [0.03, 0.03, 0.30, 0.30],
        transform=ax.transAxes,
        projection="3d",
    )
    ax_in.patch.set_facecolor("white")
    ax_in.patch.set_alpha(0.94)
    ax_in.patch.set_edgecolor("#888888")
    ax_in.patch.set_linewidth(0.6)

    L = 1.0
    kw = dict(lw=2.0, solid_capstyle="round")
    ax_in.plot([0, L], [0, 0], [0, 0], color="#c41e3a", **kw)
    ax_in.plot([0, 0], [0, L], [0, 0], color="#2e8b57", **kw)
    ax_in.plot([0, 0], [0, 0], [0, L], color="#1e6bd9", **kw)

    tip = 1.14
    ax_in.text(L * tip, 0, 0, r"$x$", color="#c41e3a", fontsize=7)
    ax_in.text(0, L * tip, 0, r"$y$", color="#2e8b57", fontsize=7)
    ax_in.text(0, 0, L * tip, r"$z$", color="#1e6bd9", fontsize=7)

    pad = 0.12
    ax_in.set_xlim(-pad, L + pad)
    ax_in.set_ylim(-pad, L + pad)
    ax_in.set_zlim(-pad, L + pad)
    ax_in.set_box_aspect((1, 1, 1))
    ax_in.view_init(elev=elev, azim=azim)
    ax_in.set_axis_off()


def plot_orbital_surface(ax, xs, ys, zs, ang, title: str, elev: float = 22, azim: float = 35):
    r = scale_radius(ang)
    X = r * xs
    Y = r * ys
    Z = r * zs

    amax = np.percentile(np.abs(ang), 99.8)
    amax = max(amax, 1e-8)
    norm = Normalize(vmin=-amax, vmax=amax)
    ang_q = quad_mean(ang)
    rgba = plt.cm.RdBu_r(norm(ang_q))
    ax.plot_surface(
        X,
        Y,
        Z,
        rstride=1,
        cstride=1,
        facecolors=rgba,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_box_aspect((1, 1, 1))
    lim = 1.35
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    draw_lower_left_axis_inset(ax, elev=elev, azim=azim)


# --- angular polynomials on (x,y,z) with x^2+y^2+z^2=1 ---


def ang_s(x, y, z):
    return np.ones_like(x)


def ang_px(x, y, z):
    return x


def ang_py(x, y, z):
    return y


def ang_pz(x, y, z):
    return z


def ang_dz2(x, y, z):
    r2 = x * x + y * y + z * z
    return 3 * z * z - r2


def ang_dxz(x, y, z):
    return x * z


def ang_dyz(x, y, z):
    return y * z


def ang_dxy(x, y, z):
    return x * y


def ang_dx2y2(x, y, z):
    return x * x - y * y


def ang_fz3(x, y, z):
    r2 = x * x + y * y + z * z
    return z * (5 * z * z - 3 * r2)


def ang_fxz2(x, y, z):
    r2 = x * x + y * y + z * z
    return x * (5 * z * z - r2)


def ang_fyz2(x, y, z):
    r2 = x * x + y * y + z * z
    return y * (5 * z * z - r2)


def ang_fxyz(x, y, z):
    return x * y * z


def ang_fz_x2y2(x, y, z):
    return z * (x * x - y * y)


def ang_fx_x2m3y2(x, y, z):
    return x * (x * x - 3 * y * y)


def ang_fy_3x2y2(x, y, z):
    return y * (3 * x * x - y * y)


def fig_samples_spdf():
    """One representative per l for the textbook table."""
    _, _, xs, ys, zs = unit_sphere_mesh(64, 80)
    x, y, z = xs, ys, zs
    fig = plt.figure(figsize=(11.2, 5.4))
    specs = [
        (ang_s, r"$s$（$l{=}0$）"),
        (ang_pz, r"$p_z$（$l{=}1$ 代表）"),
        (ang_dz2, r"$d_{z^2}$（$l{=}2$ 代表）"),
        (ang_fz3, r"$f_{z^3}$（$l{=}3$ 代表）"),
    ]
    for i, (fn, ttl) in enumerate(specs):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        plot_orbital_surface(ax, xs, ys, zs, fn(x, y, z), ttl)
    fig.suptitle(
        "角向因子三维形状（单位球方向按 $|A|^{0.5}$ 缩放半径；颜色 $=$ $A$ 正负，红蓝）",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ao3d_s_p_d_f_samples.png"), bbox_inches="tight")
    plt.close(fig)


def fig_p_all():
    _, _, xs, ys, zs = unit_sphere_mesh(56, 72)
    x, y, z = xs, ys, zs
    fig = plt.figure(figsize=(10.5, 3.2))
    for i, (fn, ttl) in enumerate(
        [
            (ang_px, r"$p_x$"),
            (ang_py, r"$p_y$"),
            (ang_pz, r"$p_z$"),
        ]
    ):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        plot_orbital_surface(ax, xs, ys, zs, fn(x, y, z), ttl)
    fig.suptitle(r"$p$ 轨道（$l{=}1$，简并度 3）", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ao3d_p_all.png"), bbox_inches="tight")
    plt.close(fig)


def fig_d_all():
    _, _, xs, ys, zs = unit_sphere_mesh(52, 66)
    x, y, z = xs, ys, zs
    fig = plt.figure(figsize=(12.0, 2.8))
    items = [
        (ang_dz2, r"$d_{z^2}$"),
        (ang_dxz, r"$d_{xz}$"),
        (ang_dyz, r"$d_{yz}$"),
        (ang_dxy, r"$d_{xy}$"),
        (ang_dx2y2, r"$d_{x^2-y^2}$"),
    ]
    for i, (fn, ttl) in enumerate(items):
        ax = fig.add_subplot(1, 5, i + 1, projection="3d")
        plot_orbital_surface(ax, xs, ys, zs, fn(x, y, z), ttl, elev=18, azim=32 + i * 5)
    fig.suptitle(r"$d$ 轨道（$l{=}2$，简并度 5）", fontsize=11, y=1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ao3d_d_all.png"), bbox_inches="tight")
    plt.close(fig)


def fig_f_all():
    _, _, xs, ys, zs = unit_sphere_mesh(48, 60)
    x, y, z = xs, ys, zs
    fig = plt.figure(figsize=(20.0, 3.1))
    items = [
        (ang_fz3, r"$f_{z^3}$"),
        (ang_fxz2, r"$f_{xz^2}$"),
        (ang_fyz2, r"$f_{yz^2}$"),
        (ang_fxyz, r"$f_{xyz}$"),
        (ang_fz_x2y2, r"$f_{z(x^2-y^2)}$"),
        (ang_fx_x2m3y2, r"$f_{x(x^2-3y^2)}$"),
        (ang_fy_3x2y2, r"$f_{y(3x^2-y^2)}$"),
    ]
    for i, (fn, ttl) in enumerate(items):
        ax = fig.add_subplot(1, 7, i + 1, projection="3d")
        plot_orbital_surface(ax, xs, ys, zs, fn(x, y, z), ttl, elev=16, azim=28 + i * 3)
    fig.suptitle(r"$f$ 轨道（$l{=}3$，简并度 7）", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ao3d_f_all.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    fig_samples_spdf()
    fig_p_all()
    fig_d_all()
    fig_f_all()
    print("Wrote 3D orbital PNGs to:", OUT)


if __name__ == "__main__":
    main()
