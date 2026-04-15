"""
Generate figures for 分裂价极化与原子轨道杂化.md
Run from repo root or this directory: python generate_orbital_figures.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets", "orbitals")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 160,
    }
)


def envelope(r: np.ndarray, l: int) -> np.ndarray:
    """Smooth Slater-like envelope; r^l damps cusp for l>0 visualization."""
    r = np.asarray(r, dtype=float)
    return np.exp(-r * 0.45) * (r**l)


def grid_xz(ylim: float = 0.0, half: float = 3.5, n: int = 180):
    x = np.linspace(-half, half, n)
    z = np.linspace(-half, half, n)
    X, Z = np.meshgrid(x, z)
    Y = np.full_like(X, ylim)
    return X, Y, Z


def grid_xy(zlim: float = 0.0, half: float = 3.5, n: int = 180):
    x = np.linspace(-half, half, n)
    y = np.linspace(-half, half, n)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, zlim)
    return X, Y, Z


def grid_yz(xlim: float = 0.0, half: float = 3.5, n: int = 180):
    """$yz$ 平面切片：$x = \texttt{xlim}$（常为 0）。"""
    y = np.linspace(-half, half, n)
    z = np.linspace(-half, half, n)
    Y, Z = np.meshgrid(y, z)
    X = np.full_like(Y, xlim)
    return X, Y, Z


def grid_plane_x_eq_y(half: float = 3.5, n: int = 180):
    """平面 $x=y$：用坐标 $u$ 与 $z$，即 $(x,y,z)=(u,u,z)$。"""
    u = np.linspace(-half, half, n)
    z = np.linspace(-half, half, n)
    U, Zm = np.meshgrid(u, z)
    X = U
    Y = U
    Z = Zm
    return X, Y, Z


def rnorm(X, Y, Z):
    r2 = X * X + Y * Y + Z * Z
    return np.sqrt(np.maximum(r2, 1e-14)), r2


def plot_slice(ax, X, Y, Z, ang, title, l: int):
    r, _ = rnorm(X, Y, Z)
    psi = envelope(r, l) * ang
    vmax = np.percentile(np.abs(psi), 99.5)
    vmax = max(vmax, 1e-8)
    cf = ax.contourf(
        X[0, :],
        Z[:, 0],
        psi,
        levels=51,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.axhline(0, color="k", lw=0.3, alpha=0.4)
    ax.axvline(0, color="k", lw=0.3, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (a.u.)")
    ax.set_ylabel("z (a.u.)")
    return cf


def plot_slice_xy(ax, X, Y, Z, ang, title, l: int):
    r, _ = rnorm(X, Y, Z)
    psi = envelope(r, l) * ang
    vmax = np.percentile(np.abs(psi), 99.5)
    vmax = max(vmax, 1e-8)
    ax.contourf(
        X[0, :],
        Y[:, 0],
        psi,
        levels=51,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.axhline(0, color="k", lw=0.3, alpha=0.4)
    ax.axvline(0, color="k", lw=0.3, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (a.u.)")
    ax.set_ylabel("y (a.u.)")


def plot_slice_yz(ax, X, Y, Z, ang, title, l: int):
    r, _ = rnorm(X, Y, Z)
    psi = envelope(r, l) * ang
    vmax = np.percentile(np.abs(psi), 99.5)
    vmax = max(vmax, 1e-8)
    ax.contourf(
        Y[0, :],
        Z[:, 0],
        psi,
        levels=51,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.axhline(0, color="k", lw=0.3, alpha=0.4)
    ax.axvline(0, color="k", lw=0.3, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("y (a.u.)")
    ax.set_ylabel("z (a.u.)")


def plot_slice_x_eq_y(ax, X, Y, Z, ang, title, l: int):
    """$x=y$ 平面：横轴为 $u$（$x=y=u$），纵轴为 $z$。"""
    r, _ = rnorm(X, Y, Z)
    psi = envelope(r, l) * ang
    vmax = np.percentile(np.abs(psi), 99.5)
    vmax = max(vmax, 1e-8)
    ax.contourf(
        X[0, :],
        Z[:, 0],
        psi,
        levels=51,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.axhline(0, color="k", lw=0.3, alpha=0.4)
    ax.axvline(0, color="k", lw=0.3, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"$u$（$x{=}y{=}u$）(a.u.)")
    ax.set_ylabel("z (a.u.)")


# --- s, p ---
def fig_sp():
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.2))
    X, Y, Z = grid_xz(0.0)
    r, r2 = rnorm(X, Y, Z)
    # xz plane
    plot_slice(axes[0, 0], X, Y, Z, np.ones_like(r), r"$s$（球对称）", 0)
    plot_slice(axes[0, 1], X, Y, Z, X, r"$p_x$", 1)
    plot_slice(axes[0, 2], X, Y, Z, Z, r"$p_z$", 1)
    X2, Y2, Z2 = grid_xy(0.0)
    r2a, _ = rnorm(X2, Y2, Z2)
    plot_slice_xy(axes[1, 0], X2, Y2, Z2, np.ones_like(r2a), r"$s$（$xy$ 平面）", 0)
    plot_slice_xy(axes[1, 1], X2, Y2, Z2, X2, r"$p_x$", 1)
    plot_slice_xy(axes[1, 2], X2, Y2, Z2, Y2, r"$p_y$", 1)
    fig.suptitle(
        "氢样径向包络 × 角向因子：$xz$ 切片（上）与 $xy$ 切片（下）\n"
        r"红/蓝表示波函数正负；$p_y$ 在 $y=0$ 的 $xz$ 截面上恒为 0，故放在 $xy$ 图展示",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ao_s_p.png"), bbox_inches="tight")
    plt.close(fig)


# --- d ---
def fig_d():
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 6.4))
    X, Y, Z = grid_xz(0.0)
    r, r2 = rnorm(X, Y, Z)
    dz2 = 3 * Z * Z - r2
    plot_slice(axes[0, 0], X, Y, Z, dz2, r"$d_{z^2}$（$xz$）", 2)
    plot_slice(axes[0, 1], X, Y, Z, X * Z, r"$d_{xz}$（$xz$）", 2)
    plot_slice(axes[0, 2], X, Y, Z, X * X, r"$d_{x^2-y^2}$ 在 $xz$（$y=0$）：$x^2$", 2)
    X2, Y2, Z2 = grid_xy(0.0)
    r2b, r2c = rnorm(X2, Y2, Z2)
    dxy = X2 * Y2
    dx2y2 = X2 * X2 - Y2 * Y2
    dz2_xy = 3 * Z2 * Z2 - r2c  # Z2=0 -> -r2 = -(x^2+y^2), donut in xy
    plot_slice_xy(axes[1, 0], X2, Y2, Z2, dxy, r"$d_{xy}$（$xy$）", 2)
    plot_slice_xy(axes[1, 1], X2, Y2, Z2, dx2y2, r"$d_{x^2-y^2}$（$xy$）", 2)
    plot_slice_xy(axes[1, 2], X2, Y2, Z2, dz2_xy, r"$d_{z^2}$（$xy$ 平面，$z=0$）", 2)
    fig.suptitle(
        r"实 $d$ 轨道（笛卡尔型）：$l=2$ 角向为二次齐次多项式 × 径向包络",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ao_d.png"), bbox_inches="tight")
    plt.close(fig)


# --- f (7 real combinations) ---
def fig_f():
    """
    含 $y$ 因子的笛卡尔 $f$ 型在 $xz$（$y{=}0$）上恒为 0，故对 $f_{yz^2}, f_{xyz}, f_{y(3x^2-y^2)}$
    分别改用 $yz$（$x{=}0$）、$x{=}y$ 斜面、$xy$（$z{=}0$）切片。
    """
    Xxz, Yxz, Zxz = grid_xz(0.0)
    Xyz, Yyz, Zyz = grid_yz(0.0)
    Xxy, Yxy, Zxy = grid_xy(0.0)
    Xdiag, Ydiag, Zdiag = grid_plane_x_eq_y()

    r2_xz = Xxz * Xxz + Yxz * Yxz + Zxz * Zxz
    fz3 = Zxz * (5 * Zxz * Zxz - 3 * r2_xz)
    fxz2 = Xxz * (5 * Zxz * Zxz - r2_xz)
    fz_x2y2 = Zxz * (Xxz * Xxz - Yxz * Yxz)
    fx_x2m3y2 = Xxz * (Xxz * Xxz - 3 * Yxz * Yxz)

    r2_yz = Xyz * Xyz + Yyz * Yyz + Zyz * Zyz
    fyz2 = Yyz * (5 * Zyz * Zyz - r2_yz)

    r2_diag = Xdiag * Xdiag + Ydiag * Ydiag + Zdiag * Zdiag
    fxyz = Xdiag * Ydiag * Zdiag

    r2_xy = Xxy * Xxy + Yxy * Yxy + Zxy * Zxy
    fy_3x2y2 = Yxy * (3 * Xxy * Xxy - Yxy * Yxy)

    panels = [
        (plot_slice, Xxz, Yxz, Zxz, fz3, r"$f_{z^3}$（$xz,\,y{=}0$）"),
        (plot_slice, Xxz, Yxz, Zxz, fxz2, r"$f_{xz^2}$（$xz,\,y{=}0$）"),
        (plot_slice_yz, Xyz, Yyz, Zyz, fyz2, r"$f_{yz^2}$（$yz,\,x{=}0$）"),
        (plot_slice_x_eq_y, Xdiag, Ydiag, Zdiag, fxyz, r"$f_{xyz}$（$x{=}y$ 面）"),
        (plot_slice, Xxz, Yxz, Zxz, fz_x2y2, r"$f_{z(x^2-y^2)}$（$xz,\,y{=}0$）"),
        (plot_slice, Xxz, Yxz, Zxz, fx_x2m3y2, r"$f_{x(x^2-3y^2)}$（$xz,\,y{=}0$）"),
        (plot_slice_xy, Xxy, Yxy, Zxy, fy_3x2y2, r"$f_{y(3x^2-y^2)}$（$xy,\,z{=}0$）"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.0))
    axes = axes.ravel()
    for ax, (plot_fn, xa, ya, za, ang, ttl) in zip(axes[:7], panels):
        plot_fn(ax, xa, ya, za, ang, ttl, 3)
    axes[7].axis("off")
    fig.suptitle(
        r"实 $f$ 轨道（$l{=}3$）：多数为 $xz$（$y{=}0$）；"
        r"$f_{yz^2}, f_{xyz}, f_{y(\cdots)}$ 在 $xz$ 上恒 0，故分别用 $yz$、$x{=}y$、$xy$ 切片",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ao_f.png"), bbox_inches="tight")
    plt.close(fig)


def plot_hybrid_panel(ax, X, Y, Z, psi, title):
    vmax = np.percentile(np.abs(psi), 99.2)
    vmax = max(vmax, 1e-8)
    ax.contourf(
        X[0, :],
        Z[:, 0],
        psi,
        levels=51,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.axhline(0, color="k", lw=0.3, alpha=0.4)
    ax.axvline(0, color="k", lw=0.3, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("z")


def fig_hybrids():
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.0))
    X, Y, Z = grid_xz(0.0, half=4.0)
    r, r2 = rnorm(X, Y, Z)
    env = envelope(r, 1)

    # sp along z: (s + pz)/sqrt(2)
    sp1 = env * (1.0 / np.sqrt(2)) * (1.0 + Z)
    sp2 = env * (1.0 / np.sqrt(2)) * (1.0 - Z)
    plot_hybrid_panel(axes[0, 0], X, Y, Z, sp1, r"$\mathrm{sp}$ 杂化之一：$(s+p_z)/\sqrt{2}$（指向 $+z$）")
    plot_hybrid_panel(axes[0, 1], X, Y, Z, sp2, r"$\mathrm{sp}$ 另一瓣：$(s-p_z)/\sqrt{2}$（指向 $-z$）")

    # sp2 in xy plane: need xy slice for lobes; show one hybrid in xz still
    inv_sqrt3 = 1.0 / np.sqrt(3)
    # textbook: three sp2 with 120°, in xy: h_k = (s + sqrt(2)(cos θ px + sin θ py))/sqrt(3)
    X2, Y2, Z2 = grid_xy(0.0)
    r2a, _ = rnorm(X2, Y2, Z2)
    env2 = envelope(r2a, 1)
    theta0 = 0.0
    sp2_a = env2 * inv_sqrt3 * (1.0 + np.sqrt(2) * (np.cos(theta0) * X2 + np.sin(theta0) * Y2))
    theta1 = 2 * np.pi / 3
    sp2_b = env2 * inv_sqrt3 * (1.0 + np.sqrt(2) * (np.cos(theta1) * X2 + np.sin(theta1) * Y2))
    vmax = np.percentile(np.abs(sp2_a), 99.2)
    vmax = max(vmax, 1e-8)
    ax = axes[1, 0]
    ax.contourf(
        X2[0, :],
        Y2[:, 0],
        sp2_a,
        levels=51,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_aspect("equal")
    ax.set_title(r"$\mathrm{sp^2}$ 之一：$xy$ 平面，沿 $+x$ 大瓣", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axhline(0, color="k", lw=0.3, alpha=0.4)
    ax.axvline(0, color="k", lw=0.3, alpha=0.4)

    vmaxb = np.percentile(np.abs(sp2_b), 99.2)
    vmaxb = max(vmaxb, 1e-8)
    ax = axes[1, 1]
    ax.contourf(
        X2[0, :],
        Y2[:, 0],
        sp2_b,
        levels=51,
        cmap="RdBu_r",
        vmin=-vmaxb,
        vmax=vmaxb,
    )
    ax.set_aspect("equal")
    ax.set_title(r"$\mathrm{sp^2}$ 另一：相对旋转 $120^\circ$", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axhline(0, color="k", lw=0.3, alpha=0.4)
    ax.axvline(0, color="k", lw=0.3, alpha=0.4)

    fig.suptitle(
        "杂化：同一原子上 $s$ 与 $p$ 的固定线性组合（系数由几何决定，非 SCF 变分自由）\n"
        r"$\mathrm{sp^3}$ 四面体四瓣见下文示意图；数值图与 $\mathrm{sp^2}$ 类似可在 $xy$ 平面观察三瓣",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_hybrids_sp_sp2.png"), bbox_inches="tight")
    plt.close(fig)

    # sp3 one lobe in xz diagonal
    fig2, ax = plt.subplots(1, 1, figsize=(5.2, 4.8))
    X, Y, Z = grid_xz(0.0, half=4.0)
    r, _ = rnorm(X, Y, Z)
    env = envelope(r, 1)
    h = 0.5 * (1.0 + X + Z)  # (s+px+pz)/2 with y=0
    psi = env * h
    vmax = np.percentile(np.abs(psi), 99.2)
    ax.contourf(X[0, :], Z[:, 0], psi, levels=51, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_title(r"$\mathrm{sp^3}$ 之一：$(s+p_x+p_z)/2$ 在 $y=0$ 截面（指向对角线）", fontsize=11)
    ax.set_xlabel("x (a.u.)")
    ax.set_ylabel("z (a.u.)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "fig_hybrid_sp3_lobe.png"), bbox_inches="tight")
    plt.close(fig2)


def fig_radial_schematic():
    """Two Gaussians different width -> split valence intuition."""
    r = np.linspace(0, 6, 400)
    tight = np.exp(-3.0 * r * r)
    loose = np.exp(-0.35 * r * r)
    tight /= tight.max()
    loose /= loose.max()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(r, tight, label=r"紧（大指数 $\zeta$）", lw=2)
    ax.plot(r, loose, label=r"松（小指数 $\zeta$）", lw=2)
    mix = 0.55 * tight + 0.45 * loose
    mix /= mix.max()
    ax.plot(r, mix, "k--", label=r"线性组合 $c_1\chi_{\mathrm{tight}}+c_2\chi_{\mathrm{loose}}$", lw=2)
    ax.set_xlabel(r"距离核 $r$（示意，任意单位）")
    ax.set_ylabel("径向因子（归一化峰值）")
    ax.set_title("分裂价：两个径向尺度提供“胖瘦”调节自由度")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_radial_split_valence.png"), bbox_inches="tight")
    plt.close(fig)


def fig_polarization_schematic():
    """s + epsilon p -> dipole deformation."""
    x = np.linspace(-3, 3, 200)
    z = np.linspace(-3, 3, 200)
    X, Z = np.meshgrid(x, z)
    Y = 0 * X
    r, _ = rnorm(X, Y, Z)
    env = envelope(r, 0)
    s = env * 1.0
    eps = 0.35
    polarized = env * (1.0 + eps * Z)  # s + eps * pz-like
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for ax, psi, ttl in [
        (axes[0], s, r"纯 $s$：等密度近似球对称"),
        (axes[1], polarized, r"$s + \varepsilon\, p_z$：沿 $z$ 极化（示意）"),
    ]:
        vmax = np.percentile(np.abs(psi), 99)
        ax.contourf(X, Z, psi, levels=41, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_aspect("equal")
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
    fig.suptitle("极化函数：给中心增加 $p$，使球对称模板可沿键轴/电场方向形变", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_polarization_s_pz.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    fig_radial_schematic()
    fig_polarization_schematic()
    fig_sp()
    fig_d()
    fig_f()
    fig_hybrids()
    print("Wrote PNGs to:", OUT)


if __name__ == "__main__":
    main()
