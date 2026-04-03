"""
Benchmark analysis — result tables and energy-error visualizations.

Provides:
    - ``BenchmarkPlotter`` : matplotlib-based plots for benchmark results.
    - ``format_table``     : text / LaTeX table formatter.
    - ``energy_errors``    : compute error metrics for multiple methods.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Matplotlib is optional (used only in plotting functions)
_MATPLOTLIB_AVAILABLE = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Table formatting helpers
# ---------------------------------------------------------------------------

def energy_errors(bench_result, reference: str = "fci") -> dict[str, float]:
    """
    Compute energy errors (mHa) relative to a reference method.

    Parameters
    ----------
    bench_result : BenchResult
    reference : str
        Method name to use as exact reference (default: ``"fci"``).

    Returns
    -------
    dict mapping method_name → error in mHa.
    """
    ref_energy = None
    for name, r in bench_result.results.items():
        if name.lower() == reference.lower():
            ref_energy = r.energy
            break

    if ref_energy is None and bench_result.fci_energy is not None:
        ref_energy = bench_result.fci_energy

    if ref_energy is None:
        logger.warning("Reference method '%s' not found in results.", reference)
        return {}

    return {
        name: (r.energy - ref_energy) * 1000.0
        for name, r in bench_result.results.items()
        if name.lower() != reference.lower()
    }


def format_table(bench_result, fmt: str = "text") -> str:
    """
    Format benchmark results as a text or LaTeX table.

    Parameters
    ----------
    bench_result : BenchResult
    fmt : str
        ``"text"`` (default) or ``"latex"``.

    Returns
    -------
    str
    """
    rows = bench_result.summary_table()
    if not rows:
        return "No results."

    if fmt == "latex":
        return _latex_table(rows)
    return _text_table(rows)


def _text_table(rows: list[dict]) -> str:
    headers = list(rows[0].keys())
    widths = [
        max(len(str(h)), max(len(_fmt_cell(r.get(h))) for r in rows))
        for h in headers
    ]
    sep = "  ".join("-" * w for w in widths)
    fmt_row = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt_row.format(*[_fmt_cell(h) for h in headers]), sep]
    for r in rows:
        lines.append(fmt_row.format(*[_fmt_cell(r.get(h)) for h in headers]))
    return "\n".join(lines)


def _latex_table(rows: list[dict]) -> str:
    headers = list(rows[0].keys())
    col_fmt = " | ".join(["l"] + ["r"] * (len(headers) - 1))
    lines = [
        r"\begin{tabular}{" + col_fmt + "}",
        r"\hline",
        " & ".join(h.replace("_", " ") for h in headers) + r" \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(" & ".join(_fmt_cell(r.get(h)) for h in headers) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


def _fmt_cell(val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        if abs(val) > 1e3 or abs(val) < 0.001 and val != 0.0:
            return f"{val:.4e}"
        return f"{val:.6f}"
    return str(val)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

class BenchmarkPlotter:
    """
    Matplotlib-based visualizations for BenchResult objects.

    Parameters
    ----------
    figsize : tuple
        Default figure size (width, height) in inches.
    style : str
        Matplotlib style (``"seaborn-v0_8-whitegrid"`` or similar).
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (9, 5),
        style: str = "seaborn-v0_8-whitegrid",
    ) -> None:
        if not _MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required: pip install matplotlib")
        self.figsize = figsize
        try:
            plt.style.use(style)
        except Exception:  # noqa: BLE001
            pass

    def energy_bar(
        self,
        bench_result,
        reference: str = "fci",
        title: str | None = None,
        save_path: str | Path | None = None,
    ) -> "plt.Figure":
        """
        Bar chart of energy errors (mHa) relative to a reference method.

        Parameters
        ----------
        bench_result : BenchResult
        reference : str
            Reference method name.
        title : str or None
            Plot title.
        save_path : str or Path or None
            If given, save figure to this path.

        Returns
        -------
        matplotlib Figure
        """
        errors = energy_errors(bench_result, reference=reference)
        if not errors:
            raise ValueError(f"No errors computed; '{reference}' may be missing.")

        names = list(errors.keys())
        vals = [errors[n] for n in names]
        colors = ["#2196F3" if v <= 0 else "#F44336" for v in vals]

        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.bar(names, vals, color=colors, width=0.6, edgecolor="k", linewidth=0.5)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Energy Error vs FCI (mHa)", fontsize=12)
        ax.set_xlabel("Method", fontsize=12)
        ax.set_title(
            title or f"Energy Error vs {reference.upper()} "
                     f"(mol: {bench_result.mol_spec.geometry[:30]}…)",
            fontsize=12,
        )
        ax.tick_params(axis="x", rotation=30)

        # Add value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(vals) - min(vals)) * 0.02,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9,
            )

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig

    def multi_method_pes(
        self,
        geometries: Sequence[float],
        results_by_method: dict[str, Sequence[float]],
        x_label: str = "Bond distance (Å)",
        y_label: str = "Energy (Ha)",
        title: str = "Potential Energy Surface",
        save_path: str | Path | None = None,
    ) -> "plt.Figure":
        """
        Plot potential energy surface (PES) for multiple methods.

        Parameters
        ----------
        geometries : sequence of float
            X-axis values (bond distances, angles, etc.).
        results_by_method : dict
            Mapping from method name to list of energies.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        linestyles = ["-", "--", "-.", ":", "-", "--"]
        markers = ["o", "s", "^", "D", "v", "P"]

        for idx, (name, energies) in enumerate(results_by_method.items()):
            ls = linestyles[idx % len(linestyles)]
            mk = markers[idx % len(markers)]
            ax.plot(
                geometries, energies,
                linestyle=ls, marker=mk, markersize=5,
                label=name,
            )

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig

    def sqd_convergence(
        self,
        iterations: Sequence[int],
        energies: Sequence[float],
        fci_energy: float | None = None,
        title: str = "SQD Convergence",
        save_path: str | Path | None = None,
    ) -> "plt.Figure":
        """
        Plot SQD energy convergence across iterations.

        Parameters
        ----------
        iterations : sequence of int
            Iteration numbers.
        energies : sequence of float
            SQD energies at each iteration.
        fci_energy : float or None
            FCI reference to draw as horizontal dashed line.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(iterations, energies, "o-", color="#2196F3", label="SQD", linewidth=1.5)

        if fci_energy is not None:
            ax.axhline(
                fci_energy, color="#F44336", linestyle="--",
                linewidth=1.5, label="FCI",
            )

        ax.set_xlabel("SQD Iteration", fontsize=12)
        ax.set_ylabel("Energy (Ha)", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig

    def zne_extrapolation(
        self,
        scale_factors: Sequence[float],
        expectations: Sequence[float],
        zero_noise_energy: float,
        exact_energy: float | None = None,
        title: str = "ZNE Extrapolation",
        save_path: str | Path | None = None,
    ) -> "plt.Figure":
        """
        Plot ZNE extrapolation curve.

        Parameters
        ----------
        scale_factors : sequence of float
        expectations : sequence of float
            ⟨H⟩ at each scale factor.
        zero_noise_energy : float
            Extrapolated zero-noise energy.
        exact_energy : float or None
            Exact reference energy for comparison.
        """
        lambdas = np.array(scale_factors, dtype=float)
        evs = np.array(expectations, dtype=float)

        # Polynomial fit for smooth curve
        poly = np.polyfit(lambdas, evs, deg=min(2, len(lambdas) - 1))
        x_fit = np.linspace(0, max(lambdas) * 1.05, 200)
        y_fit = np.polyval(poly, x_fit)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(x_fit, y_fit, "b-", linewidth=1.5, alpha=0.6, label="Fit")
        ax.scatter(lambdas, evs, color="blue", zorder=5, label="Measured")
        ax.scatter(
            [0], [zero_noise_energy],
            color="green", marker="*", s=200, zorder=6, label="ZNE (λ→0)",
        )

        if exact_energy is not None:
            ax.axhline(exact_energy, color="red", linestyle="--",
                       linewidth=1.5, label="Exact")

        ax.set_xlabel("Noise scale factor λ", fontsize=12)
        ax.set_ylabel("⟨H⟩ (Ha)", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig
