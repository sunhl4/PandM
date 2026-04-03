"""
Benchmark collector: run multiple solvers on the same fragment Hamiltonian
and produce a comparison table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.interfaces import EmbeddedHamiltonian, SolverResult
from ..core.registry import registry

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkEntry:
    solver_name: str
    energy: float
    converged: bool
    extra: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    region_name: str
    entries: list[BenchmarkEntry] = field(default_factory=list)
    reference_energy: float | None = None   # e.g. NumPy FCI

    def summary_table(self) -> str:
        """Return a human-readable comparison table."""
        lines = [
            f"\nBenchmark: region '{self.region_name}'",
            f"{'Solver':<20} {'Energy (Ha)':>18} {'ΔE vs FCI (mHa)':>20} {'Converged':>10}",
            "-" * 72,
        ]
        ref = self.reference_energy
        for e in sorted(self.entries, key=lambda x: x.energy):
            delta = (e.energy - ref) * 1000 if ref is not None else float("nan")
            lines.append(
                f"{e.solver_name:<20} {e.energy:>18.10f} "
                f"{delta:>20.4f} {str(e.converged):>10}"
            )
        return "\n".join(lines)

    def best(self) -> BenchmarkEntry | None:
        if not self.entries:
            return None
        return min(self.entries, key=lambda x: x.energy)


class BenchmarkCollector:
    """
    Run a list of solvers on a single fragment Hamiltonian and collect results.

    Usage::

        bc = BenchmarkCollector(solver_names=["numpy", "vqe", "sqd"])
        result = bc.run(emb_H, mapper, num_particles, num_spatial_orbitals)
        print(result.summary_table())
    """

    def __init__(self, solver_names: list[str], solver_kwargs: dict | None = None) -> None:
        self.solver_names = solver_names
        self.solver_kwargs = solver_kwargs or {}

    def run(
        self,
        emb_H: EmbeddedHamiltonian,
        mapper,
        solver_configs: dict[str, dict] | None = None,
    ) -> BenchmarkResult:
        """
        Execute all solvers and collect energies.

        Parameters
        ----------
        emb_H : EmbeddedHamiltonian
        mapper : _MapperWrapper  (from hamiltonian.mappers)
        solver_configs : dict mapping solver name → kwargs dict, optional

        Returns
        -------
        BenchmarkResult
        """
        H_qubit, num_particles, num_spatial_orbitals = mapper.map(emb_H)
        bench = BenchmarkResult(region_name=emb_H.region_name)

        for sname in self.solver_names:
            logger.info("[Benchmark] Running solver: %s", sname)
            cfg = dict(self.solver_kwargs.get(sname, {}))
            if solver_configs and sname in solver_configs:
                cfg.update(solver_configs[sname])
            cfg["type"] = sname

            try:
                solver = registry.build(cfg, category="solver")
                result: SolverResult = solver.solve(
                    H_qubit, num_particles, num_spatial_orbitals
                )
                entry = BenchmarkEntry(
                    solver_name=sname,
                    energy=result.energy + emb_H.e_core,
                    converged=result.converged,
                    extra=result.extra,
                )
            except Exception as exc:
                logger.warning("[Benchmark] Solver '%s' failed: %s", sname, exc)
                entry = BenchmarkEntry(
                    solver_name=sname,
                    energy=float("nan"),
                    converged=False,
                    extra={"error": str(exc)},
                )

            bench.entries.append(entry)

            # Mark the NumPy result as FCI reference
            if sname == "numpy" and not np.isnan(entry.energy):
                bench.reference_energy = entry.energy

        logger.info("\n%s", bench.summary_table())
        return bench
