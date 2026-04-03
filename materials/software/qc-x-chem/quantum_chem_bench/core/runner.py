"""
BenchRunner — orchestrates multi-solver benchmark runs.

Usage::

    from quantum_chem_bench.core.runner import BenchRunner
    from quantum_chem_bench.core.config import BenchConfig

    config = BenchConfig.from_yaml("configs/h2_sto3g.yaml")
    runner = BenchRunner(config)
    bench_result = runner.run()

    import pandas as pd
    df = pd.DataFrame(bench_result.summary_table())
    print(df.to_string(index=False))
"""

from __future__ import annotations

import logging
from typing import Any

from quantum_chem_bench.core.config import BenchConfig
from quantum_chem_bench.core.interfaces import BenchResult, MethodResult, MolSpec
from quantum_chem_bench.core.registry import registry

logger = logging.getLogger(__name__)


class BenchRunner:
    """
    Run all configured solvers on a single molecule and collect results.

    Parameters
    ----------
    config : BenchConfig
        Fully-specified benchmark configuration.
    verbose : bool
        If True, log progress at INFO level.
    """

    def __init__(self, config: BenchConfig, *, verbose: bool = True) -> None:
        self.config = config
        self.verbose = verbose
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BenchResult:
        """
        Execute all configured solvers and return aggregated BenchResult.

        Returns
        -------
        BenchResult
            Contains one MethodResult per solver plus HF/FCI reference
            energies (when those solvers are included).
        """
        mol_spec = self.config.to_mol_spec()
        bench = BenchResult(mol_spec=mol_spec)

        logger.info(
            "BenchRunner starting — molecule: %s | basis: %s | solvers: %s",
            mol_spec.geometry[:40],
            mol_spec.basis,
            self.config.solvers.all,
        )

        all_solver_names = self.config.solvers.all

        for solver_name in all_solver_names:
            result = self._run_one(solver_name, mol_spec)
            if result is not None:
                bench.add(result)

        # Populate reference energies
        if "hf" in bench.results:
            bench.hf_energy = bench.results["hf"].energy
        if "fci" in bench.results:
            bench.fci_energy = bench.results["fci"].energy

        # Back-fill correlation energies relative to HF
        if bench.hf_energy != 0.0:
            for r in bench.results.values():
                r.corr_energy = r.energy - bench.hf_energy

        logger.info(
            "BenchRunner finished — %d/%d solvers succeeded.",
            len(bench.results),
            len(all_solver_names),
        )
        return bench

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_one(
        self, solver_name: str, mol_spec: MolSpec
    ) -> MethodResult | None:
        """Build and run a single solver; log any failures gracefully."""
        opts = self.config.get_solver_opts(solver_name)
        # Inject mapper settings for quantum solvers
        opts.setdefault("mapper_type", self.config.mapper.type)
        opts.setdefault(
            "z2symmetry_reduction", self.config.mapper.z2symmetry_reduction
        )

        try:
            solver = registry.build(solver_name, category="solver", **opts)
        except KeyError:
            logger.warning(
                "Solver '%s' not registered — skipping (have you imported its module?)",
                solver_name,
            )
            return None

        logger.info("Running solver: %s …", solver_name)
        try:
            result = solver.solve(mol_spec)
            logger.info(
                "  %-20s  energy = %+.10f Ha  converged = %s  t = %.1f s",
                solver_name,
                result.energy,
                result.converged,
                result.wall_time,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("Solver '%s' raised: %s", solver_name, exc, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Convenience: print summary table
    # ------------------------------------------------------------------

    @staticmethod
    def print_summary(bench: BenchResult) -> None:
        """Print a formatted summary table to stdout."""
        rows = bench.summary_table()
        if not rows:
            print("No results collected.")
            return
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            print(df.to_string(index=False))
        except ImportError:
            # Fallback: simple text table
            headers = list(rows[0].keys())
            widths = [max(len(str(h)), max(len(str(r[h])) for r in rows)) for h in headers]
            fmt = "  ".join(f"{{:<{w}}}" for w in widths)
            print(fmt.format(*headers))
            print("  ".join("-" * w for w in widths))
            for row in rows:
                print(fmt.format(*[str(row[h]) for h in headers]))
