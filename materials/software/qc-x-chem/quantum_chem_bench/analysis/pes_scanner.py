"""
PES Scanner — scan a molecular coordinate and collect energies from multiple solvers.

Usage::

    from quantum_chem_bench.analysis.pes_scanner import PESScanner
    from quantum_chem_bench.core.config import BenchConfig

    config = BenchConfig.from_yaml("configs/h2_sto3g.yaml")
    scanner = PESScanner(config)

    # Scan H-H bond distance from 0.4 to 2.5 Å
    pes = scanner.scan_bond(
        atom_indices=(0, 1),
        distances=np.linspace(0.4, 2.5, 20),
        atom_symbols=("H", "H"),
    )
    # pes["energies"] → {method_name: [e0, e1, …]}
    # pes["geometries"] → [0.4, 0.5, …]
"""

from __future__ import annotations

import copy
import logging
from typing import Sequence

import numpy as np

from quantum_chem_bench.core.config import BenchConfig
from quantum_chem_bench.core.runner import BenchRunner

logger = logging.getLogger(__name__)


class PESScanner:
    """
    Scan a molecular coordinate (bond length, angle) using BenchRunner.

    Parameters
    ----------
    config : BenchConfig
        Base configuration; geometry will be overwritten at each scan point.
    verbose : bool
        Log progress.
    """

    def __init__(self, config: BenchConfig, *, verbose: bool = True) -> None:
        self.config = config
        self.verbose = verbose

    def scan_bond(
        self,
        atom_indices: tuple[int, int],
        distances: Sequence[float],
        atom_symbols: tuple[str, str],
        fixed_atoms: str | None = None,
    ) -> dict:
        """
        Scan a diatomic bond length.

        Parameters
        ----------
        atom_indices : (i, j)
            Indices of the two atoms whose bond is scanned.
        distances : sequence of float
            Bond distances in Angstrom.
        atom_symbols : (sym_i, sym_j)
            Element symbols of the two atoms.
        fixed_atoms : str or None
            PySCF geometry string for any additional frozen atoms.

        Returns
        -------
        dict with keys:
            ``"geometries"``   — list of distances used,
            ``"energies"``     — {method_name: [energy_at_each_d]},
            ``"wall_times"``   — {method_name: [time_at_each_d]},
            ``"converged"``    — {method_name: [bool_at_each_d]}.
        """
        sym_i, sym_j = atom_symbols
        results: dict[str, list] = {}
        times: dict[str, list] = {}
        conv: dict[str, list] = {}

        for d in distances:
            geom = f"{sym_i} 0 0 0; {sym_j} 0 0 {d:.6f}"
            if fixed_atoms:
                geom = geom + "; " + fixed_atoms

            cfg = self._clone_config(geom)
            runner = BenchRunner(cfg, verbose=False)
            bench = runner.run()

            for method, r in bench.results.items():
                results.setdefault(method, []).append(r.energy)
                times.setdefault(method, []).append(r.wall_time)
                conv.setdefault(method, []).append(r.converged)

            if self.verbose:
                logger.info(
                    "PES scan d=%.3f Å — %d methods completed.",
                    d, len(bench.results),
                )

        return {
            "geometries": list(distances),
            "energies": results,
            "wall_times": times,
            "converged": conv,
        }

    def scan_custom(
        self,
        geometry_fn,
        parameters: Sequence[float],
        param_label: str = "parameter",
    ) -> dict:
        """
        Scan using a custom geometry generator function.

        Parameters
        ----------
        geometry_fn : callable
            ``geometry_fn(param) → str``  returns PySCF geometry string.
        parameters : sequence of float
            Parameter values to scan.
        param_label : str
            Label for the x-axis (used in the returned dict).

        Returns
        -------
        dict with keys:
            ``"parameters"``, ``"energies"``, ``"wall_times"``, ``"converged"``.
        """
        results: dict[str, list] = {}
        times: dict[str, list] = {}
        conv: dict[str, list] = {}

        for p in parameters:
            geom = geometry_fn(p)
            cfg = self._clone_config(geom)
            runner = BenchRunner(cfg, verbose=False)
            bench = runner.run()

            for method, r in bench.results.items():
                results.setdefault(method, []).append(r.energy)
                times.setdefault(method, []).append(r.wall_time)
                conv.setdefault(method, []).append(r.converged)

            if self.verbose:
                logger.info(
                    "PES scan %s=%.4f — %d methods completed.",
                    param_label, p, len(bench.results),
                )

        return {
            "parameters": list(parameters),
            "energies": results,
            "wall_times": times,
            "converged": conv,
        }

    def _clone_config(self, new_geometry: str) -> BenchConfig:
        """Deep-copy config with updated geometry."""
        cfg = copy.deepcopy(self.config)
        cfg.molecule.geometry = new_geometry
        return cfg
