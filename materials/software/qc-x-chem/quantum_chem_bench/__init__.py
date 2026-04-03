"""
quantum_chem_bench — Quantum Chemistry Algorithm Testing Platform

Importing this package registers all built-in classical and quantum solvers
into the global registry.

Quick start::

    import quantum_chem_bench as qcb

    config = qcb.BenchConfig.from_yaml('configs/h2_sto3g.yaml')
    bench  = qcb.BenchRunner(config).run()
    qcb.BenchRunner.print_summary(bench)
"""

# Register all solvers by importing the subpackages
from quantum_chem_bench import classical_solvers   # noqa: F401
from quantum_chem_bench import quantum_solvers     # noqa: F401

from quantum_chem_bench.core import (
    BaseSolver,
    BenchConfig,
    BenchResult,
    BenchRunner,
    MethodResult,
    MolIntegrals,
    MolSpec,
    registry,
)
from quantum_chem_bench.analysis import BenchmarkPlotter, PESScanner, energy_errors, format_table
from quantum_chem_bench.error_mitigation import ZNEWrapper, extrapolate_zne, fold_gates

__version__ = "0.1.0"

__all__ = [
    # Core
    "BaseSolver",
    "BenchConfig",
    "BenchResult",
    "BenchRunner",
    "MethodResult",
    "MolIntegrals",
    "MolSpec",
    "registry",
    # Analysis
    "BenchmarkPlotter",
    "PESScanner",
    "energy_errors",
    "format_table",
    # Error mitigation
    "ZNEWrapper",
    "extrapolate_zne",
    "fold_gates",
]
