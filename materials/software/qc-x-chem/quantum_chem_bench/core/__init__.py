"""quantum_chem_bench.core — core interfaces, registry, config and runner."""

from quantum_chem_bench.core.interfaces import (
    BaseSolver,
    BenchResult,
    MethodResult,
    MolIntegrals,
    MolSpec,
)
from quantum_chem_bench.core.registry import registry
from quantum_chem_bench.core.config import BenchConfig, validate_bench_config
from quantum_chem_bench.core.runner import BenchRunner

__all__ = [
    "BaseSolver",
    "BenchConfig",
    "validate_bench_config",
    "BenchResult",
    "BenchRunner",
    "MethodResult",
    "MolIntegrals",
    "MolSpec",
    "registry",
]
