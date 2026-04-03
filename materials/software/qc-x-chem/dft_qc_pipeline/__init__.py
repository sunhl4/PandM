"""
dft_qc_pipeline – top-level package.

Importing this package registers all built-in backends, embeddings, and solvers
into the global registry.
"""

# Trigger registration by importing all subpackages
from . import classical_backends   # noqa: F401  registers "pyscf", "toy"
from . import embedding            # noqa: F401  registers "simple_cas", "dmet", "avas", "projector", "hubbard"
from . import quantum_solvers      # noqa: F401  registers "numpy", "vqe", "sqd", "adapt_vqe"

from .core import (
    Pipeline,
    PipelineConfig,
    BackendConfig,
    RegionConfig,
    EmbeddingConfig,
    MapperConfig,
    SolverConfig,
    registry,
)

__version__ = "0.3.1"

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "BackendConfig",
    "RegionConfig",
    "EmbeddingConfig",
    "MapperConfig",
    "SolverConfig",
    "registry",
]
