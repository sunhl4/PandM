"""core package – re-export the public API."""

from .interfaces import (
    BackendResult,
    ClassicalBackend,
    EmbeddedHamiltonian,
    EmbeddingMethod,
    PipelineResult,
    QuantumSolver,
    QubitMapperWrapper,
    SolverResult,
    SupportsEmbeddedMap,
)
from .registry import registry
from .config import (
    PipelineConfig,
    BackendConfig,
    RegionConfig,
    EmbeddingConfig,
    MapperConfig,
    SolverConfig,
    merge_registry_kwargs,
    validate_pipeline_config,
)
from .pipeline import Pipeline

__all__ = [
    "BackendResult",
    "ClassicalBackend",
    "EmbeddedHamiltonian",
    "EmbeddingMethod",
    "PipelineResult",
    "QuantumSolver",
    "QubitMapperWrapper",
    "SolverResult",
    "SupportsEmbeddedMap",
    "registry",
    "PipelineConfig",
    "BackendConfig",
    "RegionConfig",
    "EmbeddingConfig",
    "MapperConfig",
    "SolverConfig",
    "merge_registry_kwargs",
    "validate_pipeline_config",
    "Pipeline",
]
