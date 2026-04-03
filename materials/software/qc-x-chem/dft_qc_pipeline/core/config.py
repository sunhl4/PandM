"""
Configuration dataclasses and YAML loader for the pipeline.

Every section of the YAML config maps to one of the ``*Config`` dataclasses
below.  ``PipelineConfig.from_yaml`` provides the single entry-point for
loading a full pipeline configuration from a file.

YAML layout expected::

    backend:
      type: pyscf
      geometry: "H 0 0 0; H 0 0 0.735"
      basis: sto-3g
      method: hf          # hf | dft
      xc: pbe             # only for dft

    regions:
      - name: fragment_A
        atom_indices: [0, 1]
        nelec: 2
        norb: 2
        localization: iao   # boys | pm | iao | none

    embedding:
      type: simple_cas      # simple_cas | dmet | projector | avas
      max_iter: 1
      conv_tol: 1.0e-6

    mapper:
      type: parity          # jw | parity | bk
      z2symmetry_reduction: true

    solver:
      type: numpy           # numpy | vqe | sqd | adapt_vqe
      # vqe-specific:
      ansatz: uccsd         # uccsd | hea | kupccgsd
      optimizer: cobyla
      max_iter: 300
      # sqd-specific:
      shots: 10000
      sqd_iterations: 10
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class BackendConfig:
    type: str = "pyscf"
    geometry: str = "H 0 0 0; H 0 0 0.735"
    basis: str = "sto-3g"
    method: str = "hf"          # hf | dft
    xc: str = "pbe"
    charge: int = 0
    spin: int = 0               # 2S
    # PySCF: accelerate J/K build (recommended for medium/large bases)
    density_fit: bool = False
    auxbasis: str | None = None  # e.g. "weigend"; None → PySCF default for DF
    extra: dict = field(default_factory=dict)


@dataclass
class RegionConfig:
    name: str = "fragment"
    atom_indices: list[int] = field(default_factory=list)
    orbital_indices: list[int] = field(default_factory=list)  # alternative to atoms
    nelec: int = 2
    norb: int = 2
    localization: str = "iao"   # boys | pm | iao | none
    extra: dict = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    type: str = "simple_cas"    # simple_cas | dmet | projector | avas
    max_iter: int = 1
    conv_tol: float = 1e-6
    # DMET: chemical-potential update mode (extra YAML keys still merged into ctor)
    mu_update: str = "gradient"  # gradient | damped | bisection | bisection_bracket
    # DMET: clamp |μ| after each update (None = no clamp)
    mu_max_abs: float | None = None
    # AVAS: optional explicit PySCF AO patterns, e.g. ["Fe 3d", "Fe 4s"]
    ao_labels: list[str] | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class MapperConfig:
    type: str = "parity"        # jw | parity | bk
    z2symmetry_reduction: bool = True
    extra: dict = field(default_factory=dict)


@dataclass
class SolverConfig:
    type: str = "numpy"         # numpy | vqe | sqd | adapt_vqe
    # VQE options
    ansatz: str = "uccsd"       # uccsd | hea | uccsd_stack | kupccgsd (alias, see docs)
    optimizer: str = "cobyla"
    max_iter: int = 300
    shots: int | None = None    # None → statevector (simulator)
    seed: int | None = None     # RNG seed for VQE/SQD stochastic paths
    # SQD options
    sqd_iterations: int = 10
    sqd_shots: int = 10_000
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level pipeline config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    backend: BackendConfig = field(default_factory=BackendConfig)
    regions: list[RegionConfig] = field(default_factory=lambda: [RegionConfig()])
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    mapper: MapperConfig = field(default_factory=MapperConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    # When True, run all registered solvers on each fragment for comparison
    benchmark_mode: bool = False
    benchmark_solvers: list[str] = field(default_factory=list)
    # Run independent regions in parallel (thread pool). Not used with benchmark_mode.
    parallel_regions: bool = False
    max_parallel_workers: int | None = None  # None → min(32, len(regions))
    # When True and embedding is dmet and len(regions) > 1: classical Mulliken
    # point-charge estimate between region atom groups → energy_corrections
    include_inter_fragment_point_charge: bool = True

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineConfig":
        """Build from a plain Python dict (e.g. already parsed YAML)."""
        cfg = cls()
        if "backend" in d:
            cfg.backend = _from_dict(BackendConfig, d["backend"])
        if "regions" in d:
            cfg.regions = [_from_dict(RegionConfig, r) for r in d["regions"]]
        if "embedding" in d:
            cfg.embedding = _from_dict(EmbeddingConfig, d["embedding"])
        if "mapper" in d:
            cfg.mapper = _from_dict(MapperConfig, d["mapper"])
        if "solver" in d:
            cfg.solver = _from_dict(SolverConfig, d["solver"])
        cfg.benchmark_mode = d.get("benchmark_mode", False)
        cfg.benchmark_solvers = d.get("benchmark_solvers", [])
        cfg.parallel_regions = bool(d.get("parallel_regions", False))
        mpw = d.get("max_parallel_workers")
        cfg.max_parallel_workers = int(mpw) if mpw is not None else None
        if "include_inter_fragment_point_charge" in d:
            cfg.include_inter_fragment_point_charge = bool(
                d["include_inter_fragment_point_charge"]
            )
        validate_pipeline_config(cfg)
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load from a YAML file."""
        if not _YAML_AVAILABLE:
            raise ImportError("pyyaml is required: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls.from_dict(raw or {})


# ---------------------------------------------------------------------------
# Validation & registry kwargs
# ---------------------------------------------------------------------------

def validate_pipeline_config(cfg: PipelineConfig) -> None:
    """Raise ``ValueError`` if the configuration is inconsistent."""
    if not cfg.regions:
        raise ValueError("pipeline config: 'regions' must contain at least one region")
    if cfg.backend.spin < 0:
        raise ValueError("backend.spin (2S) must be non-negative")
    if cfg.embedding.max_iter < 1:
        raise ValueError("embedding.max_iter must be >= 1")
    if cfg.embedding.conv_tol <= 0:
        raise ValueError("embedding.conv_tol must be positive")
    mu_mode = (cfg.embedding.mu_update or "gradient").lower()
    if mu_mode not in ("gradient", "damped", "bisection", "bisection_bracket"):
        raise ValueError(
            "embedding.mu_update must be one of: gradient, damped, bisection, "
            f"bisection_bracket (got {cfg.embedding.mu_update!r})"
        )
    if cfg.parallel_regions and cfg.benchmark_mode:
        raise ValueError(
            "parallel_regions=True is incompatible with benchmark_mode=True "
            "(benchmark shares solver instances; run regions sequentially or disable benchmark)."
        )
    seen_names: set[str] = set()
    for i, r in enumerate(cfg.regions):
        if r.name in seen_names:
            raise ValueError(
                f"regions[{i}]: duplicate region name {r.name!r}; names must be unique"
            )
        seen_names.add(r.name)
        if r.norb < 1:
            raise ValueError(f"regions[{i}] ({r.name!r}): norb must be >= 1")
        if r.nelec < 0:
            raise ValueError(f"regions[{i}] ({r.name!r}): nelec must be >= 0")
        if r.nelec > 2 * r.norb:
            raise ValueError(
                f"regions[{i}] ({r.name!r}): nelec={r.nelec} exceeds 2*norb={2 * r.norb}"
            )


def merge_registry_kwargs(obj: Any) -> tuple[str, dict[str, Any]]:
    """
    Build kwargs for ``registry.build`` from a config dataclass.

    Unknown YAML keys are stored in ``.extra``; known fields override ``extra``
    on key collisions.
    """
    fields = dict(vars(obj))
    extra = dict(fields.pop("extra", None) or {})
    typ = fields.pop("type")
    return typ, {**extra, **fields}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _from_dict(datacls: type, d: dict[str, Any]):
    """
    Construct a dataclass from a dict; unknown keys go into ``extra``.
    """
    import dataclasses
    known = {f.name for f in dataclasses.fields(datacls)}
    known_vals = {k: v for k, v in d.items() if k in known}
    extra_vals = {k: v for k, v in d.items() if k not in known}
    if "extra" in known:
        known_vals["extra"] = extra_vals
    return datacls(**known_vals)
