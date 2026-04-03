"""
Configuration dataclasses and YAML loader for quantum_chem_bench.

All YAML config files are loaded via ``BenchConfig.from_yaml(path)``.

Example YAML structure::

    molecule:
      geometry: "H 0 0 0; H 0 0 0.735"
      basis: sto-3g
      charge: 0
      spin: 0
      n_active_electrons: [1, 1]
      n_active_orbitals: 2

    solvers:
      classical: [hf, mp2, cisd, ccsd, fci]
      quantum:   [vqe_uccsd, adapt_vqe, sqd]

    mapper:
      type: parity
      z2symmetry_reduction: true

    solver_options:
      vqe_uccsd:
        optimizer: slsqp
        max_iter: 300
        shots: null
      adapt_vqe:
        max_iter: 50
        gradient_threshold: 1.0e-3
      sqd:
        shots: 10000
        iterations: 10
        ansatz: hea
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class MoleculeConfig:
    geometry: str
    basis: str
    charge: int = 0
    spin: int = 0
    n_active_electrons: tuple[int, int] | None = None
    n_active_orbitals: int | None = None
    density_fit: bool = False
    auxbasis: str | None = None


@dataclass
class MapperConfig:
    type: str = "parity"
    z2symmetry_reduction: bool = True


@dataclass
class SolversConfig:
    classical: list[str] = field(default_factory=list)
    quantum: list[str] = field(default_factory=list)

    @property
    def all(self) -> list[str]:
        return self.classical + self.quantum


@dataclass
class BenchConfig:
    """
    Top-level configuration for a benchmark run.

    Parameters
    ----------
    molecule : MoleculeConfig
    solvers : SolversConfig
    mapper : MapperConfig
    solver_options : dict[str, dict]
        Per-solver keyword arguments forwarded to solver constructors.
    name : str
        Optional human label for this benchmark (used in plot titles).
    """
    molecule: MoleculeConfig
    solvers: SolversConfig = field(default_factory=SolversConfig)
    mapper: MapperConfig = field(default_factory=MapperConfig)
    solver_options: dict[str, dict] = field(default_factory=dict)
    name: str = "benchmark"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchConfig":
        """Load and validate a BenchConfig from a YAML file."""
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        return cls._from_dict(raw)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchConfig":
        """Construct a BenchConfig from a plain dict."""
        return cls._from_dict(copy.deepcopy(data))

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> "BenchConfig":
        mol_raw = raw.get("molecule", {})
        nae = mol_raw.get("n_active_electrons")
        if nae is not None:
            nae = tuple(int(x) for x in nae)
        mol = MoleculeConfig(
            geometry=mol_raw["geometry"],
            basis=mol_raw.get("basis", "sto-3g"),
            charge=int(mol_raw.get("charge", 0)),
            spin=int(mol_raw.get("spin", 0)),
            n_active_electrons=nae,
            n_active_orbitals=mol_raw.get("n_active_orbitals"),
            density_fit=bool(mol_raw.get("density_fit", False)),
            auxbasis=mol_raw.get("auxbasis"),
        )

        solvers_raw = raw.get("solvers", {})
        solvers = SolversConfig(
            classical=list(solvers_raw.get("classical", [])),
            quantum=list(solvers_raw.get("quantum", [])),
        )

        mapper_raw = raw.get("mapper", {})
        mapper = MapperConfig(
            type=mapper_raw.get("type", "parity"),
            z2symmetry_reduction=bool(mapper_raw.get("z2symmetry_reduction", True)),
        )

        solver_options: dict[str, dict] = {}
        for key, val in raw.get("solver_options", {}).items():
            solver_options[key] = dict(val) if val else {}

        cfg = cls(
            molecule=mol,
            solvers=solvers,
            mapper=mapper,
            solver_options=solver_options,
            name=str(raw.get("name", "benchmark")),
        )
        validate_bench_config(cfg)
        return cfg

    def get_solver_opts(self, solver_name: str) -> dict[str, Any]:
        """Return solver-specific options, falling back to empty dict."""
        return copy.deepcopy(self.solver_options.get(solver_name, {}))

    def to_mol_spec(self):
        """Convert molecule + mapper settings to a MolSpec."""
        from quantum_chem_bench.core.interfaces import MolSpec
        return MolSpec(
            geometry=self.molecule.geometry,
            basis=self.molecule.basis,
            charge=self.molecule.charge,
            spin=self.molecule.spin,
            n_active_electrons=self.molecule.n_active_electrons,
            n_active_orbitals=self.molecule.n_active_orbitals,
            mapper_type=self.mapper.type,
            z2symmetry_reduction=self.mapper.z2symmetry_reduction,
            density_fit=self.molecule.density_fit,
            auxbasis=self.molecule.auxbasis,
        )


def validate_bench_config(cfg: BenchConfig) -> None:
    """
    Raise ``ValueError`` if molecule / mapper / active-space fields are inconsistent.

    Called automatically from ``BenchConfig.from_yaml`` / ``from_dict``.
    """
    m = cfg.molecule
    if not (m.geometry or "").strip():
        raise ValueError("molecule.geometry must be non-empty")
    if not (m.basis or "").strip():
        raise ValueError("molecule.basis must be non-empty")
    if m.spin < 0:
        raise ValueError("molecule.spin (2S) must be non-negative")

    mt = (cfg.mapper.type or "parity").lower()
    if mt not in ("jw", "parity", "bk"):
        raise ValueError(f"mapper.type must be one of: jw, parity, bk (got {cfg.mapper.type!r})")

    nae = m.n_active_electrons
    nao = m.n_active_orbitals
    if (nae is None) ^ (nao is None):
        raise ValueError(
            "molecule.n_active_electrons and n_active_orbitals must be both set or both omitted"
        )
    if nae is not None and nao is not None:
        na, nb = int(nae[0]), int(nae[1])
        if na < 0 or nb < 0:
            raise ValueError("n_active_electrons components must be non-negative")
        if nao < 1:
            raise ValueError("n_active_orbitals must be >= 1")
        if na + nb > 2 * nao:
            raise ValueError(
                f"n_alpha+n_beta={na + nb} exceeds 2*n_active_orbitals={2 * nao}"
            )
