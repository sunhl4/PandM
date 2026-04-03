"""Tests for core: registry, config, interfaces, and runner."""

import pytest
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_register_and_get(self):
        from quantum_chem_bench.core.registry import Registry
        reg = Registry()

        @reg.register("test_solver", category="solver")
        class MySolver:
            pass

        assert reg.get("test_solver", category="solver") is MySolver

    def test_register_duplicate_raises(self):
        from quantum_chem_bench.core.registry import Registry
        reg = Registry()

        @reg.register("dup", category="solver")
        class A:
            pass

        with pytest.raises(KeyError, match="already registered"):
            @reg.register("dup", category="solver")
            class B:
                pass

    def test_get_missing_raises(self):
        from quantum_chem_bench.core.registry import Registry
        reg = Registry()
        with pytest.raises(KeyError, match="No 'ghost' registered"):
            reg.get("ghost", category="solver")

    def test_list_names(self):
        from quantum_chem_bench.core.registry import Registry
        reg = Registry()

        @reg.register("a", category="x")
        class A:
            pass

        @reg.register("b", category="x")
        class B:
            pass

        names = reg.list_names(category="x")
        assert "a" in names
        assert "b" in names

    def test_build(self):
        from quantum_chem_bench.core.registry import Registry
        reg = Registry()

        @reg.register("buildable", category="solver")
        class Buildable:
            def __init__(self, val=42):
                self.val = val

        obj = reg.build("buildable", category="solver", val=99)
        assert obj.val == 99


class TestQuantumSolverSeedKwargs:
    """Optional ``seed`` is accepted by all quantum solvers (registry.build)."""

    def test_registry_build_passes_seed(self):
        from quantum_chem_bench.core.registry import registry

        names = (
            "adapt_vqe",
            "sqd",
            "qse",
            "qpe",
            "qpe_full",
            "vqe_uccsd",
            "vqe_hea",
            "vqe_kupccgsd",
            "vqe_uccsd_stack",
        )
        for name in names:
            s = registry.build(name, category="solver", seed=123)
            assert getattr(s, "seed", None) == 123


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestBenchConfig:
    def test_from_dict_minimal(self):
        from quantum_chem_bench.core.config import BenchConfig
        cfg = BenchConfig.from_dict({
            "molecule": {
                "geometry": "H 0 0 0; H 0 0 0.735",
                "basis": "sto-3g",
            },
            "solvers": {"classical": ["hf"], "quantum": []},
        })
        assert cfg.molecule.geometry == "H 0 0 0; H 0 0 0.735"
        assert cfg.molecule.basis == "sto-3g"
        assert "hf" in cfg.solvers.classical

    def test_from_dict_active_space(self):
        from quantum_chem_bench.core.config import BenchConfig
        cfg = BenchConfig.from_dict({
            "molecule": {
                "geometry": "H 0 0 0; H 0 0 0.735",
                "basis": "sto-3g",
                "n_active_electrons": [1, 1],
                "n_active_orbitals": 2,
            },
            "solvers": {},
        })
        assert cfg.molecule.n_active_electrons == (1, 1)
        assert cfg.molecule.n_active_orbitals == 2

    def test_to_mol_spec(self):
        from quantum_chem_bench.core.config import BenchConfig
        cfg = BenchConfig.from_dict({
            "molecule": {"geometry": "H 0 0 0; H 0 0 0.735", "basis": "sto-3g"},
            "mapper": {"type": "jw", "z2symmetry_reduction": False},
        })
        spec = cfg.to_mol_spec()
        assert spec.mapper_type == "jw"
        assert spec.z2symmetry_reduction is False

    def test_from_yaml(self, tmp_path):
        import yaml
        from quantum_chem_bench.core.config import BenchConfig

        yaml_content = {
            "molecule": {"geometry": "H 0 0 0; H 0 0 0.5", "basis": "sto-3g"},
            "solvers": {"classical": ["hf", "fci"], "quantum": ["vqe_uccsd"]},
            "mapper": {"type": "parity", "z2symmetry_reduction": True},
        }
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml.dump(yaml_content))

        cfg = BenchConfig.from_yaml(yaml_file)
        assert "fci" in cfg.solvers.classical
        assert "vqe_uccsd" in cfg.solvers.quantum

    def test_get_solver_opts(self):
        from quantum_chem_bench.core.config import BenchConfig
        cfg = BenchConfig.from_dict({
            "molecule": {"geometry": "H 0 0 0; H 0 0 0.735", "basis": "sto-3g"},
            "solver_options": {"vqe_uccsd": {"max_iter": 500}},
        })
        opts = cfg.get_solver_opts("vqe_uccsd")
        assert opts["max_iter"] == 500
        # Missing solver returns empty dict
        assert cfg.get_solver_opts("nonexistent") == {}

    def test_validate_rejects_mismatched_active_space(self):
        from quantum_chem_bench.core.config import BenchConfig
        with pytest.raises(ValueError, match="both set or both omitted"):
            BenchConfig.from_dict({
                "molecule": {
                    "geometry": "H 0 0 0; H 0 0 0.735",
                    "basis": "sto-3g",
                    "n_active_electrons": [1, 1],
                },
            })

    def test_validate_rejects_bad_mapper(self):
        from quantum_chem_bench.core.config import BenchConfig
        with pytest.raises(ValueError, match="mapper.type"):
            BenchConfig.from_dict({
                "molecule": {"geometry": "H 0 0 0; H 0 0 0.735", "basis": "sto-3g"},
                "mapper": {"type": "invalid_mapper"},
            })

    def test_validate_rejects_nelec_gt_2norb(self):
        from quantum_chem_bench.core.config import BenchConfig
        with pytest.raises(ValueError, match="exceeds 2\\*n_active"):
            BenchConfig.from_dict({
                "molecule": {
                    "geometry": "H 0 0 0; H 0 0 0.735",
                    "basis": "sto-3g",
                    "n_active_electrons": [5, 5],
                    "n_active_orbitals": 2,
                },
            })


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

class TestInterfaces:
    def test_mol_spec_defaults(self):
        from quantum_chem_bench.core.interfaces import MolSpec
        spec = MolSpec(geometry="H 0 0 0; H 0 0 0.5", basis="sto-3g")
        assert spec.charge == 0
        assert spec.spin == 0
        assert spec.mapper_type == "parity"
        assert spec.n_active_electrons is None

    def test_bench_result_add_and_summary(self):
        from quantum_chem_bench.core.interfaces import BenchResult, MethodResult, MolSpec
        spec = MolSpec(geometry="H 0 0 0; H 0 0 0.5", basis="sto-3g")
        bench = BenchResult(mol_spec=spec, hf_energy=-1.0, fci_energy=-1.1)
        r = MethodResult(
            method_name="test", energy=-1.05, corr_energy=-0.05,
            converged=True, n_qubits=2, wall_time=0.1,
        )
        bench.add(r)
        rows = bench.summary_table()
        assert len(rows) == 1
        assert rows[0]["Method"] == "test"
        assert abs(rows[0]["Error vs FCI (mHa)"] - 50.0) < 1e-6


# ---------------------------------------------------------------------------
# Runner (integration, requires PySCF)
# ---------------------------------------------------------------------------

class TestBenchRunner:
    @pytest.mark.requires_pyscf
    def test_runner_hf_fci(self, bench_config):
        from quantum_chem_bench.core.runner import BenchRunner
        runner = BenchRunner(bench_config, verbose=False)
        bench = runner.run()

        assert "HF" in bench.results
        assert "FCI" in bench.results
        assert bench.results["HF"].converged
        assert bench.results["FCI"].converged
        # HF energy > FCI energy (variational principle)
        assert bench.results["HF"].energy > bench.results["FCI"].energy

    @pytest.mark.requires_pyscf
    def test_runner_unknown_solver_skipped(self):
        import quantum_chem_bench.classical_solvers  # noqa
        from quantum_chem_bench.core.config import BenchConfig
        from quantum_chem_bench.core.runner import BenchRunner

        cfg = BenchConfig.from_dict({
            "molecule": {"geometry": "H 0 0 0; H 0 0 0.735", "basis": "sto-3g"},
            "solvers": {"classical": ["hf", "nonexistent_method"], "quantum": []},
        })
        runner = BenchRunner(cfg, verbose=False)
        bench = runner.run()
        # nonexistent_method silently skipped; HF succeeds
        assert "HF" in bench.results
        assert "nonexistent_method" not in bench.results
