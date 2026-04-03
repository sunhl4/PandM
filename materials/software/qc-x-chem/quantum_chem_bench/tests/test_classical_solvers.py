"""Tests for classical solvers (requires PySCF)."""

import pytest
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import quantum_chem_bench.classical_solvers  # noqa: F401 — register all


# Reference energies for H₂ STO-3G from literature
H2_STO3G_HF = -1.1175_0     # Ha (approximate)
H2_STO3G_FCI = -1.1373_2    # Ha (approximate)
H2_STO3G_CCSD = -1.1373_2   # Ha (H₂ CCSD = FCI for 2e)

ENERGY_TOL = 2e-4  # 0.2 mHa tolerance


@pytest.fixture(scope="module")
def h2_spec():
    from quantum_chem_bench.core.interfaces import MolSpec
    return MolSpec(
        geometry="H 0 0 0; H 0 0 0.735",
        basis="sto-3g",
        n_active_electrons=(1, 1),
        n_active_orbitals=2,
    )


@pytest.mark.requires_pyscf
class TestHFSolver:
    def test_energy_close_to_reference(self, h2_spec):
        from quantum_chem_bench.core.registry import registry
        solver = registry.build("hf", category="solver")
        result = solver.solve(h2_spec)
        assert result.converged
        assert abs(result.energy - H2_STO3G_HF) < ENERGY_TOL
        assert result.n_qubits is None
        assert result.corr_energy == 0.0

    def test_wall_time_positive(self, h2_spec):
        from quantum_chem_bench.core.registry import registry
        solver = registry.build("hf", category="solver")
        result = solver.solve(h2_spec)
        assert result.wall_time > 0


@pytest.mark.requires_pyscf
class TestMP2Solver:
    def test_energy_below_hf(self, h2_spec):
        from quantum_chem_bench.core.registry import registry
        hf = registry.build("hf", category="solver").solve(h2_spec)
        mp2 = registry.build("mp2", category="solver").solve(h2_spec)
        # MP2 energy < HF (correlation is always negative)
        assert mp2.energy < hf.energy
        assert mp2.corr_energy < 0


@pytest.mark.requires_pyscf
class TestCISDSolver:
    def test_variational(self, h2_spec):
        from quantum_chem_bench.core.registry import registry
        hf = registry.build("hf", category="solver").solve(h2_spec)
        cisd = registry.build("cisd", category="solver").solve(h2_spec)
        assert cisd.energy < hf.energy
        assert cisd.energy >= H2_STO3G_FCI - ENERGY_TOL  # CISD ≥ FCI

    def test_method_name(self, h2_spec):
        from quantum_chem_bench.core.registry import registry
        r = registry.build("cisd", category="solver").solve(h2_spec)
        assert "CISD" in r.method_name


@pytest.mark.requires_pyscf
class TestCCSDSolver:
    def test_ccsd_equals_fci_for_h2(self, h2_spec):
        """For 2 electrons, CCSD is exact (equals FCI)."""
        from quantum_chem_bench.core.registry import registry
        ccsd = registry.build("ccsd", category="solver").solve(h2_spec)
        fci = registry.build("fci", category="solver").solve(h2_spec)
        assert abs(ccsd.energy - fci.energy) < ENERGY_TOL

    def test_ccsd_t_close_to_ccsd_for_h2(self, h2_spec):
        """For H₂, CCSD(T) triples correction is near zero."""
        from quantum_chem_bench.core.registry import registry
        ccsd = registry.build("ccsd", category="solver").solve(h2_spec)
        ccsdt = registry.build("ccsd_t", category="solver").solve(h2_spec)
        assert abs(ccsd.energy - ccsdt.energy) < 1e-4


@pytest.mark.requires_pyscf
class TestFCISolver:
    def test_energy_exact(self, h2_spec):
        from quantum_chem_bench.core.registry import registry
        fci = registry.build("fci", category="solver").solve(h2_spec)
        assert abs(fci.energy - H2_STO3G_FCI) < ENERGY_TOL
        assert fci.converged
        assert fci.n_qubits is None

    def test_fci_is_lowest(self, h2_spec):
        """FCI must give the lowest energy (variational principle)."""
        from quantum_chem_bench.core.registry import registry
        fci = registry.build("fci", category="solver").solve(h2_spec)
        for method in ["hf", "mp2", "cisd", "ccsd"]:
            r = registry.build(method, category="solver").solve(h2_spec)
            assert r.energy >= fci.energy - 1e-8, (
                f"{method} energy {r.energy:.8f} < FCI {fci.energy:.8f}"
            )


@pytest.mark.requires_pyscf
class TestClassicalSolverOrdering:
    def test_energy_hierarchy(self, h2_spec):
        """E(HF) > E(MP2) > E(CISD) ≥ E(CCSD) ≈ E(FCI)."""
        from quantum_chem_bench.core.registry import registry
        energies = {}
        for m in ["hf", "mp2", "cisd", "ccsd", "fci"]:
            energies[m] = registry.build(m, category="solver").solve(h2_spec).energy

        assert energies["hf"] > energies["mp2"] - 1e-8
        assert energies["mp2"] > energies["fci"] - ENERGY_TOL
        assert energies["cisd"] > energies["fci"] - ENERGY_TOL
        assert energies["ccsd"] > energies["fci"] - ENERGY_TOL
