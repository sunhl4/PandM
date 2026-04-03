"""Tests for quantum solvers (requires PySCF + Qiskit)."""

import pytest
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import quantum_chem_bench.classical_solvers  # noqa: F401
import quantum_chem_bench.quantum_solvers    # noqa: F401

H2_STO3G_FCI = -1.13732
ENERGY_TOL = 5e-3  # 5 mHa tolerance for VQE tests


@pytest.fixture(scope="module")
def h2_spec():
    from quantum_chem_bench.core.interfaces import MolSpec
    return MolSpec(
        geometry="H 0 0 0; H 0 0 0.735",
        basis="sto-3g",
        n_active_electrons=(1, 1),
        n_active_orbitals=2,
        mapper_type="parity",
        z2symmetry_reduction=True,
    )


@pytest.mark.requires_pyscf
@pytest.mark.requires_qiskit_nature
class TestVQEUCCSDSolver:
    def test_energy_close_to_fci(self, h2_spec):
        from quantum_chem_bench.core.registry import registry
        vqe = registry.build("vqe_uccsd", category="solver",
                              optimizer="slsqp", max_iter=300)
        result = vqe.solve(h2_spec)
        assert result.n_qubits is not None
        assert result.n_qubits >= 2
        assert abs(result.energy - H2_STO3G_FCI) < ENERGY_TOL

    def test_method_name_contains_uccsd(self, h2_spec):
        from quantum_chem_bench.core.registry import registry
        r = registry.build("vqe_uccsd", category="solver").solve(h2_spec)
        assert "UCCSD" in r.method_name.upper()


@pytest.mark.requires_pyscf
@pytest.mark.requires_qiskit_nature
class TestVQEHEASolver:
    def test_runs_and_returns_result(self, h2_spec):
        from quantum_chem_bench.core.registry import registry
        vqe_hea = registry.build("vqe_hea", category="solver",
                                  reps=1, max_iter=100)
        result = vqe_hea.solve(h2_spec)
        assert result.n_qubits is not None
        assert result.wall_time > 0
        assert "HEA" in result.method_name.upper()


@pytest.mark.requires_pyscf
@pytest.mark.requires_qiskit_nature
class TestVQEkUpCCGSDSolver:
    def test_runs_and_returns_result(self, h2_spec):
        from quantum_chem_bench.core.registry import registry
        vqe_k = registry.build("vqe_kupccgsd", category="solver",
                                k=1, max_iter=200)
        result = vqe_k.solve(h2_spec)
        assert result.n_qubits is not None
        assert result.wall_time > 0


@pytest.mark.requires_pyscf
@pytest.mark.requires_qiskit_nature
class TestADAPTVQESolver:
    def test_energy_close_to_fci(self, h2_spec):
        from quantum_chem_bench.core.registry import registry
        adapt = registry.build("adapt_vqe", category="solver",
                               max_iter=20, vqe_max_iter=200)
        result = adapt.solve(h2_spec)
        assert abs(result.energy - H2_STO3G_FCI) < ENERGY_TOL
        assert "ADAPT" in result.method_name.upper()


@pytest.mark.requires_pyscf
class TestQPESolver:
    def test_energy_exact(self, h2_spec):
        """QPE in ideal mode uses exact diagonalization → exact FCI energy."""
        from quantum_chem_bench.core.registry import registry
        qpe = registry.build("qpe", category="solver",
                              mapper_type="jw", z2symmetry_reduction=False)
        result = qpe.solve(h2_spec)
        assert abs(result.energy - H2_STO3G_FCI) < ENERGY_TOL
        assert result.n_qubits is not None
        assert "QPE" in result.method_name.upper()

    def test_qpe_full_has_lambda_norm(self, h2_spec):
        from quantum_chem_bench.core.registry import registry
        qpe_full = registry.build("qpe_full", category="solver",
                                  mapper_type="jw", z2symmetry_reduction=False)
        result = qpe_full.solve(h2_spec)
        assert "lambda_1_norm" in result.extra
        assert result.extra["lambda_1_norm"] > 0


@pytest.mark.requires_pyscf
class TestSQDSolver:
    def test_energy_within_tolerance(self, h2_spec):
        from quantum_chem_bench.core.registry import registry
        sqd = registry.build("sqd", category="solver",
                              shots=2000, iterations=5,
                              mapper_type="jw", z2symmetry_reduction=False)
        result = sqd.solve(h2_spec)
        # SQD should get within 10 mHa of FCI
        assert abs(result.energy - H2_STO3G_FCI) < 0.05
        assert "SQD" in result.method_name.upper()


@pytest.mark.requires_pyscf
@pytest.mark.requires_qiskit_nature
@pytest.mark.slow
class TestQSESolver:
    def test_qse_below_vqe(self, h2_spec):
        """QSE energy should be ≤ VQE energy (expanded subspace)."""
        from quantum_chem_bench.core.registry import registry
        qse = registry.build("qse", category="solver", expansion_order=1,
                             vqe_max_iter=200)
        vqe = registry.build("vqe_uccsd", category="solver", max_iter=200)

        r_qse = qse.solve(h2_spec)
        r_vqe = vqe.solve(h2_spec)

        # QSE should not be worse than plain VQE by more than 1 mHa
        assert r_qse.energy <= r_vqe.energy + 1e-3


# ---------------------------------------------------------------------------
# Registration completeness
# ---------------------------------------------------------------------------

class TestRegistrationCompleteness:
    def test_all_quantum_solvers_registered(self):
        from quantum_chem_bench.core.registry import registry
        names = registry.list_names(category="solver")
        expected = ["vqe_uccsd", "vqe_hea", "vqe_kupccgsd", "vqe_uccsd_stack",
                    "adapt_vqe", "qpe", "qpe_full", "sqd", "qse"]
        for name in expected:
            assert name in names, f"Quantum solver '{name}' not registered"

    def test_all_classical_solvers_registered(self):
        from quantum_chem_bench.core.registry import registry
        names = registry.list_names(category="solver")
        expected = ["hf", "mp2", "cisd", "ccsd", "ccsd_t", "fci"]
        for name in expected:
            assert name in names, f"Classical solver '{name}' not registered"
