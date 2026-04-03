"""
Shared pytest fixtures for quantum_chem_bench tests.
"""

import pytest
import sys
from pathlib import Path

# Ensure package is importable
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Common markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_pyscf: skip if PySCF is not installed"
    )
    config.addinivalue_line(
        "markers", "requires_qiskit_nature: skip if qiskit-nature is not installed"
    )
    config.addinivalue_line(
        "markers", "requires_sqd: skip if qiskit-addon-sqd is not installed"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (skipped in CI_FAST mode)"
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "requires_pyscf" in item.keywords:
            try:
                import pyscf  # noqa: F401
            except ImportError:
                item.add_marker(pytest.mark.skip(reason="PySCF not installed"))

        if "requires_qiskit_nature" in item.keywords:
            try:
                import qiskit_nature  # noqa: F401
            except ImportError:
                item.add_marker(pytest.mark.skip(reason="qiskit-nature not installed"))

        if "requires_sqd" in item.keywords:
            try:
                import qiskit_addon_sqd  # noqa: F401
            except ImportError:
                item.add_marker(pytest.mark.skip(reason="qiskit-addon-sqd not installed"))

        import os
        if "slow" in item.keywords and os.environ.get("CI_FAST_NB"):
            item.add_marker(pytest.mark.skip(reason="CI_FAST_NB=1: skipping slow test"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def h2_mol_spec():
    """Minimal H₂ STO-3G MolSpec for fast tests."""
    import quantum_chem_bench.classical_solvers  # noqa: F401 (register solvers)
    import quantum_chem_bench.quantum_solvers    # noqa: F401
    from quantum_chem_bench.core.interfaces import MolSpec
    return MolSpec(
        geometry="H 0 0 0; H 0 0 0.735",
        basis="sto-3g",
        n_active_electrons=(1, 1),
        n_active_orbitals=2,
    )


@pytest.fixture(scope="session")
def h2_integrals(h2_mol_spec):
    """Pre-built H₂ MolIntegrals."""
    pytest.importorskip("pyscf")
    from quantum_chem_bench.molecule.builder import MoleculeBuilder
    return MoleculeBuilder(verbose=0).build(h2_mol_spec)


@pytest.fixture
def bench_config():
    """Minimal BenchConfig for fast tests (HF + FCI only)."""
    import quantum_chem_bench.classical_solvers  # noqa
    from quantum_chem_bench.core.config import BenchConfig
    return BenchConfig.from_dict({
        "molecule": {"geometry": "H 0 0 0; H 0 0 0.735", "basis": "sto-3g"},
        "solvers": {"classical": ["hf", "fci"], "quantum": []},
        "mapper": {"type": "parity", "z2symmetry_reduction": True},
    })
