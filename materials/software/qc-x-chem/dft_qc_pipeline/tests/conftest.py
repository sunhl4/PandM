"""Ensure built-in plugins are registered before tests run."""

from __future__ import annotations

import pytest

import dft_qc_pipeline  # noqa: F401  # registers backends, embeddings, solvers


def pytest_collection_modifyitems(config, items):
    try:
        import pyscf  # noqa: F401
        has_pyscf = True
    except ImportError:
        has_pyscf = False
    try:
        import qiskit_nature  # noqa: F401
        has_qn = True
    except ImportError:
        has_qn = False
    for item in items:
        if "requires_pyscf" in item.keywords and not has_pyscf:
            item.add_marker(pytest.mark.skip(reason="PySCF not installed"))
        if "requires_qiskit_nature" in item.keywords and not has_qn:
            item.add_marker(pytest.mark.skip(reason="qiskit-nature not installed"))
