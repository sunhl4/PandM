"""Plugin registry smoke tests."""

from __future__ import annotations

from dft_qc_pipeline.core.registry import registry


def test_build_numpy_solver() -> None:
    sol = registry.build({"type": "numpy"}, category="solver")
    assert sol.__class__.__name__ == "NumPySolver"
