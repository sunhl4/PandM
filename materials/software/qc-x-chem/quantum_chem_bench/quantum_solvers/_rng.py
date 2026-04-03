"""RNG seeding for reproducible VQE / stochastic quantum paths (mirrors dft_qc_pipeline)."""

from __future__ import annotations


def apply_solver_seed(seed: int | None) -> None:
    if seed is None:
        return
    import numpy as np

    np.random.seed(seed)
    try:
        from qiskit_algorithms.utils import algorithm_globals

        algorithm_globals.random_seed = seed
    except Exception:  # noqa: BLE001
        pass
