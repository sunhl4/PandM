"""Shared RNG seeding for stochastic / pseudo-random solver paths."""

from __future__ import annotations

import numpy as np


def apply_solver_seed(seed: int | None) -> None:
    """Best-effort global seed for NumPy and Qiskit algorithm stacks."""
    if seed is None:
        return
    np.random.seed(seed)
    try:
        from qiskit_algorithms.utils import algorithm_globals

        algorithm_globals.random_seed = seed
    except Exception:
        try:
            from qiskit.utils import algorithm_globals

            algorithm_globals.random_seed = seed
        except Exception:
            pass
