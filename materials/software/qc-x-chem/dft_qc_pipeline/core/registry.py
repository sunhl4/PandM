"""
Plugin registry for the DFT + quantum-embedding pipeline.

Usage
-----
Register a concrete implementation::

    from dft_qc_pipeline.core.registry import registry

    @registry.register("my_solver", category="solver")
    class MySolver(QuantumSolver):
        ...

Retrieve it::

    SolverClass = registry.get("my_solver", category="solver")
    solver = SolverClass(**kwargs)

Or use the ``build`` helper which reads a config dict::

    solver = registry.build({"type": "my_solver", "shots": 1000}, category="solver")
"""

from __future__ import annotations

from typing import Any, Callable, Type


class Registry:
    """Central registry that maps (category, name) -> class."""

    def __init__(self) -> None:
        # _store[category][name] = class
        self._store: dict[str, dict[str, type]] = {}

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def register(self, name: str, category: str) -> Callable[[type], type]:
        """Decorator: ``@registry.register("vqe", category="solver")``."""
        def decorator(cls: type) -> type:
            self._store.setdefault(category, {})[name] = cls
            cls._registry_name = name          # store back-reference
            cls._registry_category = category
            return cls
        return decorator

    def register_class(self, name: str, category: str, cls: type) -> None:
        """Imperative alternative to the decorator."""
        self._store.setdefault(category, {})[name] = cls

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get(self, name: str, category: str) -> type:
        """Return the class registered under *name* in *category*."""
        cat = self._store.get(category, {})
        if name not in cat:
            available = list(cat.keys())
            raise KeyError(
                f"No '{name}' registered in category '{category}'. "
                f"Available: {available}"
            )
        return cat[name]

    def list(self, category: str) -> list[str]:
        """List all registered names in *category*."""
        return list(self._store.get(category, {}).keys())

    def build(self, cfg: dict[str, Any], category: str) -> Any:
        """
        Instantiate a class from a config dict.

        The dict must contain ``"type"`` key. All other keys are forwarded
        as keyword arguments to the class constructor.

        Example::

            solver = registry.build({"type": "vqe", "ansatz": "uccsd"}, "solver")
        """
        cfg = dict(cfg)  # don't mutate the caller's dict
        name = cfg.pop("type")
        cls = self.get(name, category)
        return cls(**cfg)


# Module-level singleton – imported everywhere in the pipeline.
registry = Registry()
