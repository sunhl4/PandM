"""
Solver registry for quantum_chem_bench.

Pattern mirrors dft_qc_pipeline/core/registry.py but is self-contained.

Usage::

    from quantum_chem_bench.core.registry import registry

    @registry.register("my_solver", category="solver")
    class MySolver(BaseSolver):
        ...

    # Later: instantiate from name
    solver = registry.build("my_solver", category="solver", max_iter=300)
"""

from __future__ import annotations

from typing import Any


class Registry:
    """
    Two-level registry: category → name → class.

    Categories are created on first use.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, type]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, *, category: str = "solver"):
        """
        Decorator that registers a class under ``(category, name)``.

        Parameters
        ----------
        name : str
            Key used to look up the class (e.g. ``"ccsd"``).
        category : str
            Logical group: ``"solver"``, ``"classical"``, ``"quantum"``, etc.
        """
        def decorator(cls: type) -> type:
            bucket = self._store.setdefault(category, {})
            if name in bucket:
                raise KeyError(
                    f"Registry conflict: '{name}' is already registered under "
                    f"category '{category}'."
                )
            bucket[name] = cls
            return cls
        return decorator

    # ------------------------------------------------------------------
    # Lookup / instantiation
    # ------------------------------------------------------------------

    def get(self, name: str, *, category: str = "solver") -> type:
        """Return the class registered under ``(category, name)``."""
        try:
            return self._store[category][name]
        except KeyError:
            available = list(self._store.get(category, {}).keys())
            raise KeyError(
                f"No '{name}' registered under category '{category}'. "
                f"Available: {available}"
            ) from None

    def build(self, name: str, *, category: str = "solver", **kwargs: Any) -> Any:
        """Instantiate the registered class with ``**kwargs``."""
        cls = self.get(name, category=category)
        return cls(**kwargs)

    def list_names(self, *, category: str = "solver") -> list[str]:
        """Return all registered names for a given category."""
        return list(self._store.get(category, {}).keys())

    def list_categories(self) -> list[str]:
        """Return all known categories."""
        return list(self._store.keys())

    def __repr__(self) -> str:
        parts = [
            f"  {cat}: {list(names.keys())}"
            for cat, names in self._store.items()
        ]
        return "Registry(\n" + "\n".join(parts) + "\n)"


# Module-level singleton
registry = Registry()
