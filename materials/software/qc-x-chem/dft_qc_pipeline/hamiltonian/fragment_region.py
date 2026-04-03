"""
FragmentRegion dataclass – describes which atoms/orbitals form the quantum region.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FragmentRegion:
    """
    Specification of a local quantum region within a larger system.

    Attributes
    ----------
    name : str
        Human-readable label (used as dict key in results).
    atom_indices : list[int]
        Indices of atoms (0-based in PySCF Mole) belonging to this fragment.
        Mutually exclusive with ``orbital_indices`` (prefer atoms; orbitals are
        derived by the embedding method).
    orbital_indices : list[int]
        Direct MO indices if you want to bypass automatic orbital selection.
        Used when ``atom_indices`` is empty.
    nelec : int
        Number of electrons in the fragment active space.
    norb : int
        Number of spatial orbitals in the fragment active space.
    localization : str
        Orbital localization scheme to apply before fragment extraction.
        One of ``"boys"``, ``"pm"`` (Pipek-Mezey), ``"iao"``, ``"none"``.
    """

    name: str = "fragment"
    atom_indices: list[int] = field(default_factory=list)
    orbital_indices: list[int] = field(default_factory=list)
    nelec: int = 2
    norb: int = 2
    localization: str = "iao"

    def validate(self) -> None:
        """Raise if the specification is internally inconsistent."""
        if not self.atom_indices and not self.orbital_indices:
            raise ValueError(
                f"FragmentRegion '{self.name}': "
                "specify either atom_indices or orbital_indices."
            )
        if self.nelec < 1:
            raise ValueError(f"FragmentRegion '{self.name}': nelec must be >= 1.")
        if self.norb < 1:
            raise ValueError(f"FragmentRegion '{self.name}': norb must be >= 1.")
        if self.nelec > 2 * self.norb:
            raise ValueError(
                f"FragmentRegion '{self.name}': "
                f"nelec ({self.nelec}) > 2 * norb ({self.norb})."
            )
        valid_loc = {"boys", "pm", "iao", "none"}
        if self.localization not in valid_loc:
            raise ValueError(
                f"FragmentRegion '{self.name}': "
                f"localization '{self.localization}' not in {valid_loc}."
            )

    @property
    def n_alpha(self) -> int:
        """Alpha electrons (ceil for odd total)."""
        return (self.nelec + 1) // 2

    @property
    def n_beta(self) -> int:
        """Beta electrons (floor for odd total)."""
        return self.nelec // 2
