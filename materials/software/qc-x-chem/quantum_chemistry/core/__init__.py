"""
Core module for quantum chemistry computations.

This module contains:
- Fermion operators and their algebra
- Qubit mappings (Jordan-Wigner, Bravyi-Kitaev, Parity)
- Molecular integral handling
"""

from .fermion_operators import (
    FermionOperator,
    creation_operator,
    annihilation_operator,
    number_operator,
    hopping_operator,
)

from .qubit_mapping import (
    jordan_wigner,
    bravyi_kitaev,
    parity_mapping,
    QubitOperator,
)

from .molecular_integrals import (
    MolecularData,
    compute_integrals_pyscf,
    build_fermionic_hamiltonian,
)
