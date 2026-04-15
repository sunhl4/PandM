"""
Quantum Chemistry Computation Framework
======================================

A from-scratch implementation of quantum chemistry algorithms for quantum computers.

This package provides:
- Second quantization formalism
- Fermion-to-qubit mappings (Jordan-Wigner, Bravyi-Kitaev)
- Molecular Hamiltonian construction
- VQE solver with various ansätze
- Utilities for visualization and analysis

Modules:
--------
- core: Fundamental operators and mappings
- ansatz: Various ansatz implementations (UCCSD, hardware-efficient, etc.)
- vqe: Variational Quantum Eigensolver
- utils: Helper functions and visualization
- tutorials: Step-by-step learning materials
"""

__version__ = "0.1.0"
__author__ = "Quantum Chemistry Learning Framework"

from . import core
from . import ansatz
from . import vqe
from . import utils
