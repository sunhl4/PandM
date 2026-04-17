"""
Fermion-to-Qubit Mappings
=========================

This module implements mappings from fermionic operators to qubit operators.

The key challenge: Fermions have anticommutation relations {a_p, a†_q} = δ_pq,
while qubits/Pauli operators have commutation relations. We need mappings that
preserve the algebraic structure.

Main mappings:
1. Jordan-Wigner (JW): Most intuitive, but produces non-local operators
2. Bravyi-Kitaev (BK): Partially local, better for some hardware
3. Parity: Good for exploiting particle number conservation

Jordan-Wigner transformation:
    a†_p → (1/2)(X_p - iY_p) ⊗ Z_{p-1} ⊗ Z_{p-2} ⊗ ... ⊗ Z_0
    a_p  → (1/2)(X_p + iY_p) ⊗ Z_{p-1} ⊗ Z_{p-2} ⊗ ... ⊗ Z_0

The Z-string ensures antisymmetry between fermions.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from copy import deepcopy


# Pauli matrices for reference
PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


@dataclass(frozen=True)
class PauliString:
    """
    A Pauli string: tensor product of Pauli operators on different qubits.
    
    Represented as a dictionary: qubit_index -> pauli_type
    where pauli_type is 'I', 'X', 'Y', or 'Z'.
    
    Example: X_0 ⊗ Y_1 ⊗ Z_2 = {0: 'X', 1: 'Y', 2: 'Z'}
    
    Identity operators are implicit (not stored).
    """
    # Using tuple of tuples for hashability
    paulis: Tuple[Tuple[int, str], ...]  # ((qubit_idx, pauli_type), ...)
    
    @classmethod
    def from_dict(cls, pauli_dict: Dict[int, str]) -> 'PauliString':
        """Create from dictionary."""
        paulis = tuple(sorted((k, v) for k, v in pauli_dict.items() if v != 'I'))
        return cls(paulis)
    
    @classmethod
    def identity(cls) -> 'PauliString':
        """Return identity operator."""
        return cls(())
    
    def to_dict(self) -> Dict[int, str]:
        """Convert to dictionary form."""
        return dict(self.paulis)
    
    def __str__(self) -> str:
        if not self.paulis:
            return "I"
        return " ⊗ ".join(f"{p}_{q}" for q, p in sorted(self.paulis))
    
    @property
    def qubits(self) -> Set[int]:
        """Return set of qubit indices with non-identity operators."""
        return {q for q, _ in self.paulis}
    
    @property
    def n_qubits(self) -> int:
        """Maximum qubit index + 1."""
        if not self.paulis:
            return 0
        return max(q for q, _ in self.paulis) + 1


class QubitOperator:
    """
    A linear combination of Pauli strings.
    
    Represents: O = Σ_i c_i P_i where P_i are Pauli strings.
    
    This is the qubit representation of operators after fermion-to-qubit mapping.
    """
    
    def __init__(self):
        """Initialize empty qubit operator."""
        # Map from PauliString to coefficient
        self._terms: Dict[PauliString, complex] = defaultdict(complex)
    
    def add_term(self, coefficient: complex, pauli_string: PauliString) -> 'QubitOperator':
        """Add a term to the operator."""
        self._terms[pauli_string] += coefficient
        if abs(self._terms[pauli_string]) < 1e-15:
            del self._terms[pauli_string]
        return self
    
    def add_pauli_string(self, coefficient: complex, pauli_dict: Dict[int, str]) -> 'QubitOperator':
        """Add a term using a dictionary of Paulis."""
        ps = PauliString.from_dict(pauli_dict)
        return self.add_term(coefficient, ps)
    
    @property
    def n_terms(self) -> int:
        """Number of Pauli terms."""
        return len(self._terms)
    
    @property
    def terms(self) -> List[Tuple[complex, PauliString]]:
        """Return list of (coefficient, PauliString) tuples."""
        return [(coef, ps) for ps, coef in self._terms.items()]
    
    def __str__(self) -> str:
        if not self._terms:
            return "0"
        
        parts = []
        for ps, coef in self._terms.items():
            if abs(coef.imag) < 1e-10:
                coef_str = f"{coef.real:.6f}"
            else:
                coef_str = f"({coef:.6f})"
            parts.append(f"{coef_str} * {ps}")
        return "\n + ".join(parts)
    
    def __repr__(self) -> str:
        return f"QubitOperator({self.n_terms} terms)"
    
    def __add__(self, other: 'QubitOperator') -> 'QubitOperator':
        result = QubitOperator()
        result._terms = deepcopy(self._terms)
        for ps, coef in other._terms.items():
            result._terms[ps] += coef
            if abs(result._terms[ps]) < 1e-15:
                del result._terms[ps]
        return result
    
    def __sub__(self, other: 'QubitOperator') -> 'QubitOperator':
        result = QubitOperator()
        result._terms = deepcopy(self._terms)
        for ps, coef in other._terms.items():
            result._terms[ps] -= coef
            if abs(result._terms[ps]) < 1e-15:
                del result._terms[ps]
        return result
    
    def __mul__(self, other: Union['QubitOperator', complex, float]) -> 'QubitOperator':
        if isinstance(other, (int, float, complex)):
            result = QubitOperator()
            for ps, coef in self._terms.items():
                result._terms[ps] = coef * other
            return result
        elif isinstance(other, QubitOperator):
            result = QubitOperator()
            for ps1, coef1 in self._terms.items():
                for ps2, coef2 in other._terms.items():
                    new_ps, phase = multiply_pauli_strings(ps1, ps2)
                    result._terms[new_ps] += coef1 * coef2 * phase
            result._terms = {k: v for k, v in result._terms.items() if abs(v) > 1e-15}
            return result
        else:
            raise TypeError(f"Cannot multiply QubitOperator with {type(other)}")
    
    def __rmul__(self, other: Union[complex, float]) -> 'QubitOperator':
        return self.__mul__(other)
    
    def dagger(self) -> 'QubitOperator':
        """Return Hermitian conjugate (Pauli matrices are Hermitian)."""
        result = QubitOperator()
        for ps, coef in self._terms.items():
            result._terms[ps] = np.conj(coef)
        return result
    
    def get_n_qubits(self) -> int:
        """Return the number of qubits needed."""
        max_q = -1
        for ps in self._terms.keys():
            if ps.paulis:
                max_q = max(max_q, max(q for q, _ in ps.paulis))
        return max_q + 1 if max_q >= 0 else 0
    
    def to_matrix(self, n_qubits: Optional[int] = None) -> np.ndarray:
        """
        Convert to dense matrix representation.
        
        Useful for exact diagonalization and verification.
        """
        if n_qubits is None:
            n_qubits = self.get_n_qubits()
        
        dim = 2 ** n_qubits
        matrix = np.zeros((dim, dim), dtype=complex)
        
        for ps, coef in self._terms.items():
            ps_matrix = pauli_string_to_matrix(ps, n_qubits)
            matrix += coef * ps_matrix
        
        return matrix


def multiply_pauli_strings(ps1: PauliString, ps2: PauliString) -> Tuple[PauliString, complex]:
    """
    Multiply two Pauli strings and return (result, phase).
    
    Pauli multiplication rules:
        XX = I, YY = I, ZZ = I
        XY = iZ, YZ = iX, ZX = iY
        YX = -iZ, ZY = -iX, XZ = -iY
    """
    result_dict = {}
    phase = 1.0
    
    # Get all qubits
    d1 = ps1.to_dict()
    d2 = ps2.to_dict()
    all_qubits = set(d1.keys()) | set(d2.keys())
    
    for q in all_qubits:
        p1 = d1.get(q, 'I')
        p2 = d2.get(q, 'I')
        
        result_pauli, local_phase = multiply_paulis(p1, p2)
        phase *= local_phase
        
        if result_pauli != 'I':
            result_dict[q] = result_pauli
    
    return PauliString.from_dict(result_dict), phase


def multiply_paulis(p1: str, p2: str) -> Tuple[str, complex]:
    """
    Multiply two single-qubit Pauli operators.
    
    Returns (result_pauli, phase).
    """
    if p1 == 'I':
        return p2, 1.0
    if p2 == 'I':
        return p1, 1.0
    if p1 == p2:
        return 'I', 1.0
    
    # XY = iZ, YZ = iX, ZX = iY
    # YX = -iZ, ZY = -iX, XZ = -iY
    table = {
        ('X', 'Y'): ('Z', 1j),
        ('Y', 'X'): ('Z', -1j),
        ('Y', 'Z'): ('X', 1j),
        ('Z', 'Y'): ('X', -1j),
        ('Z', 'X'): ('Y', 1j),
        ('X', 'Z'): ('Y', -1j),
    }
    
    return table[(p1, p2)]


def pauli_string_to_matrix(ps: PauliString, n_qubits: int) -> np.ndarray:
    """Convert a Pauli string to a matrix."""
    pauli_map = {'I': PAULI_I, 'X': PAULI_X, 'Y': PAULI_Y, 'Z': PAULI_Z}
    ps_dict = ps.to_dict()
    
    result = np.array([[1.0]], dtype=complex)
    
    for q in range(n_qubits):
        p = ps_dict.get(q, 'I')
        result = np.kron(result, pauli_map[p])
    
    return result


# ============================================================================
# Jordan-Wigner Transformation
# ============================================================================

def jordan_wigner_creation(orbital: int) -> QubitOperator:
    """
    Apply Jordan-Wigner transformation to creation operator a†_p.
    
    a†_p → (1/2)(X_p - iY_p) ⊗ Z_{p-1} ⊗ ... ⊗ Z_0
    
    The Z-string enforces the fermionic antisymmetry.
    """
    # (X - iY)/2 = |1><0| raises the qubit from |0> to |1>
    # We need to apply Z to all qubits with index < p (the Z-string)
    
    result = QubitOperator()
    
    # Build Z-string for qubits 0 to p-1
    z_string = {q: 'Z' for q in range(orbital)}
    
    # X_p term with coefficient 1/2
    x_term = {**z_string, orbital: 'X'}
    result.add_pauli_string(0.5, x_term)
    
    # -iY_p term with coefficient 1/2
    y_term = {**z_string, orbital: 'Y'}
    result.add_pauli_string(-0.5j, y_term)
    
    return result


def jordan_wigner_annihilation(orbital: int) -> QubitOperator:
    """
    Apply Jordan-Wigner transformation to annihilation operator a_p.
    
    a_p → (1/2)(X_p + iY_p) ⊗ Z_{p-1} ⊗ ... ⊗ Z_0
    
    This is the Hermitian conjugate of the creation operator mapping.
    """
    result = QubitOperator()
    
    z_string = {q: 'Z' for q in range(orbital)}
    
    # X_p term
    x_term = {**z_string, orbital: 'X'}
    result.add_pauli_string(0.5, x_term)
    
    # +iY_p term
    y_term = {**z_string, orbital: 'Y'}
    result.add_pauli_string(0.5j, y_term)
    
    return result


def jordan_wigner(fermion_operator) -> QubitOperator:
    """
    Transform a fermionic operator to a qubit operator using Jordan-Wigner.
    
    Args:
        fermion_operator: A FermionOperator object
    
    Returns:
        QubitOperator: The transformed operator
    """
    from .fermion_operators import FermionOperator
    
    if not isinstance(fermion_operator, FermionOperator):
        raise TypeError("Input must be a FermionOperator")
    
    result = QubitOperator()
    result.add_term(0, PauliString.identity())  # Start with zero
    
    for term in fermion_operator.terms:
        # Transform each term
        term_result = QubitOperator()
        term_result.add_term(term.coefficient, PauliString.identity())
        
        for orbital, is_creation in term.operators:
            if is_creation:
                op = jordan_wigner_creation(orbital)
            else:
                op = jordan_wigner_annihilation(orbital)
            
            term_result = term_result * op
        
        result = result + term_result
    
    return result


# ============================================================================
# Bravyi-Kitaev Transformation
# ============================================================================

def bravyi_kitaev_creation(orbital: int, n_orbitals: int) -> QubitOperator:
    """
    Apply Bravyi-Kitaev transformation to creation operator.
    
    The BK transformation uses a different encoding that improves locality
    for certain operations at the cost of more complex transformations.
    
    This is a simplified implementation for small systems.
    """
    # For a complete implementation, one needs to construct the BK transformation
    # matrix which depends on the total number of orbitals.
    # For simplicity, we fall back to Jordan-Wigner here.
    # A full implementation would use the Fenwick tree structure.
    
    # TODO: Implement full BK transformation
    # For now, return JW as placeholder
    return jordan_wigner_creation(orbital)


def bravyi_kitaev_annihilation(orbital: int, n_orbitals: int) -> QubitOperator:
    """Apply Bravyi-Kitaev transformation to annihilation operator."""
    return jordan_wigner_annihilation(orbital)


def bravyi_kitaev(fermion_operator, n_orbitals: Optional[int] = None) -> QubitOperator:
    """
    Transform using Bravyi-Kitaev mapping.
    
    For now, this is identical to Jordan-Wigner.
    A full implementation would provide better locality.
    """
    # TODO: Implement proper BK transformation
    return jordan_wigner(fermion_operator)


# ============================================================================
# Parity Mapping
# ============================================================================

def parity_mapping(fermion_operator) -> QubitOperator:
    """
    Transform using parity mapping.
    
    The parity mapping stores the parity (even/odd) of occupation numbers
    rather than the occupation numbers directly.
    
    This can simplify exploiting symmetries like particle number conservation.
    """
    # TODO: Implement parity mapping
    # For now, return JW
    return jordan_wigner(fermion_operator)


# ============================================================================
# Useful Qubit Operators
# ============================================================================

def pauli_x(qubit: int) -> QubitOperator:
    """Return Pauli X operator on given qubit."""
    op = QubitOperator()
    op.add_pauli_string(1.0, {qubit: 'X'})
    return op


def pauli_y(qubit: int) -> QubitOperator:
    """Return Pauli Y operator on given qubit."""
    op = QubitOperator()
    op.add_pauli_string(1.0, {qubit: 'Y'})
    return op


def pauli_z(qubit: int) -> QubitOperator:
    """Return Pauli Z operator on given qubit."""
    op = QubitOperator()
    op.add_pauli_string(1.0, {qubit: 'Z'})
    return op


def identity_operator() -> QubitOperator:
    """Return identity operator."""
    op = QubitOperator()
    op.add_term(1.0, PauliString.identity())
    return op


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Fermion-to-Qubit Mapping Demo (Jordan-Wigner)")
    print("=" * 60)
    
    from .fermion_operators import (
        FermionOperator, creation_operator, annihilation_operator,
        number_operator, hopping_operator
    )
    
    print("\n1. Single Creation Operator:")
    print("-" * 40)
    
    # a†_0: simplest case, no Z string
    a0_dag = creation_operator(0)
    print(f"Fermion: {a0_dag}")
    q_a0_dag = jordan_wigner(a0_dag)
    print(f"Qubit (JW): {q_a0_dag}")
    
    print("\n2. Creation on Higher Orbital (with Z-string):")
    print("-" * 40)
    
    # a†_2: needs Z_1 ⊗ Z_0
    a2_dag = creation_operator(2)
    print(f"Fermion: {a2_dag}")
    q_a2_dag = jordan_wigner(a2_dag)
    print(f"Qubit (JW): {q_a2_dag}")
    print("Note: Z-string Z_1 ⊗ Z_0 appears to track fermion parity")
    
    print("\n3. Number Operator n_p = a†_p a_p:")
    print("-" * 40)
    
    n0 = number_operator(0)
    print(f"Fermion: {n0}")
    q_n0 = jordan_wigner(n0)
    print(f"Qubit (JW): {q_n0}")
    print("Simplifies to (1/2)(I - Z_0), counting occupation of qubit 0")
    
    print("\n4. Hopping Operator a†_0 a_1:")
    print("-" * 40)
    
    hop = hopping_operator(0, 1, 1.0)
    print(f"Fermion: {hop}")
    q_hop = jordan_wigner(hop)
    print(f"Qubit (JW):")
    print(q_hop)
    
    print("\n5. Building and Diagonalizing H2 Hamiltonian:")
    print("-" * 40)
    
    # Simplified H2 Hamiltonian (2 spin-orbitals for illustration)
    # H = h_00 n_0 + h_11 n_1 + g_0110 a†_0 a†_1 a_1 a_0
    #   = h_00 n_0 + h_11 n_1 + g_0110 n_0 n_1
    
    # Example coefficients (not realistic, just for demo)
    h00, h11 = -1.0, -1.0
    g0110 = 0.5
    
    H_fermion = FermionOperator()
    H_fermion = H_fermion + number_operator(0, h00)
    H_fermion = H_fermion + number_operator(1, h11)
    
    # Two-body term: n_0 n_1 = a†_0 a_0 a†_1 a_1
    # For simplicity, we just add the density-density term
    H_fermion.add_term(g0110, [(0, True), (0, False), (1, True), (1, False)])
    
    print(f"Fermion Hamiltonian has {H_fermion.n_terms} terms")
    
    H_qubit = jordan_wigner(H_fermion)
    print(f"Qubit Hamiltonian has {H_qubit.n_terms} Pauli terms:")
    print(H_qubit)
    
    # Convert to matrix and diagonalize
    print("\n6. Exact Diagonalization:")
    print("-" * 40)
    
    H_matrix = H_qubit.to_matrix(n_qubits=2)
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    
    print(f"Matrix dimension: {H_matrix.shape}")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Ground state energy: {eigenvalues[0]:.6f}")
    
    print("\n" + "=" * 60)
    print("Mapping complete! The qubit Hamiltonian can now be used with VQE")
    print("=" * 60)
