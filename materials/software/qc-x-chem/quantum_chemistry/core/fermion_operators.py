"""
Fermion Operators for Second Quantization
==========================================

This module implements the fundamental fermion operators used in quantum chemistry:
- Creation operator: a†_p creates an electron in orbital p
- Annihilation operator: a_p destroys an electron in orbital p

Key properties (anticommutation relations):
    {a_p, a†_q} = δ_pq
    {a_p, a_q} = 0
    {a†_p, a†_q} = 0

The Hamiltonian in second quantization is:
    H = Σ_pq h_pq a†_p a_q + (1/2) Σ_pqrs g_pqrs a†_p a†_q a_r a_s

where h_pq are one-electron integrals and g_pqrs are two-electron integrals.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from copy import deepcopy


@dataclass(frozen=True)
class FermionTerm:
    """
    A single term in a fermionic operator.
    
    Represents: coefficient * a†_i a†_j ... a_k a_l ...
    
    The operators list contains tuples (index, is_creation):
        - (0, True) means a†_0
        - (1, False) means a_1
    
    Example:
        FermionTerm(1.0, [(0, True), (1, False)]) = 1.0 * a†_0 a_1
    """
    coefficient: complex
    operators: Tuple[Tuple[int, bool], ...]  # ((orbital_idx, is_creation), ...)
    
    def __str__(self) -> str:
        if not self.operators:
            return f"{self.coefficient}"
        
        ops_str = []
        for idx, is_creation in self.operators:
            if is_creation:
                ops_str.append(f"a†_{idx}")
            else:
                ops_str.append(f"a_{idx}")
        
        coef_str = f"{self.coefficient:.4f}" if abs(self.coefficient.imag) < 1e-10 else f"({self.coefficient:.4f})"
        return f"{coef_str} * {' '.join(ops_str)}"
    
    def dagger(self) -> 'FermionTerm':
        """Return the Hermitian conjugate of this term."""
        new_ops = tuple((idx, not is_creation) for idx, is_creation in reversed(self.operators))
        return FermionTerm(np.conj(self.coefficient), new_ops)
    
    def is_hermitian(self) -> bool:
        """Check if this term is Hermitian."""
        return self == self.dagger()


class FermionOperator:
    """
    A general fermionic operator as a sum of FermionTerms.
    
    This class represents operators in second quantization:
        O = Σ_i c_i * (product of creation/annihilation operators)
    
    Example usage:
    >>> # Create a†_0 a_1 (electron hopping from orbital 1 to 0)
    >>> op = FermionOperator()
    >>> op.add_term(1.0, [(0, True), (1, False)])
    >>> 
    >>> # Or use helper functions:
    >>> hop = hopping_operator(0, 1, coefficient=1.0)
    """
    
    def __init__(self):
        """Initialize an empty fermion operator."""
        # Dictionary: operators_tuple -> coefficient
        self._terms: Dict[Tuple[Tuple[int, bool], ...], complex] = defaultdict(complex)
    
    def add_term(self, coefficient: complex, operators: List[Tuple[int, bool]]) -> 'FermionOperator':
        """
        Add a term to the operator.
        
        Args:
            coefficient: The coefficient of this term
            operators: List of (orbital_index, is_creation) tuples
        
        Returns:
            self for method chaining
        """
        ops_tuple = tuple(operators)
        self._terms[ops_tuple] += coefficient
        
        # Remove zero terms
        if abs(self._terms[ops_tuple]) < 1e-15:
            del self._terms[ops_tuple]
        
        return self
    
    @property
    def terms(self) -> List[FermionTerm]:
        """Return all terms as a list of FermionTerm objects."""
        return [FermionTerm(coef, ops) for ops, coef in self._terms.items()]
    
    @property
    def n_terms(self) -> int:
        """Number of terms in the operator."""
        return len(self._terms)
    
    def __str__(self) -> str:
        if not self._terms:
            return "0"
        return " + ".join(str(term) for term in self.terms)
    
    def __repr__(self) -> str:
        return f"FermionOperator({self.n_terms} terms)"
    
    def __add__(self, other: 'FermionOperator') -> 'FermionOperator':
        """Add two fermion operators."""
        result = FermionOperator()
        result._terms = deepcopy(self._terms)
        for ops, coef in other._terms.items():
            result._terms[ops] += coef
            if abs(result._terms[ops]) < 1e-15:
                del result._terms[ops]
        return result
    
    def __sub__(self, other: 'FermionOperator') -> 'FermionOperator':
        """Subtract two fermion operators."""
        result = FermionOperator()
        result._terms = deepcopy(self._terms)
        for ops, coef in other._terms.items():
            result._terms[ops] -= coef
            if abs(result._terms[ops]) < 1e-15:
                del result._terms[ops]
        return result
    
    def __mul__(self, other: Union['FermionOperator', complex, float]) -> 'FermionOperator':
        """
        Multiply operator by a scalar or another operator.
        
        For operator multiplication, we need to handle anticommutation.
        """
        if isinstance(other, (int, float, complex)):
            result = FermionOperator()
            for ops, coef in self._terms.items():
                result._terms[ops] = coef * other
            return result
        elif isinstance(other, FermionOperator):
            # Operator multiplication: concatenate operator strings
            # (proper normal ordering would require more complex logic)
            result = FermionOperator()
            for ops1, coef1 in self._terms.items():
                for ops2, coef2 in other._terms.items():
                    new_ops = ops1 + ops2
                    result._terms[new_ops] += coef1 * coef2
            # Clean up zeros
            result._terms = {k: v for k, v in result._terms.items() if abs(v) > 1e-15}
            return result
        else:
            raise TypeError(f"Cannot multiply FermionOperator with {type(other)}")
    
    def __rmul__(self, other: Union[complex, float]) -> 'FermionOperator':
        """Right multiplication by scalar."""
        return self.__mul__(other)
    
    def dagger(self) -> 'FermionOperator':
        """Return the Hermitian conjugate of this operator."""
        result = FermionOperator()
        for term in self.terms:
            dagger_term = term.dagger()
            result._terms[dagger_term.operators] += dagger_term.coefficient
        return result
    
    def is_hermitian(self, tol: float = 1e-10) -> bool:
        """Check if the operator is Hermitian: O† = O."""
        diff = self - self.dagger()
        return all(abs(coef) < tol for coef in diff._terms.values())
    
    def get_max_orbital_index(self) -> int:
        """Return the maximum orbital index in this operator."""
        max_idx = -1
        for ops in self._terms.keys():
            for idx, _ in ops:
                max_idx = max(max_idx, idx)
        return max_idx
    
    def normal_order(self) -> 'FermionOperator':
        """
        Transform operator to normal order (all creation operators to the left).
        
        Uses the anticommutation relations:
            a_p a†_q = δ_pq - a†_q a_p
            
        This is a simplified implementation that works for common cases.
        """
        result = FermionOperator()
        
        for ops, coef in self._terms.items():
            # Convert to list for manipulation
            ops_list = list(ops)
            
            # Bubble sort to bring creation operators to the left
            # Track sign changes from anticommutation
            sign = 1
            n = len(ops_list)
            
            # Simple case: if already normal ordered or single operator
            if n <= 1:
                result._terms[ops] += coef
                continue
            
            # Perform bubble sort with anticommutation
            sorted_ops = list(ops_list)
            for i in range(n):
                for j in range(n - 1, i, -1):
                    idx_j, is_creation_j = sorted_ops[j]
                    idx_k, is_creation_k = sorted_ops[j-1]
                    
                    # Want creation operators (True) to come before annihilation (False)
                    if is_creation_j and not is_creation_k:
                        # Swap and pick up a minus sign (anticommutation)
                        sorted_ops[j], sorted_ops[j-1] = sorted_ops[j-1], sorted_ops[j]
                        sign *= -1
                        
                        # If same index, also add delta term
                        if idx_j == idx_k:
                            # a_p a†_p = 1 - a†_p a_p
                            # We're swapping, so we need to add the identity contribution
                            # This simplified implementation doesn't fully handle this
                            pass
            
            result._terms[tuple(sorted_ops)] += sign * coef
        
        # Clean up zeros
        result._terms = {k: v for k, v in result._terms.items() if abs(v) > 1e-15}
        return result


def creation_operator(orbital: int, coefficient: complex = 1.0) -> FermionOperator:
    """
    Create a creation operator a†_p.
    
    Args:
        orbital: The orbital index p
        coefficient: Optional coefficient
    
    Returns:
        FermionOperator representing coefficient * a†_p
    """
    op = FermionOperator()
    op.add_term(coefficient, [(orbital, True)])
    return op


def annihilation_operator(orbital: int, coefficient: complex = 1.0) -> FermionOperator:
    """
    Create an annihilation operator a_p.
    
    Args:
        orbital: The orbital index p
        coefficient: Optional coefficient
    
    Returns:
        FermionOperator representing coefficient * a_p
    """
    op = FermionOperator()
    op.add_term(coefficient, [(orbital, False)])
    return op


def number_operator(orbital: int, coefficient: complex = 1.0) -> FermionOperator:
    """
    Create a number operator n_p = a†_p a_p.
    
    The number operator counts electrons in orbital p.
    Eigenvalues are 0 (empty) or 1 (occupied).
    
    Args:
        orbital: The orbital index p
        coefficient: Optional coefficient
    
    Returns:
        FermionOperator representing coefficient * a†_p a_p
    """
    op = FermionOperator()
    op.add_term(coefficient, [(orbital, True), (orbital, False)])
    return op


def hopping_operator(p: int, q: int, coefficient: complex = 1.0) -> FermionOperator:
    """
    Create a hopping operator a†_p a_q.
    
    This represents electron hopping from orbital q to orbital p.
    
    Args:
        p: Target orbital
        q: Source orbital
        coefficient: Optional coefficient (typically h_pq)
    
    Returns:
        FermionOperator representing coefficient * a†_p a_q
    """
    op = FermionOperator()
    op.add_term(coefficient, [(p, True), (q, False)])
    return op


def two_body_operator(p: int, q: int, r: int, s: int, coefficient: complex = 1.0) -> FermionOperator:
    """
    Create a two-body operator a†_p a†_q a_r a_s.
    
    This appears in the two-electron interaction term of the Hamiltonian.
    
    Args:
        p, q, r, s: Orbital indices
        coefficient: The two-electron integral g_pqrs
    
    Returns:
        FermionOperator representing coefficient * a†_p a†_q a_r a_s
    """
    op = FermionOperator()
    op.add_term(coefficient, [(p, True), (q, True), (r, False), (s, False)])
    return op


def excitation_operator(occupied: List[int], virtual: List[int], coefficient: complex = 1.0) -> FermionOperator:
    """
    Create an excitation operator that excites electrons from occupied to virtual orbitals.
    
    Single excitation (one occupied, one virtual):
        T_i^a = a†_a a_i
    
    Double excitation (two occupied, two virtual):
        T_ij^ab = a†_a a†_b a_j a_i
    
    Args:
        occupied: List of occupied orbital indices to excite from
        virtual: List of virtual orbital indices to excite to
        coefficient: Optional coefficient
    
    Returns:
        FermionOperator representing the excitation
    """
    if len(occupied) != len(virtual):
        raise ValueError("Must have same number of occupied and virtual orbitals")
    
    op = FermionOperator()
    
    # Build operator string: creation operators for virtual, then annihilation for occupied
    ops = []
    for v in virtual:
        ops.append((v, True))  # a†_v
    for o in reversed(occupied):
        ops.append((o, False))  # a_o
    
    op.add_term(coefficient, ops)
    return op


# ============================================================================
# Demonstration and Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Second Quantization - Fermion Operators Demo")
    print("=" * 60)
    
    print("\n1. Basic Operators:")
    print("-" * 40)
    
    # Creation and annihilation
    a0_dag = creation_operator(0)
    a0 = annihilation_operator(0)
    print(f"Creation operator a†_0: {a0_dag}")
    print(f"Annihilation operator a_0: {a0}")
    
    # Number operator
    n0 = number_operator(0)
    print(f"Number operator n_0 = a†_0 a_0: {n0}")
    
    print("\n2. Hopping (One-body) Operator:")
    print("-" * 40)
    
    # Hopping with coefficient
    h01 = 0.5  # One-electron integral h_01
    hop_01 = hopping_operator(0, 1, h01)
    print(f"Hopping a†_0 a_1 with h_01 = {h01}: {hop_01}")
    
    print("\n3. Two-body Operator:")
    print("-" * 40)
    
    # Two-electron term
    g0101 = 0.25  # Two-electron integral
    two_body = two_body_operator(0, 1, 0, 1, g0101)
    print(f"Two-body a†_0 a†_1 a_0 a_1 with g = {g0101}: {two_body}")
    
    print("\n4. Excitation Operators (for UCCSD):")
    print("-" * 40)
    
    # Single excitation: electron from orbital 0 to orbital 2
    T_0_2 = excitation_operator([0], [2])
    print(f"Single excitation T_0^2 = a†_2 a_0: {T_0_2}")
    
    # Double excitation: electrons from orbitals 0,1 to orbitals 2,3
    T_01_23 = excitation_operator([0, 1], [2, 3])
    print(f"Double excitation T_01^23 = a†_2 a†_3 a_1 a_0: {T_01_23}")
    
    print("\n5. Operator Algebra:")
    print("-" * 40)
    
    # Adding operators
    H_one_body = hopping_operator(0, 1, 0.5) + hopping_operator(1, 0, 0.5)
    print(f"Sum of hopping terms: {H_one_body}")
    
    # Scalar multiplication
    scaled = 2.0 * number_operator(0)
    print(f"2.0 * n_0: {scaled}")
    
    print("\n6. Hermitian Conjugate:")
    print("-" * 40)
    
    # a†_0 a_1 and its Hermitian conjugate a†_1 a_0
    t = hopping_operator(0, 1, 1.0)
    t_dag = t.dagger()
    print(f"Original: {t}")
    print(f"Hermitian conjugate: {t_dag}")
    
    # Hermitian combination
    H_hermitian = t + t_dag
    print(f"Hermitian sum (a†_0 a_1 + a†_1 a_0): {H_hermitian}")
    print(f"Is Hermitian? {H_hermitian.is_hermitian()}")
    
    print("\n7. Building a Simple Molecular Hamiltonian (H2 in minimal basis):")
    print("-" * 40)
    
    # Example: H2 molecule in STO-3G basis
    # This is a simplified version - real integrals come from PySCF
    # H = Σ_pq h_pq a†_p a_q + (1/2) Σ_pqrs g_pqrs a†_p a†_q a_r a_s
    
    # One-electron integrals (example values)
    h_00, h_11 = -1.25, -1.25
    h_01, h_10 = -0.475, -0.475
    
    # Build one-body part
    H = FermionOperator()
    H = H + number_operator(0, h_00)
    H = H + number_operator(1, h_11)
    H = H + hopping_operator(0, 1, h_01)
    H = H + hopping_operator(1, 0, h_10)
    
    print(f"One-body Hamiltonian has {H.n_terms} terms")
    print(f"H_one-body = {H}")
    
    print("\n" + "=" * 60)
    print("Demo complete! Next: Map to qubits using Jordan-Wigner")
    print("=" * 60)
