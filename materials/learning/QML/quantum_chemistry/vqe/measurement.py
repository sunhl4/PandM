"""
Measurement Strategies for VQE
==============================

This module provides utilities for measuring Hamiltonian expectation values
on quantum computers.

The Hamiltonian is expressed as a sum of Pauli strings:
    H = Σ_i c_i P_i

where each P_i is a tensor product of Pauli operators (I, X, Y, Z).

Key challenges:
1. Each Pauli string requires separate measurement
2. Number of measurements scales with number of terms
3. Statistical error from finite shots

Strategies:
1. Term-by-term measurement (simple but expensive)
2. Grouping commuting terms (reduces measurements)
3. Classical shadows (efficient for many observables)
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict


@dataclass
class MeasurementResult:
    """Result of Hamiltonian measurement."""
    expectation_value: float
    variance: float
    n_measurements: int
    term_values: Dict[str, float]  # Expectation of each Pauli term


def measure_hamiltonian_expectation(state_vector: np.ndarray,
                                    hamiltonian,
                                    n_qubits: int) -> float:
    """
    Compute exact expectation value from state vector.
    
    E = ⟨ψ|H|ψ⟩
    
    Args:
        state_vector: Quantum state as array
        hamiltonian: QubitOperator or matrix
        n_qubits: Number of qubits
    
    Returns:
        Expectation value
    """
    if hasattr(hamiltonian, 'to_matrix'):
        H_matrix = hamiltonian.to_matrix(n_qubits)
    elif isinstance(hamiltonian, np.ndarray):
        H_matrix = hamiltonian
    else:
        raise TypeError(f"Unknown hamiltonian type: {type(hamiltonian)}")
    
    return float(np.real(state_vector.conj() @ H_matrix @ state_vector))


def measure_pauli_string(state_vector: np.ndarray, pauli_dict: Dict[int, str],
                        n_qubits: int) -> float:
    """
    Measure expectation value of a single Pauli string.
    
    Args:
        state_vector: Quantum state
        pauli_dict: Dictionary mapping qubit index to Pauli ('X', 'Y', 'Z')
        n_qubits: Number of qubits
    
    Returns:
        Expectation value ⟨ψ|P|ψ⟩
    """
    from ..core.qubit_mapping import PauliString, pauli_string_to_matrix
    
    ps = PauliString.from_dict(pauli_dict)
    P_matrix = pauli_string_to_matrix(ps, n_qubits)
    
    return float(np.real(state_vector.conj() @ P_matrix @ state_vector))


# ============================================================================
# Pauli Grouping
# ============================================================================

def commute(pauli1: Dict[int, str], pauli2: Dict[int, str]) -> bool:
    """
    Check if two Pauli strings commute.
    
    Two Pauli strings commute if they share an even number of qubits
    where they have different non-identity Pauli operators.
    
    Args:
        pauli1, pauli2: Dictionaries mapping qubit index to Pauli type
    
    Returns:
        True if they commute
    """
    # Count anti-commuting pairs
    n_anticommute = 0
    
    all_qubits = set(pauli1.keys()) | set(pauli2.keys())
    
    for q in all_qubits:
        p1 = pauli1.get(q, 'I')
        p2 = pauli2.get(q, 'I')
        
        if p1 != 'I' and p2 != 'I' and p1 != p2:
            n_anticommute += 1
    
    return n_anticommute % 2 == 0


def qubitwise_commute(pauli1: Dict[int, str], pauli2: Dict[int, str]) -> bool:
    """
    Check if two Pauli strings qubitwise commute.
    
    QWC means on each qubit, the Paulis are either equal or one is identity.
    Stronger condition than general commutativity.
    
    QWC operators can be measured simultaneously using the same measurement basis.
    """
    all_qubits = set(pauli1.keys()) | set(pauli2.keys())
    
    for q in all_qubits:
        p1 = pauli1.get(q, 'I')
        p2 = pauli2.get(q, 'I')
        
        # Both non-identity and different -> not QWC
        if p1 != 'I' and p2 != 'I' and p1 != p2:
            return False
    
    return True


def group_commuting_terms(hamiltonian) -> List[List[Tuple[complex, Dict[int, str]]]]:
    """
    Group Hamiltonian terms into qubitwise commuting groups.
    
    Each group can be measured with a single measurement basis.
    Uses a greedy algorithm.
    
    Args:
        hamiltonian: QubitOperator
    
    Returns:
        List of groups, each group is a list of (coefficient, pauli_dict) tuples
    """
    groups = []
    
    for coef, ps in hamiltonian.terms:
        pauli_dict = ps.to_dict()
        
        # Try to add to existing group
        added = False
        for group in groups:
            # Check if QWC with all terms in group
            can_add = all(qubitwise_commute(pauli_dict, term[1]) for term in group)
            
            if can_add:
                group.append((coef, pauli_dict))
                added = True
                break
        
        if not added:
            groups.append([(coef, pauli_dict)])
    
    return groups


def get_measurement_basis(group: List[Tuple[complex, Dict[int, str]]],
                         n_qubits: int) -> Dict[int, str]:
    """
    Get the measurement basis for a QWC group.
    
    Returns dictionary mapping qubit -> measurement basis ('X', 'Y', or 'Z').
    Default is 'Z' for qubits not appearing in any term.
    """
    basis = {}
    
    for _, pauli_dict in group:
        for q, p in pauli_dict.items():
            if p != 'I':
                if q in basis and basis[q] != p:
                    raise ValueError(f"Group is not QWC: qubit {q} has {basis[q]} and {p}")
                basis[q] = p
    
    # Default to Z for unmeasured qubits
    for q in range(n_qubits):
        if q not in basis:
            basis[q] = 'Z'
    
    return basis


# ============================================================================
# Shot-based Measurement Simulation
# ============================================================================

def sample_measurements(state_vector: np.ndarray, n_shots: int,
                       measurement_basis: Optional[Dict[int, str]] = None,
                       n_qubits: Optional[int] = None) -> np.ndarray:
    """
    Simulate measurement outcomes from a quantum state.
    
    Args:
        state_vector: Quantum state
        n_shots: Number of measurement shots
        measurement_basis: Measurement basis per qubit (default: all Z)
        n_qubits: Number of qubits
    
    Returns:
        Array of shape (n_shots, n_qubits) with measurement outcomes (0 or 1)
    """
    if n_qubits is None:
        n_qubits = int(np.log2(len(state_vector)))
    
    # If not measuring in Z basis, need to rotate state first
    if measurement_basis is not None:
        state_vector = rotate_to_measurement_basis(state_vector, measurement_basis, n_qubits)
    
    # Sample from probability distribution
    probabilities = np.abs(state_vector)**2
    outcomes = np.random.choice(len(probabilities), size=n_shots, p=probabilities)
    
    # Convert to binary
    measurements = np.zeros((n_shots, n_qubits), dtype=int)
    for i, outcome in enumerate(outcomes):
        for q in range(n_qubits):
            measurements[i, q] = (outcome >> q) & 1
    
    return measurements


def rotate_to_measurement_basis(state_vector: np.ndarray, 
                                basis: Dict[int, str],
                                n_qubits: int) -> np.ndarray:
    """
    Rotate state to computational basis for measurement in given basis.
    
    To measure in X basis: apply H
    To measure in Y basis: apply HSdag
    """
    state = state_vector.copy()
    
    # Single qubit rotation matrices
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    Sdag = np.array([[1, 0], [0, -1j]])
    HSdag = H @ Sdag
    
    for q in range(n_qubits):
        b = basis.get(q, 'Z')
        
        if b == 'X':
            state = apply_single_qubit_gate(state, H, q, n_qubits)
        elif b == 'Y':
            state = apply_single_qubit_gate(state, HSdag, q, n_qubits)
        # Z basis needs no rotation
    
    return state


def apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, 
                           qubit: int, n_qubits: int) -> np.ndarray:
    """Apply a single-qubit gate to a state vector."""
    dim = 2 ** n_qubits
    new_state = np.zeros_like(state)
    
    for i in range(dim):
        bit = (i >> qubit) & 1
        i_flipped = i ^ (1 << qubit)
        
        if bit == 0:
            new_state[i] += gate[0, 0] * state[i] + gate[0, 1] * state[i_flipped]
            new_state[i_flipped] += gate[1, 0] * state[i] + gate[1, 1] * state[i_flipped]
    
    return new_state


def estimate_expectation_from_shots(measurements: np.ndarray,
                                   pauli_dict: Dict[int, str]) -> Tuple[float, float]:
    """
    Estimate Pauli string expectation value from measurement outcomes.
    
    For Z basis measurements, ⟨Z_i⟩ = (n_0 - n_1) / n_total
    For product of Z's, multiply the single-qubit outcomes.
    
    Args:
        measurements: Array of measurement outcomes (n_shots, n_qubits)
        pauli_dict: Pauli string specification
    
    Returns:
        (expectation_value, standard_error)
    """
    n_shots = len(measurements)
    
    # For each shot, compute the eigenvalue of the Pauli string
    eigenvalues = np.ones(n_shots)
    
    for q, p in pauli_dict.items():
        if p != 'I':
            # Eigenvalue is +1 if measurement is 0, -1 if measurement is 1
            eigenvalues *= (1 - 2 * measurements[:, q])
    
    mean = np.mean(eigenvalues)
    std = np.std(eigenvalues) / np.sqrt(n_shots)
    
    return mean, std


# ============================================================================
# Classical Shadows (simplified)
# ============================================================================

def random_pauli_measurement(state_vector: np.ndarray, n_qubits: int) -> Tuple[str, np.ndarray]:
    """
    Perform a random Pauli measurement (classical shadow).
    
    Each qubit is measured in a randomly chosen basis (X, Y, or Z).
    
    Returns:
        (basis_string, outcome_array)
    """
    # Random basis for each qubit
    bases = np.random.choice(['X', 'Y', 'Z'], size=n_qubits)
    basis_str = ''.join(bases)
    
    # Measure in this basis
    basis_dict = {q: bases[q] for q in range(n_qubits)}
    measurements = sample_measurements(state_vector, n_shots=1, 
                                       measurement_basis=basis_dict, n_qubits=n_qubits)
    
    return basis_str, measurements[0]


def estimate_from_shadows(shadows: List[Tuple[str, np.ndarray]], 
                         pauli_dict: Dict[int, str]) -> float:
    """
    Estimate Pauli expectation from classical shadows.
    
    This is a simplified implementation. The full classical shadow
    protocol provides efficient estimation of many observables.
    """
    estimates = []
    
    for basis_str, outcomes in shadows:
        # Check if this shadow is compatible with the observable
        compatible = True
        estimate = 1.0
        
        for q, p in pauli_dict.items():
            if p == 'I':
                continue
            
            if basis_str[q] != p:
                compatible = False
                break
            
            # Contribution from this qubit
            outcome = outcomes[q]
            estimate *= 3 * (1 - 2 * outcome)  # Factor of 3 from shadow tomography
        
        if compatible:
            estimates.append(estimate)
    
    if not estimates:
        return 0.0  # No compatible shadows
    
    return np.mean(estimates)


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Measurement Strategies Demo")
    print("=" * 60)
    
    # Create a simple 2-qubit state
    n_qubits = 2
    
    # Bell state: |00⟩ + |11⟩ / sqrt(2)
    bell_state = np.zeros(4, dtype=complex)
    bell_state[0] = 1 / np.sqrt(2)  # |00⟩
    bell_state[3] = 1 / np.sqrt(2)  # |11⟩
    
    print("\n1. Bell State Measurements:")
    print("-" * 50)
    
    # Measure Z⊗Z
    zz = {'0': 'Z', '1': 'Z'}
    exp_zz = measure_pauli_string(bell_state, {0: 'Z', 1: 'Z'}, n_qubits)
    print(f"⟨Z⊗Z⟩ = {exp_zz:.4f} (expected: 1.0)")
    
    # Measure X⊗X
    exp_xx = measure_pauli_string(bell_state, {0: 'X', 1: 'X'}, n_qubits)
    print(f"⟨X⊗X⟩ = {exp_xx:.4f} (expected: 1.0)")
    
    # Measure Z⊗I
    exp_zi = measure_pauli_string(bell_state, {0: 'Z'}, n_qubits)
    print(f"⟨Z⊗I⟩ = {exp_zi:.4f} (expected: 0.0)")
    
    print("\n2. Shot-based Measurement:")
    print("-" * 50)
    
    n_shots = 1000
    
    # Sample Z basis
    measurements = sample_measurements(bell_state, n_shots, n_qubits=n_qubits)
    
    # Count outcomes
    unique, counts = np.unique(measurements @ [1, 2], return_counts=True)
    print(f"Measurement outcomes ({n_shots} shots):")
    for u, c in zip(unique, counts):
        bits = bin(u)[2:].zfill(2)[::-1]
        print(f"  |{bits}⟩: {c} ({100*c/n_shots:.1f}%)")
    
    # Estimate Z⊗Z from shots
    mean, std = estimate_expectation_from_shots(measurements, {0: 'Z', 1: 'Z'})
    print(f"\n⟨Z⊗Z⟩ from {n_shots} shots: {mean:.4f} ± {std:.4f}")
    
    print("\n3. Pauli Grouping:")
    print("-" * 50)
    
    # Create example Hamiltonian terms
    # H = 0.5*ZZ + 0.3*XI + 0.2*IX + 0.1*XX
    class MockHamiltonian:
        @property
        def terms(self):
            return [
                (0.5, MockPauliString({0: 'Z', 1: 'Z'})),
                (0.3, MockPauliString({0: 'X'})),
                (0.2, MockPauliString({1: 'X'})),
                (0.1, MockPauliString({0: 'X', 1: 'X'})),
            ]
    
    class MockPauliString:
        def __init__(self, d):
            self._d = d
        def to_dict(self):
            return self._d
    
    H = MockHamiltonian()
    groups = group_commuting_terms(H)
    
    print(f"Number of groups: {len(groups)}")
    for i, group in enumerate(groups):
        terms = [str(dict(t[1])) for t in group]
        print(f"  Group {i+1}: {terms}")
    
    print("\n4. Commutation Check:")
    print("-" * 50)
    
    p1 = {0: 'Z', 1: 'Z'}
    p2 = {0: 'X', 1: 'X'}
    p3 = {0: 'Z', 1: 'X'}
    
    print(f"ZZ and XX commute: {commute(p1, p2)}")
    print(f"ZZ and ZX commute: {commute(p1, p3)}")
    print(f"ZZ and XX QWC: {qubitwise_commute(p1, p2)}")
    
    print("\n" + "=" * 60)
    print("Measurement Demo Complete!")
    print("=" * 60)
