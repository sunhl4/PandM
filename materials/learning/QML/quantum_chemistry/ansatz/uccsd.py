"""
UCCSD Ansatz (Unitary Coupled Cluster Singles and Doubles)
==========================================================

The UCCSD ansatz is the most widely used chemistry-inspired ansatz for VQE.

Theory:
-------
The coupled cluster (CC) wave function is:
    |ψ_CC⟩ = e^T |HF⟩

where T = T_1 + T_2 + ... contains excitation operators.

For UCCSD (singles and doubles):
    T = T_1 + T_2
    T_1 = Σ_{ia} t_i^a (a†_a a_i - a†_i a_a)  [single excitations]
    T_2 = Σ_{ijab} t_{ij}^{ab} (a†_a a†_b a_j a_i - h.c.)  [double excitations]

The unitary version uses:
    |ψ_UCCSD⟩ = e^{T - T†} |HF⟩

This ensures the ansatz is unitary (can be implemented on quantum computer).

Implementation:
--------------
We use Trotterization to decompose the exponential:
    e^{T - T†} ≈ ∏_k e^{θ_k (τ_k - τ_k†)}

where each τ_k is an individual excitation operator.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class UCCSDConfig:
    """Configuration for UCCSD ansatz."""
    n_qubits: int
    n_electrons: int
    n_orbitals: int  # Number of spatial orbitals
    trotter_steps: int = 1
    include_singles: bool = True
    include_doubles: bool = True
    
    @property
    def n_occupied(self) -> int:
        """Number of occupied spin orbitals."""
        return self.n_electrons
    
    @property
    def n_virtual(self) -> int:
        """Number of virtual spin orbitals."""
        return 2 * self.n_orbitals - self.n_electrons


def get_uccsd_excitations(n_electrons: int, n_orbitals: int,
                          singles: bool = True, doubles: bool = True
                          ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
    """
    Generate all single and double excitations.
    
    Args:
        n_electrons: Number of electrons
        n_orbitals: Number of spatial orbitals
        singles: Include single excitations
        doubles: Include double excitations
    
    Returns:
        Tuple of (singles_list, doubles_list)
        singles_list: List of (i, a) tuples (occupied -> virtual)
        doubles_list: List of (i, j, a, b) tuples
    """
    n_spin_orbitals = 2 * n_orbitals
    
    # Occupied orbitals: 0, 1, ..., n_electrons-1
    # Virtual orbitals: n_electrons, n_electrons+1, ..., n_spin_orbitals-1
    occupied = list(range(n_electrons))
    virtual = list(range(n_electrons, n_spin_orbitals))
    
    singles_list = []
    doubles_list = []
    
    if singles:
        for i in occupied:
            for a in virtual:
                singles_list.append((i, a))
    
    if doubles:
        for i in occupied:
            for j in occupied:
                if i < j:  # Avoid double counting
                    for a in virtual:
                        for b in virtual:
                            if a < b:  # Avoid double counting
                                doubles_list.append((i, j, a, b))
    
    return singles_list, doubles_list


def count_uccsd_parameters(n_electrons: int, n_orbitals: int,
                           singles: bool = True, doubles: bool = True) -> int:
    """Count the number of parameters in UCCSD ansatz."""
    singles_list, doubles_list = get_uccsd_excitations(
        n_electrons, n_orbitals, singles, doubles
    )
    return len(singles_list) + len(doubles_list)


class UCCSD:
    """
    UCCSD ansatz implementation.
    
    This class provides methods to:
    1. Generate the excitation operators
    2. Build the quantum circuit (using PennyLane)
    3. Prepare the initial Hartree-Fock state
    
    Example usage:
    >>> config = UCCSDConfig(n_qubits=4, n_electrons=2, n_orbitals=2)
    >>> uccsd = UCCSD(config)
    >>> params = np.zeros(uccsd.n_parameters)
    >>> # Use with PennyLane circuit
    """
    
    def __init__(self, config: UCCSDConfig):
        """Initialize UCCSD ansatz."""
        self.config = config
        
        # Get excitations
        self.singles, self.doubles = get_uccsd_excitations(
            config.n_electrons,
            config.n_orbitals,
            config.include_singles,
            config.include_doubles
        )
        
        self.n_parameters = len(self.singles) + len(self.doubles)
    
    def hf_state(self) -> List[int]:
        """
        Return the Hartree-Fock state as a list of qubit values.
        
        In occupation number basis: |1...1 0...0⟩
        where the first n_electrons qubits are |1⟩.
        """
        state = [1] * self.config.n_electrons + [0] * (self.config.n_qubits - self.config.n_electrons)
        return state
    
    def get_circuit_pennylane(self):
        """
        Return a PennyLane circuit template for UCCSD.
        
        Requires PennyLane to be installed.
        """
        try:
            import pennylane as qml
        except ImportError:
            raise ImportError("PennyLane is required. Install with: pip install pennylane")
        
        config = self.config
        singles = self.singles
        doubles = self.doubles
        
        def circuit(params, wires):
            """
            UCCSD circuit.
            
            Args:
                params: Array of parameters [theta_singles..., theta_doubles...]
                wires: Qubit wires to use
            """
            # Prepare HF state
            hf = self.hf_state()
            qml.BasisState(np.array(hf), wires=wires)
            
            # Apply excitation operators
            param_idx = 0
            
            # Single excitations
            for i, a in singles:
                qml.SingleExcitation(params[param_idx], wires=[i, a])
                param_idx += 1
            
            # Double excitations
            for i, j, a, b in doubles:
                qml.DoubleExcitation(params[param_idx], wires=[i, j, a, b])
                param_idx += 1
        
        return circuit
    
    def get_initial_params(self, method: str = 'zeros') -> np.ndarray:
        """
        Get initial parameters.
        
        Args:
            method: 'zeros', 'random', or 'mp2' (for MP2-based initialization)
        
        Returns:
            Initial parameter array
        """
        if method == 'zeros':
            return np.zeros(self.n_parameters)
        elif method == 'random':
            return np.random.uniform(-0.1, 0.1, self.n_parameters)
        elif method == 'mp2':
            # MP2-based initialization would require integrals
            # For now, return small random values
            return np.random.uniform(-0.01, 0.01, self.n_parameters)
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    
    def excitation_operators_fermion(self):
        """
        Return the fermionic excitation operators.
        
        Returns:
            List of FermionOperator objects for each excitation.
        """
        from ..core.fermion_operators import FermionOperator
        
        operators = []
        
        # Single excitations: T_i^a = a†_a a_i
        for i, a in self.singles:
            T = FermionOperator()
            T.add_term(1.0, [(a, True), (i, False)])  # a†_a a_i
            T_dag = FermionOperator()
            T_dag.add_term(1.0, [(i, True), (a, False)])  # a†_i a_a
            
            # T - T† for unitarity
            excitation = T - T_dag
            operators.append(excitation)
        
        # Double excitations: T_ij^ab = a†_a a†_b a_j a_i
        for i, j, a, b in self.doubles:
            T = FermionOperator()
            T.add_term(1.0, [(a, True), (b, True), (j, False), (i, False)])
            T_dag = FermionOperator()
            T_dag.add_term(1.0, [(i, True), (j, True), (b, False), (a, False)])
            
            excitation = T - T_dag
            operators.append(excitation)
        
        return operators


def uccsd_circuit(params: np.ndarray, n_qubits: int, n_electrons: int,
                  n_orbitals: int) -> Callable:
    """
    Convenience function to create UCCSD circuit.
    
    Args:
        params: Parameter array
        n_qubits: Number of qubits
        n_electrons: Number of electrons
        n_orbitals: Number of spatial orbitals
    
    Returns:
        PennyLane circuit function
    """
    config = UCCSDConfig(
        n_qubits=n_qubits,
        n_electrons=n_electrons,
        n_orbitals=n_orbitals
    )
    uccsd = UCCSD(config)
    return uccsd.get_circuit_pennylane()


# ============================================================================
# Manual UCCSD implementation without PennyLane built-in operations
# ============================================================================

def single_excitation_gate(theta: float, qubit_i: int, qubit_a: int):
    """
    Implement single excitation gate using basic gates.
    
    The single excitation operator is:
        G(θ) = exp(θ (a†_a a_i - a†_i a_a))
    
    This can be decomposed into CNOT and rotation gates.
    """
    try:
        import pennylane as qml
    except ImportError:
        raise ImportError("PennyLane is required")
    
    # Implementation using Givens rotation
    # For qubits i < a:
    # This implements the unitary that rotates |01⟩ ↔ |10⟩
    
    phi = theta / 2
    
    # Decomposition into basic gates
    qml.CNOT(wires=[qubit_a, qubit_i])
    qml.RY(phi, wires=qubit_a)
    qml.RY(phi, wires=qubit_i)
    qml.CNOT(wires=[qubit_i, qubit_a])
    qml.RY(-phi, wires=qubit_i)
    qml.CNOT(wires=[qubit_a, qubit_i])


def double_excitation_gate(theta: float, i: int, j: int, a: int, b: int):
    """
    Implement double excitation gate using basic gates.
    
    The double excitation operator involves 4 qubits.
    This is a simplified implementation.
    """
    try:
        import pennylane as qml
    except ImportError:
        raise ImportError("PennyLane is required")
    
    # This is a placeholder - the full decomposition is complex
    # In practice, use qml.DoubleExcitation directly
    qml.DoubleExcitation(theta, wires=[i, j, a, b])


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("UCCSD Ansatz Demo")
    print("=" * 60)
    
    print("\n1. H2 molecule (2 electrons, 2 spatial orbitals):")
    print("-" * 50)
    
    config = UCCSDConfig(
        n_qubits=4,  # 2 spatial orbitals * 2 spins
        n_electrons=2,
        n_orbitals=2
    )
    
    uccsd = UCCSD(config)
    
    print(f"Number of qubits: {config.n_qubits}")
    print(f"Number of electrons: {config.n_electrons}")
    print(f"Number of parameters: {uccsd.n_parameters}")
    print(f"HF state: {uccsd.hf_state()}")
    
    print(f"\nSingle excitations ({len(uccsd.singles)}):")
    for i, a in uccsd.singles:
        print(f"  {i} → {a}")
    
    print(f"\nDouble excitations ({len(uccsd.doubles)}):")
    for i, j, a, b in uccsd.doubles:
        print(f"  ({i}, {j}) → ({a}, {b})")
    
    print("\n2. LiH molecule (4 electrons, 6 spatial orbitals):")
    print("-" * 50)
    
    config_lih = UCCSDConfig(
        n_qubits=12,  # 6 spatial orbitals * 2 spins
        n_electrons=4,
        n_orbitals=6
    )
    
    uccsd_lih = UCCSD(config_lih)
    
    print(f"Number of qubits: {config_lih.n_qubits}")
    print(f"Number of electrons: {config_lih.n_electrons}")
    print(f"Number of parameters: {uccsd_lih.n_parameters}")
    print(f"  - Single excitations: {len(uccsd_lih.singles)}")
    print(f"  - Double excitations: {len(uccsd_lih.doubles)}")
    
    print("\n3. Parameter scaling:")
    print("-" * 50)
    
    for n_e, n_o in [(2, 2), (4, 4), (6, 6), (8, 8), (10, 10)]:
        n_params = count_uccsd_parameters(n_e, n_o)
        print(f"  {n_e} electrons, {n_o} orbitals: {n_params} parameters")
    
    # Try PennyLane circuit if available
    try:
        import pennylane as qml
        
        print("\n4. PennyLane Circuit (H2):")
        print("-" * 50)
        
        n_qubits = 4
        dev = qml.device('default.qubit', wires=n_qubits)
        
        circuit = uccsd.get_circuit_pennylane()
        
        @qml.qnode(dev)
        def state_prep(params):
            circuit(params, wires=range(n_qubits))
            return qml.state()
        
        # Test with zero parameters (should give HF state)
        params = np.zeros(uccsd.n_parameters)
        state = state_prep(params)
        
        # HF state should be |0011⟩ in computational basis
        # In PennyLane ordering, this is index 3 (binary 0011)
        print(f"State with θ=0 (HF state):")
        for i, amp in enumerate(state):
            if abs(amp) > 0.01:
                print(f"  |{i:04b}⟩: {amp:.4f}")
        
        print("\nCircuit structure:")
        print(qml.draw(state_prep, expansion_strategy="device")(params))
        
    except ImportError:
        print("\n4. PennyLane not available - skipping circuit demo")
    
    print("\n" + "=" * 60)
    print("UCCSD Demo Complete!")
    print("=" * 60)
