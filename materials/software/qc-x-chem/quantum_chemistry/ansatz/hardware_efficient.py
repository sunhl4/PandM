"""
Hardware-Efficient Ansatz
=========================

Hardware-efficient ansätze are designed to be directly implementable on
near-term quantum devices with limited connectivity and gate fidelity.

Key features:
- Use native gates of the hardware (e.g., RY, RZ, CNOT)
- Respect hardware topology (qubit connectivity)
- Shallow circuits to minimize decoherence

Trade-offs:
- (+) Easy to implement on real hardware
- (+) Flexible expressibility
- (-) May suffer from barren plateaus (vanishing gradients)
- (-) Less physical motivation than chemistry-inspired ansätze

Structure:
    |ψ(θ)⟩ = U_L(θ_L) ... U_2(θ_2) U_1(θ_1) |0⟩

Each layer U_l typically contains:
1. Single-qubit rotation gates (parameterized)
2. Entangling gates (fixed or parameterized)
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass
import numpy as np


@dataclass
class HardwareEfficientConfig:
    """Configuration for hardware-efficient ansatz."""
    n_qubits: int
    n_layers: int
    rotation_gates: List[str] = None  # Default: ['RY', 'RZ']
    entangling_gate: str = 'CNOT'  # 'CNOT', 'CZ', 'CRZ', 'CRY'
    entanglement: str = 'linear'  # 'linear', 'circular', 'full', 'custom'
    custom_entanglement: Optional[List[Tuple[int, int]]] = None
    
    def __post_init__(self):
        if self.rotation_gates is None:
            self.rotation_gates = ['RY', 'RZ']


class HardwareEfficientAnsatz:
    """
    Hardware-efficient ansatz implementation.
    
    Structure of each layer:
    1. Rotation sublayer: Apply parameterized rotations to each qubit
    2. Entanglement sublayer: Apply entangling gates between qubits
    
    Example:
    >>> config = HardwareEfficientConfig(n_qubits=4, n_layers=2)
    >>> ansatz = HardwareEfficientAnsatz(config)
    >>> params = np.random.uniform(0, 2*np.pi, ansatz.n_parameters)
    """
    
    def __init__(self, config: HardwareEfficientConfig):
        """Initialize the ansatz."""
        self.config = config
        
        # Calculate number of parameters
        n_rotations_per_qubit = len(config.rotation_gates)
        n_rotation_params = config.n_qubits * n_rotations_per_qubit * config.n_layers
        
        # Entangling gates may also have parameters
        if config.entangling_gate in ['CRZ', 'CRY', 'CRX']:
            n_entangling_pairs = self._count_entangling_pairs()
            n_entangle_params = n_entangling_pairs * config.n_layers
        else:
            n_entangle_params = 0
        
        self.n_parameters = n_rotation_params + n_entangle_params
        self.n_rotation_params = n_rotation_params
        self.n_entangle_params = n_entangle_params
    
    def _count_entangling_pairs(self) -> int:
        """Count the number of entangling gate pairs."""
        n = self.config.n_qubits
        
        if self.config.entanglement == 'linear':
            return n - 1
        elif self.config.entanglement == 'circular':
            return n
        elif self.config.entanglement == 'full':
            return n * (n - 1) // 2
        elif self.config.entanglement == 'custom':
            return len(self.config.custom_entanglement or [])
        else:
            raise ValueError(f"Unknown entanglement: {self.config.entanglement}")
    
    def _get_entangling_pairs(self) -> List[Tuple[int, int]]:
        """Get list of qubit pairs for entangling gates."""
        n = self.config.n_qubits
        
        if self.config.entanglement == 'linear':
            return [(i, i+1) for i in range(n-1)]
        elif self.config.entanglement == 'circular':
            pairs = [(i, i+1) for i in range(n-1)]
            pairs.append((n-1, 0))
            return pairs
        elif self.config.entanglement == 'full':
            pairs = []
            for i in range(n):
                for j in range(i+1, n):
                    pairs.append((i, j))
            return pairs
        elif self.config.entanglement == 'custom':
            return self.config.custom_entanglement or []
        else:
            raise ValueError(f"Unknown entanglement: {self.config.entanglement}")
    
    def get_circuit_pennylane(self):
        """
        Return a PennyLane circuit template.
        
        Returns:
            A function that applies the ansatz to given wires.
        """
        try:
            import pennylane as qml
        except ImportError:
            raise ImportError("PennyLane is required. Install with: pip install pennylane")
        
        config = self.config
        entangling_pairs = self._get_entangling_pairs()
        
        # Gate mapping
        rotation_gates = {
            'RX': qml.RX,
            'RY': qml.RY,
            'RZ': qml.RZ,
        }
        
        def circuit(params, wires):
            """
            Hardware-efficient circuit.
            
            Args:
                params: Parameter array
                wires: Qubit wires
            """
            param_idx = 0
            
            for layer in range(config.n_layers):
                # Rotation sublayer
                for qubit in range(config.n_qubits):
                    for gate_name in config.rotation_gates:
                        gate = rotation_gates[gate_name]
                        gate(params[param_idx], wires=wires[qubit])
                        param_idx += 1
                
                # Entanglement sublayer
                for i, j in entangling_pairs:
                    if config.entangling_gate == 'CNOT':
                        qml.CNOT(wires=[wires[i], wires[j]])
                    elif config.entangling_gate == 'CZ':
                        qml.CZ(wires=[wires[i], wires[j]])
                    elif config.entangling_gate == 'CRZ':
                        qml.CRZ(params[param_idx], wires=[wires[i], wires[j]])
                        param_idx += 1
                    elif config.entangling_gate == 'CRY':
                        qml.CRY(params[param_idx], wires=[wires[i], wires[j]])
                        param_idx += 1
        
        return circuit
    
    def get_initial_params(self, method: str = 'random') -> np.ndarray:
        """
        Get initial parameters.
        
        Args:
            method: 'zeros', 'random', 'small_random'
        
        Returns:
            Initial parameter array
        """
        if method == 'zeros':
            return np.zeros(self.n_parameters)
        elif method == 'random':
            return np.random.uniform(0, 2 * np.pi, self.n_parameters)
        elif method == 'small_random':
            return np.random.uniform(-0.1, 0.1, self.n_parameters)
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    
    def estimate_circuit_depth(self) -> int:
        """
        Estimate the circuit depth.
        
        Depth = number of layers × (rotations + entanglement)
        """
        n_rotation_gates = len(self.config.rotation_gates)
        
        # Assuming rotations are parallelized and entanglement is sequential
        depth_per_layer = n_rotation_gates + len(self._get_entangling_pairs())
        
        return depth_per_layer * self.config.n_layers


def create_hardware_efficient_ansatz(
    n_qubits: int,
    n_layers: int,
    rotations: List[str] = None,
    entanglement: str = 'linear'
) -> HardwareEfficientAnsatz:
    """
    Convenience function to create hardware-efficient ansatz.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of layers
        rotations: List of rotation gates (default: ['RY', 'RZ'])
        entanglement: Entanglement pattern
    
    Returns:
        HardwareEfficientAnsatz instance
    """
    config = HardwareEfficientConfig(
        n_qubits=n_qubits,
        n_layers=n_layers,
        rotation_gates=rotations,
        entanglement=entanglement
    )
    return HardwareEfficientAnsatz(config)


# ============================================================================
# Specialized Hardware-Efficient Variants
# ============================================================================

class RealAmplitudesAnsatz(HardwareEfficientAnsatz):
    """
    Real amplitudes ansatz (also known as RY ansatz).
    
    Uses only RY rotations and CNOT gates.
    The resulting states have real amplitudes.
    
    Good for:
    - Problems with real Hamiltonians
    - Reducing parameter count
    """
    
    def __init__(self, n_qubits: int, n_layers: int, entanglement: str = 'linear'):
        config = HardwareEfficientConfig(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_gates=['RY'],
            entangling_gate='CNOT',
            entanglement=entanglement
        )
        super().__init__(config)


class EfficientSU2Ansatz(HardwareEfficientAnsatz):
    """
    Efficient SU(2) ansatz.
    
    Uses RY and RZ rotations with CNOT entanglement.
    Can represent any state in the SU(2) subspace.
    """
    
    def __init__(self, n_qubits: int, n_layers: int, entanglement: str = 'linear'):
        config = HardwareEfficientConfig(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_gates=['RY', 'RZ'],
            entangling_gate='CNOT',
            entanglement=entanglement
        )
        super().__init__(config)


class StronglyEntanglingAnsatz(HardwareEfficientAnsatz):
    """
    Strongly entangling layers.
    
    Full three-rotation gates (RX, RY, RZ) with full entanglement.
    Maximum expressibility but more prone to barren plateaus.
    """
    
    def __init__(self, n_qubits: int, n_layers: int):
        config = HardwareEfficientConfig(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_gates=['RX', 'RY', 'RZ'],
            entangling_gate='CNOT',
            entanglement='full'
        )
        super().__init__(config)


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Hardware-Efficient Ansatz Demo")
    print("=" * 60)
    
    print("\n1. Basic Hardware-Efficient Ansatz:")
    print("-" * 50)
    
    config = HardwareEfficientConfig(
        n_qubits=4,
        n_layers=2,
        rotation_gates=['RY', 'RZ'],
        entangling_gate='CNOT',
        entanglement='linear'
    )
    
    ansatz = HardwareEfficientAnsatz(config)
    
    print(f"Number of qubits: {config.n_qubits}")
    print(f"Number of layers: {config.n_layers}")
    print(f"Rotation gates: {config.rotation_gates}")
    print(f"Entanglement: {config.entanglement}")
    print(f"Number of parameters: {ansatz.n_parameters}")
    print(f"  - Rotation parameters: {ansatz.n_rotation_params}")
    print(f"  - Entanglement parameters: {ansatz.n_entangle_params}")
    print(f"Estimated circuit depth: {ansatz.estimate_circuit_depth()}")
    
    print("\n2. Entanglement Patterns:")
    print("-" * 50)
    
    for pattern in ['linear', 'circular', 'full']:
        cfg = HardwareEfficientConfig(n_qubits=4, n_layers=1, entanglement=pattern)
        ans = HardwareEfficientAnsatz(cfg)
        pairs = ans._get_entangling_pairs()
        print(f"  {pattern}: {pairs}")
    
    print("\n3. Specialized Ansätze:")
    print("-" * 50)
    
    real_amp = RealAmplitudesAnsatz(n_qubits=4, n_layers=2)
    print(f"RealAmplitudes: {real_amp.n_parameters} params")
    
    eff_su2 = EfficientSU2Ansatz(n_qubits=4, n_layers=2)
    print(f"EfficientSU2: {eff_su2.n_parameters} params")
    
    strong = StronglyEntanglingAnsatz(n_qubits=4, n_layers=2)
    print(f"StronglyEntangling: {strong.n_parameters} params")
    
    print("\n4. Parameter Scaling:")
    print("-" * 50)
    
    print("Parameters for different configurations (n_layers=2):")
    for n_q in [2, 4, 6, 8, 10]:
        ans = EfficientSU2Ansatz(n_qubits=n_q, n_layers=2)
        print(f"  {n_q} qubits: {ans.n_parameters} parameters")
    
    # Try PennyLane circuit if available
    try:
        import pennylane as qml
        
        print("\n5. PennyLane Circuit:")
        print("-" * 50)
        
        n_qubits = 4
        dev = qml.device('default.qubit', wires=n_qubits)
        
        ansatz = EfficientSU2Ansatz(n_qubits=n_qubits, n_layers=2)
        circuit = ansatz.get_circuit_pennylane()
        
        @qml.qnode(dev)
        def state_prep(params):
            circuit(params, wires=range(n_qubits))
            return qml.state()
        
        params = ansatz.get_initial_params('random')
        state = state_prep(params)
        
        print(f"State vector dimension: {len(state)}")
        print(f"State norm: {np.linalg.norm(state):.6f}")
        
        print("\nCircuit structure:")
        print(qml.draw(state_prep, expansion_strategy="device")(params))
        
    except ImportError:
        print("\n5. PennyLane not available - skipping circuit demo")
    
    print("\n" + "=" * 60)
    print("Hardware-Efficient Ansatz Demo Complete!")
    print("=" * 60)
