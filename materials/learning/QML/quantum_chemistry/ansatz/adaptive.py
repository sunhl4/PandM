"""
ADAPT-VQE (Adaptive Derivative-Assembled Pseudo-Trotter)
========================================================

ADAPT-VQE is an adaptive algorithm that iteratively builds the ansatz
by selecting operators from a pool based on their gradient magnitude.

Key idea:
- Start with a simple ansatz (e.g., HF state)
- Compute gradients with respect to all operators in the pool
- Add the operator with the largest gradient
- Optimize all parameters
- Repeat until convergence

Advantages:
- Produces compact circuits
- Avoids unnecessary parameters
- Better gradient landscapes than fixed ansätze

Reference:
Grimsley et al. (2019) "An adaptive variational algorithm for exact 
molecular simulations on a quantum computer"
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ADAPTConfig:
    """Configuration for ADAPT-VQE."""
    n_qubits: int
    n_electrons: int
    n_orbitals: int
    
    # Convergence criteria
    gradient_threshold: float = 1e-3  # Stop when max gradient < threshold
    max_iterations: int = 100
    energy_threshold: float = 1e-6  # Stop when energy change < threshold
    
    # Operator pool type
    pool_type: str = 'fermionic_sd'  # 'fermionic_sd', 'qubit', 'generalized'


@dataclass
class ADAPTResult:
    """Results from ADAPT-VQE optimization."""
    energy: float
    parameters: np.ndarray
    n_iterations: int
    selected_operators: List[int]  # Indices of selected operators
    energy_history: List[float]
    gradient_history: List[float]
    converged: bool


class ADAPTAnsatz:
    """
    ADAPT-VQE implementation.
    
    This class provides:
    1. Operator pool generation
    2. Gradient computation for operator selection
    3. Iterative ansatz construction
    
    Example usage:
    >>> config = ADAPTConfig(n_qubits=4, n_electrons=2, n_orbitals=2)
    >>> adapt = ADAPTAnsatz(config)
    >>> result = adapt.run(hamiltonian, optimizer)
    """
    
    def __init__(self, config: ADAPTConfig):
        """Initialize ADAPT-VQE."""
        self.config = config
        
        # Generate operator pool
        self.operator_pool = self._generate_operator_pool()
        
        # Track selected operators and parameters
        self.selected_operators: List[int] = []
        self.parameters: List[float] = []
    
    def _generate_operator_pool(self) -> List[Dict[str, Any]]:
        """
        Generate the operator pool.
        
        For fermionic_sd pool: all singles and doubles excitation operators.
        """
        pool = []
        
        if self.config.pool_type == 'fermionic_sd':
            n_e = self.config.n_electrons
            n_o = self.config.n_orbitals
            n_spin = 2 * n_o
            
            occupied = list(range(n_e))
            virtual = list(range(n_e, n_spin))
            
            # Single excitations
            for i in occupied:
                for a in virtual:
                    pool.append({
                        'type': 'single',
                        'indices': (i, a),
                        'generator': self._single_excitation_generator(i, a)
                    })
            
            # Double excitations
            for i in occupied:
                for j in occupied:
                    if i < j:
                        for a in virtual:
                            for b in virtual:
                                if a < b:
                                    pool.append({
                                        'type': 'double',
                                        'indices': (i, j, a, b),
                                        'generator': self._double_excitation_generator(i, j, a, b)
                                    })
        
        elif self.config.pool_type == 'qubit':
            # Qubit pool: all single-qubit rotations and two-qubit entangling
            n_q = self.config.n_qubits
            
            # Single-qubit generators: X, Y, Z on each qubit
            for q in range(n_q):
                for pauli in ['X', 'Y', 'Z']:
                    pool.append({
                        'type': f'qubit_{pauli}',
                        'indices': (q,),
                        'pauli': pauli
                    })
            
            # Two-qubit generators: XX, YY, ZZ, XY, etc.
            for i in range(n_q):
                for j in range(i+1, n_q):
                    for p1 in ['X', 'Y', 'Z']:
                        for p2 in ['X', 'Y', 'Z']:
                            pool.append({
                                'type': f'qubit_{p1}{p2}',
                                'indices': (i, j),
                                'paulis': (p1, p2)
                            })
        
        return pool
    
    def _single_excitation_generator(self, i: int, a: int):
        """Return generator for single excitation."""
        from ..core.fermion_operators import FermionOperator
        
        # T_i^a - T_i^a† = a†_a a_i - a†_i a_a
        T = FermionOperator()
        T.add_term(1.0, [(a, True), (i, False)])
        T.add_term(-1.0, [(i, True), (a, False)])
        return T
    
    def _double_excitation_generator(self, i: int, j: int, a: int, b: int):
        """Return generator for double excitation."""
        from ..core.fermion_operators import FermionOperator
        
        # T_ij^ab - T_ij^ab†
        T = FermionOperator()
        T.add_term(1.0, [(a, True), (b, True), (j, False), (i, False)])
        T.add_term(-1.0, [(i, True), (j, True), (b, False), (a, False)])
        return T
    
    def compute_operator_gradient(self, op_index: int, state_prep_fn: Callable,
                                  hamiltonian, current_params: np.ndarray) -> float:
        """
        Compute the gradient of energy with respect to operator op_index.
        
        This is used to select which operator to add next.
        
        For operator A_k, the gradient is:
            ∂E/∂θ_k |_{θ_k=0} = 2 Re⟨ψ|[H, A_k]|ψ⟩
        
        Args:
            op_index: Index of operator in pool
            state_prep_fn: Function that prepares current state
            hamiltonian: Hamiltonian operator
            current_params: Current parameter values
        
        Returns:
            Gradient magnitude
        """
        # This is a simplified placeholder
        # Full implementation would compute the commutator expectation value
        return 0.0
    
    def get_circuit_pennylane(self):
        """
        Return the current ADAPT circuit for PennyLane.
        
        The circuit is built from the sequence of selected operators.
        """
        try:
            import pennylane as qml
        except ImportError:
            raise ImportError("PennyLane is required")
        
        selected = self.selected_operators
        pool = self.operator_pool
        
        def circuit(params, wires):
            """ADAPT circuit with current operators."""
            n_e = self.config.n_electrons
            
            # Prepare HF state
            hf = [1] * n_e + [0] * (len(wires) - n_e)
            qml.BasisState(np.array(hf), wires=wires)
            
            # Apply selected operators
            for param_idx, op_idx in enumerate(selected):
                op = pool[op_idx]
                theta = params[param_idx]
                
                if op['type'] == 'single':
                    i, a = op['indices']
                    qml.SingleExcitation(theta, wires=[i, a])
                elif op['type'] == 'double':
                    i, j, a, b = op['indices']
                    qml.DoubleExcitation(theta, wires=[i, j, a, b])
        
        return circuit
    
    def add_operator(self, op_index: int, initial_param: float = 0.0):
        """Add an operator to the ansatz."""
        self.selected_operators.append(op_index)
        self.parameters.append(initial_param)
    
    @property
    def n_parameters(self) -> int:
        """Current number of parameters."""
        return len(self.parameters)
    
    def run_adapt_iteration(self, hamiltonian, optimizer, current_energy: float) -> Tuple[float, bool]:
        """
        Run one iteration of ADAPT-VQE.
        
        1. Compute gradients for all operators in pool
        2. Select operator with largest gradient
        3. Add to ansatz
        4. Optimize all parameters
        
        Returns:
            (new_energy, converged)
        """
        # This is a placeholder - full implementation needs
        # gradient computation and optimization infrastructure
        return current_energy, False


def adapt_vqe_gradients(adapt: ADAPTAnsatz, state_vector: np.ndarray,
                        H_matrix: np.ndarray) -> np.ndarray:
    """
    Compute gradients for all operators in the ADAPT pool.
    
    For each operator A_k:
        gradient_k = 2 Re⟨ψ|[H, A_k]|ψ⟩ = 2 Re⟨ψ|H A_k|ψ⟩ - 2 Re⟨ψ|A_k H|ψ⟩
    
    Simplification: At θ_k = 0, this reduces to:
        gradient_k = 2 Im⟨ψ|H A_k|ψ⟩
    
    Args:
        adapt: ADAPTAnsatz instance
        state_vector: Current state |ψ⟩
        H_matrix: Hamiltonian matrix
    
    Returns:
        Array of gradient magnitudes for each operator in pool
    """
    n_ops = len(adapt.operator_pool)
    gradients = np.zeros(n_ops)
    
    # For each operator, compute ⟨ψ|H A|ψ⟩
    # This requires converting fermionic operators to matrices
    
    # Placeholder - would need full implementation
    return gradients


# ============================================================================
# Simplified ADAPT-VQE Implementation with PennyLane
# ============================================================================

def run_adapt_vqe_simple(n_qubits: int, n_electrons: int, hamiltonian_matrix: np.ndarray,
                        max_iterations: int = 20, gradient_threshold: float = 1e-3,
                        verbose: bool = True) -> ADAPTResult:
    """
    Simplified ADAPT-VQE implementation.
    
    This is a demonstration implementation that shows the core algorithm.
    A production implementation would use more sophisticated gradient computation.
    
    Args:
        n_qubits: Number of qubits
        n_electrons: Number of electrons
        hamiltonian_matrix: Hamiltonian as matrix
        max_iterations: Maximum ADAPT iterations
        gradient_threshold: Convergence threshold
        verbose: Print progress
    
    Returns:
        ADAPTResult with optimization history
    """
    try:
        import pennylane as qml
    except ImportError:
        raise ImportError("PennyLane is required")
    
    from scipy.optimize import minimize
    
    n_orbitals = n_qubits // 2
    
    # Generate operator pool (singles and doubles)
    occupied = list(range(n_electrons))
    virtual = list(range(n_electrons, n_qubits))
    
    operators = []
    
    # Singles
    for i in occupied:
        for a in virtual:
            operators.append(('single', i, a))
    
    # Doubles
    for i in occupied:
        for j in occupied:
            if i < j:
                for a in virtual:
                    for b in virtual:
                        if a < b:
                            operators.append(('double', i, j, a, b))
    
    if verbose:
        print(f"Operator pool size: {len(operators)}")
    
    # Device
    dev = qml.device('default.qubit', wires=n_qubits)
    
    # Track selected operators
    selected_ops = []
    all_params = []
    energy_history = []
    gradient_history = []
    
    # Start with HF state
    hf_state = [1] * n_electrons + [0] * (n_qubits - n_electrons)
    
    def build_circuit(params, ops_list):
        """Build circuit with selected operators."""
        qml.BasisState(np.array(hf_state), wires=range(n_qubits))
        
        for idx, op_idx in enumerate(ops_list):
            op = operators[op_idx]
            if op[0] == 'single':
                _, i, a = op
                qml.SingleExcitation(params[idx], wires=[i, a])
            else:  # double
                _, i, j, a, b = op
                qml.DoubleExcitation(params[idx], wires=[i, j, a, b])
    
    @qml.qnode(dev)
    def energy_circuit(params, ops_list):
        """Compute energy expectation value."""
        build_circuit(params, ops_list)
        return qml.expval(qml.Hermitian(hamiltonian_matrix, wires=range(n_qubits)))
    
    # Main ADAPT loop
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n--- ADAPT Iteration {iteration + 1} ---")
        
        # Compute gradients for all operators in pool
        # Using parameter shift rule approximation
        max_grad = 0.0
        best_op_idx = -1
        
        for op_idx in range(len(operators)):
            if op_idx in selected_ops:
                continue  # Skip already selected
            
            # Test adding this operator
            test_ops = selected_ops + [op_idx]
            test_params = list(all_params) + [0.0]
            
            # Compute gradient at θ=0
            shift = np.pi / 2
            params_plus = np.array(test_params)
            params_plus[-1] = shift
            params_minus = np.array(test_params)
            params_minus[-1] = -shift
            
            E_plus = energy_circuit(params_plus, test_ops)
            E_minus = energy_circuit(params_minus, test_ops)
            
            grad = (E_plus - E_minus) / 2
            
            if abs(grad) > abs(max_grad):
                max_grad = grad
                best_op_idx = op_idx
        
        gradient_history.append(abs(max_grad))
        
        if verbose:
            print(f"Max gradient: {abs(max_grad):.6f}")
        
        # Check convergence
        if abs(max_grad) < gradient_threshold:
            if verbose:
                print("Converged! Max gradient below threshold.")
            break
        
        # Add best operator
        selected_ops.append(best_op_idx)
        all_params.append(0.0)
        
        if verbose:
            op = operators[best_op_idx]
            print(f"Added operator: {op}")
        
        # Optimize all parameters
        def cost_fn(params):
            return float(energy_circuit(params, selected_ops))
        
        result = minimize(
            cost_fn,
            x0=np.array(all_params),
            method='COBYLA',
            options={'maxiter': 500}
        )
        
        all_params = list(result.x)
        current_energy = result.fun
        energy_history.append(current_energy)
        
        if verbose:
            print(f"Energy: {current_energy:.8f}")
            print(f"Parameters: {all_params}")
    
    converged = abs(max_grad) < gradient_threshold if gradient_history else False
    
    return ADAPTResult(
        energy=energy_history[-1] if energy_history else 0.0,
        parameters=np.array(all_params),
        n_iterations=len(energy_history),
        selected_operators=selected_ops,
        energy_history=energy_history,
        gradient_history=gradient_history,
        converged=converged
    )


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ADAPT-VQE Demo")
    print("=" * 60)
    
    print("\n1. ADAPT Configuration:")
    print("-" * 50)
    
    config = ADAPTConfig(
        n_qubits=4,
        n_electrons=2,
        n_orbitals=2,
        gradient_threshold=1e-3,
        max_iterations=20
    )
    
    adapt = ADAPTAnsatz(config)
    
    print(f"Number of qubits: {config.n_qubits}")
    print(f"Number of electrons: {config.n_electrons}")
    print(f"Pool type: {config.pool_type}")
    print(f"Operator pool size: {len(adapt.operator_pool)}")
    
    print("\n2. Operator Pool Contents:")
    print("-" * 50)
    
    singles = [op for op in adapt.operator_pool if op['type'] == 'single']
    doubles = [op for op in adapt.operator_pool if op['type'] == 'double']
    
    print(f"Single excitations: {len(singles)}")
    for op in singles[:3]:
        print(f"  {op['indices']}")
    if len(singles) > 3:
        print(f"  ... and {len(singles) - 3} more")
    
    print(f"Double excitations: {len(doubles)}")
    for op in doubles[:3]:
        print(f"  {op['indices']}")
    if len(doubles) > 3:
        print(f"  ... and {len(doubles) - 3} more")
    
    # Try running ADAPT-VQE on H2 if PennyLane is available
    try:
        import pennylane as qml
        
        print("\n3. ADAPT-VQE on Simple H2-like System:")
        print("-" * 50)
        
        # Simple 4-qubit Hamiltonian (not real H2, just for demo)
        n_qubits = 4
        H_diag = np.diag([-1.5, -0.5, -0.5, 0.0, -0.5, 0.5, 0.5, 1.5,
                         -0.5, 0.5, 0.5, 1.5, 0.0, 1.0, 1.0, 2.0])
        
        # Add some off-diagonal terms
        H = H_diag.copy()
        H[3, 12] = H[12, 3] = -0.2
        H[5, 10] = H[10, 5] = -0.2
        H[6, 9] = H[9, 6] = -0.2
        
        print(f"Hamiltonian eigenvalues (lowest 5): {np.linalg.eigvalsh(H)[:5]}")
        exact_gs = np.linalg.eigvalsh(H)[0]
        print(f"Exact ground state energy: {exact_gs:.6f}")
        
        print("\nRunning ADAPT-VQE...")
        result = run_adapt_vqe_simple(
            n_qubits=4,
            n_electrons=2,
            hamiltonian_matrix=H,
            max_iterations=10,
            gradient_threshold=1e-3,
            verbose=True
        )
        
        print(f"\n--- Results ---")
        print(f"Final energy: {result.energy:.6f}")
        print(f"Exact energy: {exact_gs:.6f}")
        print(f"Error: {abs(result.energy - exact_gs):.6f}")
        print(f"Number of operators used: {len(result.selected_operators)}")
        print(f"Converged: {result.converged}")
        
    except ImportError:
        print("\n3. PennyLane not available - skipping ADAPT-VQE demo")
    
    print("\n" + "=" * 60)
    print("ADAPT-VQE Demo Complete!")
    print("=" * 60)
