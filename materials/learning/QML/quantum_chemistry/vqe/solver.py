"""
VQE (Variational Quantum Eigensolver) Solver
=============================================

VQE is a hybrid quantum-classical algorithm for finding ground state energies.

Algorithm:
1. Prepare a parameterized quantum state |ψ(θ)⟩ using an ansatz
2. Measure the energy expectation value E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
3. Use classical optimizer to update parameters θ
4. Repeat until convergence

Key components:
- Ansatz: The parameterized quantum circuit
- Hamiltonian: Expressed as sum of Pauli strings
- Optimizer: Classical optimization algorithm
- Measurement: Grouping and sampling strategies
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Callable, Dict, Any, Union
from dataclasses import dataclass, field
import numpy as np
import time


@dataclass
class VQEConfig:
    """Configuration for VQE solver."""
    # Optimization settings
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    
    # Measurement settings
    shots: Optional[int] = None  # None = exact expectation values
    
    # Optimizer settings
    optimizer: str = 'COBYLA'  # 'COBYLA', 'BFGS', 'Adam', 'SPSA', 'gradient_descent'
    learning_rate: float = 0.1
    
    # Output settings
    verbose: bool = True
    callback: Optional[Callable] = None


@dataclass
class VQEResult:
    """Results from VQE optimization."""
    energy: float
    parameters: np.ndarray
    n_iterations: int
    energy_history: List[float]
    parameter_history: List[np.ndarray]
    success: bool
    runtime: float
    optimizer_result: Any = None


class VQESolver:
    """
    Variational Quantum Eigensolver implementation.
    
    This solver supports:
    - Multiple ansatz types (UCCSD, hardware-efficient, etc.)
    - Various classical optimizers
    - Exact simulation and shot-based measurement
    - Energy gradient computation
    
    Example usage:
    >>> from quantum_chemistry.core import MolecularData, compute_integrals_pyscf
    >>> from quantum_chemistry.ansatz import UCCSD, UCCSDConfig
    >>> 
    >>> # Setup molecule
    >>> mol = MolecularData.h2(bond_length=0.74)
    >>> mol = compute_integrals_pyscf(mol)
    >>> H_qubit = get_qubit_hamiltonian(mol)
    >>> 
    >>> # Setup ansatz
    >>> config = UCCSDConfig(n_qubits=4, n_electrons=2, n_orbitals=2)
    >>> ansatz = UCCSD(config)
    >>> 
    >>> # Run VQE
    >>> solver = VQESolver(config=VQEConfig())
    >>> result = solver.run(H_qubit, ansatz)
    """
    
    def __init__(self, config: Optional[VQEConfig] = None):
        """Initialize VQE solver."""
        self.config = config or VQEConfig()
        
        # Storage for optimization
        self.energy_history: List[float] = []
        self.parameter_history: List[np.ndarray] = []
        self.n_function_calls = 0
    
    def run(self, hamiltonian, ansatz, initial_params: Optional[np.ndarray] = None,
            backend: str = 'pennylane') -> VQEResult:
        """
        Run VQE optimization.
        
        Args:
            hamiltonian: Qubit Hamiltonian (QubitOperator or matrix)
            ansatz: Ansatz object with get_circuit_pennylane() method
            initial_params: Initial parameter values (optional)
            backend: 'pennylane' or 'matrix'
        
        Returns:
            VQEResult with optimization results
        """
        start_time = time.time()
        
        # Reset history
        self.energy_history = []
        self.parameter_history = []
        self.n_function_calls = 0
        
        # Get initial parameters
        if initial_params is None:
            initial_params = ansatz.get_initial_params('zeros')
        
        n_params = len(initial_params)
        
        if self.config.verbose:
            print("=" * 60)
            print("VQE Optimization")
            print("=" * 60)
            print(f"Number of parameters: {n_params}")
            print(f"Optimizer: {self.config.optimizer}")
            print(f"Max iterations: {self.config.max_iterations}")
        
        # Setup cost function
        if backend == 'pennylane':
            cost_fn = self._setup_pennylane_cost(hamiltonian, ansatz)
        elif backend == 'matrix':
            cost_fn = self._setup_matrix_cost(hamiltonian, ansatz)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        # Run optimization
        if self.config.optimizer in ['COBYLA', 'BFGS', 'L-BFGS-B', 'Nelder-Mead', 'Powell']:
            result = self._run_scipy_optimizer(cost_fn, initial_params)
        elif self.config.optimizer == 'gradient_descent':
            result = self._run_gradient_descent(cost_fn, initial_params, ansatz)
        elif self.config.optimizer == 'Adam':
            result = self._run_adam(cost_fn, initial_params, ansatz)
        elif self.config.optimizer == 'SPSA':
            result = self._run_spsa(cost_fn, initial_params)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        runtime = time.time() - start_time
        
        if self.config.verbose:
            print("-" * 60)
            print(f"Optimization complete!")
            print(f"Final energy: {result.energy:.10f}")
            print(f"Iterations: {result.n_iterations}")
            print(f"Runtime: {runtime:.2f} s")
            print("=" * 60)
        
        result.runtime = runtime
        return result
    
    def _setup_pennylane_cost(self, hamiltonian, ansatz) -> Callable:
        """Setup cost function using PennyLane."""
        try:
            import pennylane as qml
        except ImportError:
            raise ImportError("PennyLane is required. Install with: pip install pennylane")
        
        # Get number of qubits from ansatz
        n_qubits = ansatz.config.n_qubits
        
        # Create device
        dev = qml.device('default.qubit', wires=n_qubits, shots=self.config.shots)
        
        # Get circuit template
        circuit_fn = ansatz.get_circuit_pennylane()
        
        # Convert hamiltonian to PennyLane observable
        if hasattr(hamiltonian, 'to_matrix'):
            # QubitOperator - convert to matrix
            H_matrix = hamiltonian.to_matrix(n_qubits)
            observable = qml.Hermitian(H_matrix, wires=range(n_qubits))
        elif isinstance(hamiltonian, np.ndarray):
            observable = qml.Hermitian(hamiltonian, wires=range(n_qubits))
        else:
            raise TypeError(f"Unknown hamiltonian type: {type(hamiltonian)}")
        
        @qml.qnode(dev)
        def cost_circuit(params):
            circuit_fn(params, wires=range(n_qubits))
            return qml.expval(observable)
        
        def cost_fn(params):
            self.n_function_calls += 1
            energy = float(cost_circuit(params))
            self.energy_history.append(energy)
            self.parameter_history.append(params.copy())
            
            if self.config.callback:
                self.config.callback(params, energy)
            
            return energy
        
        return cost_fn
    
    def _setup_matrix_cost(self, hamiltonian, ansatz) -> Callable:
        """Setup cost function using matrix simulation."""
        try:
            import pennylane as qml
        except ImportError:
            raise ImportError("PennyLane is required")
        
        n_qubits = ansatz.config.n_qubits
        dev = qml.device('default.qubit', wires=n_qubits)
        circuit_fn = ansatz.get_circuit_pennylane()
        
        # Get Hamiltonian matrix
        if hasattr(hamiltonian, 'to_matrix'):
            H_matrix = hamiltonian.to_matrix(n_qubits)
        elif isinstance(hamiltonian, np.ndarray):
            H_matrix = hamiltonian
        else:
            raise TypeError(f"Unknown hamiltonian type: {type(hamiltonian)}")
        
        @qml.qnode(dev)
        def state_circuit(params):
            circuit_fn(params, wires=range(n_qubits))
            return qml.state()
        
        def cost_fn(params):
            self.n_function_calls += 1
            state = state_circuit(params)
            energy = float(np.real(state.conj() @ H_matrix @ state))
            self.energy_history.append(energy)
            self.parameter_history.append(params.copy())
            
            if self.config.callback:
                self.config.callback(params, energy)
            
            return energy
        
        return cost_fn
    
    def _run_scipy_optimizer(self, cost_fn: Callable, initial_params: np.ndarray) -> VQEResult:
        """Run optimization using scipy.optimize.minimize."""
        from scipy.optimize import minimize
        
        result = minimize(
            cost_fn,
            x0=initial_params,
            method=self.config.optimizer,
            options={
                'maxiter': self.config.max_iterations,
                'disp': self.config.verbose
            }
        )
        
        return VQEResult(
            energy=result.fun,
            parameters=result.x,
            n_iterations=result.nfev,
            energy_history=self.energy_history,
            parameter_history=self.parameter_history,
            success=result.success,
            runtime=0.0,
            optimizer_result=result
        )
    
    def _run_gradient_descent(self, cost_fn: Callable, initial_params: np.ndarray,
                              ansatz) -> VQEResult:
        """Run gradient descent optimization."""
        params = initial_params.copy()
        lr = self.config.learning_rate
        
        for iteration in range(self.config.max_iterations):
            # Compute gradient using parameter shift
            grad = self._compute_gradient_parameter_shift(cost_fn, params)
            
            # Update parameters
            params = params - lr * grad
            
            # Compute current energy
            energy = cost_fn(params)
            
            if self.config.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Energy = {energy:.8f}")
            
            # Check convergence
            if len(self.energy_history) > 1:
                if abs(self.energy_history[-1] - self.energy_history[-2]) < self.config.convergence_threshold:
                    break
        
        return VQEResult(
            energy=self.energy_history[-1],
            parameters=params,
            n_iterations=iteration + 1,
            energy_history=self.energy_history,
            parameter_history=self.parameter_history,
            success=True,
            runtime=0.0
        )
    
    def _run_adam(self, cost_fn: Callable, initial_params: np.ndarray,
                  ansatz) -> VQEResult:
        """Run Adam optimization."""
        params = initial_params.copy()
        lr = self.config.learning_rate
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        m = np.zeros_like(params)  # First moment
        v = np.zeros_like(params)  # Second moment
        
        for iteration in range(self.config.max_iterations):
            # Compute gradient
            grad = self._compute_gradient_parameter_shift(cost_fn, params)
            
            # Update moments
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            
            # Bias correction
            m_hat = m / (1 - beta1**(iteration + 1))
            v_hat = v / (1 - beta2**(iteration + 1))
            
            # Update parameters
            params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Compute current energy
            energy = cost_fn(params)
            
            if self.config.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Energy = {energy:.8f}")
            
            # Check convergence
            if len(self.energy_history) > 1:
                if abs(self.energy_history[-1] - self.energy_history[-2]) < self.config.convergence_threshold:
                    break
        
        return VQEResult(
            energy=self.energy_history[-1],
            parameters=params,
            n_iterations=iteration + 1,
            energy_history=self.energy_history,
            parameter_history=self.parameter_history,
            success=True,
            runtime=0.0
        )
    
    def _run_spsa(self, cost_fn: Callable, initial_params: np.ndarray) -> VQEResult:
        """
        Run SPSA (Simultaneous Perturbation Stochastic Approximation).
        
        SPSA is gradient-free and uses only 2 function evaluations per iteration
        regardless of the number of parameters.
        """
        params = initial_params.copy()
        
        # SPSA hyperparameters
        a = self.config.learning_rate
        c = 0.1
        A = self.config.max_iterations // 10
        alpha = 0.602
        gamma = 0.101
        
        for iteration in range(self.config.max_iterations):
            k = iteration + 1
            
            # Decaying step sizes
            ak = a / (k + A)**alpha
            ck = c / k**gamma
            
            # Random perturbation direction
            delta = 2 * np.random.binomial(1, 0.5, size=len(params)) - 1
            
            # Perturbed evaluations
            E_plus = cost_fn(params + ck * delta)
            E_minus = cost_fn(params - ck * delta)
            
            # Gradient estimate
            grad_estimate = (E_plus - E_minus) / (2 * ck * delta)
            
            # Update parameters
            params = params - ak * grad_estimate
            
            # Current energy
            energy = cost_fn(params)
            
            if self.config.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Energy = {energy:.8f}")
            
            # Check convergence
            if len(self.energy_history) > 1:
                if abs(self.energy_history[-1] - self.energy_history[-2]) < self.config.convergence_threshold:
                    break
        
        return VQEResult(
            energy=self.energy_history[-1],
            parameters=params,
            n_iterations=iteration + 1,
            energy_history=self.energy_history,
            parameter_history=self.parameter_history,
            success=True,
            runtime=0.0
        )
    
    def _compute_gradient_parameter_shift(self, cost_fn: Callable, params: np.ndarray,
                                          shift: float = np.pi/2) -> np.ndarray:
        """
        Compute gradient using parameter shift rule.
        
        For a rotation gate R(θ), the derivative is:
            dE/dθ = (E(θ+π/2) - E(θ-π/2)) / 2
        
        Args:
            cost_fn: Cost function
            params: Current parameters
            shift: Shift amount (π/2 for standard gates)
        
        Returns:
            Gradient array
        """
        grad = np.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += shift
            
            params_minus = params.copy()
            params_minus[i] -= shift
            
            # Note: These calls will update energy_history
            # We need to handle this carefully
            E_plus = cost_fn(params_plus)
            E_minus = cost_fn(params_minus)
            
            grad[i] = (E_plus - E_minus) / (2 * np.sin(shift))
        
        return grad


# ============================================================================
# Convenience functions
# ============================================================================

def run_vqe_h2(bond_length: float = 0.74, basis: str = 'sto-3g',
               optimizer: str = 'COBYLA', verbose: bool = True) -> VQEResult:
    """
    Run VQE on H2 molecule.
    
    This is a convenience function for quick testing.
    
    Args:
        bond_length: H-H bond length in Angstroms
        basis: Basis set name
        optimizer: Optimizer to use
        verbose: Print progress
    
    Returns:
        VQEResult
    """
    from ..core.molecular_integrals import MolecularData, compute_integrals_pyscf, get_qubit_hamiltonian
    from ..ansatz.uccsd import UCCSD, UCCSDConfig
    
    # Setup molecule
    mol = MolecularData.h2(bond_length=bond_length, basis=basis)
    mol = compute_integrals_pyscf(mol, run_fci=True)
    
    # Get qubit Hamiltonian
    H = get_qubit_hamiltonian(mol)
    
    # Setup ansatz
    config = UCCSDConfig(
        n_qubits=2 * mol.n_orbitals,
        n_electrons=mol.n_electrons,
        n_orbitals=mol.n_orbitals
    )
    ansatz = UCCSD(config)
    
    # Run VQE
    vqe_config = VQEConfig(
        optimizer=optimizer,
        max_iterations=500,
        verbose=verbose
    )
    solver = VQESolver(vqe_config)
    result = solver.run(H, ansatz)
    
    if verbose:
        print(f"\nComparison:")
        print(f"  VQE energy:  {result.energy:.8f} Ha")
        print(f"  HF energy:   {mol.hf_energy:.8f} Ha")
        print(f"  FCI energy:  {mol.fci_energy:.8f} Ha")
        print(f"  Error vs FCI: {abs(result.energy - mol.fci_energy):.8f} Ha")
    
    return result


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VQE Solver Demo")
    print("=" * 60)
    
    # Check if PennyLane is available
    try:
        import pennylane as qml
        pennylane_available = True
    except ImportError:
        pennylane_available = False
        print("PennyLane not available. Install with: pip install pennylane")
    
    if pennylane_available:
        print("\n1. Simple 2-qubit test:")
        print("-" * 50)
        
        from ..ansatz.hardware_efficient import EfficientSU2Ansatz
        
        # Simple Hamiltonian
        H = np.array([
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0, -1.0]
        ])
        
        exact_energy = np.min(np.linalg.eigvalsh(H))
        print(f"Exact ground state energy: {exact_energy:.6f}")
        
        # Create ansatz
        ansatz = EfficientSU2Ansatz(n_qubits=2, n_layers=2)
        
        # Run VQE
        config = VQEConfig(
            optimizer='COBYLA',
            max_iterations=200,
            verbose=False
        )
        solver = VQESolver(config)
        result = solver.run(H, ansatz, backend='matrix')
        
        print(f"VQE energy: {result.energy:.6f}")
        print(f"Error: {abs(result.energy - exact_energy):.6f}")
        print(f"Iterations: {result.n_iterations}")
        
        print("\n2. Testing different optimizers:")
        print("-" * 50)
        
        for opt in ['COBYLA', 'Nelder-Mead', 'Powell']:
            config = VQEConfig(optimizer=opt, max_iterations=200, verbose=False)
            solver = VQESolver(config)
            result = solver.run(H, ansatz, backend='matrix')
            print(f"  {opt:15s}: E = {result.energy:.6f}, iterations = {result.n_iterations}")
        
        # Try H2 if PySCF is available
        try:
            import pyscf
            
            print("\n3. H2 molecule VQE:")
            print("-" * 50)
            
            result = run_vqe_h2(bond_length=0.74, optimizer='COBYLA', verbose=True)
            
        except ImportError:
            print("\n3. PySCF not available - skipping H2 demo")
            print("   Install with: pip install pyscf")
    
    print("\n" + "=" * 60)
    print("VQE Solver Demo Complete!")
    print("=" * 60)
