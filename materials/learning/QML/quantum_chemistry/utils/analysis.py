"""
Analysis Utilities for Quantum Chemistry
========================================

Tools for analyzing:
- VQE results
- Circuit properties
- Hamiltonian structure
- Quantum state fidelity
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


def compute_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Compute fidelity between two quantum states.
    
    F = |⟨ψ1|ψ2⟩|²
    
    Args:
        state1, state2: State vectors (normalized)
    
    Returns:
        Fidelity in [0, 1]
    """
    state1 = np.asarray(state1)
    state2 = np.asarray(state2)
    
    overlap = np.abs(np.vdot(state1, state2))**2
    return float(overlap)


def compare_with_exact(vqe_energy: float, exact_energy: float,
                       hf_energy: Optional[float] = None) -> Dict[str, float]:
    """
    Compare VQE result with exact solution.
    
    Args:
        vqe_energy: VQE optimized energy
        exact_energy: Exact ground state energy
        hf_energy: Hartree-Fock energy (optional)
    
    Returns:
        Dictionary with error metrics
    """
    results = {
        'vqe_energy': vqe_energy,
        'exact_energy': exact_energy,
        'absolute_error': abs(vqe_energy - exact_energy),
        'error_mhartree': abs(vqe_energy - exact_energy) * 1000,
        'relative_error': abs((vqe_energy - exact_energy) / exact_energy) if exact_energy != 0 else float('inf'),
    }
    
    if hf_energy is not None:
        correlation_energy = exact_energy - hf_energy
        captured_correlation = hf_energy - vqe_energy
        
        results['hf_energy'] = hf_energy
        results['correlation_energy'] = correlation_energy
        results['captured_correlation'] = captured_correlation
        results['correlation_recovery'] = captured_correlation / correlation_energy if correlation_energy != 0 else 0
    
    return results


def analyze_circuit_depth(circuit, backend: str = 'pennylane') -> Dict[str, int]:
    """
    Analyze circuit depth and gate counts.
    
    Args:
        circuit: Quantum circuit object
        backend: 'pennylane', 'qiskit', or 'cirq'
    
    Returns:
        Dictionary with circuit statistics
    """
    if backend == 'pennylane':
        try:
            import pennylane as qml
            
            # Get circuit resources
            specs = qml.specs(circuit)
            
            return {
                'depth': specs.get('depth', 0),
                'num_operations': specs.get('num_operations', 0),
                'num_parameters': specs.get('num_trainable_params', 0),
            }
        except Exception as e:
            return {'error': str(e)}
    
    elif backend == 'qiskit':
        try:
            return {
                'depth': circuit.depth(),
                'num_operations': circuit.size(),
                'num_parameters': circuit.num_parameters,
            }
        except Exception as e:
            return {'error': str(e)}
    
    else:
        return {'error': f'Unknown backend: {backend}'}


def analyze_hamiltonian(hamiltonian, n_qubits: Optional[int] = None) -> Dict[str, Any]:
    """
    Analyze properties of a qubit Hamiltonian.
    
    Args:
        hamiltonian: Hamiltonian (matrix or QubitOperator)
        n_qubits: Number of qubits
    
    Returns:
        Dictionary with Hamiltonian properties
    """
    # Convert to matrix if needed
    if hasattr(hamiltonian, 'to_matrix'):
        if n_qubits is None:
            n_qubits = hamiltonian.get_n_qubits()
        H = hamiltonian.to_matrix(n_qubits)
    elif isinstance(hamiltonian, np.ndarray):
        H = hamiltonian
        if n_qubits is None:
            n_qubits = int(np.log2(H.shape[0]))
    else:
        return {'error': f'Unknown hamiltonian type: {type(hamiltonian)}'}
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(H)
    
    # Analyze spectrum
    results = {
        'n_qubits': n_qubits,
        'matrix_dimension': H.shape[0],
        'ground_state_energy': float(eigenvalues[0]),
        'first_excited_energy': float(eigenvalues[1]) if len(eigenvalues) > 1 else None,
        'spectral_gap': float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else None,
        'spectral_range': float(eigenvalues[-1] - eigenvalues[0]),
        'frobenius_norm': float(np.linalg.norm(H, 'fro')),
        'is_hermitian': np.allclose(H, H.conj().T),
    }
    
    # Count non-zero elements
    nonzero = np.count_nonzero(np.abs(H) > 1e-10)
    total = H.shape[0] * H.shape[1]
    results['sparsity'] = 1 - nonzero / total
    
    return results


def chemical_accuracy_check(vqe_energy: float, exact_energy: float,
                           threshold_hartree: float = 0.0016) -> Dict[str, Any]:
    """
    Check if VQE result achieves chemical accuracy.
    
    Chemical accuracy is typically defined as 1 kcal/mol ≈ 1.6 mHa.
    
    Args:
        vqe_energy: VQE energy
        exact_energy: Exact (FCI) energy
        threshold_hartree: Accuracy threshold in Hartree (default: ~1 kcal/mol)
    
    Returns:
        Dictionary with accuracy analysis
    """
    error = abs(vqe_energy - exact_energy)
    error_mha = error * 1000
    error_kcal_mol = error * 627.5  # Hartree to kcal/mol
    
    return {
        'error_hartree': error,
        'error_mhartree': error_mha,
        'error_kcal_mol': error_kcal_mol,
        'threshold_hartree': threshold_hartree,
        'achieves_chemical_accuracy': error < threshold_hartree,
        'accuracy_ratio': threshold_hartree / error if error > 0 else float('inf'),
    }


def analyze_convergence(energy_history: List[float],
                       window_size: int = 10) -> Dict[str, Any]:
    """
    Analyze VQE convergence behavior.
    
    Args:
        energy_history: List of energies during optimization
        window_size: Window for computing rolling statistics
    
    Returns:
        Dictionary with convergence analysis
    """
    if len(energy_history) < 2:
        return {'error': 'Need at least 2 points for analysis'}
    
    E = np.array(energy_history)
    
    results = {
        'n_iterations': len(E),
        'initial_energy': float(E[0]),
        'final_energy': float(E[-1]),
        'total_improvement': float(E[0] - E[-1]),
        'min_energy': float(np.min(E)),
        'min_energy_iteration': int(np.argmin(E)),
    }
    
    # Compute gradients (energy changes)
    dE = np.diff(E)
    results['mean_energy_change'] = float(np.mean(dE))
    results['std_energy_change'] = float(np.std(dE))
    
    # Check for monotonic decrease
    results['is_monotonic'] = bool(np.all(dE <= 0))
    results['n_increases'] = int(np.sum(dE > 0))
    
    # Rolling average (final window)
    if len(E) >= window_size:
        final_window = E[-window_size:]
        results['final_window_mean'] = float(np.mean(final_window))
        results['final_window_std'] = float(np.std(final_window))
        results['is_converged'] = results['final_window_std'] < 1e-6
    
    return results


def parameter_sensitivity(cost_fn, params: np.ndarray,
                         perturbation: float = 0.01) -> np.ndarray:
    """
    Compute sensitivity of cost to each parameter.
    
    Uses finite difference to estimate |∂E/∂θ_i|.
    
    Args:
        cost_fn: Cost function
        params: Current parameters
        perturbation: Size of perturbation
    
    Returns:
        Array of sensitivities for each parameter
    """
    sensitivities = np.zeros(len(params))
    
    E0 = cost_fn(params)
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += perturbation
        
        params_minus = params.copy()
        params_minus[i] -= perturbation
        
        E_plus = cost_fn(params_plus)
        E_minus = cost_fn(params_minus)
        
        # Gradient magnitude
        sensitivities[i] = abs(E_plus - E_minus) / (2 * perturbation)
    
    return sensitivities


def barren_plateau_check(cost_fn, n_params: int,
                        n_samples: int = 100,
                        param_range: Tuple[float, float] = (0, 2*np.pi)) -> Dict[str, Any]:
    """
    Check for barren plateau by analyzing gradient variance.
    
    In barren plateaus, gradient variance vanishes exponentially with system size.
    
    Args:
        cost_fn: Cost function
        n_params: Number of parameters
        n_samples: Number of random parameter samples
        param_range: Range for random parameters
    
    Returns:
        Dictionary with barren plateau analysis
    """
    gradients = []
    
    for _ in range(n_samples):
        # Random parameters
        params = np.random.uniform(param_range[0], param_range[1], n_params)
        
        # Compute gradient (using parameter shift)
        grad = np.zeros(n_params)
        shift = np.pi / 2
        
        for i in range(n_params):
            params_plus = params.copy()
            params_plus[i] += shift
            
            params_minus = params.copy()
            params_minus[i] -= shift
            
            grad[i] = (cost_fn(params_plus) - cost_fn(params_minus)) / 2
        
        gradients.append(grad)
    
    gradients = np.array(gradients)
    
    # Compute statistics
    mean_grad = np.mean(gradients, axis=0)
    var_grad = np.var(gradients, axis=0)
    
    results = {
        'n_samples': n_samples,
        'mean_gradient': mean_grad.tolist(),
        'variance_gradient': var_grad.tolist(),
        'avg_variance': float(np.mean(var_grad)),
        'max_variance': float(np.max(var_grad)),
        'min_variance': float(np.min(var_grad)),
    }
    
    # Warning if variance is very small
    results['barren_plateau_warning'] = np.mean(var_grad) < 1e-6
    
    return results


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Analysis Utilities Demo")
    print("=" * 60)
    
    print("\n1. Compare with Exact:")
    print("-" * 50)
    
    comparison = compare_with_exact(
        vqe_energy=-1.136,
        exact_energy=-1.137,
        hf_energy=-1.117
    )
    for key, value in comparison.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\n2. Chemical Accuracy Check:")
    print("-" * 50)
    
    accuracy = chemical_accuracy_check(-1.136, -1.137)
    for key, value in accuracy.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n3. Convergence Analysis:")
    print("-" * 50)
    
    # Simulate convergence
    np.random.seed(42)
    energy_history = -1.0 - 0.1 * (1 - np.exp(-np.arange(50) / 10))
    energy_history += 0.002 * np.random.randn(50)
    
    conv_analysis = analyze_convergence(energy_history)
    for key, value in conv_analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n4. Hamiltonian Analysis:")
    print("-" * 50)
    
    # Simple 2-qubit Hamiltonian
    H = np.array([
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0, -1.0]
    ])
    
    h_analysis = analyze_hamiltonian(H)
    for key, value in h_analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
