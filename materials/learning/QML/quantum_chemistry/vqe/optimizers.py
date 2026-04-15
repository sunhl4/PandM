"""
Optimizers for VQE
==================

This module provides various optimization algorithms for VQE:

1. Classical optimizers (scipy-based):
   - COBYLA: Gradient-free, good for noisy cost functions
   - L-BFGS-B: Quasi-Newton, efficient for smooth functions
   - Nelder-Mead: Simplex method, robust but slow

2. Quantum-aware optimizers:
   - SPSA: Stochastic gradient approximation, 2 evaluations per step
   - Quantum Natural Gradient: Uses Fisher information matrix

3. Gradient-based optimizers:
   - Adam: Adaptive learning rate
   - Gradient Descent: Simple baseline
"""

from __future__ import annotations
from typing import Callable, Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class OptimizerResult:
    """Result from optimizer."""
    x: np.ndarray  # Optimal parameters
    fun: float  # Optimal function value
    nfev: int  # Number of function evaluations
    nit: int  # Number of iterations
    success: bool
    message: str = ""


class BaseOptimizer:
    """Base class for optimizers."""
    
    def __init__(self, max_iterations: int = 1000, tol: float = 1e-6):
        self.max_iterations = max_iterations
        self.tol = tol
        self.history: List[float] = []
    
    def optimize(self, cost_fn: Callable, x0: np.ndarray) -> OptimizerResult:
        """Run optimization. Override in subclasses."""
        raise NotImplementedError


class COBYLA(BaseOptimizer):
    """
    COBYLA (Constrained Optimization BY Linear Approximation).
    
    A gradient-free optimizer that works well with noisy cost functions.
    Good choice for VQE on real quantum hardware.
    """
    
    def __init__(self, max_iterations: int = 1000, tol: float = 1e-6, rhobeg: float = 0.5):
        super().__init__(max_iterations, tol)
        self.rhobeg = rhobeg
    
    def optimize(self, cost_fn: Callable, x0: np.ndarray) -> OptimizerResult:
        from scipy.optimize import minimize
        
        self.history = []
        
        def tracked_cost(x):
            val = cost_fn(x)
            self.history.append(val)
            return val
        
        result = minimize(
            tracked_cost,
            x0,
            method='COBYLA',
            options={
                'maxiter': self.max_iterations,
                'rhobeg': self.rhobeg,
            }
        )
        
        return OptimizerResult(
            x=result.x,
            fun=result.fun,
            nfev=result.nfev,
            nit=len(self.history),
            success=result.success,
            message=result.message
        )


class Adam(BaseOptimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).
    
    Combines momentum and adaptive learning rates.
    Requires gradient computation.
    """
    
    def __init__(self, max_iterations: int = 1000, tol: float = 1e-6,
                 learning_rate: float = 0.01, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(max_iterations, tol)
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
    
    def optimize(self, cost_fn: Callable, x0: np.ndarray,
                 grad_fn: Optional[Callable] = None) -> OptimizerResult:
        """
        Run Adam optimization.
        
        Args:
            cost_fn: Cost function
            x0: Initial parameters
            grad_fn: Gradient function. If None, uses parameter shift.
        """
        x = x0.copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        
        self.history = []
        nfev = 0
        
        if grad_fn is None:
            grad_fn = lambda x: self._parameter_shift_gradient(cost_fn, x)
        
        for t in range(1, self.max_iterations + 1):
            # Compute gradient
            grad = grad_fn(x)
            nfev += 2 * len(x)  # Parameter shift uses 2 evaluations per param
            
            # Update moments
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * grad**2
            
            # Bias correction
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)
            
            # Update parameters
            x = x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
            # Track cost
            cost = cost_fn(x)
            nfev += 1
            self.history.append(cost)
            
            # Check convergence
            if len(self.history) > 1 and abs(self.history[-1] - self.history[-2]) < self.tol:
                return OptimizerResult(
                    x=x, fun=cost, nfev=nfev, nit=t, success=True,
                    message="Converged"
                )
        
        return OptimizerResult(
            x=x, fun=self.history[-1], nfev=nfev, nit=self.max_iterations,
            success=False, message="Max iterations reached"
        )
    
    def _parameter_shift_gradient(self, cost_fn: Callable, x: np.ndarray,
                                  shift: float = np.pi/2) -> np.ndarray:
        """Compute gradient using parameter shift rule."""
        grad = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += shift
            
            x_minus = x.copy()
            x_minus[i] -= shift
            
            grad[i] = (cost_fn(x_plus) - cost_fn(x_minus)) / (2 * np.sin(shift))
        
        return grad


class SPSA(BaseOptimizer):
    """
    SPSA (Simultaneous Perturbation Stochastic Approximation).
    
    A stochastic gradient approximation method that uses only 2 function
    evaluations per iteration regardless of parameter dimension.
    
    Excellent for noisy cost functions and high-dimensional problems.
    """
    
    def __init__(self, max_iterations: int = 1000, tol: float = 1e-6,
                 a: float = 0.1, c: float = 0.1, A: Optional[int] = None,
                 alpha: float = 0.602, gamma: float = 0.101):
        super().__init__(max_iterations, tol)
        self.a = a
        self.c = c
        self.A = A or max_iterations // 10
        self.alpha = alpha
        self.gamma = gamma
    
    def optimize(self, cost_fn: Callable, x0: np.ndarray) -> OptimizerResult:
        x = x0.copy()
        self.history = []
        nfev = 0
        
        for k in range(1, self.max_iterations + 1):
            # Decaying step sizes
            ak = self.a / (k + self.A)**self.alpha
            ck = self.c / k**self.gamma
            
            # Random perturbation (Bernoulli ±1)
            delta = 2 * np.random.randint(0, 2, size=len(x)) - 1
            
            # Perturbed evaluations
            x_plus = x + ck * delta
            x_minus = x - ck * delta
            
            y_plus = cost_fn(x_plus)
            y_minus = cost_fn(x_minus)
            nfev += 2
            
            # Gradient estimate
            g_hat = (y_plus - y_minus) / (2 * ck * delta)
            
            # Update
            x = x - ak * g_hat
            
            # Track cost
            cost = cost_fn(x)
            nfev += 1
            self.history.append(cost)
            
            # Check convergence
            if len(self.history) > 1 and abs(self.history[-1] - self.history[-2]) < self.tol:
                return OptimizerResult(
                    x=x, fun=cost, nfev=nfev, nit=k, success=True,
                    message="Converged"
                )
        
        return OptimizerResult(
            x=x, fun=self.history[-1], nfev=nfev, nit=self.max_iterations,
            success=False, message="Max iterations reached"
        )


class QuantumNaturalGradient(BaseOptimizer):
    """
    Quantum Natural Gradient optimizer.
    
    Uses the quantum Fisher information matrix (QFIM) to precondition
    the gradient, leading to faster convergence in many cases.
    
    The update rule is:
        θ_{t+1} = θ_t - η F^{-1}(θ_t) ∇E(θ_t)
    
    where F is the quantum Fisher information matrix.
    
    Note: Computing QFIM is expensive (O(n^2) circuit evaluations).
    """
    
    def __init__(self, max_iterations: int = 100, tol: float = 1e-6,
                 learning_rate: float = 0.01, regularization: float = 1e-4):
        super().__init__(max_iterations, tol)
        self.lr = learning_rate
        self.reg = regularization
    
    def optimize(self, cost_fn: Callable, x0: np.ndarray,
                 qfim_fn: Optional[Callable] = None,
                 grad_fn: Optional[Callable] = None) -> OptimizerResult:
        """
        Run QNG optimization.
        
        Args:
            cost_fn: Cost function
            x0: Initial parameters
            qfim_fn: Function to compute QFIM (if None, uses identity)
            grad_fn: Gradient function (if None, uses parameter shift)
        """
        x = x0.copy()
        self.history = []
        nfev = 0
        
        for k in range(1, self.max_iterations + 1):
            # Compute gradient
            if grad_fn is None:
                grad = self._parameter_shift_gradient(cost_fn, x)
                nfev += 2 * len(x)
            else:
                grad = grad_fn(x)
            
            # Compute QFIM (or use identity)
            if qfim_fn is not None:
                F = qfim_fn(x)
                # Regularize to ensure invertibility
                F_reg = F + self.reg * np.eye(len(x))
                # Natural gradient direction
                nat_grad = np.linalg.solve(F_reg, grad)
            else:
                nat_grad = grad
            
            # Update
            x = x - self.lr * nat_grad
            
            # Track cost
            cost = cost_fn(x)
            nfev += 1
            self.history.append(cost)
            
            # Check convergence
            if len(self.history) > 1 and abs(self.history[-1] - self.history[-2]) < self.tol:
                return OptimizerResult(
                    x=x, fun=cost, nfev=nfev, nit=k, success=True,
                    message="Converged"
                )
        
        return OptimizerResult(
            x=x, fun=self.history[-1], nfev=nfev, nit=self.max_iterations,
            success=False, message="Max iterations reached"
        )
    
    def _parameter_shift_gradient(self, cost_fn: Callable, x: np.ndarray,
                                  shift: float = np.pi/2) -> np.ndarray:
        """Compute gradient using parameter shift rule."""
        grad = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += shift
            
            x_minus = x.copy()
            x_minus[i] -= shift
            
            grad[i] = (cost_fn(x_plus) - cost_fn(x_minus)) / (2 * np.sin(shift))
        
        return grad


# ============================================================================
# Helper Functions
# ============================================================================

def compute_qfim_pennylane(circuit_fn: Callable, params: np.ndarray,
                           n_qubits: int) -> np.ndarray:
    """
    Compute the Quantum Fisher Information Matrix using PennyLane.
    
    The QFIM F_ij is defined as:
        F_ij = 4 Re(⟨∂_i ψ|∂_j ψ⟩ - ⟨∂_i ψ|ψ⟩⟨ψ|∂_j ψ⟩)
    
    Args:
        circuit_fn: Function that applies the parameterized circuit
        params: Current parameter values
        n_qubits: Number of qubits
    
    Returns:
        QFIM as numpy array
    """
    try:
        import pennylane as qml
    except ImportError:
        raise ImportError("PennyLane is required")
    
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev)
    def state_circuit(p):
        circuit_fn(p, wires=range(n_qubits))
        return qml.state()
    
    # Use PennyLane's metric tensor computation
    # This is a simplified approach
    n_params = len(params)
    F = np.zeros((n_params, n_params))
    
    shift = np.pi / 2
    
    for i in range(n_params):
        for j in range(i, n_params):
            # Compute F_ij using parameter shift
            # This is an approximation
            
            if i == j:
                # Diagonal elements
                p_plus = params.copy()
                p_plus[i] += shift
                p_minus = params.copy()
                p_minus[i] -= shift
                
                psi = state_circuit(params)
                psi_plus = state_circuit(p_plus)
                psi_minus = state_circuit(p_minus)
                
                # F_ii ≈ 1 - |⟨ψ|ψ+⟩|^2
                overlap_plus = abs(np.vdot(psi, psi_plus))**2
                overlap_minus = abs(np.vdot(psi, psi_minus))**2
                
                F[i, i] = 1 - 0.5 * (overlap_plus + overlap_minus)
            else:
                # Off-diagonal elements (simplified)
                F[i, j] = 0.0
                F[j, i] = 0.0
    
    return F


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VQE Optimizers Demo")
    print("=" * 60)
    
    # Test function: Rosenbrock
    def rosenbrock(x):
        return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    # Test function: Simple quadratic
    def quadratic(x):
        return np.sum(x**2)
    
    x0 = np.array([0.5, 0.5])
    
    print("\n1. Testing on quadratic function:")
    print("-" * 50)
    
    optimizers = [
        ("COBYLA", COBYLA(max_iterations=200)),
        ("Adam", Adam(max_iterations=200, learning_rate=0.1)),
        ("SPSA", SPSA(max_iterations=200, a=0.5)),
    ]
    
    for name, opt in optimizers:
        if name == "Adam":
            result = opt.optimize(quadratic, x0.copy(),
                                  grad_fn=lambda x: 2*x)
        else:
            result = opt.optimize(quadratic, x0.copy())
        
        print(f"  {name:10s}: x = {result.x}, f = {result.fun:.6f}, "
              f"nfev = {result.nfev}, nit = {result.nit}")
    
    print("\n2. Optimizer comparison on noisy function:")
    print("-" * 50)
    
    # Add noise to simulate quantum measurement
    noise_level = 0.1
    
    def noisy_quadratic(x):
        return quadratic(x) + noise_level * np.random.randn()
    
    for name, opt in optimizers[:2]:  # Skip Adam for noisy
        opt = COBYLA(max_iterations=100) if name == "COBYLA" else SPSA(max_iterations=100)
        result = opt.optimize(noisy_quadratic, x0.copy())
        print(f"  {name:10s}: f = {result.fun:.4f} ± noise, nit = {result.nit}")
    
    print("\n" + "=" * 60)
    print("Optimizers Demo Complete!")
    print("=" * 60)
