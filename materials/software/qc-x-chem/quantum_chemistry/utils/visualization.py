"""
Visualization Utilities for Quantum Chemistry
=============================================

Tools for visualizing:
- Energy convergence
- Potential energy surfaces
- Molecular structures
- Circuit diagrams
- Optimization landscapes
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np

# Check matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_energy_convergence(energy_history: List[float],
                           exact_energy: Optional[float] = None,
                           hf_energy: Optional[float] = None,
                           title: str = "VQE Energy Convergence",
                           save_path: Optional[str] = None) -> None:
    """
    Plot energy convergence during VQE optimization.
    
    Args:
        energy_history: List of energies at each iteration
        exact_energy: Exact ground state energy (reference line)
        hf_energy: Hartree-Fock energy (reference line)
        title: Plot title
        save_path: If provided, save figure to this path
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(1, len(energy_history) + 1)
    ax.plot(iterations, energy_history, 'b-', linewidth=2, label='VQE', marker='o', markersize=3)
    
    if exact_energy is not None:
        ax.axhline(y=exact_energy, color='r', linestyle='--', linewidth=2, label='Exact GS')
    
    if hf_energy is not None:
        ax.axhline(y=hf_energy, color='g', linestyle=':', linewidth=2, label='Hartree-Fock')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Energy (Hartree)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add error annotation if exact energy is known
    if exact_energy is not None and energy_history:
        final_error = abs(energy_history[-1] - exact_energy)
        ax.annotate(f'Final error: {final_error:.2e} Ha',
                   xy=(0.98, 0.02), xycoords='axes fraction',
                   ha='right', va='bottom', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_potential_energy_surface(bond_lengths: np.ndarray,
                                  energies: Dict[str, np.ndarray],
                                  title: str = "Potential Energy Surface",
                                  xlabel: str = "Bond Length (Å)",
                                  save_path: Optional[str] = None) -> None:
    """
    Plot potential energy surface comparison.
    
    Args:
        bond_lengths: Array of bond lengths
        energies: Dictionary of {method_name: energy_array}
        title: Plot title
        xlabel: X-axis label
        save_path: If provided, save figure to this path
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for plotting")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), height_ratios=[3, 1])
    
    # Main plot
    colors = plt.cm.tab10(np.linspace(0, 1, len(energies)))
    styles = ['-', '--', '-.', ':']
    
    for i, (method, E) in enumerate(energies.items()):
        style = styles[i % len(styles)]
        ax1.plot(bond_lengths, E, color=colors[i], linestyle=style,
                linewidth=2, label=method, marker='o', markersize=4)
    
    ax1.set_ylabel('Energy (Hartree)', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Error plot (relative to first method assumed to be reference)
    ref_method = list(energies.keys())[0]
    ref_E = energies[ref_method]
    
    for i, (method, E) in enumerate(energies.items()):
        if method == ref_method:
            continue
        error = (E - ref_E) * 1000  # Convert to mHa
        ax2.plot(bond_lengths, error, color=colors[i], linestyle=styles[i % len(styles)],
                linewidth=2, label=f'{method} - {ref_method}')
    
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel('Error (mHa)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_parameter_landscape(cost_fn, params: np.ndarray,
                            param_indices: Tuple[int, int] = (0, 1),
                            param_range: Tuple[float, float] = (-np.pi, np.pi),
                            n_points: int = 50,
                            title: str = "Energy Landscape",
                            save_path: Optional[str] = None) -> None:
    """
    Plot 2D energy landscape for two parameters.
    
    Args:
        cost_fn: Cost function that takes parameter array
        params: Current parameter values (other params held fixed)
        param_indices: Which two parameters to vary
        param_range: Range for parameter sweep
        n_points: Number of points along each axis
        title: Plot title
        save_path: If provided, save figure to this path
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for plotting")
        return
    
    i, j = param_indices
    
    theta_i = np.linspace(param_range[0], param_range[1], n_points)
    theta_j = np.linspace(param_range[0], param_range[1], n_points)
    
    Ti, Tj = np.meshgrid(theta_i, theta_j)
    E = np.zeros_like(Ti)
    
    print(f"Computing energy landscape ({n_points}x{n_points} points)...")
    for ii in range(n_points):
        for jj in range(n_points):
            test_params = params.copy()
            test_params[i] = Ti[ii, jj]
            test_params[j] = Tj[ii, jj]
            E[ii, jj] = cost_fn(test_params)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    contour = ax.contourf(Ti, Tj, E, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Energy (Ha)')
    
    # Mark current parameters
    ax.scatter([params[i]], [params[j]], color='r', s=100, marker='*',
              label=f'Current: E={cost_fn(params):.4f}')
    
    ax.set_xlabel(f'θ_{i}', fontsize=12)
    ax.set_ylabel(f'θ_{j}', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_molecule(atoms: List[Tuple[str, Tuple[float, float, float]]],
                       title: str = "Molecular Structure",
                       save_path: Optional[str] = None) -> None:
    """
    Simple 3D visualization of molecular structure.
    
    Args:
        atoms: List of (element_symbol, (x, y, z)) tuples
        title: Plot title
        save_path: If provided, save figure to this path
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for plotting")
        return
    
    from mpl_toolkits.mplot3d import Axes3D
    
    # Element colors and sizes
    element_colors = {
        'H': 'white', 'He': 'cyan',
        'Li': 'violet', 'Be': 'darkgreen', 'B': 'salmon', 'C': 'gray',
        'N': 'blue', 'O': 'red', 'F': 'green', 'Ne': 'cyan',
        'Na': 'violet', 'Mg': 'darkgreen', 'Al': 'silver', 'Si': 'tan',
        'P': 'orange', 'S': 'yellow', 'Cl': 'green', 'Ar': 'cyan',
    }
    element_sizes = {
        'H': 100, 'He': 80,
        'Li': 200, 'Be': 180, 'B': 160, 'C': 150,
        'N': 145, 'O': 140, 'F': 135, 'Ne': 130,
    }
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for element, (x, y, z) in atoms:
        color = element_colors.get(element, 'gray')
        size = element_sizes.get(element, 100)
        ax.scatter([x], [y], [z], c=color, s=size, edgecolors='black', linewidth=1)
        ax.text(x, y, z + 0.1, element, fontsize=10, ha='center')
    
    # Draw bonds (simple distance-based)
    coords = np.array([pos for _, pos in atoms])
    n_atoms = len(atoms)
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < 2.0:  # Bond threshold in Å
                ax.plot([coords[i, 0], coords[j, 0]],
                       [coords[i, 1], coords[j, 1]],
                       [coords[i, 2], coords[j, 2]], 'k-', linewidth=2)
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_ansatz_comparison(results: Dict[str, Dict],
                          title: str = "Ansatz Comparison",
                          save_path: Optional[str] = None) -> None:
    """
    Compare different ansätze performance.
    
    Args:
        results: Dict of {ansatz_name: {'energy': float, 'params': int, 'depth': int, ...}}
        title: Plot title
        save_path: If provided, save figure to this path
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for plotting")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = list(results.keys())
    energies = [results[n].get('energy', 0) for n in names]
    n_params = [results[n].get('params', 0) for n in names]
    depths = [results[n].get('depth', 0) for n in names]
    
    # Energy comparison
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    axes[0].bar(names, energies, color=colors)
    axes[0].set_ylabel('Energy (Ha)')
    axes[0].set_title('Final Energy')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Parameter count
    axes[1].bar(names, n_params, color=colors)
    axes[1].set_ylabel('Number of Parameters')
    axes[1].set_title('Parameter Count')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Circuit depth
    axes[2].bar(names, depths, color=colors)
    axes[2].set_ylabel('Circuit Depth')
    axes[2].set_title('Circuit Depth')
    axes[2].tick_params(axis='x', rotation=45)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Visualization Utilities Demo")
    print("=" * 60)
    
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Install with: pip install matplotlib")
    else:
        # Demo: Energy convergence
        print("\n1. Energy Convergence Plot:")
        np.random.seed(42)
        energy_history = -1.0 - 0.1 * (1 - np.exp(-np.arange(50) / 10))
        energy_history += 0.01 * np.random.randn(50)
        
        plot_energy_convergence(
            energy_history,
            exact_energy=-1.1,
            hf_energy=-0.9,
            title="Demo VQE Convergence"
        )
        
        # Demo: PES
        print("\n2. Potential Energy Surface:")
        r = np.linspace(0.5, 2.5, 20)
        E_hf = -1.0 + 0.5 * (r - 0.74)**2
        E_vqe = -1.1 + 0.4 * (r - 0.74)**2
        E_fci = -1.12 + 0.4 * (r - 0.74)**2
        
        plot_potential_energy_surface(
            r,
            {'FCI': E_fci, 'VQE': E_vqe, 'HF': E_hf},
            title="Demo H2 PES"
        )
        
        # Demo: Molecule
        print("\n3. Molecule Visualization:")
        h2o = [
            ('O', (0.0, 0.0, 0.0)),
            ('H', (0.96, 0.0, 0.0)),
            ('H', (-0.24, 0.93, 0.0))
        ]
        visualize_molecule(h2o, title="H2O Molecule")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
