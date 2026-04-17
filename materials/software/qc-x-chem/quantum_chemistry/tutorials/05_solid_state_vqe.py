#!/usr/bin/env python3
"""
VQE for Solid State Materials
=============================

This tutorial covers extending VQE to solid-state systems:

1. Challenges with periodic systems
2. Embedding methods (DMET)
3. Simple model Hamiltonians (Hubbard)
4. Towards real materials simulations

Key differences from molecular systems:
- Infinite periodicity → k-space sampling
- Many electrons → active space essential
- Strong correlation in materials

Reference methods:
- DMET (Density Matrix Embedding Theory)
- VQE-DMET hybrid approaches
- Periodic VQE with plane waves
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict


# ============================================================================
# Part 1: Hubbard Model - Prototype for Strongly Correlated Systems
# ============================================================================

def build_hubbard_hamiltonian(n_sites: int, t: float = 1.0, U: float = 4.0,
                              periodic: bool = True) -> Tuple[np.ndarray, int]:
    """
    Build the Hubbard model Hamiltonian.
    
    H = -t Σ_{<ij>,σ} (a†_iσ a_jσ + h.c.) + U Σ_i n_i↑ n_i↓
    
    This is a fundamental model for strongly correlated electrons in materials.
    
    Args:
        n_sites: Number of lattice sites
        t: Hopping parameter
        U: On-site Coulomb repulsion
        periodic: Use periodic boundary conditions
    
    Returns:
        (Hamiltonian matrix, number of spin orbitals)
    """
    n_spin_orbitals = 2 * n_sites  # spin up and spin down
    dim = 2 ** n_spin_orbitals
    
    H = np.zeros((dim, dim), dtype=complex)
    
    # Helper: occupation of spin orbital j in basis state i
    def occupation(state: int, orbital: int) -> int:
        return (state >> orbital) & 1
    
    # Hopping terms: -t Σ (a†_i a_j + h.c.)
    for i in range(n_sites):
        j = (i + 1) % n_sites if periodic else (i + 1 if i + 1 < n_sites else -1)
        
        if j < 0:
            continue
        
        for spin in [0, 1]:  # 0 = up, 1 = down
            # Spin orbital indices
            orb_i = 2 * i + spin
            orb_j = 2 * j + spin
            
            for state in range(dim):
                # a†_i a_j: destroy at j, create at i
                if occupation(state, orb_j) == 1 and occupation(state, orb_i) == 0:
                    # Count sign from fermion anticommutation
                    new_state = state ^ (1 << orb_j)  # Remove electron at j
                    
                    # Count electrons between orb_i and orb_j for sign
                    n_between = 0
                    for k in range(min(orb_i, orb_j) + 1, max(orb_i, orb_j)):
                        n_between += occupation(new_state, k)
                    sign = (-1) ** n_between
                    
                    new_state = new_state ^ (1 << orb_i)  # Add electron at i
                    
                    H[new_state, state] += -t * sign
                
                # Hermitian conjugate: a†_j a_i
                if occupation(state, orb_i) == 1 and occupation(state, orb_j) == 0:
                    new_state = state ^ (1 << orb_i)
                    
                    n_between = 0
                    for k in range(min(orb_i, orb_j) + 1, max(orb_i, orb_j)):
                        n_between += occupation(new_state, k)
                    sign = (-1) ** n_between
                    
                    new_state = new_state ^ (1 << orb_j)
                    
                    H[new_state, state] += -t * sign
    
    # On-site repulsion: U Σ_i n_i↑ n_i↓
    for i in range(n_sites):
        orb_up = 2 * i
        orb_down = 2 * i + 1
        
        for state in range(dim):
            if occupation(state, orb_up) == 1 and occupation(state, orb_down) == 1:
                H[state, state] += U
    
    return H, n_spin_orbitals


def analyze_hubbard_spectrum(H: np.ndarray, n_sites: int) -> Dict:
    """Analyze the Hubbard model spectrum."""
    eigenvalues = np.linalg.eigvalsh(H)
    
    return {
        'ground_state': eigenvalues[0],
        'first_excited': eigenvalues[1],
        'gap': eigenvalues[1] - eigenvalues[0],
        'bandwidth': eigenvalues[-1] - eigenvalues[0],
        'n_states': len(eigenvalues),
    }


# ============================================================================
# Part 2: VQE for Hubbard Model
# ============================================================================

def run_hubbard_vqe(n_sites: int = 2, t: float = 1.0, U: float = 4.0,
                    verbose: bool = True) -> Dict:
    """
    Run VQE on the Hubbard model.
    
    For small systems, demonstrates VQE on a strongly correlated model.
    """
    try:
        import pennylane as qml
        from scipy.optimize import minimize
    except ImportError:
        print("PennyLane required. Install with: pip install pennylane")
        return {}
    
    if verbose:
        print("=" * 60)
        print(f"VQE for {n_sites}-site Hubbard Model (t={t}, U={U})")
        print("=" * 60)
    
    # Build Hamiltonian
    H, n_qubits = build_hubbard_hamiltonian(n_sites, t, U)
    
    if verbose:
        print(f"\nSystem: {n_sites} sites, {n_qubits} qubits")
        print(f"Hilbert space dimension: {H.shape[0]}")
    
    # Exact solution
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    exact_gs = eigenvalues[0]
    
    if verbose:
        print(f"Exact ground state: {exact_gs:.6f}")
        print(f"Gap to first excited: {eigenvalues[1] - eigenvalues[0]:.6f}")
    
    # VQE setup
    dev = qml.device('default.qubit', wires=n_qubits)
    
    # Hardware-efficient ansatz for demonstration
    n_layers = 2
    n_params = n_qubits * 2 * n_layers  # RY, RZ per qubit per layer
    
    @qml.qnode(dev)
    def circuit(params):
        # Initial state: half-filled (alternating up/down)
        for i in range(n_sites):
            qml.PauliX(wires=2*i)  # Put one electron per site (spin up)
        
        # Variational layers
        param_idx = 0
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(params[param_idx], wires=q)
                param_idx += 1
                qml.RZ(params[param_idx], wires=q)
                param_idx += 1
            
            # Entangling
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q+1])
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits-1, 0])
        
        return qml.expval(qml.Hermitian(H, wires=range(n_qubits)))
    
    # Optimize
    if verbose:
        print(f"\nRunning VQE with {n_params} parameters...")
    
    energy_history = []
    
    def cost_fn(params):
        E = float(circuit(params))
        energy_history.append(E)
        return E
    
    params0 = np.random.uniform(-0.1, 0.1, n_params)
    result = minimize(cost_fn, params0, method='COBYLA', options={'maxiter': 300})
    
    vqe_energy = result.fun
    
    if verbose:
        print(f"\nResults:")
        print(f"  VQE energy: {vqe_energy:.6f}")
        print(f"  Exact energy: {exact_gs:.6f}")
        print(f"  Error: {abs(vqe_energy - exact_gs):.6f}")
        print(f"  Iterations: {len(energy_history)}")
    
    return {
        'vqe_energy': vqe_energy,
        'exact_energy': exact_gs,
        'error': abs(vqe_energy - exact_gs),
        'energy_history': energy_history,
        'n_sites': n_sites,
        't': t,
        'U': U,
    }


# ============================================================================
# Part 3: DMET (Density Matrix Embedding Theory) Concept
# ============================================================================

def explain_dmet():
    """
    Explain DMET (Density Matrix Embedding Theory) concept.
    
    DMET is an embedding method that combines:
    - Mean-field treatment for environment
    - High-accuracy solver (like VQE) for impurity/fragment
    """
    print("""
DMET (Density Matrix Embedding Theory) for VQE
==============================================

Idea: Divide and Conquer for Large Systems
------------------------------------------

1. Fragment the system:
   - Choose an "impurity" (fragment of interest)
   - Treat the rest as "environment" (bath)

2. Mean-field approximation:
   - Solve whole system at mean-field level (HF/DFT)
   - Use to construct bath orbitals

3. Embedding Hamiltonian:
   - Project onto impurity + bath subspace
   - Much smaller than full system!

4. High-accuracy solver:
   - Use VQE/FCI on the embedded problem
   - Gets correlation energy for impurity

5. Self-consistency:
   - Match density matrices between embedding and mean-field
   - Iterate until converged


VQE-DMET Workflow:
------------------

                     Full System (many electrons)
                              |
                      Mean-field (HF/DFT)
                              |
                     Bath orbital construction
                              |
    +-------------------+-------------------+
    |                   |                   |
 Fragment 1        Fragment 2        Fragment 3 ...
    |                   |                   |
  VQE (small!)       VQE (small!)       VQE (small!)
    |                   |                   |
    +-------------------+-------------------+
                              |
                       Combine results
                              |
                    Total energy estimate


Advantages:
-----------
- Reduces qubit requirements dramatically
- Captures local correlation effects
- Systematic improvability
- Compatible with existing VQE implementations

Challenges:
-----------
- Bath construction can be tricky
- Self-consistency loop needed
- Edge effects at fragment boundaries

For a 100-atom material:
- Direct VQE: 200+ qubits (impossible now)
- DMET + VQE: ~10-20 qubits per fragment (feasible!)
""")


# ============================================================================
# Part 4: Periodic Systems and k-space
# ============================================================================

def explain_periodic_vqe():
    """Explain approaches for periodic systems."""
    print("""
VQE for Periodic Systems
========================

Challenge: Infinite Systems
---------------------------
Real materials are periodic → infinite number of atoms
Cannot directly map to finite quantum computer!

Approaches:
-----------

1. Finite Cluster + Periodic Boundary Conditions
   - Treat small cluster with PBC
   - Size extrapolation to infinite limit
   - Simple but limited accuracy

2. k-space Sampling
   - Bloch's theorem: ψ_k(r+R) = e^{ik·R} ψ_k(r)
   - Sample Brillouin zone at k-points
   - Each k-point is independent calculation
   
   Workflow:
   - Generate k-point mesh
   - Run VQE at each k-point
   - Integrate over k-space

3. Wannier Function Basis
   - Localized basis from Bloch states
   - Can truncate to finite range
   - Natural for embedding methods

4. Hybrid Classical-Quantum
   - Classical DFT/HF for most of system
   - VQE for correlated impurity
   - DMET, DMFT frameworks

Example: Band Gap Calculation
-----------------------------

For a semiconductor/insulator:

1. Run DFT to get band structure
2. Identify valence and conduction bands
3. Use VQE for correlation at band edges
4. Correct DFT gap with VQE correlation

Required Resources:
-------------------

Material    | Naive Qubits | With DMET | Status
------------|--------------|-----------|--------
H2          | 4            | 4         | Easy
Graphene    | ~100         | ~10-20    | Challenging
Silicon     | ~200         | ~20-30    | Research
Transition  | ~500         | ~30-50    | Future
metals
""")


# ============================================================================
# Part 5: Simple Solid State Example
# ============================================================================

def hydrogen_chain_pes(n_atoms: int = 4, n_points: int = 10) -> Dict:
    """
    Compute potential energy surface for a hydrogen chain.
    
    This is a simple 1D model system relevant for understanding
    correlation in extended systems.
    """
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
    except ImportError:
        print("PennyLane required")
        return {}
    
    print("=" * 60)
    print(f"Hydrogen Chain: {n_atoms} atoms")
    print("=" * 60)
    
    # This would require PySCF for real calculation
    # Here we show the conceptual workflow
    
    print("""
Hydrogen Chain Analysis (Conceptual)
------------------------------------

The hydrogen chain (H_n) is important because:
1. Simplest extended system
2. Shows metal-insulator transition
3. Benchmark for correlation methods

At equilibrium spacing (~0.74 Å):
- Metallic (delocalized electrons)
- Weakly correlated

At large spacing (> 2 Å):
- Insulating (localized electrons)
- Strongly correlated
- HF fails dramatically

VQE Approach:
1. Define chain geometry at various spacings
2. Compute molecular integrals (PySCF)
3. Map to qubits (Jordan-Wigner)
4. Run VQE with UCCSD ansatz
5. Compare with FCI/DMRG

Expected Results:
- VQE captures correlation at all distances
- HF only good at small distance
- Important for materials with localized electrons
""")
    
    return {'n_atoms': n_atoms, 'status': 'conceptual_demo'}


# ============================================================================
# Main Tutorial
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" VQE for Solid State Materials")
    print(" From Model Hamiltonians to Real Materials")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("Part 1: Hubbard Model")
    print("=" * 70)
    
    # Hubbard model analysis
    for n_sites in [2, 3]:
        for U_ratio in [2.0, 4.0, 8.0]:
            H, n_qubits = build_hubbard_hamiltonian(n_sites, t=1.0, U=U_ratio)
            analysis = analyze_hubbard_spectrum(H, n_sites)
            print(f"\n{n_sites} sites, U/t = {U_ratio:.1f}:")
            print(f"  Ground state: {analysis['ground_state']:.4f}")
            print(f"  Gap: {analysis['gap']:.4f}")
    
    print("\n" + "=" * 70)
    print("Part 2: VQE on Hubbard Model")
    print("=" * 70)
    
    try:
        result = run_hubbard_vqe(n_sites=2, t=1.0, U=4.0)
    except Exception as e:
        print(f"VQE failed: {e}")
    
    print("\n" + "=" * 70)
    print("Part 3: DMET Concept")
    print("=" * 70)
    
    explain_dmet()
    
    print("\n" + "=" * 70)
    print("Part 4: Periodic Systems")
    print("=" * 70)
    
    explain_periodic_vqe()
    
    print("\n" + "=" * 70)
    print("Part 5: Hydrogen Chain")
    print("=" * 70)
    
    hydrogen_chain_pes()
    
    print("\n" + "=" * 70)
    print(" Tutorial Complete!")
    print("=" * 70)
    print("""
Key Takeaways:
--------------
1. Hubbard model captures essential physics of correlation
2. VQE works well on small model systems
3. DMET enables treating larger systems
4. Periodic systems require k-space or embedding approaches
5. Real materials calculations require hybrid methods

Next Steps:
-----------
1. Implement full DMET-VQE workflow
2. Study larger Hubbard clusters
3. Try hydrogen chain with real integrals
4. Explore k-space VQE implementations
""")
