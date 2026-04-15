#!/usr/bin/env python3
"""
PennyLane Quantum Chemistry Tutorial
=====================================

PennyLane's qchem module provides a seamless integration of quantum chemistry
with quantum machine learning. It offers:

- Molecular Hamiltonian construction
- Built-in ansätze (UCCSD, AllSinglesDoubles)
- Automatic differentiation of quantum circuits
- Integration with multiple backends (default.qubit, qiskit, cirq, etc.)

This tutorial covers:
1. Building molecular Hamiltonians
2. Using built-in excitation operators
3. VQE with automatic differentiation
4. Advanced features (active space, dipole moments)

Installation:
    pip install pennylane pennylane-qchem

Reference: https://pennylane.ai/qml/demos_quantum_chemistry.html
"""

from __future__ import annotations
import numpy as np

# Check PennyLane availability
try:
    import pennylane as qml
    from pennylane import numpy as pnp  # For automatic differentiation
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("PennyLane not installed. Install with: pip install pennylane")

# Check if qchem is available (needs PySCF or other backend)
QCHEM_AVAILABLE = False
if PENNYLANE_AVAILABLE:
    try:
        from pennylane import qchem
        # Test if PySCF is available
        import pyscf
        QCHEM_AVAILABLE = True
    except ImportError:
        print("PennyLane qchem or PySCF not available.")
        print("Install with: pip install pennylane pyscf")


def tutorial_1_molecular_hamiltonian():
    """
    Tutorial 1: Building Molecular Hamiltonians with PennyLane
    """
    print("=" * 60)
    print("Tutorial 1: Molecular Hamiltonians in PennyLane")
    print("=" * 60)
    
    if not PENNYLANE_AVAILABLE:
        print("PennyLane not available. Skipping tutorial.")
        return None, None, None
    
    if not QCHEM_AVAILABLE:
        print("PennyLane qchem not available. Using example Hamiltonian.")
        return create_example_hamiltonian()
    
    print("\n1.1 Define Molecule:")
    print("-" * 50)
    
    # Define H2 molecule
    symbols = ['H', 'H']
    coordinates = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.74]])  # Angstroms
    
    print(f"Molecule: H2")
    print(f"Symbols: {symbols}")
    print(f"Coordinates (Å):\n{coordinates}")
    
    print("\n1.2 Build Hamiltonian:")
    print("-" * 50)
    
    # Build molecular Hamiltonian
    H, n_qubits = qchem.molecular_hamiltonian(
        symbols=symbols,
        coordinates=coordinates,
        basis='sto-3g',
        method='pyscf',  # or 'openfermion'
        active_electrons=2,
        active_orbitals=2
    )
    
    print(f"Number of qubits: {n_qubits}")
    print(f"Number of Pauli terms: {len(H.ops)}")
    
    print("\n1.3 Inspect Hamiltonian:")
    print("-" * 50)
    
    # Print first few terms
    print("First 10 Pauli terms:")
    for i, (coef, op) in enumerate(zip(H.coeffs[:10], H.ops[:10])):
        print(f"  {coef.real:+.6f} * {op}")
    
    if len(H.ops) > 10:
        print(f"  ... and {len(H.ops) - 10} more terms")
    
    # Get exact energy by matrix diagonalization
    print("\n1.4 Exact Diagonalization:")
    print("-" * 50)
    
    H_matrix = qml.matrix(H, wire_order=range(n_qubits))
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    exact_gs = eigenvalues[0]
    
    print(f"Matrix dimension: {H_matrix.shape}")
    print(f"Exact ground state energy: {exact_gs:.8f} Ha")
    
    return H, n_qubits, exact_gs


def create_example_hamiltonian():
    """Create a simple example Hamiltonian when qchem is not available."""
    print("\n1.1 Creating Example H2 Hamiltonian:")
    print("-" * 50)
    
    # Coefficients from a simplified H2 Hamiltonian
    coeffs = [
        -0.04207897,  # Identity
        0.17771287,   # Z0
        0.17771287,   # Z1
        -0.24274281,  # Z2
        -0.24274281,  # Z3
        0.17059738,   # Z0 Z1
        0.04475014,   # Z0 Z2
        0.04475014,   # Z0 Z3
        0.04475014,   # Z1 Z2
        0.04475014,   # Z1 Z3
        0.12293305,   # Z2 Z3
        0.04475014,   # X0 X1 Y2 Y3
        -0.04475014,  # X0 Y1 Y2 X3
        -0.04475014,  # Y0 X1 X2 Y3
        0.04475014,   # Y0 Y1 X2 X3
    ]
    
    obs = [
        qml.Identity(0),
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliZ(2),
        qml.PauliZ(3),
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliZ(0) @ qml.PauliZ(2),
        qml.PauliZ(0) @ qml.PauliZ(3),
        qml.PauliZ(1) @ qml.PauliZ(2),
        qml.PauliZ(1) @ qml.PauliZ(3),
        qml.PauliZ(2) @ qml.PauliZ(3),
        qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliY(3),
        qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliX(3),
        qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliY(3),
        qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliX(3),
    ]
    
    H = qml.Hamiltonian(coeffs, obs)
    n_qubits = 4
    
    print(f"Number of qubits: {n_qubits}")
    print(f"Number of Pauli terms: {len(coeffs)}")
    
    # Exact diagonalization
    H_matrix = qml.matrix(H, wire_order=range(n_qubits))
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    exact_gs = eigenvalues[0]
    
    print(f"Exact ground state energy: {exact_gs:.8f} Ha")
    
    return H, n_qubits, exact_gs


def tutorial_2_excitation_operators():
    """
    Tutorial 2: Excitation Operators and Ansätze
    """
    print("\n" + "=" * 60)
    print("Tutorial 2: Excitation Operators in PennyLane")
    print("=" * 60)
    
    if not PENNYLANE_AVAILABLE:
        print("PennyLane not available. Skipping tutorial.")
        return
    
    print("\n2.1 Single and Double Excitations:")
    print("-" * 50)
    
    n_qubits = 4
    dev = qml.device('default.qubit', wires=n_qubits)
    
    # Single excitation: |0⟩ -> |1⟩ on one electron
    @qml.qnode(dev)
    def single_excitation_demo(theta):
        # Prepare |0011⟩ (2 electrons in orbitals 0,1)
        qml.BasisState(np.array([1, 1, 0, 0]), wires=range(4))
        # Excite electron from orbital 0 to orbital 2
        qml.SingleExcitation(theta, wires=[0, 2])
        return qml.state()
    
    print("SingleExcitation: moves one electron between orbitals")
    print("Example: |0011⟩ -> superposition of |0011⟩ and |0110⟩")
    
    theta = np.pi / 4
    state = single_excitation_demo(theta)
    print(f"\nWith θ = π/4:")
    for i, amp in enumerate(state):
        if abs(amp) > 0.01:
            print(f"  |{i:04b}⟩: {amp:.4f}")
    
    # Double excitation
    @qml.qnode(dev)
    def double_excitation_demo(theta):
        qml.BasisState(np.array([1, 1, 0, 0]), wires=range(4))
        # Excite two electrons: (0,1) -> (2,3)
        qml.DoubleExcitation(theta, wires=[0, 1, 2, 3])
        return qml.state()
    
    print("\nDoubleExcitation: moves two electrons between orbital pairs")
    print("Example: |0011⟩ -> superposition of |0011⟩ and |1100⟩")
    
    state = double_excitation_demo(np.pi / 4)
    print(f"\nWith θ = π/4:")
    for i, amp in enumerate(state):
        if abs(amp) > 0.01:
            print(f"  |{i:04b}⟩: {amp:.4f}")
    
    print("\n2.2 Using qml.AllSinglesDoubles:")
    print("-" * 50)
    
    if QCHEM_AVAILABLE:
        # Get excitation generators from qchem
        singles, doubles = qchem.excitations(electrons=2, orbitals=4)
        print(f"Single excitations: {singles}")
        print(f"Double excitations: {doubles}")
    else:
        # Manual definition
        singles = [[0, 2], [0, 3], [1, 2], [1, 3]]
        doubles = [[0, 1, 2, 3]]
        print(f"Single excitations: {singles}")
        print(f"Double excitations: {doubles}")
    
    # Create AllSinglesDoubles ansatz
    n_params = len(singles) + len(doubles)
    print(f"\nTotal parameters: {n_params}")
    
    @qml.qnode(dev)
    def uccsd_circuit(params):
        # HF reference state
        qml.BasisState(np.array([1, 1, 0, 0]), wires=range(4))
        
        # Apply all single and double excitations
        qml.AllSinglesDoubles(
            weights=params,
            wires=range(4),
            hf_state=np.array([1, 1, 0, 0]),
            singles=singles,
            doubles=doubles
        )
        return qml.state()
    
    params = np.zeros(n_params)
    print("\nCircuit at θ=0 (HF state):")
    state = uccsd_circuit(params)
    for i, amp in enumerate(state):
        if abs(amp) > 0.01:
            print(f"  |{i:04b}⟩: {amp:.4f}")


def tutorial_3_vqe_with_autodiff():
    """
    Tutorial 3: VQE with Automatic Differentiation
    """
    print("\n" + "=" * 60)
    print("Tutorial 3: VQE with PennyLane Autodiff")
    print("=" * 60)
    
    if not PENNYLANE_AVAILABLE:
        print("PennyLane not available. Skipping tutorial.")
        return
    
    # Get Hamiltonian
    H, n_qubits, exact_gs = tutorial_1_molecular_hamiltonian()
    if H is None:
        return
    
    print("\n3.1 Define VQE Circuit:")
    print("-" * 50)
    
    dev = qml.device('default.qubit', wires=n_qubits)
    
    # Define excitations
    if QCHEM_AVAILABLE:
        singles, doubles = qchem.excitations(electrons=2, orbitals=n_qubits)
    else:
        singles = [[0, 2], [0, 3], [1, 2], [1, 3]]
        doubles = [[0, 1, 2, 3]]
    
    hf_state = np.array([1, 1] + [0] * (n_qubits - 2))
    n_params = len(singles) + len(doubles)
    
    @qml.qnode(dev, interface='autograd')
    def cost_fn(params):
        qml.AllSinglesDoubles(
            weights=params,
            wires=range(n_qubits),
            hf_state=hf_state,
            singles=singles,
            doubles=doubles
        )
        return qml.expval(H)
    
    print(f"Number of parameters: {n_params}")
    print(f"HF state: {hf_state}")
    
    print("\n3.2 Gradient-Based Optimization:")
    print("-" * 50)
    
    # Use PennyLane's numpy for autodiff
    params = pnp.zeros(n_params, requires_grad=True)
    
    # Compute HF energy
    hf_energy = cost_fn(params)
    print(f"HF energy: {hf_energy:.8f} Ha")
    
    # Compute gradient at HF point
    grad_fn = qml.grad(cost_fn)
    grad = grad_fn(params)
    print(f"Gradient at HF: {grad}")
    
    print("\n3.3 Run Optimization:")
    print("-" * 50)
    
    # Gradient descent optimization
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    
    energy_history = []
    
    for i in range(50):
        params, energy = opt.step_and_cost(cost_fn, params)
        energy_history.append(energy)
        
        if i % 10 == 0:
            print(f"  Iteration {i:3d}: E = {energy:.8f} Ha")
    
    print(f"\n3.4 Results:")
    print("-" * 50)
    print(f"Final VQE energy: {energy_history[-1]:.8f} Ha")
    print(f"Exact GS energy:  {exact_gs:.8f} Ha")
    print(f"Error: {abs(energy_history[-1] - exact_gs):.2e} Ha")
    print(f"Optimal parameters: {params}")
    
    return energy_history


def tutorial_4_advanced_features():
    """
    Tutorial 4: Advanced PennyLane QChem Features
    """
    print("\n" + "=" * 60)
    print("Tutorial 4: Advanced Features")
    print("=" * 60)
    
    if not PENNYLANE_AVAILABLE:
        print("PennyLane not available. Skipping tutorial.")
        return
    
    if not QCHEM_AVAILABLE:
        print("Full qchem not available. Showing conceptual examples.")
        show_conceptual_examples()
        return
    
    print("\n4.1 Active Space Selection:")
    print("-" * 50)
    
    # LiH with active space
    symbols = ['Li', 'H']
    coordinates = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.6]])
    
    # Full space
    H_full, n_full = qchem.molecular_hamiltonian(
        symbols=symbols,
        coordinates=coordinates,
        basis='sto-3g'
    )
    print(f"Full space: {n_full} qubits, {len(H_full.ops)} Pauli terms")
    
    # Reduced active space: 2 electrons in 3 orbitals
    H_active, n_active = qchem.molecular_hamiltonian(
        symbols=symbols,
        coordinates=coordinates,
        basis='sto-3g',
        active_electrons=2,
        active_orbitals=3
    )
    print(f"Active space (2e, 3o): {n_active} qubits, {len(H_active.ops)} Pauli terms")
    
    print("\n4.2 Different Basis Sets:")
    print("-" * 50)
    
    # Compare basis sets for H2
    symbols_h2 = ['H', 'H']
    coords_h2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    
    for basis in ['sto-3g', '6-31g']:
        try:
            H, n = qchem.molecular_hamiltonian(
                symbols=symbols_h2,
                coordinates=coords_h2,
                basis=basis,
                active_electrons=2,
                active_orbitals=2
            )
            H_mat = qml.matrix(H, wire_order=range(n))
            gs = np.linalg.eigvalsh(H_mat)[0]
            print(f"  {basis:10s}: {n} qubits, GS = {gs:.6f} Ha")
        except Exception as e:
            print(f"  {basis:10s}: Error - {e}")
    
    print("\n4.3 Dipole Moment:")
    print("-" * 50)
    
    try:
        # Get dipole moment operator
        dipole = qchem.dipole_of(symbols_h2, coords_h2, basis='sto-3g')
        print(f"Dipole operators: {len(dipole)} components (x, y, z)")
        
        # For H2, dipole should be zero at equilibrium
        # but varies with bond stretching
    except Exception as e:
        print(f"Dipole moment calculation: {e}")


def show_conceptual_examples():
    """Show conceptual examples without full qchem."""
    print("\n4.1 Active Space Concept:")
    print("-" * 50)
    print("""
Active Space Selection:
-----------------------
Full space: All electrons in all orbitals
  - LiH: 4 electrons, 6 orbitals = 12 qubits
  - H2O: 10 electrons, 7 orbitals = 14 qubits

Active space: Only frontier orbitals
  - Choose most important orbitals (HOMO, LUMO, nearby)
  - Freeze core electrons
  - Reduces qubit count significantly
  
Example: LiH (2e, 3o) active space = 6 qubits vs 12 qubits full
""")
    
    print("\n4.2 Basis Set Effects:")
    print("-" * 50)
    print("""
Basis Set       Orbitals   Accuracy    Cost
-------------------------------------------------
STO-3G          Minimal    Low         Cheap
3-21G           Split      Medium      Medium
6-31G           Split      Good        Medium
6-311G*         Polarized  Better      Higher
cc-pVDZ         Corr.      Good        Higher
cc-pVTZ         Corr.      Very Good   Expensive

For VQE: Start with STO-3G, then increase as resources allow
""")


# ============================================================================
# Comparison with Classical Methods
# ============================================================================

def tutorial_5_comparison():
    """
    Tutorial 5: Comparing VQE with Classical Methods
    """
    print("\n" + "=" * 60)
    print("Tutorial 5: VQE vs Classical Methods")
    print("=" * 60)
    
    if not PENNYLANE_AVAILABLE:
        print("PennyLane not available. Skipping tutorial.")
        return
    
    if not QCHEM_AVAILABLE:
        print("Full qchem not available. Showing expected results.")
        print("""
Expected Results for H2 at R = 0.74 Å (STO-3G):
-----------------------------------------------
Method              Energy (Ha)    Error (mHa)
-----------------------------------------------
Hartree-Fock        -1.1167        19.3
VQE (UCCSD)         -1.1372        ~0.1
FCI (Exact)         -1.1373        0.0
-----------------------------------------------

VQE captures most correlation energy!
""")
        return
    
    print("\n5.1 H2 at Different Bond Lengths:")
    print("-" * 50)
    
    bond_lengths = [0.5, 0.7, 0.74, 1.0, 1.5, 2.0]
    
    print("R (Å)     HF (Ha)     VQE (Ha)    FCI (Ha)    VQE Error")
    print("-" * 65)
    
    dev = qml.device('default.qubit', wires=4)
    
    for r in bond_lengths:
        symbols = ['H', 'H']
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]])
        
        try:
            H, n_qubits = qchem.molecular_hamiltonian(
                symbols=symbols,
                coordinates=coords,
                basis='sto-3g'
            )
            
            # Get exact energies
            H_mat = qml.matrix(H, wire_order=range(n_qubits))
            eigenvalues = np.linalg.eigvalsh(H_mat)
            fci_energy = eigenvalues[0]
            
            # HF energy (just the expectation in HF state)
            hf_state = np.zeros(2**n_qubits)
            hf_state[3] = 1.0  # |0011⟩
            hf_energy = hf_state @ H_mat @ hf_state
            
            # Quick VQE (few iterations for demo)
            singles, doubles = qchem.excitations(electrons=2, orbitals=4)
            
            @qml.qnode(dev, interface='autograd')
            def cost(params):
                qml.AllSinglesDoubles(
                    weights=params,
                    wires=range(4),
                    hf_state=np.array([1, 1, 0, 0]),
                    singles=singles,
                    doubles=doubles
                )
                return qml.expval(H)
            
            # Optimize briefly
            params = pnp.zeros(len(singles) + len(doubles), requires_grad=True)
            opt = qml.GradientDescentOptimizer(stepsize=0.4)
            
            for _ in range(30):
                params = opt.step(cost, params)
            
            vqe_energy = float(cost(params))
            error = abs(vqe_energy - fci_energy) * 1000  # mHa
            
            print(f"{r:.2f}     {hf_energy:.4f}      {vqe_energy:.4f}      "
                  f"{fci_energy:.4f}      {error:.2f} mHa")
        
        except Exception as e:
            print(f"{r:.2f}     Error: {e}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" PennyLane Quantum Chemistry Tutorial")
    print(" Seamless Integration of QML and Quantum Chemistry")
    print("=" * 70)
    
    tutorial_1_molecular_hamiltonian()
    tutorial_2_excitation_operators()
    energy_history = tutorial_3_vqe_with_autodiff()
    tutorial_4_advanced_features()
    tutorial_5_comparison()
    
    print("\n" + "=" * 70)
    print(" Tutorial Complete!")
    print("=" * 70)
    print("""
Key PennyLane QChem Functions:
------------------------------
- qchem.molecular_hamiltonian(): Build H from geometry
- qchem.excitations(): Get single/double excitations
- qml.SingleExcitation(): Single excitation gate
- qml.DoubleExcitation(): Double excitation gate
- qml.AllSinglesDoubles(): Full UCCSD-like ansatz

Advantages of PennyLane:
------------------------
1. Automatic differentiation (no finite differences!)
2. Multiple backends (simulators and real hardware)
3. Easy integration with ML workflows
4. GPU acceleration available
5. Good documentation and community

Next Steps:
-----------
1. Try larger molecules with active space
2. Implement custom cost functions
3. Add noise models
4. Use quantum natural gradient
5. Compare with classical VQE libraries
""")
