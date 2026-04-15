#!/usr/bin/env python3
"""
OpenFermion Tutorial
====================

OpenFermion is Google's open-source library for quantum chemistry.
It provides tools for:
- Defining fermionic and qubit operators
- Fermion-to-qubit transformations
- Integration with classical chemistry packages (PySCF, Psi4)

This tutorial covers:
1. Basic operator manipulation
2. Building molecular Hamiltonians
3. Jordan-Wigner and Bravyi-Kitaev transformations
4. Integration with PySCF

Installation:
    pip install openfermion openfermionpyscf

Reference: https://quantumai.google/openfermion
"""

from __future__ import annotations
import numpy as np

# Check if OpenFermion is available
try:
    import openfermion as of
    from openfermion import (
        FermionOperator, QubitOperator, MolecularData,
        jordan_wigner, bravyi_kitaev, get_fermion_operator,
        hermitian_conjugated, normal_ordered
    )
    OPENFERMION_AVAILABLE = True
except ImportError:
    OPENFERMION_AVAILABLE = False
    print("OpenFermion not installed. Install with: pip install openfermion")

# Check if OpenFermion-PySCF is available
try:
    from openfermionpyscf import run_pyscf
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False


def tutorial_1_basic_operators():
    """
    Tutorial 1: Basic Fermionic and Qubit Operators
    """
    print("=" * 60)
    print("Tutorial 1: Basic Operators in OpenFermion")
    print("=" * 60)
    
    if not OPENFERMION_AVAILABLE:
        print("OpenFermion not available. Skipping tutorial.")
        return
    
    # Creating FermionOperators
    print("\n1.1 Creating Fermionic Operators:")
    print("-" * 50)
    
    # Creation operator a†_0
    a0_dag = FermionOperator('0^')
    print(f"Creation a†_0: {a0_dag}")
    
    # Annihilation operator a_0
    a0 = FermionOperator('0')
    print(f"Annihilation a_0: {a0}")
    
    # Number operator n_0 = a†_0 a_0
    n0 = FermionOperator('0^ 0')
    print(f"Number operator n_0: {n0}")
    
    # Hopping term with coefficient
    hop = FermionOperator('0^ 1', coefficient=0.5)
    print(f"Hopping 0.5 * a†_0 a_1: {hop}")
    
    # Two-body operator
    two_body = FermionOperator('0^ 1^ 2 3', coefficient=0.25)
    print(f"Two-body term: {two_body}")
    
    print("\n1.2 Operator Algebra:")
    print("-" * 50)
    
    # Addition
    H = FermionOperator('0^ 0', -1.0) + FermionOperator('1^ 1', -1.0)
    print(f"H = -n_0 - n_1: {H}")
    
    # Multiplication
    product = FermionOperator('0^') * FermionOperator('0')
    print(f"a†_0 * a_0: {product}")
    
    # Hermitian conjugate
    op = FermionOperator('0^ 1', coefficient=1.0)
    op_dag = hermitian_conjugated(op)
    print(f"Original: {op}")
    print(f"Hermitian conjugate: {op_dag}")
    
    # Normal ordering
    not_normal = FermionOperator('0 0^')
    normal = normal_ordered(not_normal)
    print(f"Not normal ordered: {not_normal}")
    print(f"Normal ordered: {normal}")
    
    print("\n1.3 Qubit Operators:")
    print("-" * 50)
    
    # Pauli operators
    X0 = QubitOperator('X0')
    Y1 = QubitOperator('Y1')
    Z0Z1 = QubitOperator('Z0 Z1')
    
    print(f"X0: {X0}")
    print(f"Y1: {Y1}")
    print(f"Z0 Z1: {Z0Z1}")
    
    # Combination
    H_qubit = 0.5 * QubitOperator('X0 X1') + 0.3 * QubitOperator('Z0')
    print(f"Qubit Hamiltonian: {H_qubit}")


def tutorial_2_jordan_wigner():
    """
    Tutorial 2: Jordan-Wigner Transformation
    """
    print("\n" + "=" * 60)
    print("Tutorial 2: Jordan-Wigner Transformation")
    print("=" * 60)
    
    if not OPENFERMION_AVAILABLE:
        print("OpenFermion not available. Skipping tutorial.")
        return
    
    print("\n2.1 Single Operators:")
    print("-" * 50)
    
    # Creation operator
    a0_dag = FermionOperator('0^')
    jw_a0_dag = jordan_wigner(a0_dag)
    print(f"Fermion a†_0: {a0_dag}")
    print(f"JW transform: {jw_a0_dag}")
    
    # Creation on qubit 2 (with Z-string)
    a2_dag = FermionOperator('2^')
    jw_a2_dag = jordan_wigner(a2_dag)
    print(f"\nFermion a†_2: {a2_dag}")
    print(f"JW transform: {jw_a2_dag}")
    print("(Note the Z0 Z1 string!)")
    
    print("\n2.2 Number Operator:")
    print("-" * 50)
    
    n0 = FermionOperator('0^ 0')
    jw_n0 = jordan_wigner(n0)
    print(f"Fermion n_0 = a†_0 a_0: {n0}")
    print(f"JW transform: {jw_n0}")
    print("Simplifies to (I - Z)/2")
    
    print("\n2.3 Hopping Term:")
    print("-" * 50)
    
    hop = FermionOperator('0^ 1') + FermionOperator('1^ 0')
    jw_hop = jordan_wigner(hop)
    print(f"Fermion: a†_0 a_1 + h.c.")
    print(f"JW transform:\n{jw_hop}")
    
    print("\n2.4 Comparison with Bravyi-Kitaev:")
    print("-" * 50)
    
    # Same operator, different mapping
    test_op = FermionOperator('0^ 2') + FermionOperator('2^ 0')
    
    jw_result = jordan_wigner(test_op)
    bk_result = bravyi_kitaev(test_op)
    
    print(f"Fermion: a†_0 a_2 + h.c.")
    print(f"\nJordan-Wigner ({len(jw_result.terms)} terms):")
    print(f"{jw_result}")
    print(f"\nBravyi-Kitaev ({len(bk_result.terms)} terms):")
    print(f"{bk_result}")


def tutorial_3_molecular_hamiltonian():
    """
    Tutorial 3: Building Molecular Hamiltonians
    """
    print("\n" + "=" * 60)
    print("Tutorial 3: Molecular Hamiltonians")
    print("=" * 60)
    
    if not OPENFERMION_AVAILABLE:
        print("OpenFermion not available. Skipping tutorial.")
        return
    
    if not PYSCF_AVAILABLE:
        print("openfermionpyscf not available.")
        print("Install with: pip install openfermionpyscf")
        print("Running with example H2 data instead.\n")
        
        # Use example data
        demonstrate_with_example_data()
        return
    
    print("\n3.1 Define Molecule:")
    print("-" * 50)
    
    # H2 molecule
    geometry = [
        ('H', (0.0, 0.0, 0.0)),
        ('H', (0.0, 0.0, 0.74))
    ]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    
    h2_molecule = MolecularData(geometry, basis, multiplicity, charge,
                                description='H2 at 0.74 Angstrom')
    
    print(f"Molecule: H2")
    print(f"Geometry: {geometry}")
    print(f"Basis: {basis}")
    
    print("\n3.2 Run PySCF Calculation:")
    print("-" * 50)
    
    # Run electronic structure calculation
    h2_molecule = run_pyscf(h2_molecule, run_fci=True)
    
    print(f"Number of spatial orbitals: {h2_molecule.n_orbitals}")
    print(f"Number of electrons: {h2_molecule.n_electrons}")
    print(f"Nuclear repulsion: {h2_molecule.nuclear_repulsion:.6f} Ha")
    print(f"HF energy: {h2_molecule.hf_energy:.6f} Ha")
    print(f"FCI energy: {h2_molecule.fci_energy:.6f} Ha")
    
    print("\n3.3 Get Molecular Hamiltonian:")
    print("-" * 50)
    
    # Get the molecular Hamiltonian
    molecular_hamiltonian = h2_molecule.get_molecular_hamiltonian()
    print(f"Type: {type(molecular_hamiltonian).__name__}")
    
    # Convert to FermionOperator
    fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
    print(f"Number of fermionic terms: {len(fermion_hamiltonian.terms)}")
    
    print("\n3.4 Convert to Qubit Hamiltonian:")
    print("-" * 50)
    
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    print(f"Number of Pauli terms: {len(qubit_hamiltonian.terms)}")
    
    # Get the matrix representation
    from openfermion import get_sparse_operator
    H_sparse = get_sparse_operator(qubit_hamiltonian)
    H_dense = H_sparse.toarray()
    
    eigenvalues = np.linalg.eigvalsh(H_dense)
    print(f"\nExact diagonalization:")
    print(f"  Ground state energy: {eigenvalues[0]:.6f} Ha")
    print(f"  FCI energy (ref):    {h2_molecule.fci_energy:.6f} Ha")


def demonstrate_with_example_data():
    """Demonstrate with hardcoded H2 data when PySCF is not available."""
    print("3.1 Using Example H2 Data:")
    print("-" * 50)
    
    # Create a simple H2 Hamiltonian manually
    # H = -1.25 (n_0 + n_1) + 0.7 (a†_0 a_1 + h.c.) + 0.5 n_0 n_1
    
    H = FermionOperator('0^ 0', -1.25)
    H += FermionOperator('1^ 1', -1.25)
    H += FermionOperator('0^ 1', 0.7)
    H += FermionOperator('1^ 0', 0.7)
    H += FermionOperator('0^ 0 1^ 1', 0.5)
    
    print(f"Example Fermionic Hamiltonian:")
    print(f"{H}")
    
    print("\n3.2 Jordan-Wigner Transform:")
    print("-" * 50)
    
    H_qubit = jordan_wigner(H)
    print(f"Qubit Hamiltonian ({len(H_qubit.terms)} terms):")
    print(f"{H_qubit}")
    
    print("\n3.3 Exact Diagonalization:")
    print("-" * 50)
    
    from openfermion import get_sparse_operator
    H_sparse = get_sparse_operator(H_qubit)
    H_dense = H_sparse.toarray()
    
    eigenvalues = np.linalg.eigvalsh(H_dense)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Ground state energy: {eigenvalues[0]:.6f}")


def tutorial_4_vqe_with_openfermion():
    """
    Tutorial 4: VQE using OpenFermion + Cirq
    """
    print("\n" + "=" * 60)
    print("Tutorial 4: VQE with OpenFermion")
    print("=" * 60)
    
    if not OPENFERMION_AVAILABLE:
        print("OpenFermion not available. Skipping tutorial.")
        return
    
    try:
        import cirq
        from openfermion import get_sparse_operator
        from scipy.optimize import minimize
        CIRQ_AVAILABLE = True
    except ImportError:
        CIRQ_AVAILABLE = False
        print("Cirq not available. Install with: pip install cirq")
        print("Showing OpenFermion setup only.\n")
    
    print("\n4.1 Setup Hamiltonian:")
    print("-" * 50)
    
    # Simple 4-qubit Hamiltonian
    H = FermionOperator('0^ 0', -1.0) + FermionOperator('1^ 1', -1.0)
    H += FermionOperator('2^ 2', -0.5) + FermionOperator('3^ 3', -0.5)
    H += FermionOperator('0^ 2', 0.3) + FermionOperator('2^ 0', 0.3)
    H += FermionOperator('1^ 3', 0.3) + FermionOperator('3^ 1', 0.3)
    
    H_qubit = jordan_wigner(H)
    H_sparse = get_sparse_operator(H_qubit)
    H_matrix = H_sparse.toarray()
    
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    print(f"Number of qubits: {int(np.log2(H_matrix.shape[0]))}")
    print(f"Exact ground state: {eigenvalues[0]:.6f}")
    
    if not CIRQ_AVAILABLE:
        return
    
    print("\n4.2 Define Ansatz Circuit:")
    print("-" * 50)
    
    n_qubits = 4
    qubits = cirq.LineQubit.range(n_qubits)
    
    def create_ansatz(params):
        """Create variational circuit."""
        circuit = cirq.Circuit()
        
        # Initial state: |0011⟩
        circuit.append(cirq.X(qubits[0]))
        circuit.append(cirq.X(qubits[1]))
        
        # Variational layers
        for i in range(n_qubits):
            circuit.append(cirq.ry(params[i]).on(qubits[i]))
        
        for i in range(n_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
        
        for i in range(n_qubits):
            circuit.append(cirq.ry(params[n_qubits + i]).on(qubits[i]))
        
        return circuit
    
    print("Ansatz circuit structure:")
    test_params = np.zeros(2 * n_qubits)
    print(create_ansatz(test_params))
    
    print("\n4.3 VQE Optimization:")
    print("-" * 50)
    
    simulator = cirq.Simulator()
    
    def cost_function(params):
        circuit = create_ansatz(params)
        result = simulator.simulate(circuit)
        state = result.final_state_vector
        energy = np.real(state.conj() @ H_matrix @ state)
        return energy
    
    # Optimize
    initial_params = np.random.uniform(-0.1, 0.1, 2 * n_qubits)
    
    result = minimize(cost_function, initial_params, method='COBYLA',
                     options={'maxiter': 200})
    
    print(f"VQE energy: {result.fun:.6f}")
    print(f"Exact energy: {eigenvalues[0]:.6f}")
    print(f"Error: {abs(result.fun - eigenvalues[0]):.6f}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" OpenFermion Tutorial")
    print(" Quantum Chemistry with Google's Open-Source Library")
    print("=" * 70)
    
    tutorial_1_basic_operators()
    tutorial_2_jordan_wigner()
    tutorial_3_molecular_hamiltonian()
    tutorial_4_vqe_with_openfermion()
    
    print("\n" + "=" * 70)
    print(" Tutorial Complete!")
    print("=" * 70)
    print("""
Key OpenFermion Classes:
------------------------
- FermionOperator: Fermionic creation/annihilation operators
- QubitOperator: Pauli operators
- MolecularData: Container for molecular information
- jordan_wigner(): JW transformation
- bravyi_kitaev(): BK transformation
- get_sparse_operator(): Convert to sparse matrix

Useful Functions:
-----------------
- hermitian_conjugated(): Get H.c.
- normal_ordered(): Normal ordering
- get_fermion_operator(): Get fermionic H from molecular data
- run_pyscf(): Interface with PySCF

Next Steps:
-----------
1. Try different molecules (LiH, H2O)
2. Compare JW vs BK for larger systems
3. Implement custom ansätze
4. Study active space selection
""")
