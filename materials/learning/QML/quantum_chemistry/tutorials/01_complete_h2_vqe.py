#!/usr/bin/env python3
"""
Complete H2 VQE Tutorial - From Zero to Ground State Energy
===========================================================

This tutorial implements a complete VQE calculation for the H2 molecule,
demonstrating every step of the process:

1. Define molecular geometry
2. Compute molecular integrals (using PySCF or hardcoded values)
3. Build the fermionic Hamiltonian in second quantization
4. Transform to qubit Hamiltonian using Jordan-Wigner
5. Design the ansatz (UCCSD)
6. Run VQE optimization
7. Compare with exact results

This is a self-contained tutorial that can run with or without PySCF.

Requirements:
    - numpy
    - pennylane
    - scipy
    - (optional) pyscf for computing real integrals
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
import time


# ============================================================================
# Part 1: Molecular Data
# ============================================================================

def get_h2_integrals(bond_length: float = 0.74) -> Dict:
    """
    Get molecular integrals for H2 in STO-3G basis.
    
    For H2 at ~0.74 Å, there are 2 spatial orbitals = 4 spin orbitals.
    
    If PySCF is available, compute real integrals.
    Otherwise, use literature values for demonstration.
    """
    try:
        from pyscf import gto, scf, fci, ao2mo
        
        # Build H2 molecule
        mol = gto.Mole()
        mol.atom = f'H 0 0 0; H 0 0 {bond_length}'
        mol.basis = 'sto-3g'
        mol.build()
        
        # Run Hartree-Fock
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.kernel()
        
        # Get integrals
        C = mf.mo_coeff
        h1_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        h1_mo = C.T @ h1_ao @ C
        
        eri_ao = mol.intor('int2e')
        eri_mo = ao2mo.incore.full(eri_ao, C).reshape(2, 2, 2, 2)
        
        # FCI for exact reference
        cisolver = fci.FCI(mf)
        cisolver.verbose = 0
        fci_energy, _ = cisolver.kernel()
        
        return {
            'n_orbitals': 2,
            'n_electrons': 2,
            'nuclear_repulsion': mol.energy_nuc(),
            'one_body': h1_mo,
            'two_body': eri_mo,
            'hf_energy': mf.e_tot,
            'fci_energy': fci_energy,
            'from_pyscf': True
        }
        
    except ImportError:
        print("PySCF not found. Using pre-computed integrals for H2 at 0.74 Å.")
        
        # Pre-computed values for H2 STO-3G at R = 0.74 Å
        # These are approximate but sufficient for learning
        return {
            'n_orbitals': 2,
            'n_electrons': 2,
            'nuclear_repulsion': 0.7137539936876182,
            'one_body': np.array([
                [-1.2527629, -0.47594398],
                [-0.47594398, -0.47594398]
            ]),
            'two_body': np.array([
                [[[0.67460573, 0.0], [0.0, 0.18128117]],
                 [[0.0, 0.6636314], [0.66363144, 0.0]]],
                [[[0.0, 0.66363144], [0.6636314, 0.0]],
                 [[0.18128117, 0.0], [0.0, 0.67460573]]]
            ]),
            'hf_energy': -1.1167593073,
            'fci_energy': -1.1372838344,
            'from_pyscf': False
        }


# ============================================================================
# Part 2: Fermionic Hamiltonian
# ============================================================================

class SimpleFermionOperator:
    """
    Simplified fermionic operator for this tutorial.
    
    Stores terms as a dictionary: (tuple of (orbital, is_creation)) -> coefficient
    """
    
    def __init__(self):
        self.terms: Dict[Tuple, complex] = {}
    
    def add_term(self, coef: complex, operators: List[Tuple[int, bool]]):
        """Add a term: coef * a†_i a†_j ... a_k a_l ..."""
        key = tuple(operators)
        if key in self.terms:
            self.terms[key] += coef
        else:
            self.terms[key] = coef
        
        # Remove near-zero terms
        if abs(self.terms[key]) < 1e-12:
            del self.terms[key]
    
    def __str__(self):
        if not self.terms:
            return "0"
        
        parts = []
        for ops, coef in self.terms.items():
            ops_str = " ".join(f"a†_{o}" if c else f"a_{o}" for o, c in ops)
            if not ops:
                parts.append(f"{coef.real:.6f}")
            else:
                parts.append(f"{coef.real:.6f} * {ops_str}")
        return " + ".join(parts)


def build_h2_fermionic_hamiltonian(integrals: Dict) -> SimpleFermionOperator:
    """
    Build the fermionic Hamiltonian for H2 in spin-orbital basis.
    
    H = E_nuc + Σ_pq h_pq a†_p a_q + (1/2) Σ_pqrs g_pqrs a†_p a†_q a_s a_r
    
    For 2 spatial orbitals, we have 4 spin orbitals:
    - 0: spatial orbital 0, spin α
    - 1: spatial orbital 0, spin β
    - 2: spatial orbital 1, spin α
    - 3: spatial orbital 1, spin β
    """
    H = SimpleFermionOperator()
    
    h1 = integrals['one_body']  # 2x2
    h2 = integrals['two_body']  # 2x2x2x2
    E_nuc = integrals['nuclear_repulsion']
    
    # Add nuclear repulsion (constant term)
    H.add_term(E_nuc, [])
    
    # One-body terms: Σ h_pq a†_pσ a_qσ
    # For each spatial orbital pair and each spin
    for p in range(2):  # spatial orbitals
        for q in range(2):
            if abs(h1[p, q]) < 1e-12:
                continue
            
            # Alpha spin: spin orbital = 2*p
            H.add_term(h1[p, q], [(2*p, True), (2*q, False)])
            # Beta spin: spin orbital = 2*p + 1
            H.add_term(h1[p, q], [(2*p+1, True), (2*q+1, False)])
    
    # Two-body terms: (1/2) Σ g_pqrs a†_p a†_q a_s a_r
    # PySCF returns chemist's notation (pq|rs)
    # We convert to physicist's notation <pq|rs> = (pr|qs) before applying
    # the standard operator ordering a†_p a†_q a_s a_r.
    for p in range(2):
        for q in range(2):
            for r in range(2):
                for s in range(2):
                    # Convert (pq|rs) -> <pq|rs> by swapping q and r: (pr|qs)
                    g = h2[p, r, q, s]
                    if abs(g) < 1e-12:
                        continue
                    
                    # Four spin combinations: a†_pσ a†_qτ a_sτ a_rσ
                    # For chemist's notation (pq|rs), operator is a†_p a†_q a_s a_r
                    # αα: pα, qα -> sα, rα
                    H.add_term(0.5 * g, [(2*p, True), (2*q, True), (2*s, False), (2*r, False)])
                    # αβ: pα, qβ -> sβ, rα
                    H.add_term(0.5 * g, [(2*p, True), (2*q+1, True), (2*s+1, False), (2*r, False)])
                    # βα: pβ, qα -> sα, rβ
                    H.add_term(0.5 * g, [(2*p+1, True), (2*q, True), (2*s, False), (2*r+1, False)])
                    # ββ: pβ, qβ -> sβ, rβ
                    H.add_term(0.5 * g, [(2*p+1, True), (2*q+1, True), (2*s+1, False), (2*r+1, False)])
    
    return H


# ============================================================================
# Part 3: Jordan-Wigner Transformation
# ============================================================================

class SimplePauliOperator:
    """
    Simplified Pauli operator for this tutorial.
    
    Stores terms as: (tuple of (qubit, pauli_type)) -> coefficient
    where pauli_type is 'I', 'X', 'Y', 'Z'
    """
    
    def __init__(self):
        self.terms: Dict[Tuple, complex] = {}
    
    def add_term(self, coef: complex, paulis: Dict[int, str]):
        """Add a Pauli string term."""
        # Convert to sorted tuple for consistent hashing
        key = tuple(sorted(paulis.items()))
        if key in self.terms:
            self.terms[key] += coef
        else:
            self.terms[key] = coef
        
        if abs(self.terms[key]) < 1e-12:
            del self.terms[key]
    
    def __add__(self, other):
        result = SimplePauliOperator()
        result.terms = dict(self.terms)
        for key, coef in other.terms.items():
            if key in result.terms:
                result.terms[key] += coef
            else:
                result.terms[key] = coef
            if abs(result.terms[key]) < 1e-12:
                del result.terms[key]
        return result
    
    def __mul__(self, other):
        """Multiply two Pauli operators."""
        if isinstance(other, (int, float, complex)):
            result = SimplePauliOperator()
            for key, coef in self.terms.items():
                result.terms[key] = coef * other
            return result
        
        result = SimplePauliOperator()
        for key1, coef1 in self.terms.items():
            for key2, coef2 in other.terms.items():
                new_paulis, phase = multiply_pauli_strings(dict(key1), dict(key2))
                new_key = tuple(sorted(new_paulis.items()))
                new_coef = coef1 * coef2 * phase
                
                # Only add if coefficient is significant
                if abs(new_coef) >= 1e-12:
                    if new_key in result.terms:
                        result.terms[new_key] += new_coef
                        # Remove if becomes too small
                        if abs(result.terms[new_key]) < 1e-12:
                            del result.terms[new_key]
                    else:
                        result.terms[new_key] = new_coef
        return result
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    @property
    def n_terms(self):
        return len(self.terms)
    
    def to_matrix(self, n_qubits: int) -> np.ndarray:
        """Convert to dense matrix representation."""
        dim = 2 ** n_qubits
        matrix = np.zeros((dim, dim), dtype=complex)
        
        for key, coef in self.terms.items():
            pauli_dict = dict(key)
            ps_matrix = pauli_string_to_matrix(pauli_dict, n_qubits)
            matrix += coef * ps_matrix
        
        return matrix


def multiply_pauli_strings(p1: Dict[int, str], p2: Dict[int, str]) -> Tuple[Dict[int, str], complex]:
    """Multiply two Pauli strings, return (result, phase)."""
    result = {}
    phase = 1.0
    
    all_qubits = set(p1.keys()) | set(p2.keys())
    
    for q in all_qubits:
        a = p1.get(q, 'I')
        b = p2.get(q, 'I')
        
        r, p = multiply_single_paulis(a, b)
        phase *= p
        if r != 'I':
            result[q] = r
    
    return result, phase


def multiply_single_paulis(a: str, b: str) -> Tuple[str, complex]:
    """Multiply two Pauli matrices."""
    if a == 'I':
        return b, 1.0
    if b == 'I':
        return a, 1.0
    if a == b:
        return 'I', 1.0
    
    # XY = iZ, YZ = iX, ZX = iY, etc.
    table = {
        ('X', 'Y'): ('Z', 1j),
        ('Y', 'X'): ('Z', -1j),
        ('Y', 'Z'): ('X', 1j),
        ('Z', 'Y'): ('X', -1j),
        ('Z', 'X'): ('Y', 1j),
        ('X', 'Z'): ('Y', -1j),
    }
    
    key = (a, b)
    if key in table:
        return table[key]
    else:
        raise ValueError(f"Unknown Pauli multiplication: {a} * {b}")


def pauli_string_to_matrix(paulis: Dict[int, str], n_qubits: int) -> np.ndarray:
    """Convert Pauli string to matrix."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    
    result = np.array([[1.0]], dtype=complex)
    for q in range(n_qubits):
        p = paulis.get(q, 'I')
        result = np.kron(result, pauli_map[p])
    
    return result


def jordan_wigner_transform(fermion_op: SimpleFermionOperator, n_qubits: int) -> SimplePauliOperator:
    """
    Apply Jordan-Wigner transformation to convert fermionic to qubit operator.
    
    a†_p → (1/2)(X_p - iY_p) ⊗ Z_{p-1} ⊗ ... ⊗ Z_0
    a_p  → (1/2)(X_p + iY_p) ⊗ Z_{p-1} ⊗ ... ⊗ Z_0
    """
    result = SimplePauliOperator()
    
    for ops, coef in fermion_op.terms.items():
        if not ops:
            # Constant term (identity)
            result.add_term(coef, {})
            continue
        
        # Transform each fermionic term
        term_result = SimplePauliOperator()
        term_result.add_term(coef, {})  # Start with coefficient
        
        for orbital, is_creation in ops:
            # Get qubit operator for this creation/annihilation
            qubit_op = SimplePauliOperator()
            
            # Z string for all qubits below this one
            z_string = {q: 'Z' for q in range(orbital)}
            
            if is_creation:
                # a† = (X - iY)/2 with Z string
                qubit_op.add_term(0.5, {**z_string, orbital: 'X'})
                qubit_op.add_term(-0.5j, {**z_string, orbital: 'Y'})
            else:
                # a = (X + iY)/2 with Z string
                qubit_op.add_term(0.5, {**z_string, orbital: 'X'})
                qubit_op.add_term(0.5j, {**z_string, orbital: 'Y'})
            
            term_result = term_result * qubit_op
        
        result = result + term_result
    
    return result


# ============================================================================
# Part 4: VQE Circuit and Optimization
# ============================================================================

def create_h2_vqe_circuit():
    """
    Create VQE circuit for H2 using PennyLane.
    
    For H2 with 2 electrons in 4 spin orbitals:
    - HF state: |0011⟩ (electrons in lowest two spin orbitals)
    - UCCSD has 1 double excitation parameter
    """
    try:
        import pennylane as qml
    except ImportError:
        raise ImportError("PennyLane is required. Install with: pip install pennylane")
    
    n_qubits = 4
    
    def circuit(params, wires):
        """
        UCCSD-like circuit for H2.
        
        params[0]: Double excitation amplitude (0,1) -> (2,3)
        """
        # Prepare HF state |0011⟩ (qubits 0,1 occupied)
        qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
        
        # Double excitation: (0,1) -> (2,3)
        # This is the only significant excitation for H2
        qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
    
    return circuit, 1  # circuit and number of parameters


def run_h2_vqe(integrals: Dict, verbose: bool = True) -> Dict:
    """
    Run complete VQE for H2.
    
    Returns dictionary with results.
    """
    try:
        import pennylane as qml
        from scipy.optimize import minimize
    except ImportError as e:
        raise ImportError(f"Required package not found: {e}")
    
    start_time = time.time()
    
    # Build Hamiltonians
    if verbose:
        print("\n" + "=" * 60)
        print("H2 VQE Calculation")
        print("=" * 60)
        print("\n1. Building Hamiltonians...")
    
    H_fermion = build_h2_fermionic_hamiltonian(integrals)
    if verbose:
        print(f"   Fermionic Hamiltonian: {len(H_fermion.terms)} terms")
    
    n_qubits = 4
    H_qubit = jordan_wigner_transform(H_fermion, n_qubits)
    if verbose:
        print(f"   Qubit Hamiltonian: {H_qubit.n_terms} Pauli terms")
    
    # Get Hamiltonian matrix
    # Note: Matrix uses standard (big-endian) ordering: qubit 0 = LSB
    # PennyLane uses little-endian: qubit 0 = MSB
    H_matrix = H_qubit.to_matrix(n_qubits)
    
    # Verify with exact diagonalization
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    exact_gs = eigenvalues[0]
    
    if verbose:
        print(f"\n2. Exact diagonalization:")
        print(f"   Ground state energy: {exact_gs:.8f} Ha")
        print(f"   Reference FCI energy: {integrals['fci_energy']:.8f} Ha")
        error = abs(exact_gs - integrals['fci_energy'])
        if error > 1e-5:
            print(f"   ⚠ WARNING: Error {error:.2e} Ha - check Hamiltonian construction!")
        else:
            print(f"   ✓ Error: {error:.2e} Ha (within tolerance)")
    
    # Create VQE circuit
    if verbose:
        print("\n3. Setting up VQE...")
    
    dev = qml.device('default.qubit', wires=n_qubits)
    circuit_fn, n_params = create_h2_vqe_circuit()
    
    @qml.qnode(dev)
    def cost_fn(params):
        circuit_fn(params, wires=range(n_qubits))
        return qml.expval(qml.Hermitian(H_matrix, wires=range(n_qubits)))
    
    # Track optimization
    energy_history = []
    
    def callback_fn(params):
        energy = float(cost_fn(params))
        energy_history.append(energy)
        return energy
    
    # Run optimization
    if verbose:
        print(f"   Number of parameters: {n_params}")
        print("\n4. Running optimization...")
    
    # Initial parameters
    params0 = np.array([0.0])
    
    # Test HF energy (params = 0)
    hf_energy_vqe = float(cost_fn(params0))
    if verbose:
        print(f"   HF energy (VQE): {hf_energy_vqe:.8f} Ha")
        print(f"   HF energy (ref): {integrals['hf_energy']:.8f} Ha")
    
    # Optimize
    result = minimize(
        callback_fn,
        params0,
        method='COBYLA',
        options={'maxiter': 200, 'rhobeg': 0.5}
    )
    
    final_energy = result.fun
    optimal_params = result.x
    
    runtime = time.time() - start_time
    
    if verbose:
        print(f"\n5. Results:")
        print("-" * 50)
        print(f"   VQE energy:     {final_energy:.8f} Ha")
        print(f"   Exact GS:       {exact_gs:.8f} Ha")
        print(f"   FCI reference:  {integrals['fci_energy']:.8f} Ha")
        print(f"   HF energy:      {integrals['hf_energy']:.8f} Ha")
        print(f"   Correlation energy: {integrals['hf_energy'] - final_energy:.8f} Ha")
        print(f"\n   Error vs exact: {abs(final_energy - exact_gs):.2e} Ha")
        print(f"   Optimal parameter: θ = {optimal_params[0]:.6f}")
        print(f"   Iterations: {len(energy_history)}")
        print(f"   Runtime: {runtime:.2f} s")
        print("=" * 60)
    
    return {
        'vqe_energy': final_energy,
        'exact_energy': exact_gs,
        'fci_energy': integrals['fci_energy'],
        'hf_energy': integrals['hf_energy'],
        'optimal_params': optimal_params,
        'energy_history': energy_history,
        'n_iterations': len(energy_history),
        'runtime': runtime
    }


def plot_energy_convergence(results: Dict, save_path: Optional[str] = None):
    """Plot energy convergence during optimization."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    iterations = range(1, len(results['energy_history']) + 1)
    ax.plot(iterations, results['energy_history'], 'b-', linewidth=2, label='VQE')
    ax.axhline(y=results['exact_energy'], color='r', linestyle='--', label='Exact GS')
    ax.axhline(y=results['hf_energy'], color='g', linestyle=':', label='HF')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Energy (Hartree)', fontsize=12)
    ax.set_title('H2 VQE Convergence', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def scan_potential_energy_surface(bond_lengths: np.ndarray = None,
                                  verbose: bool = True) -> Dict:
    """
    Scan the H2 potential energy surface at multiple bond lengths.
    
    Returns dictionary with bond lengths and energies.
    
    Note: H2 equilibrium bond length is ~0.74 Å. The scan includes
    more points around this region for better accuracy.
    """
    if bond_lengths is None:
        # Create a more refined scan around equilibrium (~0.74 Å)
        # Combine dense sampling near equilibrium with sparse sampling elsewhere
        short_range = np.linspace(0.4, 0.9, 11)  # Dense around equilibrium
        long_range = np.linspace(1.0, 2.5, 6)    # Sparse for long distances
        bond_lengths = np.concatenate([short_range, long_range])
        bond_lengths = np.sort(bond_lengths)  # Ensure sorted
    
    if verbose:
        print("\n" + "=" * 60)
        print("H2 Potential Energy Surface Scan")
        print("=" * 60)
    
    # If PySCF is unavailable, we only have precomputed integrals at 0.74 Å.
    # In that case, avoid generating a misleading PES scan.
    test_integrals = get_h2_integrals(0.74)
    if not test_integrals['from_pyscf']:
        if verbose:
            print("\nPySCF not available. PES scan requires bond-length-dependent integrals.")
            print("Using a single-point result at 0.74 Å instead of a scan.")
        bond_lengths = np.array([0.74])
        energies_vqe = []
        energies_hf = []
        energies_fci = []
        integrals = test_integrals
        results = run_h2_vqe(integrals, verbose=False)
        energies_vqe.append(results['vqe_energy'])
        energies_hf.append(results['hf_energy'])
        energies_fci.append(results['fci_energy'])

        min_idx = 0
        if verbose:
            print("\n" + "-" * 60)
            print(f"Equilibrium bond length: {bond_lengths[min_idx]:.2f} Å")
            print(f"Equilibrium energy (FCI): {energies_fci[min_idx]:.6f} Ha")

        return {
            'bond_lengths': bond_lengths,
            'vqe_energies': np.array(energies_vqe),
            'hf_energies': np.array(energies_hf),
            'fci_energies': np.array(energies_fci),
            'equilibrium_r': bond_lengths[min_idx],
            'equilibrium_e': energies_fci[min_idx]
        }

    energies_vqe = []
    energies_hf = []
    energies_fci = []
    
    for r in bond_lengths:
        if verbose:
            print(f"\nBond length: {r:.2f} Å")
        
        integrals = get_h2_integrals(r)
        results = run_h2_vqe(integrals, verbose=False)
        
        energies_vqe.append(results['vqe_energy'])
        energies_hf.append(results['hf_energy'])
        energies_fci.append(results['fci_energy'])
        
        if verbose:
            print(f"  HF:  {results['hf_energy']:.6f}  VQE: {results['vqe_energy']:.6f}  "
                  f"FCI: {results['fci_energy']:.6f}")
    
    # Find equilibrium (minimum energy)
    # Note: H2 experimental equilibrium is ~0.74 Å
    min_idx = np.argmin(energies_fci)
    equilibrium_r = bond_lengths[min_idx]
    
    if verbose:
        print("\n" + "-" * 60)
        print(f"Equilibrium bond length: {equilibrium_r:.2f} Å")
        print(f"Equilibrium energy (FCI): {energies_fci[min_idx]:.6f} Ha")
        if abs(equilibrium_r - 0.74) > 0.1:
            print(f"  Note: Expected equilibrium ~0.74 Å. Consider finer sampling.")
    
    return {
        'bond_lengths': bond_lengths,
        'vqe_energies': np.array(energies_vqe),
        'hf_energies': np.array(energies_hf),
        'fci_energies': np.array(energies_fci),
        'equilibrium_r': bond_lengths[min_idx],
        'equilibrium_e': energies_fci[min_idx]
    }


def plot_potential_energy_surface(pes_results: Dict, save_path: Optional[str] = None):
    """Plot the potential energy surface."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    r = pes_results['bond_lengths']
    
    ax.plot(r, pes_results['hf_energies'], 'g--', linewidth=2, label='Hartree-Fock')
    ax.plot(r, pes_results['vqe_energies'], 'b-o', linewidth=2, markersize=6, label='VQE')
    ax.plot(r, pes_results['fci_energies'], 'r--', linewidth=2, label='FCI (Exact)')
    
    ax.set_xlabel('Bond Length (Å)', fontsize=12)
    ax.set_ylabel('Energy (Hartree)', fontsize=12)
    ax.set_title('H2 Potential Energy Surface', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Mark equilibrium
    ax.axvline(x=pes_results['equilibrium_r'], color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


# ============================================================================
# Main Tutorial Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" H2 VQE TUTORIAL: From Molecular Integrals to Ground State Energy")
    print("=" * 70)
    
    # Step 1: Get molecular integrals
    print("\n>>> Step 1: Loading molecular integrals...")
    integrals = get_h2_integrals(bond_length=0.74)
    
    print(f"    Number of spatial orbitals: {integrals['n_orbitals']}")
    print(f"    Number of electrons: {integrals['n_electrons']}")
    print(f"    Nuclear repulsion: {integrals['nuclear_repulsion']:.6f} Ha")
    print(f"    From PySCF: {integrals['from_pyscf']}")
    
    # Step 2: Build fermionic Hamiltonian
    print("\n>>> Step 2: Building fermionic Hamiltonian...")
    H_fermion = build_h2_fermionic_hamiltonian(integrals)
    print(f"    Number of terms: {len(H_fermion.terms)}")
    
    # Step 3: Jordan-Wigner transformation
    print("\n>>> Step 3: Applying Jordan-Wigner transformation...")
    n_qubits = 4
    H_qubit = jordan_wigner_transform(H_fermion, n_qubits)
    print(f"    Number of Pauli terms: {H_qubit.n_terms}")
    print(f"    Number of qubits: {n_qubits}")
    
    # Step 4: Verify with exact diagonalization
    print("\n>>> Step 4: Exact diagonalization check...")
    H_matrix = H_qubit.to_matrix(n_qubits)
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    print(f"    Matrix dimension: {H_matrix.shape}")
    print(f"    Lowest eigenvalue: {eigenvalues[0]:.8f} Ha")
    print(f"    FCI reference:     {integrals['fci_energy']:.8f} Ha")
    
    # Step 5: Run VQE
    print("\n>>> Step 5: Running VQE...")
    try:
        results = run_h2_vqe(integrals, verbose=True)
        
        # Plot convergence
        print("\n>>> Step 6: Plotting results...")
        try:
            plot_energy_convergence(results, save_path='h2_vqe_convergence.png')
        except Exception as e:
            print(f"    Could not create plot: {e}")
        
        # Optionally scan PES
        response = input("\n>>> Would you like to scan the potential energy surface? (y/n): ")
        if response.lower() == 'y':
            pes_results = scan_potential_energy_surface(verbose=True)
            try:
                plot_potential_energy_surface(pes_results, save_path='h2_pes.png')
            except Exception as e:
                print(f"    Could not create plot: {e}")
        
    except ImportError as e:
        print(f"\n    Cannot run VQE: {e}")
        print("    Please install PennyLane: pip install pennylane")
    
    print("\n" + "=" * 70)
    print(" Tutorial Complete!")
    print("=" * 70)
    print("""
Next Steps:
-----------
1. Modify the bond length to see how the energy changes
2. Try different ansätze (hardware-efficient, ADAPT-VQE)
3. Explore larger molecules (LiH, BeH2, H2O)
4. Study the effect of noise on VQE
5. Implement error mitigation techniques

Key Files in This Framework:
---------------------------
- core/fermion_operators.py  : Second quantization formalism
- core/qubit_mapping.py      : Jordan-Wigner, Bravyi-Kitaev
- core/molecular_integrals.py: PySCF interface
- ansatz/uccsd.py           : UCCSD ansatz
- ansatz/hardware_efficient.py: Hardware-efficient ansatz
- vqe/solver.py             : VQE optimization
""")
