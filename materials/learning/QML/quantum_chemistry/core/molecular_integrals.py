"""
Molecular Integrals and Hamiltonian Construction
================================================

This module provides tools to:
1. Define molecular geometry
2. Compute molecular integrals using PySCF
3. Build the fermionic Hamiltonian in second quantization
4. Convert to qubit Hamiltonian for quantum simulation

The molecular electronic Hamiltonian in second quantization is:
    H = E_nuc + Σ_pq h_pq a†_p a_q + (1/2) Σ_pqrs g_pqrs a†_p a†_q a_s a_r

where:
- E_nuc: Nuclear repulsion energy (constant)
- h_pq: One-electron integrals (kinetic + nuclear attraction)
- g_pqrs: Two-electron integrals (electron-electron repulsion)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MolecularData:
    """
    Container for molecular data.
    
    Stores geometry, basis set, and computed integrals.
    
    Attributes:
        atoms: List of (element_symbol, (x, y, z)) tuples
        basis: Basis set name (e.g., 'sto-3g', '6-31g')
        charge: Total molecular charge
        multiplicity: Spin multiplicity (2S+1)
        n_electrons: Number of electrons
        n_orbitals: Number of molecular orbitals
        nuclear_repulsion: Nuclear repulsion energy
        one_body_integrals: h_pq matrix
        two_body_integrals: g_pqrs tensor
        hf_energy: Hartree-Fock energy
        fci_energy: Full CI energy (if computed)
    """
    atoms: List[Tuple[str, Tuple[float, float, float]]]
    basis: str = 'sto-3g'
    charge: int = 0
    multiplicity: int = 1
    
    # Computed properties
    n_electrons: int = field(default=0, init=False)
    n_orbitals: int = field(default=0, init=False)
    nuclear_repulsion: float = field(default=0.0, init=False)
    one_body_integrals: Optional[np.ndarray] = field(default=None, init=False)
    two_body_integrals: Optional[np.ndarray] = field(default=None, init=False)
    hf_energy: Optional[float] = field(default=None, init=False)
    fci_energy: Optional[float] = field(default=None, init=False)
    
    # PySCF objects (stored for reference)
    _mol: Any = field(default=None, init=False, repr=False)
    _mf: Any = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Compute number of electrons from atoms and charge."""
        atomic_numbers = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20
        }
        total_z = sum(atomic_numbers.get(atom[0], 0) for atom in self.atoms)
        self.n_electrons = total_z - self.charge
    
    @classmethod
    def from_xyz_string(cls, xyz_string: str, basis: str = 'sto-3g',
                        charge: int = 0, multiplicity: int = 1) -> 'MolecularData':
        """
        Create MolecularData from XYZ format string.
        
        Example:
            xyz = '''
            H 0.0 0.0 0.0
            H 0.0 0.0 0.74
            '''
            mol = MolecularData.from_xyz_string(xyz, basis='sto-3g')
        """
        atoms = []
        for line in xyz_string.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 4:
                element = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                atoms.append((element, (x, y, z)))
        
        return cls(atoms=atoms, basis=basis, charge=charge, multiplicity=multiplicity)
    
    @classmethod
    def h2(cls, bond_length: float = 0.74, basis: str = 'sto-3g') -> 'MolecularData':
        """Create H2 molecule with given bond length (in Angstroms)."""
        atoms = [
            ('H', (0.0, 0.0, 0.0)),
            ('H', (0.0, 0.0, bond_length))
        ]
        return cls(atoms=atoms, basis=basis)
    
    @classmethod
    def lih(cls, bond_length: float = 1.6, basis: str = 'sto-3g') -> 'MolecularData':
        """Create LiH molecule with given bond length (in Angstroms)."""
        atoms = [
            ('Li', (0.0, 0.0, 0.0)),
            ('H', (0.0, 0.0, bond_length))
        ]
        return cls(atoms=atoms, basis=basis)
    
    @classmethod
    def beh2(cls, bond_length: float = 1.3, basis: str = 'sto-3g') -> 'MolecularData':
        """Create BeH2 molecule (linear) with given Be-H bond length."""
        atoms = [
            ('H', (0.0, 0.0, -bond_length)),
            ('Be', (0.0, 0.0, 0.0)),
            ('H', (0.0, 0.0, bond_length))
        ]
        return cls(atoms=atoms, basis=basis)
    
    @classmethod
    def h2o(cls, bond_length: float = 0.96, angle: float = 104.5,
            basis: str = 'sto-3g') -> 'MolecularData':
        """Create H2O molecule with given O-H bond length and H-O-H angle."""
        # Convert angle to radians
        theta = np.radians(angle / 2)
        r = bond_length
        
        atoms = [
            ('O', (0.0, 0.0, 0.0)),
            ('H', (r * np.sin(theta), 0.0, r * np.cos(theta))),
            ('H', (-r * np.sin(theta), 0.0, r * np.cos(theta)))
        ]
        return cls(atoms=atoms, basis=basis)
    
    def get_geometry_string(self) -> str:
        """Return geometry in PySCF format."""
        lines = []
        for elem, (x, y, z) in self.atoms:
            lines.append(f"{elem} {x:.10f} {y:.10f} {z:.10f}")
        return "; ".join(lines)
    
    @property
    def n_spin_orbitals(self) -> int:
        """Number of spin orbitals (2 * n_spatial_orbitals)."""
        return 2 * self.n_orbitals if self.n_orbitals else 0


def compute_integrals_pyscf(molecular_data: MolecularData,
                            run_fci: bool = False) -> MolecularData:
    """
    Compute molecular integrals using PySCF.
    
    This function:
    1. Builds the molecule in PySCF
    2. Runs Hartree-Fock calculation
    3. Extracts one-electron and two-electron integrals
    4. Optionally runs Full CI for exact ground state
    
    Args:
        molecular_data: MolecularData object with geometry
        run_fci: Whether to compute exact FCI energy
    
    Returns:
        The same MolecularData object with integrals filled in
    """
    try:
        from pyscf import gto, scf, fci, ao2mo
    except ImportError:
        raise ImportError(
            "PySCF is required for computing molecular integrals.\n"
            "Install with: pip install pyscf"
        )
    
    # Build PySCF molecule
    mol = gto.Mole()
    mol.atom = molecular_data.get_geometry_string()
    mol.basis = molecular_data.basis
    mol.charge = molecular_data.charge
    mol.spin = molecular_data.multiplicity - 1  # PySCF uses n_alpha - n_beta
    mol.unit = 'angstrom'
    mol.build()
    
    # Store molecule object
    molecular_data._mol = mol
    molecular_data.n_orbitals = mol.nao  # Number of atomic orbitals = number of MOs in minimal basis
    molecular_data.nuclear_repulsion = mol.energy_nuc()
    
    # Run Hartree-Fock
    if molecular_data.multiplicity == 1:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    
    mf.verbose = 0
    mf.kernel()
    
    molecular_data._mf = mf
    molecular_data.hf_energy = mf.e_tot
    
    # Get MO coefficients
    C = mf.mo_coeff
    
    # One-electron integrals in MO basis
    # h_pq = <p|h|q> where h = kinetic + nuclear attraction
    h1_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    h1_mo = C.T @ h1_ao @ C
    molecular_data.one_body_integrals = h1_mo
    
    # Two-electron integrals in MO basis
    # g_pqrs = (pq|rs) = ∫∫ φ_p(1) φ_q(1) (1/r_12) φ_r(2) φ_s(2) dr_1 dr_2
    # Using physicist's notation (chemist's notation from PySCF is (pr|qs))
    eri_ao = mol.intor('int2e')
    eri_mo = ao2mo.incore.full(eri_ao, C)
    
    # Reshape to 4D tensor
    n = molecular_data.n_orbitals
    eri_mo = eri_mo.reshape((n, n, n, n))
    
    # Convert from chemist's to physicist's notation
    # chemist: (pq|rs) -> physicist: <pr|qs>
    # In chemist's notation: (pq|rs) means integral of φ_p(1)φ_q(1)φ_r(2)φ_s(2)/r12
    # We keep chemist's notation for now (easier with ao2mo)
    molecular_data.two_body_integrals = eri_mo
    
    # Optionally run FCI
    if run_fci:
        cisolver = fci.FCI(mf)
        cisolver.verbose = 0
        fci_energy, _ = cisolver.kernel()
        molecular_data.fci_energy = fci_energy
    
    return molecular_data


def build_fermionic_hamiltonian(molecular_data: MolecularData,
                                spin_orbital: bool = True):
    """
    Build the fermionic Hamiltonian from molecular integrals.
    
    The Hamiltonian in second quantization is:
        H = E_nuc + Σ_pq h_pq a†_p a_q + (1/2) Σ_pqrs g_pqrs a†_p a†_q a_s a_r
    
    Args:
        molecular_data: MolecularData with computed integrals
        spin_orbital: If True, use spin orbitals (2N orbitals for N spatial)
                     If False, use spatial orbitals with spin symmetry
    
    Returns:
        FermionOperator: The molecular Hamiltonian
    """
    from .fermion_operators import FermionOperator, number_operator, hopping_operator
    
    if molecular_data.one_body_integrals is None:
        raise ValueError("Must compute integrals first using compute_integrals_pyscf")
    
    h1 = molecular_data.one_body_integrals
    h2 = molecular_data.two_body_integrals
    E_nuc = molecular_data.nuclear_repulsion
    n_spatial = molecular_data.n_orbitals
    
    H = FermionOperator()
    
    # Add nuclear repulsion as constant (will be identity term after mapping)
    H.add_term(E_nuc, [])
    
    if spin_orbital:
        # Use spin orbitals: p = 2*i (alpha) or 2*i+1 (beta)
        n_spin = 2 * n_spatial
        
        # One-body terms: Σ_pσ,qτ h_pq δ_στ a†_pσ a_qτ
        for p in range(n_spatial):
            for q in range(n_spatial):
                if abs(h1[p, q]) > 1e-12:
                    # Alpha spin
                    H.add_term(h1[p, q], [(2*p, True), (2*q, False)])
                    # Beta spin
                    H.add_term(h1[p, q], [(2*p+1, True), (2*q+1, False)])
        
        # Two-body terms: (1/2) Σ g_pqrs a†_p a†_q a_s a_r
        # Note: We use chemist's notation (pq|rs)
        # In physicist's notation <pq|rs>, the operator is a†_p a†_q a_r a_s
        # The relationship is: <pq|rs>_phys = (pr|qs)_chem
        for p in range(n_spatial):
            for q in range(n_spatial):
                for r in range(n_spatial):
                    for s in range(n_spatial):
                        # g_pqrs in chemist's notation
                        integral = h2[p, q, r, s]
                        if abs(integral) < 1e-12:
                            continue
                        
                        # The two-electron term in spin-orbital basis
                        # with chemist's notation (pq|rs):
                        # (1/2) Σ (pq|rs) [a†_pα a†_rα a_sα a_qα +
                        #                  a†_pα a†_rβ a_sβ a_qα +
                        #                  a†_pβ a†_rα a_sα a_qβ +
                        #                  a†_pβ a†_rβ a_sβ a_qβ]
                        
                        # We need to be careful about the index ordering
                        # For two-body: a†_p a†_q a_r a_s where integral is <pq||rs>
                        # Using physicist's notation, integral <pq|rs> = (pr|qs) in chemist
                        
                        # Simplest approach: direct correspondence
                        # αα term
                        H.add_term(0.5 * integral, 
                                   [(2*p, True), (2*r, True), (2*s, False), (2*q, False)])
                        # αβ term
                        H.add_term(0.5 * integral,
                                   [(2*p, True), (2*r+1, True), (2*s+1, False), (2*q, False)])
                        # βα term
                        H.add_term(0.5 * integral,
                                   [(2*p+1, True), (2*r, True), (2*s, False), (2*q+1, False)])
                        # ββ term
                        H.add_term(0.5 * integral,
                                   [(2*p+1, True), (2*r+1, True), (2*s+1, False), (2*q+1, False)])
    else:
        # Spatial orbital version (simpler but requires spin adaptation)
        for p in range(n_spatial):
            for q in range(n_spatial):
                if abs(h1[p, q]) > 1e-12:
                    H.add_term(h1[p, q], [(p, True), (q, False)])
        
        for p in range(n_spatial):
            for q in range(n_spatial):
                for r in range(n_spatial):
                    for s in range(n_spatial):
                        integral = h2[p, q, r, s]
                        if abs(integral) > 1e-12:
                            H.add_term(0.5 * integral,
                                       [(p, True), (r, True), (s, False), (q, False)])
    
    return H


def get_qubit_hamiltonian(molecular_data: MolecularData,
                          mapping: str = 'jordan_wigner'):
    """
    Get the qubit Hamiltonian from molecular data.
    
    This is a convenience function that:
    1. Builds the fermionic Hamiltonian
    2. Applies the specified mapping
    
    Args:
        molecular_data: MolecularData with computed integrals
        mapping: 'jordan_wigner', 'bravyi_kitaev', or 'parity'
    
    Returns:
        QubitOperator: The qubit Hamiltonian
    """
    from .qubit_mapping import jordan_wigner, bravyi_kitaev, parity_mapping
    
    H_fermion = build_fermionic_hamiltonian(molecular_data, spin_orbital=True)
    
    if mapping == 'jordan_wigner':
        return jordan_wigner(H_fermion)
    elif mapping == 'bravyi_kitaev':
        return bravyi_kitaev(H_fermion)
    elif mapping == 'parity':
        return parity_mapping(H_fermion)
    else:
        raise ValueError(f"Unknown mapping: {mapping}")


def compute_exact_energy(molecular_data: MolecularData,
                         n_electrons: Optional[int] = None) -> float:
    """
    Compute exact ground state energy by diagonalizing the qubit Hamiltonian.
    
    This is useful for verifying VQE results on small systems.
    
    Args:
        molecular_data: MolecularData with computed integrals
        n_electrons: Number of electrons (for particle number sector)
    
    Returns:
        Exact ground state energy
    """
    H_qubit = get_qubit_hamiltonian(molecular_data)
    
    n_qubits = H_qubit.get_n_qubits()
    H_matrix = H_qubit.to_matrix(n_qubits)
    
    eigenvalues, _ = np.linalg.eigh(H_matrix)
    
    return eigenvalues[0]


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Molecular Hamiltonian Construction Demo")
    print("=" * 70)
    
    # Check if PySCF is available
    try:
        import pyscf
        pyscf_available = True
        print("\nPySCF found! Running full demo with real integrals.")
    except ImportError:
        pyscf_available = False
        print("\nPySCF not found. Running demo with example H2 values.")
        print("Install PySCF for full functionality: pip install pyscf")
    
    print("\n" + "-" * 70)
    print("1. Creating H2 Molecule")
    print("-" * 70)
    
    h2 = MolecularData.h2(bond_length=0.74, basis='sto-3g')
    print(f"Molecule: H2")
    print(f"Bond length: 0.74 Å")
    print(f"Basis: {h2.basis}")
    print(f"Number of electrons: {h2.n_electrons}")
    print(f"Geometry: {h2.get_geometry_string()}")
    
    if pyscf_available:
        print("\n" + "-" * 70)
        print("2. Computing Molecular Integrals")
        print("-" * 70)
        
        h2 = compute_integrals_pyscf(h2, run_fci=True)
        
        print(f"Number of spatial orbitals: {h2.n_orbitals}")
        print(f"Number of spin orbitals: {h2.n_spin_orbitals}")
        print(f"Nuclear repulsion energy: {h2.nuclear_repulsion:.6f} Ha")
        print(f"Hartree-Fock energy: {h2.hf_energy:.6f} Ha")
        print(f"FCI energy (exact): {h2.fci_energy:.6f} Ha")
        
        print("\nOne-electron integrals h_pq:")
        print(h2.one_body_integrals)
        
        print("\n" + "-" * 70)
        print("3. Building Fermionic Hamiltonian")
        print("-" * 70)
        
        H_fermion = build_fermionic_hamiltonian(h2, spin_orbital=True)
        print(f"Fermionic Hamiltonian has {H_fermion.n_terms} terms")
        
        print("\n" + "-" * 70)
        print("4. Converting to Qubit Hamiltonian (Jordan-Wigner)")
        print("-" * 70)
        
        from .qubit_mapping import jordan_wigner
        H_qubit = jordan_wigner(H_fermion)
        print(f"Qubit Hamiltonian has {H_qubit.n_terms} Pauli terms")
        print(f"Number of qubits needed: {H_qubit.get_n_qubits()}")
        
        print("\n" + "-" * 70)
        print("5. Exact Diagonalization")
        print("-" * 70)
        
        n_qubits = H_qubit.get_n_qubits()
        H_matrix = H_qubit.to_matrix(n_qubits)
        eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
        
        print(f"Matrix dimension: {H_matrix.shape}")
        print(f"Lowest 5 eigenvalues: {eigenvalues[:5]}")
        print(f"\nGround state energy from diagonalization: {eigenvalues[0]:.6f} Ha")
        print(f"FCI energy from PySCF:                    {h2.fci_energy:.6f} Ha")
        print(f"Difference: {abs(eigenvalues[0] - h2.fci_energy):.2e} Ha")
        
        print("\n" + "-" * 70)
        print("6. Potential Energy Surface Scan")
        print("-" * 70)
        
        bond_lengths = np.linspace(0.4, 2.5, 11)
        energies_hf = []
        energies_fci = []
        
        print("Bond Length (Å) | HF Energy (Ha) | FCI Energy (Ha)")
        print("-" * 50)
        
        for r in bond_lengths:
            mol = MolecularData.h2(bond_length=r, basis='sto-3g')
            mol = compute_integrals_pyscf(mol, run_fci=True)
            energies_hf.append(mol.hf_energy)
            energies_fci.append(mol.fci_energy)
            print(f"     {r:.2f}       |   {mol.hf_energy:.6f}   |   {mol.fci_energy:.6f}")
        
        print(f"\nEquilibrium bond length (FCI): ~{bond_lengths[np.argmin(energies_fci)]:.2f} Å")
        print(f"Minimum FCI energy: {min(energies_fci):.6f} Ha")
    
    else:
        # Demo without PySCF using hardcoded H2 values
        print("\n" + "-" * 70)
        print("2. Using Example H2 Integrals (STO-3G at 0.74 Å)")
        print("-" * 70)
        
        # These are approximate values for H2 in STO-3G
        h2.n_orbitals = 2
        h2.nuclear_repulsion = 0.7137539936876182
        h2.one_body_integrals = np.array([
            [-1.2527, -0.4761],
            [-0.4761, -0.4761]
        ])
        h2.two_body_integrals = np.zeros((2, 2, 2, 2))
        h2.two_body_integrals[0, 0, 0, 0] = 0.6746
        h2.two_body_integrals[1, 1, 1, 1] = 0.6746
        h2.two_body_integrals[0, 1, 0, 1] = 0.6636
        h2.two_body_integrals[1, 0, 1, 0] = 0.6636
        h2.two_body_integrals[0, 1, 1, 0] = 0.1813
        h2.two_body_integrals[1, 0, 0, 1] = 0.1813
        
        print("Loaded example integrals")
        print(f"Nuclear repulsion: {h2.nuclear_repulsion:.6f}")
        
        print("\n" + "-" * 70)
        print("3. Building Fermionic Hamiltonian")
        print("-" * 70)
        
        H_fermion = build_fermionic_hamiltonian(h2, spin_orbital=True)
        print(f"Fermionic Hamiltonian has {H_fermion.n_terms} terms")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("Next: Use this Hamiltonian with VQE to find the ground state")
    print("=" * 70)
