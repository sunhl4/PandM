#!/usr/bin/env python3
"""
Qiskit Nature Tutorial
======================

Qiskit Nature is IBM's framework for quantum chemistry and materials simulation.
It provides:

- Molecular problem definition
- Multiple chemistry drivers (PySCF, PSI4, Gaussian)
- Fermion-to-qubit mappers
- Built-in VQE with various ansätze
- Error mitigation techniques
- Integration with IBM Quantum hardware

This tutorial covers:
1. Setting up electronic structure problems
2. Qubit mapping strategies
3. VQE with different ansätze
4. Running on IBM quantum hardware (simulation)

Installation:
    pip install qiskit-nature qiskit-aer
    pip install pyscf  # For chemistry driver

Reference: https://qiskit.org/ecosystem/nature/
"""

from __future__ import annotations
import numpy as np

# Check Qiskit availability
try:
    import qiskit
    from qiskit import QuantumCircuit
    from qiskit_aer import Aer
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Qiskit not installed. Install with: pip install qiskit qiskit-aer")

# Check Qiskit Nature availability
NATURE_AVAILABLE = False
if QISKIT_AVAILABLE:
    try:
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
        from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
        from qiskit_nature.second_q.algorithms import GroundStateEigensolver
        from qiskit_algorithms import VQE, NumPyMinimumEigensolver
        from qiskit_algorithms.optimizers import COBYLA, SLSQP
        from qiskit.primitives import Estimator
        NATURE_AVAILABLE = True
    except ImportError:
        print("Qiskit Nature not fully installed.")
        print("Install with: pip install qiskit-nature qiskit-algorithms")


def tutorial_1_electronic_structure_problem():
    """
    Tutorial 1: Setting Up Electronic Structure Problems
    """
    print("=" * 60)
    print("Tutorial 1: Electronic Structure Problems in Qiskit Nature")
    print("=" * 60)
    
    if not QISKIT_AVAILABLE:
        print("Qiskit not available. Skipping tutorial.")
        return None
    
    if not NATURE_AVAILABLE:
        print("Qiskit Nature not available. Showing conceptual example.")
        show_conceptual_qiskit_nature()
        return None
    
    print("\n1.1 Define Molecule with PySCF Driver:")
    print("-" * 50)
    
    # Define H2 molecule
    try:
        driver = PySCFDriver(
            atom='H 0 0 0; H 0 0 0.74',
            basis='sto3g',
            charge=0,
            spin=0
        )
        
        # Run the driver to get the electronic structure problem
        problem = driver.run()
        
        print(f"Molecule: H2 at 0.74 Å")
        print(f"Number of spatial orbitals: {problem.num_spatial_orbitals}")
        print(f"Number of particles: {problem.num_particles}")
        
        # Get second-quantized operators
        second_q_op = problem.hamiltonian.second_q_op()
        print(f"Number of fermionic terms: {len(second_q_op)}")
        
        return problem
        
    except Exception as e:
        print(f"Error setting up problem: {e}")
        print("Make sure PySCF is installed: pip install pyscf")
        return None


def show_conceptual_qiskit_nature():
    """Show conceptual example when Nature is not available."""
    print("""
Qiskit Nature Structure:
------------------------

1. Driver (PySCFDriver):
   - Reads molecular geometry
   - Runs classical HF calculation
   - Extracts one/two-body integrals

2. ElectronicStructureProblem:
   - Contains Hamiltonian in second quantization
   - Stores molecular properties
   - Defines number of electrons/orbitals

3. QubitMapper (JordanWigner, Parity, BravyiKitaev):
   - Transforms fermionic H to qubit H
   - Different trade-offs in locality/depth

4. Ansatz (UCCSD, HartreeFock):
   - Parameterized quantum circuit
   - Chemistry-inspired or hardware-efficient

5. VQE:
   - Hybrid quantum-classical optimization
   - Uses Estimator primitive for expectation values
""")


def tutorial_2_qubit_mapping():
    """
    Tutorial 2: Qubit Mapping Strategies
    """
    print("\n" + "=" * 60)
    print("Tutorial 2: Qubit Mapping in Qiskit Nature")
    print("=" * 60)
    
    if not NATURE_AVAILABLE:
        print("Qiskit Nature not available. Showing mapping comparison.")
        show_mapping_comparison()
        return
    
    # Set up problem
    try:
        driver = PySCFDriver(
            atom='H 0 0 0; H 0 0 0.74',
            basis='sto3g',
            charge=0,
            spin=0
        )
        problem = driver.run()
        second_q_op = problem.hamiltonian.second_q_op()
    except Exception as e:
        print(f"Error: {e}")
        show_mapping_comparison()
        return
    
    print("\n2.1 Jordan-Wigner Mapping:")
    print("-" * 50)
    
    jw_mapper = JordanWignerMapper()
    jw_hamiltonian = jw_mapper.map(second_q_op)
    
    print(f"Number of Pauli terms: {len(jw_hamiltonian)}")
    print("First few terms:")
    for i, (pauli, coef) in enumerate(jw_hamiltonian.items()):
        if i < 5:
            print(f"  {coef.real:+.6f} * {pauli}")
    
    print("\n2.2 Parity Mapping:")
    print("-" * 50)
    
    parity_mapper = ParityMapper(num_particles=problem.num_particles)
    parity_hamiltonian = parity_mapper.map(second_q_op)
    
    print(f"Number of Pauli terms: {len(parity_hamiltonian)}")
    print("Note: Parity mapping can exploit symmetries to reduce qubits")
    
    print("\n2.3 Comparison:")
    print("-" * 50)
    
    # Number of qubits
    jw_qubits = jw_hamiltonian.num_qubits
    parity_qubits = parity_hamiltonian.num_qubits
    
    print(f"Jordan-Wigner: {jw_qubits} qubits, {len(jw_hamiltonian)} terms")
    print(f"Parity:        {parity_qubits} qubits, {len(parity_hamiltonian)} terms")


def show_mapping_comparison():
    """Show mapping comparison conceptually."""
    print("""
Qubit Mapping Comparison:
-------------------------

Mapping         | Qubits | Locality  | Best For
----------------|--------|-----------|------------------
Jordan-Wigner   | N      | Non-local | Small systems
Bravyi-Kitaev   | N      | O(log N)  | Medium systems
Parity          | N      | Varies    | Number conservation

For H2 (2 spatial orbitals = 4 spin orbitals):
- Jordan-Wigner: 4 qubits
- Parity (with 2-qubit reduction): 2 qubits!

The parity mapping can exploit:
- Particle number conservation
- Z2 symmetries
- Reduces qubit requirements
""")


def tutorial_3_vqe_with_qiskit():
    """
    Tutorial 3: VQE with Qiskit Nature
    """
    print("\n" + "=" * 60)
    print("Tutorial 3: VQE in Qiskit Nature")
    print("=" * 60)
    
    if not NATURE_AVAILABLE:
        print("Qiskit Nature not available. Showing VQE conceptually.")
        show_vqe_conceptual()
        return
    
    print("\n3.1 Setup:")
    print("-" * 50)
    
    try:
        # Create problem
        driver = PySCFDriver(
            atom='H 0 0 0; H 0 0 0.74',
            basis='sto3g'
        )
        problem = driver.run()
        
        # Map to qubits
        mapper = JordanWignerMapper()
        
        print(f"Problem: H2 molecule")
        print(f"Basis: STO-3G")
        
        print("\n3.2 Exact Solver (Reference):")
        print("-" * 50)
        
        # Exact solver for reference
        exact_solver = NumPyMinimumEigensolver()
        exact_gs_solver = GroundStateEigensolver(mapper, exact_solver)
        exact_result = exact_gs_solver.solve(problem)
        
        exact_energy = exact_result.total_energies[0]
        print(f"Exact ground state energy: {exact_energy:.8f} Ha")
        
        print("\n3.3 VQE with UCCSD Ansatz:")
        print("-" * 50)
        
        # Setup ansatz
        ansatz = UCCSD(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
            initial_state=HartreeFock(
                problem.num_spatial_orbitals,
                problem.num_particles,
                mapper
            )
        )
        
        print(f"UCCSD parameters: {ansatz.num_parameters}")
        print(f"Circuit depth: {ansatz.decompose().depth()}")
        
        # Setup VQE
        estimator = Estimator()
        optimizer = COBYLA(maxiter=200)
        
        vqe = VQE(estimator, ansatz, optimizer)
        vqe.initial_point = np.zeros(ansatz.num_parameters)
        
        # Solve
        vqe_solver = GroundStateEigensolver(mapper, vqe)
        
        print("\nRunning VQE optimization...")
        result = vqe_solver.solve(problem)
        
        vqe_energy = result.total_energies[0]
        print(f"\nVQE ground state energy: {vqe_energy:.8f} Ha")
        print(f"Exact energy:            {exact_energy:.8f} Ha")
        print(f"Error:                   {abs(vqe_energy - exact_energy):.2e} Ha")
        
        return result
        
    except Exception as e:
        print(f"Error in VQE: {e}")
        show_vqe_conceptual()
        return None


def show_vqe_conceptual():
    """Show VQE conceptually."""
    print("""
Qiskit Nature VQE Workflow:
---------------------------

1. Create Driver and Problem:
   driver = PySCFDriver(atom='H 0 0 0; H 0 0 0.74')
   problem = driver.run()

2. Choose Mapper:
   mapper = JordanWignerMapper()
   # or ParityMapper(num_particles=...)

3. Create Ansatz:
   ansatz = UCCSD(
       num_spatial_orbitals,
       num_particles,
       mapper,
       initial_state=HartreeFock(...)
   )

4. Setup VQE:
   estimator = Estimator()  # Or use Sampler
   optimizer = COBYLA(maxiter=200)
   vqe = VQE(estimator, ansatz, optimizer)

5. Solve:
   solver = GroundStateEigensolver(mapper, vqe)
   result = solver.solve(problem)

Key Components:
- Estimator: Computes expectation values
- Optimizer: Classical optimization (COBYLA, SPSA, etc.)
- Ansatz: Parameterized circuit (UCCSD, hardware-efficient)
""")


def tutorial_4_hardware_efficient():
    """
    Tutorial 4: Hardware-Efficient Ansatz
    """
    print("\n" + "=" * 60)
    print("Tutorial 4: Hardware-Efficient Ansatz")
    print("=" * 60)
    
    if not QISKIT_AVAILABLE:
        print("Qiskit not available. Skipping tutorial.")
        return
    
    print("\n4.1 Creating Hardware-Efficient Circuit:")
    print("-" * 50)
    
    from qiskit.circuit.library import EfficientSU2
    
    n_qubits = 4
    
    # EfficientSU2 is a popular hardware-efficient ansatz
    ansatz = EfficientSU2(
        num_qubits=n_qubits,
        reps=2,  # Number of repetitions
        entanglement='linear'
    )
    
    print(f"Number of qubits: {n_qubits}")
    print(f"Number of parameters: {ansatz.num_parameters}")
    print(f"Circuit depth: {ansatz.decompose().depth()}")
    
    print("\nCircuit structure:")
    print(ansatz.decompose().draw(output='text', fold=80))
    
    print("\n4.2 Different Entanglement Patterns:")
    print("-" * 50)
    
    for ent in ['linear', 'circular', 'full']:
        ans = EfficientSU2(num_qubits=4, reps=1, entanglement=ent)
        print(f"  {ent:10s}: {ans.num_parameters} params, depth {ans.decompose().depth()}")


def tutorial_5_error_mitigation():
    """
    Tutorial 5: Error Mitigation Techniques
    """
    print("\n" + "=" * 60)
    print("Tutorial 5: Error Mitigation")
    print("=" * 60)
    
    print("""
Error Mitigation in Qiskit:
---------------------------

1. Zero-Noise Extrapolation (ZNE):
   - Run circuit at multiple noise levels
   - Extrapolate to zero noise
   - Built into Estimator with resilience_level

2. Probabilistic Error Cancellation (PEC):
   - Learn the noise model
   - Apply inverse operations statistically
   - More overhead but can be more accurate

3. Twirled Readout Error Extinction (TREX):
   - Mitigate measurement errors
   - Uses randomized measurements

Usage with Qiskit Runtime:
--------------------------
from qiskit_ibm_runtime import Estimator, Options

options = Options()
options.resilience_level = 1  # Enable error mitigation
options.optimization_level = 3  # Circuit optimization

estimator = Estimator(backend=backend, options=options)

Resilience Levels:
- 0: No error mitigation
- 1: Readout error mitigation + ZNE
- 2: More aggressive mitigation
""")


def tutorial_6_ibm_quantum():
    """
    Tutorial 6: Running on IBM Quantum Hardware
    """
    print("\n" + "=" * 60)
    print("Tutorial 6: IBM Quantum Hardware")
    print("=" * 60)
    
    print("""
Running VQE on Real Hardware:
-----------------------------

1. Setup IBM Quantum Account:
   from qiskit_ibm_runtime import QiskitRuntimeService
   
   # Save your account (one-time)
   QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")
   
   # Load service
   service = QiskitRuntimeService()

2. Choose Backend:
   # List available backends
   backends = service.backends()
   
   # Get a specific backend
   backend = service.backend("ibm_brisbane")  # Example

3. Run with Qiskit Runtime:
   from qiskit_ibm_runtime import Estimator, Session
   
   with Session(service=service, backend=backend) as session:
       estimator = Estimator(session=session)
       
       # Run VQE with hardware Estimator
       vqe = VQE(estimator, ansatz, optimizer)
       result = vqe.compute_minimum_eigenvalue(hamiltonian)

Best Practices for Hardware:
----------------------------
- Use active space reduction (fewer qubits)
- Choose hardware-efficient ansatz
- Enable error mitigation
- Use transpiler optimization
- Consider qubit topology

Typical Results:
- H2: Chemical accuracy achievable (~1 mHa)
- LiH: ~10 mHa error typical with current hardware
- Larger molecules: Challenging, active research area
""")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" Qiskit Nature Tutorial")
    print(" IBM's Framework for Quantum Chemistry")
    print("=" * 70)
    
    problem = tutorial_1_electronic_structure_problem()
    tutorial_2_qubit_mapping()
    result = tutorial_3_vqe_with_qiskit()
    tutorial_4_hardware_efficient()
    tutorial_5_error_mitigation()
    tutorial_6_ibm_quantum()
    
    print("\n" + "=" * 70)
    print(" Tutorial Complete!")
    print("=" * 70)
    print("""
Key Qiskit Nature Classes:
--------------------------
- PySCFDriver: Interface to PySCF
- ElectronicStructureProblem: Problem definition
- JordanWignerMapper: Fermion-to-qubit mapping
- UCCSD: Chemistry-inspired ansatz
- VQE: Variational eigensolver
- GroundStateEigensolver: Complete workflow

Advantages of Qiskit Nature:
----------------------------
1. Direct integration with IBM quantum hardware
2. Built-in error mitigation
3. Good documentation
4. Active development
5. Support for materials (not just molecules)

Comparison:
-----------
                PennyLane       Qiskit Nature    OpenFermion
Autodiff        Yes             No               No
IBM Hardware    Via plugin      Native           Via Cirq
Active Space    Yes             Yes              Yes
Error Mitigation Limited        Built-in         External
ML Integration  Strong          Moderate         Basic

Choose based on:
- Hardware access needed
- ML integration needs
- Error mitigation requirements
""")
