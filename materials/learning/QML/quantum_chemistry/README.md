# Quantum Chemistry Computation Framework

A comprehensive, from-scratch implementation of quantum chemistry algorithms for quantum computers.

## Overview

This framework provides educational and practical tools for:

- **Second quantization formalism**: Fermion operators and their algebra
- **Fermion-to-qubit mappings**: Jordan-Wigner, Bravyi-Kitaev
- **Molecular Hamiltonian construction**: PySCF integration
- **VQE (Variational Quantum Eigensolver)**: Multiple ansätze and optimizers
- **Tutorials**: Step-by-step learning materials

## Related curriculum (Chinese)

For a four-week **written** track on quantum chemistry foundations, classical ML for electronic structure, and quantum algorithms (VQE, mappings, etc.)—merged from the former `QC-learn` repo—see [`../docs/qc_learn/README.md`](../docs/qc_learn/README.md). That material complements this package’s code-first tutorials under `tutorials/` and `docs/`.

Repo-wide **goals and maintenance cadence** (expert path, literature → reproduction → research): [`../docs/DOMAIN_EXPERT_ROADMAP.md`](../docs/DOMAIN_EXPERT_ROADMAP.md).

**Single theory + derivations document** (generated): [`../docs/theory_and_derivations.md`](../docs/theory_and_derivations.md) — update via `python3 tools/build_theory_master.py`.  
Legacy split index: [`../docs/unified_chemistry_theory/README.md`](../docs/unified_chemistry_theory/README.md).

## Installation

```bash
# Core dependencies
pip install numpy scipy pennylane

# For molecular integrals
pip install pyscf

# Optional: Alternative frameworks
pip install openfermion qiskit-nature

# Full installation
pip install -r requirements.txt
```

## Quick Start

### 1. Simple H2 VQE Calculation

```python
from quantum_chemistry.tutorials.complete_h2_vqe import (
    get_h2_integrals,
    run_h2_vqe
)

# Get molecular integrals
integrals = get_h2_integrals(bond_length=0.74)

# Run VQE
results = run_h2_vqe(integrals, verbose=True)

print(f"VQE Energy: {results['vqe_energy']:.6f} Ha")
print(f"FCI Energy: {results['fci_energy']:.6f} Ha")
```

### 2. Using the Core Modules

```python
from quantum_chemistry.core.fermion_operators import (
    FermionOperator, creation_operator, number_operator
)
from quantum_chemistry.core.qubit_mapping import jordan_wigner

# Create number operator n_0 = a†_0 a_0
n0 = number_operator(0)

# Transform to qubits
n0_qubit = jordan_wigner(n0)
print(n0_qubit)  # (1/2)(I - Z_0)
```

### 3. Running Tutorials

```bash
# Phase 1: Second quantization basics
python -m quantum_chemistry.core.fermion_operators

# Complete H2 VQE tutorial
python -m quantum_chemistry.tutorials.01_complete_h2_vqe

# OpenFermion integration
python -m quantum_chemistry.tutorials.02_openfermion_tutorial

# PennyLane QChem
python -m quantum_chemistry.tutorials.03_pennylane_qchem_tutorial
```

## Phase 0 baseline (H₂ VQE, reproducible figures)

Fixed conventions (0.74 Å, STO-3G, JW, COBYLA) and non-interactive figure regeneration are documented in **[`docs/PHASE0_H2_BASELINE.md`](../docs/PHASE0_H2_BASELINE.md)**. To regenerate `docs/figures/h2_*.png` and sync copies to the repo root:

```bash
python quantum_chemistry/tutorials/plot_h2_baseline_figures.py
```

Full PES scans require **PySCF**; without it, the PES plot is a single-point placeholder consistent with the tutorial.

## Package Structure

```
quantum_chemistry/
├── __init__.py                 # Package initialization
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── core/                       # Core implementations
│   ├── __init__.py
│   ├── fermion_operators.py    # Second quantization
│   ├── qubit_mapping.py        # Jordan-Wigner, Bravyi-Kitaev
│   └── molecular_integrals.py  # PySCF interface
│
├── ansatz/                     # Ansatz implementations
│   ├── __init__.py
│   ├── uccsd.py               # UCCSD ansatz
│   ├── hardware_efficient.py   # Hardware-efficient ansatz
│   └── adaptive.py            # ADAPT-VQE
│
├── vqe/                        # VQE solver
│   ├── __init__.py
│   ├── solver.py              # Main VQE implementation
│   ├── optimizers.py          # COBYLA, Adam, SPSA, QNG
│   └── measurement.py         # Measurement strategies
│
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── visualization.py       # Plotting tools
│   └── analysis.py            # Analysis utilities
│
└── tutorials/                  # Learning materials
    ├── __init__.py
    ├── 01_complete_h2_vqe.py  # Complete H2 VQE walkthrough
    ├── 02_openfermion_tutorial.py
    ├── 03_pennylane_qchem_tutorial.py
    ├── 04_qiskit_nature_tutorial.py
    └── 05_solid_state_vqe.py  # Materials applications
```

## Key Concepts

### Second Quantization

Fermion operators satisfy anticommutation relations:
```
{a_p, a†_q} = δ_pq
{a_p, a_q} = 0
{a†_p, a†_q} = 0
```

The molecular Hamiltonian:
```
H = E_nuc + Σ_pq h_pq a†_p a_q + (1/2) Σ_pqrs g_pqrs a†_p a†_q a_s a_r
```

### Jordan-Wigner Transformation

Maps fermion operators to qubit operators:
```
a†_p → (1/2)(X_p - iY_p) ⊗ Z_{p-1} ⊗ ... ⊗ Z_0
a_p  → (1/2)(X_p + iY_p) ⊗ Z_{p-1} ⊗ ... ⊗ Z_0
```

### VQE Algorithm

1. Prepare parameterized state: |ψ(θ)⟩ = U(θ)|0⟩
2. Measure energy: E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
3. Update parameters with classical optimizer
4. Repeat until convergence

## Ansätze

| Ansatz | Use Case | Advantages | Disadvantages |
|--------|----------|------------|---------------|
| UCCSD | Chemistry | Physical, fast convergence | Deep circuits |
| Hardware-Efficient | NISQ devices | Shallow, flexible | Barren plateaus |
| ADAPT-VQE | Adaptive | Compact circuits | Many measurements |

## Tutorials

### Phase 1: Theory Foundation
- `core/fermion_operators.py`: Second quantization basics
- `core/qubit_mapping.py`: Jordan-Wigner transformation
- `core/molecular_integrals.py`: Building Hamiltonians

### Phase 2: Core Algorithms
- `ansatz/uccsd.py`: UCCSD ansatz design
- `vqe/solver.py`: VQE optimization
- `vqe/measurement.py`: Hamiltonian measurement

### Phase 3: External Frameworks
- `tutorials/02_openfermion_tutorial.py`: Google's OpenFermion
- `tutorials/03_pennylane_qchem_tutorial.py`: Xanadu's PennyLane
- `tutorials/04_qiskit_nature_tutorial.py`: IBM's Qiskit Nature

### Phase 4: Advanced Topics
- `tutorials/05_solid_state_vqe.py`: Materials applications
- Hubbard model
- DMET embedding
- Periodic systems

## Example Results

### H2 Molecule (STO-3G, R = 0.74 Å)

| Method | Energy (Ha) | Error (mHa) |
|--------|-------------|-------------|
| Hartree-Fock | -1.1168 | 20.5 |
| VQE (UCCSD) | -1.1372 | ~0.1 |
| FCI (Exact) | -1.1373 | 0.0 |

### Hubbard Model (2 sites, U/t = 4)

| Method | Energy | Error |
|--------|--------|-------|
| Mean-Field | -1.23 | Large |
| VQE | -1.56 | ~0.01 |
| Exact | -1.56 | 0.0 |

## References

### Key Papers
1. Peruzzo et al. (2014) - VQE original paper
2. McClean et al. (2016) - Theory of VQE
3. Kandala et al. (2017) - Hardware VQE
4. Grimsley et al. (2019) - ADAPT-VQE
5. McArdle et al. (2020) - Quantum chemistry review

### Books
- Szabo & Ostlund: Modern Quantum Chemistry
- Nielsen & Chuang: Quantum Computation

### Software Documentation
- [PennyLane QChem](https://pennylane.ai/qml/demos_quantum_chemistry.html)
- [OpenFermion](https://quantumai.google/openfermion)
- [Qiskit Nature](https://qiskit.org/ecosystem/nature/)

## Contributing

This is an educational framework. Contributions welcome:

1. Bug fixes and improvements
2. Additional tutorials
3. New ansatz implementations
4. Performance optimizations

## License

MIT License - Feel free to use for learning and research.

## Acknowledgments

This framework was created as part of a quantum chemistry learning curriculum,
integrating concepts from multiple sources and providing a unified implementation
for educational purposes.
