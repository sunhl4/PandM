/**
 * Initial graph data — PandM Knowledge Graph
 * Three main modules: Learning Resources / Research Plans / Software Engineering
 *
 * Node schema:
 *   id          – unique string
 *   label       – display label (English)
 *   type        – 'root' | 'module' | 'category' | 'topic' | 'leaf'
 *   description – one-line summary
 *   content     – markdown shown in NodePanel
 *   tags        – string[]
 *   links       – { label, url }[]
 *
 * Edge schema:
 *   source / target – node ids
 *   type            – undefined (hierarchy) | 'cross' (dashed cross-module)
 */

// Notebook 链接用 raw + nbviewer /urls/，避免 github/blob 解析在部分环境下 404（如 “not found among 60 files”）
const BASE = 'https://nbviewer.org/urls/raw.githubusercontent.com/sunhl4/PandM/main/materials'
const RAW  = 'https://github.com/sunhl4/PandM/blob/main/materials'

export const INITIAL_GRAPH = {
  nodes: [
    /* ─── ROOT ─────────────────────────────────────────────── */
    {
      id: 'root',
      label: 'Knowledge Universe',
      type: 'root',
      description: 'Three modules: Learning · Research Plans · Software Engineering',
      content: `# Knowledge Universe

Personal knowledge management system. Three core modules:

| Module | Branches | Focus |
|--------|----------|-------|
| **Learning Resources** | 6 directions | Classical Chem · QC · ML + 3 cross-disciplinary |
| **Research Plans** | 2 directions | QC×Comp.Chem · QC×ML |
| **Software Engineering** | 3 directions | QC×Comp.Chem · QC×ML · QC×Mol.Dynamics |

Click any node to expand, double-click to focus subgraph.`,
      tags: ['navigation', 'system'],
      links: [],
    },

    /* ════════════════════════════════════════════════════════
       MODULE 1: Learning Resources
       ════════════════════════════════════════════════════════ */
    {
      id: 'learning',
      label: 'Learning Resources',
      type: 'module',
      description: 'Systematic knowledge base across 6 directions',
      content: `# Learning Resources

Six learning directions:

## Classical Foundations
- **Classical Comp. Chem.** — DFT, CCSD, Mol. Dynamics, PySCF
- **Quantum Computing** — Qubits, gates, Qiskit, NISQ landscape
- **Machine Learning** — NNP, GNN, active learning

## Cross-Disciplinary
- **QC × Comp. Chem.** — VQE, SQD/SKQD, ADAPT-VQE, QPE, DMET
- **QC × Machine Learning** — QML, quantum kernels, VQC, PennyLane
- **QC × Mol. Dynamics** — Quantum-accuracy PES, QM/MM, enhanced sampling

## Literature Archive
\`materials/learning/quantum-chem/literature/\`  
5 papers with Chinese translations and Q&A files.`,
      tags: ['learning', 'knowledge base'],
      links: [],
    },

    /* ── 1-1  Classical Comp. Chem. ────────────────────────── */
    {
      id: 'learn-classical-chem',
      label: 'Classical Comp. Chem.',
      type: 'category',
      description: 'DFT, HF, CCSD, molecular dynamics, PySCF toolchain',
      content: `# Classical Computational Chemistry

Foundation and benchmark for quantum methods. Understanding classical limits reveals quantum entry points.

## Theory Hierarchy

\`\`\`
HF → MP2 → CCSD → CCSD(T) → FCI
DFT (LDA → GGA → Hybrid → meta-GGA)
\`\`\`

## Key Concepts

- **Born-Oppenheimer approximation** — nuclear/electronic separation
- **Basis sets** (STO-3G, 6-31G, cc-pVDZ)
- **Active space** (CASSCF, CASPT2, NEVPT2)
- **Embedding** (DMET, DFT/WF)

## Molecular Dynamics

| Type | Representatives | Feature |
|------|----------------|---------|
| Classical MD | AMBER, CHARMM, ReaxFF | Fast, limited accuracy |
| AIMD | Born-Oppenheimer MD | DFT/QM force-driven |
| Enhanced sampling | metadynamics, REMD | Conformational exploration |
| Neural network FF | DeepMD, NequIP, MACE | High accuracy + efficiency |

## Toolchain

- **PySCF** — Python ab initio framework
- **Psi4 / ORCA** — Advanced quantum chemistry
- **LAMMPS / GROMACS** — MD simulation engines`,
      tags: ['DFT', 'HF', 'CCSD', 'PySCF', 'MD', 'active space'],
      links: [],
    },

    /* ── 1-2  Quantum Computing ─────────────────────────────── */
    {
      id: 'learn-qc',
      label: 'Quantum Computing',
      type: 'category',
      description: 'Qubits, gates, Qiskit basics, NISQ landscape',
      content: `# Quantum Computing Fundamentals

## Core Concepts

- **Qubits** — superposition, entanglement, measurement collapse
- **Quantum gates** — H, X, CNOT, Rz, universal gate sets
- **Quantum circuits** — circuit construction, depth, noise models

## Quantum Chemistry Mapping

| Classical concept | Quantum computing analog |
|-------------------|--------------------------|
| Hamiltonian | Pauli string operator |
| Wavefunction | Quantum state \`|ψ⟩\` |
| Expectation value | Quantum measurement |
| Ground state energy | Minimum expectation |

## NISQ Landscape

- 50–1000 noisy qubits today
- Shallow variational circuits (VQE, ADAPT-VQE)
- Sampling-based methods (SQD)
- Hybrid quantum-classical workflows

## Learning Materials

\`materials/learning/quantum-computing/Phase1_Fundamentals/\`
1. \`01_量子计算基础与概念映射.ipynb\`
2. \`02_Qiskit入门实践.ipynb\``,
      tags: ['Qiskit', 'NISQ', 'qubit', 'quantum gate', 'quantum circuit'],
      links: [
        { label: 'Phase1: QC Fundamentals', url: `${BASE}/learning/quantum-computing/Phase1_Fundamentals/01_量子计算基础与概念映射.ipynb` },
        { label: 'Phase1: Qiskit Practice', url: `${BASE}/learning/quantum-computing/Phase1_Fundamentals/02_Qiskit入门实践.ipynb` },
      ],
    },
    {
      id: 'phase1',
      label: 'Phase 1: Fundamentals',
      type: 'leaf',
      description: 'QC fundamentals & concept mapping, Qiskit practice',
      content: `# Phase 1 — Quantum Computing Fundamentals

## Learning Objectives

Build a conceptual bridge between quantum computing and quantum chemistry; master basic Qiskit operations.

## Notebook 1: QC Fundamentals & Concept Mapping
- Quantum states, superposition, entanglement
- Basic gates (H, X, CNOT, Rz)
- Hamiltonian → Pauli string mapping

## Notebook 2: Qiskit Hands-on Practice
- QuantumCircuit construction & visualization
- Quantum measurement statistics
- Statevector / AerSimulator usage

## Path

\`materials/learning/quantum-computing/Phase1_Fundamentals/\``,
      tags: ['Qiskit', 'qubit', 'Phase1', 'fundamentals'],
      links: [
        { label: '01_QC Fundamentals', url: `${BASE}/learning/quantum-computing/Phase1_Fundamentals/01_量子计算基础与概念映射.ipynb` },
        { label: '02_Qiskit Practice', url: `${BASE}/learning/quantum-computing/Phase1_Fundamentals/02_Qiskit入门实践.ipynb` },
      ],
    },

    /* ── 1-3  Machine Learning ──────────────────────────────── */
    {
      id: 'learn-ml',
      label: 'Machine Learning',
      type: 'category',
      description: 'NNP, GNN, active learning, transfer learning',
      content: `# Machine Learning for Computational Chemistry

## Main Applications

- **Neural Network Potentials (NNP)** — DeepMD-kit, NequIP, MACE
- **Molecular property prediction** — HOMO/LUMO, solvation energy
- **Graph Neural Networks (GNN)** — SchNet, DimeNet, PaiNN
- **Active learning** — uncertainty sampling for training set construction

## Input Representations

- ACSF / SOAP descriptors
- E(3)-equivariant graph neural networks
- Atomic coordinates + element types

## Quantum-Accuracy Data Generation

\`\`\`
CCSD(T) / VQE → PES data points → NNP training → MD simulation
\`\`\`

This is one of the most actionable near-term industrial value paths for quantum computing: quantum computers don't need to cover all data points—only the most critical, scarce high-value ones.`,
      tags: ['NNP', 'GNN', 'DeepMD', 'active learning', 'MACE', 'SOAP'],
      links: [],
    },

    /* ── 1-4  QC × Comp. Chem. ──────────────────────────────── */
    {
      id: 'learn-qc-chem',
      label: 'QC × Comp. Chem.',
      type: 'category',
      description: 'VQE, SQD, ADAPT-VQE, QPE, DMET, TC, GBS, Gibbs, CQE, NEO — full method landscape',
      content: `# Quantum Computing × Computational Chemistry

**Core goal**: Solve strongly-correlated electronic structure problems on quantum computers.

## Algorithm Landscape (to 2026-04)

### NISQ / Near-Term Algorithms
- **VQE** — Variational Quantum Eigensolver (+ UCCSD, HEA, OO-VQE ansätze)
- **ADAPT-VQE** — Adaptive ansatz construction via gradient selection
- **SQD** — Sample-based Quantum Diagonalization
- **SKQD** — Krylov-enhanced version of SQD
- **VQD / QSE / qEOM** — Excited states & spectroscopy
- **TC-VQE** — Transcorrelated method for chemical accuracy on NISQ
- **CQE** — Contracted quantum eigensolver (RDM-based)
- **Classical shadows** — Measurement cost reduction for VQE

### Fault-Tolerant Algorithms
- **QPE** — Quantum Phase Estimation (qubitization, LCU, block encoding)
- **Low-rank factorization / THC / SCDF** — Hamiltonian compression

### Specialized Platforms
- **GBS** — Gaussian Boson Sampling for vibronic spectra (photonic)
- **Quantum annealing** — Molecular conformation, docking (D-Wave)

### Supporting Methods
- **Fermionic encodings** — JW, BK, Parity, tapering
- **Embedding / DMET** — Active-space partitioning
- **Error mitigation** — ZNE, PEC, symmetry verification
- **Finite-temperature** — Gibbs state preparation, quantum MCMC
- **NEO-VQE** — Nuclear quantum effects, proton tunneling

## Quantum Embedding

\`DFT → DMET → active-space Hamiltonian → VQE / SQD / QPE\`

## 4-Phase Learning Path

| Phase | Content | Path |
|-------|---------|------|
| 1 | Qubits, gates, concept mapping | \`learning/quantum-computing/\` |
| 2 | VQE, H₂/LiH | \`learning/quantum-chem/learning-ms/\` |
| 3 | ADAPT-VQE, SQD, quantum advantage | \`learning/quantum-chem/learning-ms/\` |
| 4 | Catalysis, force field applications | \`software/qc-x-chem/Phase4_Applications/\` |

## Full Method Reference

See: \`literature/量子计算在计算化学中的方法与文献地图.md\``,
      tags: ['VQE', 'ADAPT-VQE', 'SQD', 'QPE', 'DMET', 'TC', 'GBS', 'Gibbs', 'CQE', 'NEO', 'quantum embedding', 'barren plateau'],
      links: [
        { label: 'Phase2: VQE Theory & Impl.', url: `${BASE}/learning/quantum-chem/learning-ms/01_VQE原理与实现.ipynb` },
        { label: 'Phase2: H2/LiH', url: `${BASE}/learning/quantum-chem/learning-ms/02_Qiskit_Nature_H2_LiH.ipynb` },
        { label: 'Phase3: Advanced Algorithms', url: `${BASE}/learning/quantum-chem/learning-ms/01_进阶算法综述.ipynb` },
        { label: 'Phase3: Quantum Advantage', url: `${BASE}/learning/quantum-chem/learning-ms/02_量子优势分析.ipynb` },
        { label: 'Method & Literature Map', url: `${RAW}/learning/quantum-chem/literature/量子计算在计算化学中的方法与文献地图.md` },
      ],
    },
    {
      id: 'qc-vqe',
      label: 'VQE Theory & Impl.',
      type: 'leaf',
      description: 'Variational Quantum Eigensolver — theory, ansatz, Qiskit implementation',
      content: `# VQE — Variational Quantum Eigensolver

## Core Idea

Variational principle: $\\langle \\psi(\\theta) | H | \\psi(\\theta) \\rangle \\geq E_0$

Minimize energy expectation by optimizing parameters $\\theta$ to approximate the ground state.

## Algorithm Flow

\`\`\`
Initial θ
  ↓
Prepare |ψ(θ)⟩  (ansatz circuit)
  ↓
Measure Pauli strings ⟨H⟩
  ↓
Classical optimizer updates θ (COBYLA / BFGS / Adam)
  ↓
Convergence → approximate ground-state energy
\`\`\`

## Ansatz Types

| Ansatz | Characteristics |
|--------|----------------|
| UCCSD | Chemistry-inspired, compact params, deep circuit |
| HEA | Hardware-efficient, shallow, limited expressibility |
| ADAPT | Adaptive, adds operators on demand |

## Path

\`materials/learning/quantum-chem/learning-ms/01_VQE原理与实现.ipynb\``,
      tags: ['VQE', 'ansatz', 'UCCSD', 'Qiskit', 'variational'],
      links: [
        { label: 'VQE Notebook', url: `${BASE}/learning/quantum-chem/learning-ms/01_VQE原理与实现.ipynb` },
      ],
    },
    {
      id: 'qc-sqd',
      label: 'SQD / SKQD',
      type: 'leaf',
      description: 'Sample-based Quantum Diagonalization + Krylov enhancement',
      content: `# SQD — Sample-based Quantum Diagonalization

## Core Idea

\`Bitstrings → valid configurations → projected subspace → classical diagonalization\`

## Workflow

1. Quantum circuit produces bitstring distribution
2. Filter bitstrings satisfying particle-number / spin symmetry
3. Project onto the subspace spanned by valid configurations
4. Classical sparse diagonalization (Lanczos) in the subspace

## Advantages over VQE

| | VQE | SQD |
|--|-----|-----|
| Quantum output | Energy expectation | Bitstring / config samples |
| Classical work | Optimizer loop | Subspace construction + diag. |
| AI interface | Weak | Strong (bitstrings as quantum data) |
| NISQ-friendliness | Medium | High |

## SKQD Enhancement

Introduces Krylov time-evolved states to SQD:
- Systematic subspace expansion via time evolution
- Combines with qDRIFT for Hamiltonian simulation
- Stronger theoretical convergence structure

## Path

\`materials/learning/quantum-chem/literature/SQD.md\``,
      tags: ['SQD', 'SKQD', 'quantum data', 'diagonalization', 'Krylov'],
      links: [
        { label: 'SQD.md', url: `${RAW}/learning/quantum-chem/literature/SQD.md` },
      ],
    },
    {
      id: 'qc-frontier',
      label: 'Frontier Methods (2026)',
      type: 'topic',
      description: 'Advanced & emerging QC-chemistry methods: TC, GBS, Gibbs, CQE, NEO, periodic, classical shadows',
      content: `# Frontier & Advanced Methods in QC × Chemistry (to 2026-04)

Beyond the core VQE / SQD / QPE triad, these directions are actively shaping the field.

## Method Overview

| Family | Key representatives | Status |
|--------|--------------------|----|
| **Fermionic encodings** | JW, Bravyi-Kitaev, Parity, tapering | Foundation layer |
| **Transcorrelated (TC)** | TC-VQE, TC+AVQITE | 2024 IBM milestone |
| **Barren plateaus** | mitigation: ADAPT, local cost, layered init | Critical VQE scaling challenge |
| **OO-VQE / SA-OO-VQE** | orbital optimization + circuit params | 2024 JOSS paper |
| **Classical shadows** | ShadowGrouping, Pauli grouping | Cuts measurement overhead |
| **CQE** | contracted Schrödinger eq. | RDM-based, strong correlation |
| **GBS / photonic** | Gaussian Boson Sampling | Vibronic spectra, Xanadu |
| **Finite-temperature** | Gibbs state, quantum MCMC | Nature 2025 breakthrough |
| **Periodic / solid-state** | SQD for band gaps, VQE+Wannier | Band structure, defects |
| **Quantum annealing** | QAE, QUBO, D-Wave | Conformation, docking |
| **Response / gradients** | VQE forces, dipoles, freq. | Geometry opt., IR/Raman |
| **NEO-VQE** | nuclear quantum effects | Proton tunneling, H-bonds |

## Full Reference

See: \`materials/learning/quantum-chem/literature/量子计算在计算化学中的方法与文献地图.md\`  
Sections §4.0–§4.18 cover each method with key papers and GitHub links.`,
      tags: ['TC', 'GBS', 'Gibbs', 'CQE', 'NEO', 'barren plateau', 'OO-VQE', 'classical shadows', 'periodic', 'QA', 'frontier', '2026'],
      links: [
        { label: 'Full Method & Literature Map', url: `${RAW}/learning/quantum-chem/literature/量子计算在计算化学中的方法与文献地图.md` },
      ],
    },
    {
      id: 'qc-adaptvqe',
      label: 'ADAPT-VQE',
      type: 'leaf',
      description: 'Adaptive ansatz construction via gradient-driven operator selection',
      content: `# ADAPT-VQE — Adaptive Variational Quantum Eigensolver

## Core Idea

Rather than fixing the ansatz structure, **greedily select the operator with the largest gradient** from a pool and add it iteratively.

## Operator Pools

- Fermionic spin operators (FGSD)
- Pauli string operator pool

## Iteration

\`\`\`
while not converged:
    Compute gradient |∂E/∂θ| for all pool operators
    Select operator Aₖ with max gradient
    Append exp(iθₖAₖ) to ansatz
    Re-optimize all parameters
\`\`\`

## Advantages

- Far fewer parameters than UCCSD
- Can theoretically reach FCI accuracy
- Circuit depth grows adaptively

## Reference

\`materials/learning/quantum-chem/literature/ADAPT-VQE.md\``,
      tags: ['ADAPT-VQE', 'ansatz', 'variational', 'gradient'],
      links: [
        { label: 'ADAPT-VQE.md', url: `${RAW}/learning/quantum-chem/literature/ADAPT-VQE.md` },
      ],
    },

    /* ── 1-4a  Literature ───────────────────────────────────── */
    {
      id: 'literature',
      label: 'Literature',
      type: 'topic',
      description: '5 key papers with Chinese translations and Q&A files',
      content: `# Literature Archive

Path: \`materials/learning/quantum-chem/literature/\`

## Papers

| Paper | Type | Status |
|-------|------|--------|
| RevModPhys 92, 015003 | Review | ✅ zh-CN + QA |
| QC in the Age of Quantum Computing | Review | ✅ zh-CN + QA |
| Towards Quantum Advantage in Chemistry | Perspective | ✅ zh-CN + QA |
| Quantum Advantage in Computational Chemistry | Study | ✅ PDF |
| arXiv 2508.02578v2 | Preprint | ✅ zh-CN |

## Workflow Standard

\`学习问答记录.md\` — workflow, literature index, appendix A (QA template), Q&A entries

## Q&A Records

- \`QA_波函数集中性.md\` — wave-function localization Q&A
- \`学习问答记录.md\` — 1500+ lines interactive learning log`,
      tags: ['literature', 'translation', 'Q&A', 'review'],
      links: [],
    },
    {
      id: 'lit-revmodphys',
      label: 'RevModPhys 92, 015003',
      type: 'leaf',
      description: 'Quantum chemistry on quantum computers — comprehensive review',
      content: `# RevModPhys 92, 015003

**Title**: Quantum computational chemistry  
**Journal**: Reviews of Modern Physics, 2020

## Key Content

Comprehensive review of quantum algorithms for chemical simulation:
- Jordan-Wigner and Bravyi-Kitaev mappings
- VQE theory and implementations
- QPE for phase estimation
- Resource estimates for real molecules

## Files

- \`RevModPhys.92.015003.pdf\` — original
- \`RevModPhys_92_015003.zh-CN.md\` — Chinese translation
- \`RevModPhys_92_015003.QA.zh-CN.md\` — Q&A notes

## Path

\`materials/learning/quantum-chem/literature/\``,
      tags: ['RevModPhys', 'review', 'VQE', 'QPE', 'qubit mapping'],
      links: [
        { label: 'zh-CN Translation', url: `${RAW}/learning/quantum-chem/literature/RevModPhys_92_015003.zh-CN.md` },
      ],
    },
    {
      id: 'lit-qcage',
      label: 'QC in the Age of QC',
      type: 'leaf',
      description: 'Quantum chemistry in the age of quantum computing — Cao et al.',
      content: `# Quantum Chemistry in the Age of Quantum Computing

**Authors**: Cao et al.  
**Journal**: Chemical Reviews, 2019

## Key Content

- Overview of quantum algorithms for electronic structure
- Comparison of VQE, QPE, quantum phase kick-back
- Near-term and long-term perspectives
- Resource analysis for FCI-level calculations

## Files

- \`Quantum_Chemistry_in_the_Age_of_Quantum_Computing.zh-CN.md\`
- \`Quantum_Chemistry_in_the_Age_of_Quantum_Computing.QA.zh-CN.md\`

## Path

\`materials/learning/quantum-chem/literature/\``,
      tags: ['quantum chemistry', 'VQE', 'QPE', 'review', 'Chem. Reviews'],
      links: [
        { label: 'zh-CN Translation', url: `${RAW}/learning/quantum-chem/literature/Quantum_Chemistry_in_the_Age_of_Quantum_Computing.zh-CN.md` },
      ],
    },
    {
      id: 'lit-towards',
      label: 'Towards QC Advantage',
      type: 'leaf',
      description: 'iQCC-based study calibrating the quantum advantage threshold in OLED emitter calculations',
      content: `# Towards Quantum Advantage in Chemistry

**Authors**: Genin, Kwon et al. (OTI Lumionics & Samsung SAIT)  
**arXiv**: 2512.13657v2 — March 2026

## Core Contribution

Uses a large-scale classical simulation of the iQCC quantum solver (up to ~200 logical qubits, ~10⁷ two-qubit gates) to **benchmark** quantum-native methods against DFT/TD-DFT/CCSD on Ir(III)/Pt(II) phosphorescent OLED emitters, and calibrate where quantum advantage may emerge.

## Key Content

- iQCC (iterative qubit coupled-cluster): VQE-type algorithm designed for fault-tolerant hardware
- Scalability: linear wall-clock time vs. qubit count × entangler count
- Result: iQCC+PT mean absolute error ~0.05 eV, R² ≈ 0.94 vs. experiment
- Conclusion: these systems remain classically tractable to ~200 logical qubits, setting a quantum-advantage threshold

## Files

- \`2512.13657v2.pdf\` (arXiv)
- \`Towards_Quantum_Advantage_in_Chemistry.zh-CN.md\`
- \`Towards_Quantum_Advantage_in_Chemistry.QA.zh-CN.md\`

## Path

\`materials/learning/quantum-chem/literature/\``,
      tags: ['iQCC', 'quantum advantage', 'OLED', 'threshold', 'fault-tolerant', 'iQCC+PT'],
      links: [
        { label: 'arXiv 2512.13657', url: 'https://arxiv.org/abs/2512.13657' },
        { label: 'zh-CN Translation', url: `${RAW}/learning/quantum-chem/literature/Towards_Quantum_Advantage_in_Chemistry.zh-CN.md` },
      ],
    },
    {
      id: 'lit-2508',
      label: 'arXiv 2508.02578',
      type: 'leaf',
      description: 'Latest preprint — Chinese translation available',
      content: `# arXiv 2508.02578v2

**Status**: Latest preprint with Chinese translation

## Files

- \`2508.02578v2.pdf\` — original preprint
- \`2508.02578v2.zh-CN.md\` — Chinese translation

## Path

\`materials/learning/quantum-chem/literature/\``,
      tags: ['arXiv', 'preprint', 'latest'],
      links: [
        { label: 'arXiv 2508.02578', url: 'https://arxiv.org/abs/2508.02578' },
        { label: 'zh-CN Translation', url: `${RAW}/learning/quantum-chem/literature/2508.02578v2.zh-CN.md` },
      ],
    },
    {
      id: 'lit-method-map',
      label: 'Method & Lit. Map (2026-04)',
      type: 'leaf',
      description: 'Comprehensive map of quantum-computing-for-chemistry methods and references through April 2026',
      content: `# 量子计算在计算化学中的方法与文献地图（截至 2026-04）

A comprehensive, extensible reference covering all major quantum-computing approaches for computational chemistry.

## Coverage

| Section | Content |
|---------|---------|
| §2 | Full method landscape (18 method families) |
| §3 | Chemistry problem types covered |
| §4.0–4.18 | Per-method literature maps |
| §5 | 2024–2026 trend analysis |
| §6 | GitHub ecosystem (6 subsections) |
| §7–8 | Reading order & practical judgments |

## Key additions vs. prior surveys

- Fermionic encodings (JW / BK / Parity / tapering)
- Transcorrelated methods (TC-VQE, TC+AVQITE)
- Classical shadows & measurement reduction
- Contracted quantum eigensolver (CQE)
- Gaussian Boson Sampling for vibronic spectra
- Finite-temperature / Gibbs-state chemistry
- Periodic & solid-state quantum chemistry
- Quantum annealing & adiabatic QC
- Response properties, forces, gradients
- Nuclear quantum effects / NEO-VQE
- Barren plateau problem & mitigation

## File

\`materials/learning/quantum-chem/literature/量子计算在计算化学中的方法与文献地图.md\``,
      tags: ['review', 'method map', 'VQE', 'QPE', 'TC', 'GBS', 'Gibbs', 'barren plateau', 'CQE', 'NEO', 'QML', '2026'],
      links: [
        { label: 'Method & Literature Map', url: `${RAW}/learning/quantum-chem/literature/量子计算在计算化学中的方法与文献地图.md` },
      ],
    },

    /* ── 1-5  QC × Machine Learning ─────────────────────────── */
    {
      id: 'learn-qc-ml',
      label: 'QC × Machine Learning',
      type: 'category',
      description: 'QML, quantum kernels, VQC, quantum data → classical ML',
      content: `# Quantum Computing × Machine Learning

## Main Directions

### Quantum Kernel Methods
- Quantum feature map: $\\phi(x) \\to |\\phi(x)\\rangle$
- Quantum kernel matrix: $k(x,x') = |\\langle\\phi(x)|\\phi(x')\\rangle|^2$
- Quantum SVM / Quantum Kernel Ridge Regression

### Variational Quantum Classifier (VQC)
- Data encoding layer + variational layer
- Gradient-based parameter optimization

### Quantum Generative Models
- QGAN (quantum generative adversarial network)
- Quantum Boltzmann machine

### Quantum Data → Classical ML
- SQD bitstring distributions as ML features
- Configuration entropy, subspace dimension tracking
- Statistical features from quantum sampling

## Frameworks

- **PennyLane** — quantum machine learning
- **Qiskit Machine Learning** — VQC, QSVC`,
      tags: ['QML', 'quantum kernel', 'VQC', 'PennyLane', 'quantum feature map'],
      links: [],
    },

    /* ── 1-6  QC × Mol. Dynamics ────────────────────────────── */
    {
      id: 'learn-qc-md',
      label: 'QC × Mol. Dynamics',
      type: 'category',
      description: 'Quantum-accuracy PES generation, quantum-enhanced sampling, QM/MM',
      content: `# Quantum Computing × Molecular Dynamics

## Main Directions

### 1. Quantum-Accuracy PES Data Generation

Compute high-accuracy potential energy surfaces with quantum computers to train neural network force fields:

\`\`\`
VQE/SQD on critical configurations
  ↓
High-accuracy PES data points (DFT-weak regions)
  ↓
Augment NNP / ReaxFF training set
  ↓
Higher-fidelity MD simulation
\`\`\`

**Near-term viable path**: quantum computing generates high-quality training data → classical NNP runs MD.

### 2. Quantum-Enhanced Sampling

- Quantum annealing (QA) aided conformational search
- Quantum Monte Carlo
- Quantum random walk

*(Requires fault-tolerant QC — medium/long-term)*

### 3. QM/MM with Quantum Solver

Active region solved by quantum computer; environment by MM force field.

## Connection to NNP

| Quantum contribution | NNP usage |
|---------------------|-----------|
| High-accuracy TS energies | Improve reaction description |
| Strongly-correlated active-site configs | Enrich training diversity |
| Multi-spin-state energy gaps | Improve catalytic simulation |`,
      tags: ['MD', 'QM/MM', 'quantum sampling', 'NNP', 'PES', 'force field'],
      links: [],
    },

    /* ════════════════════════════════════════════════════════
       MODULE 2: Research Plans
       ════════════════════════════════════════════════════════ */
    {
      id: 'work-plan',
      label: 'Research Plans',
      type: 'module',
      description: 'Work roadmaps for two research directions',
      content: `# Research Plans

Two core research directions:

## QC × Comp. Chem.

**Logic**: \`concept mapping → few-qubit algorithms → quantum advantage → SQD quantum data → catalysis & force field\`

- **Few-Qubit Applications** — method validation, active-space workflows
- **Quantum Advantage** — when / what system / what resources beats classical
- **Chem. Applications** — strongly-correlated active sites, catalysis

## QC × Machine Learning

- **Quantum Data** — SQD bitstrings → quantum data assets → AI collaboration
- **QML & Force Fields** — quantum-accuracy PES augmentation + quantum kernel

## Full Reference

\`materials/work-plan/qc-x-chem/工作计划.md\`  
Complete research report: 10 sections including PPT outline, milestones, Q&A bank.`,
      tags: ['planning', 'roadmap', 'research direction'],
      links: [],
    },

    /* ── 2-1  QC×Comp.Chem. Plans ───────────────────────────── */
    {
      id: 'wp-qc-chem',
      label: 'QC×Comp.Chem. Plans',
      type: 'category',
      description: 'Few-qubit applications, quantum advantage, catalytic active sites',
      content: `# Research Plans: QC × Computational Chemistry

## Three Core Work Lines

### 1. Few-Qubit Applications
*"Does quantum computing still have value with few qubits?"*

- Validate algorithms and active-space workflows
- Model strongly-correlated local problems
- Build quantum data assets and hybrid QC-classical pipelines

Benchmark systems: H₂ (2Q), LiH (4Q), 2-site Hubbard (2Q), Fe-N4 toy (4Q)

### 2. Quantum Advantage
**Four-dimensional judgment**: problem type × precision × resource budget × classical baseline

- Near-term (NISQ): shallow circuits, SQD subspace, quantum embedding
- Long-term (FT): QPE → FCI-level polynomial complexity advantage

### 3. Chemical Applications
Chemistry is a natural quantum many-body problem. Strongly-correlated systems are the entry point:

| Application | System | Method |
|-------------|--------|--------|
| Heterogeneous catalysis | Fe-N4, Co-N4, TM oxide | active space + DMET embedding |
| Force field data | Critical PES points | VQE/SQD → NNP training |

## Reference

\`materials/work-plan/qc-x-chem/工作计划.md\``,
      tags: ['planning', 'few-qubit', 'quantum advantage', 'catalysis', 'active space'],
      links: [
        { label: 'Full Research Report', url: `${RAW}/work-plan/qc-x-chem/工作计划.md` },
      ],
    },
    {
      id: 'wp-fewqubit',
      label: 'Few-Qubit Applications',
      type: 'topic',
      description: 'Algorithm validation, strongly-correlated systems, active-space workflow',
      content: `# Few-Qubit Applications

**Core question**: With limited qubits, does quantum computing still have value?

## Value

- Validate algorithm correctness
- Characterize strongly-correlated local problems
- Build active-space workflows
- Generate quantum data assets for hybrid pipelines

## Benchmark Systems

| System | Size | Significance |
|--------|------|-------------|
| H₂ | 2 qubits | Minimal chemical benchmark |
| LiH | 4 qubits | Bond dissociation verification |
| 2-site Hubbard | 2 qubits | Strongly-correlated model |
| Fe-N4 toy | 4 qubits | Catalytic active site |

The real value of the few-qubit stage is **identifying methods and data pipelines that will scale**.`,
      tags: ['NISQ', 'few-qubit', 'active space', 'H2', 'LiH', 'Hubbard'],
      links: [
        { label: 'Phase2: H2/LiH Notebook', url: `${BASE}/learning/quantum-chem/learning-ms/02_Qiskit_Nature_H2_LiH.ipynb` },
      ],
    },
    {
      id: 'wp-advantage',
      label: 'Quantum Advantage',
      type: 'topic',
      description: 'Multi-dimensional analysis: precision × resource × classical baseline',
      content: `# Quantum Advantage Analysis

Quantum advantage is not simply "faster than classical" — it is a **four-dimensional judgment**:
**problem type × required precision × resource budget × classical baseline**.

## Near-Term Opportunities (NISQ)

- Shallow-circuit algorithms
- Sampling-based subspace methods (SQD)
- Quantum embedding (DMET + VQE)
- Quantum data + QML

## Long-Term Advantage (Fault-Tolerant)

- QPE: polynomial quantum complexity for FCI-level electronic structure
- Large strongly-correlated systems, FCI-accuracy
- Requires large physical qubit count and error correction overhead

## Key Judgment Dimensions

| Dimension | Key question |
|-----------|-------------|
| Problem type | Does the problem have quantum structural advantage? |
| Resource | Qubit count, gate depth, sampling shots |
| Precision | Chemical accuracy / excited states / dynamics? |
| Classical baseline | vs. HF / DFT / CCSD(T) / DMRG / FCI? |`,
      tags: ['quantum advantage', 'NISQ', 'fault-tolerant', 'QPE', 'FCI'],
      links: [
        { label: 'Quantum Advantage Notebook', url: `${BASE}/learning/quantum-chem/learning-ms/02_量子优势分析.ipynb` },
      ],
    },
    {
      id: 'wp-chemistry',
      label: 'Chem. Applications',
      type: 'topic',
      description: 'Strongly-correlated active sites, heterogeneous catalysis, force field data',
      content: `# Chemical Applications

Chemistry is a quantum many-body problem. **The problem structure is highly isomorphic with quantum computing language.**

## Target Systems

- Strongly-correlated systems, near-degenerate states
- Spin-state competition
- Bond breaking / forming processes
- Transition metal active sites (TM oxide, Fe-N4, Co-N4)

## Two Application Stories

### Heterogeneous Catalysis
\`Fe-N4 / Co-N4 / TM oxide / active space / embedding\`

DFT fails here: strong correlation + near-degeneracy + parameter sensitivity.

Quantum embedding route:
\`\`\`
PySCF (DFT full system) → DMET active-site → qubit Hamiltonian → VQE/SQD
\`\`\`

### Force Field Data Generation
\`Critical PES points → VQE/SQD high-accuracy calculation → ReaxFF/NNP/QML force field\`

Classical DFT is weakest exactly where industry needs it most.`,
      tags: ['catalysis', 'force field', 'strongly-correlated', 'DMET', 'Fe-N4'],
      links: [
        { label: 'Catalysis Notebook', url: `${BASE}/software/qc-x-chem/Phase4_Applications/01_非均相催化量子计算.ipynb` },
        { label: 'Force Field Notebook', url: `${BASE}/software/qc-x-chem/Phase4_Applications/02_量子计算辅助力场开发.ipynb` },
      ],
    },

    /* ── 2-2  QC×ML Plans ───────────────────────────────────── */
    {
      id: 'wp-qc-ml',
      label: 'QC×ML Plans',
      type: 'category',
      description: 'Quantum data + AI, QML force fields, bitstring feature engineering',
      content: `# Research Plans: QC × Machine Learning

## Core Logic

**Quantum computing provides new high-value data and features; AI amplifies their value.**

\`Quantum data = outputs of quantum state preparation and measurement that can be exploited by physical constraints, statistical processing, and classical algorithms\`

## Two Work Lines

### Quantum Data Line (SQD → AI)
- Bitstring distribution → valid configuration recovery → subspace feature extraction
- AI-assisted configuration recovery and sample filtering
- Quantum-sample-derived features as classical ML inputs

### QML Force Field Direction
- Quantum kernel exploration on PES
- Quantum-accuracy PES data augmentation for NNP/ReaxFF
- Generating high-value sparse data points for force field training

## Four Most Viable Paths (by priority)

1. **Quantum-generated high-value training data** — highest priority
2. **AI-assisted SQD sample recovery** — engineering entry point
3. **Quantum sample-derived features** — feature engineering direction
4. **Quantum kernel methods for force fields / QML** — differentiation`,
      tags: ['quantum data', 'QML', 'SQD', 'AI', 'force field', 'bitstring'],
      links: [],
    },
    {
      id: 'wp-qdata',
      label: 'Quantum Data',
      type: 'topic',
      description: 'SQD bitstrings as quantum data assets for AI/ML collaboration',
      content: `# Quantum Data

**Why do quantum device bitstrings have value?**

## SQD Workflow

\`Bitstrings → valid configurations → projected subspace → classical diagonalization\`

The quantum device outputs not just one energy number, but:
- Bitstring distributions
- Configuration samples  
- Time-evolved sampling snapshots

## Value as Quantum Data

These samples can be **recovered, filtered, projected, and featurized**:

- **High-frequency configuration fraction** → electronic structure signal
- **Configuration entropy / distribution compactness** → correlation strength indicator
- **Subspace dimension & convergence rate** → method efficiency metric
- **Sample fidelity under physical constraints** → noise robustness measure

They enable **collaboration** with classical AI, not mere replacement.

## Reference

\`materials/learning/quantum-chem/literature/SQD.md\``,
      tags: ['SQD', 'quantum data', 'QML', 'SKQD', 'bitstring', 'configuration'],
      links: [
        { label: 'SQD.md', url: `${RAW}/learning/quantum-chem/literature/SQD.md` },
      ],
    },
    {
      id: 'wp-qml',
      label: 'QML & Force Fields',
      type: 'topic',
      description: 'Quantum kernel methods, quantum-accuracy PES augmentation, QML force fields',
      content: `# QML & Force Field Direction

## Positioning

Quantum computing's value in ML / force fields is not replacing classical ML, but:
1. **Providing high-accuracy training data inaccessible to classical methods**
2. **Exploring quantum feature-map expressibility**
3. **Differentiated competition: quantum-data-driven AI force fields**

## Quantum-Accuracy Data Augmentation

\`\`\`
Identify DFT-weak configurations
  ↓
VQE/SQD: compute quantum-accuracy energy (TS, strongly-correlated points)
  ↓
Augment NNP/ReaxFF training set with sparse high-value points
  ↓
Force field accuracy improved in critical regimes
\`\`\`

## Quantum Kernel Methods

- Quantum feature map defines kernel
- Quantum Kernel Ridge vs. classical RBF kernel
- Identify regimes where quantum kernels offer better expressibility

## Frameworks

- **PennyLane** — quantum kernel implementation
- **Qiskit Machine Learning** — VQC, QSVC
- \`quantum_chem_bench\` — benchmarking framework`,
      tags: ['QML', 'quantum kernel', 'force field', 'NNP', 'PES', 'data augmentation'],
      links: [],
    },

    /* ════════════════════════════════════════════════════════
       MODULE 3: Software Engineering
       ════════════════════════════════════════════════════════ */
    {
      id: 'software',
      label: 'Software Engineering',
      type: 'module',
      description: 'Three engineering directions with full Python packages',
      content: `# Software Engineering

Three engineering directions, all source code under \`materials/software/\`.

## QC × Comp. Chem.
- **dft_qc_pipeline** — full Python package: DFT → DMET → VQE/SQD workflow
- **quantum_chem_bench** — benchmark platform: HF/CCSD/VQE/SQD/QPE comparison
- **Reproductions** — SQD Nature Chem. 2024, HF Sycamore Science 2020
- **Phase4 Notebooks** — catalysis and force field application demos

## QC × Machine Learning
- QML toolchain (PennyLane + Qiskit ML)
- Quantum data processing pipeline
- Quantum kernel vs. classical kernel benchmarks

## QC × Mol. Dynamics
- Quantum-accuracy PES data generation workflow
- NNP training data augmentation tools
- QM/MM interface framework`,
      tags: ['software engineering', 'toolchain', 'Python package'],
      links: [],
    },

    /* ── 3-1  Soft. Eng. / QC×Comp.Chem. ───────────────────── */
    {
      id: 'sw-qc-chem',
      label: 'QC×Comp.Chem.',
      type: 'category',
      description: 'dft_qc_pipeline, quantum_chem_bench, reproductions, Phase4 notebooks',
      content: `# Software Engineering: QC × Computational Chemistry

## Engineering Assets

| Package | Description | Path |
|---------|-------------|------|
| \`dft_qc_pipeline\` | DFT → DMET → VQE/SQD pipeline | \`software/qc-x-chem/dft_qc_pipeline/\` |
| \`quantum_chem_bench\` | Multi-method benchmark platform | \`software/qc-x-chem/quantum_chem_bench/\` |
| Phase4 notebooks | Catalysis & force field demos | \`software/qc-x-chem/Phase4_Applications/\` |
| Install scripts | PySCF on Windows / WSL | \`software/qc-x-chem/\` |

## Quick Start

\`\`\`bash
cd materials/software/qc-x-chem
pip install -e .    # installs both packages via pyproject.toml
\`\`\``,
      tags: ['DMET', 'DFT', 'VQE', 'pipeline', 'benchmark', 'PySCF'],
      links: [],
    },
    {
      id: 'sw-pipeline',
      label: 'DFT-QC Pipeline',
      type: 'topic',
      description: 'dft_qc_pipeline: DFT → DMET → active-space → VQE/SQD full workflow',
      content: `# DFT-QC Pipeline

**Package**: \`dft_qc_pipeline\`  
**Path**: \`materials/software/qc-x-chem/dft_qc_pipeline/\`

## Architecture

\`\`\`
core/                   # pipeline.py, config.py, registry.py
embedding/              # dmet.py, avas.py, simple_cas.py, projector.py
hamiltonian/            # builder.py, mappers.py, localizer.py
quantum_solvers/        # vqe_solver.py, sqd_solver.py, adapt_vqe_solver.py
classical_backends/     # pyscf_backend.py, toy_backend.py
postprocessing/         # ml_export.py, rdm_extractor.py, benchmark.py
configs/                # YAML config files (h2, lih, n2, fen4, hubbard)
examples/               # 3 demo notebooks
tests/                  # 12 test modules with pytest
\`\`\`

## Typical Workflow

1. DFT full-system calculation (PySCF backend)
2. DMET active-site partitioning
3. Active-space Hamiltonian construction
4. VQE / SQD solver
5. Result export (energy, RDM, ML features)

## Example Notebooks

- \`01_H2_minimal.ipynb\` — minimal working example
- \`02_N2_multisolver.ipynb\` — multi-solver comparison
- \`03_FeN4_DMET_SQD.ipynb\` — catalytic active-site with SQD

## YAML Config Example

\`\`\`yaml
# configs/h2_vqe.yaml
molecule: h2
basis: sto-3g
embedding: simple_cas
solver: vqe
active_space: [2, 2]
\`\`\``,
      tags: ['DMET', 'DFT', 'VQE', 'SQD', 'pipeline', 'PySCF', 'Python package'],
      links: [
        { label: 'Phase4: Catalysis Demo', url: `${BASE}/software/qc-x-chem/Phase4_Applications/01_非均相催化量子计算.ipynb` },
        { label: 'Phase4: Force Field Demo', url: `${BASE}/software/qc-x-chem/Phase4_Applications/02_量子计算辅助力场开发.ipynb` },
      ],
    },
    {
      id: 'sw-bench',
      label: 'Benchmark Platform',
      type: 'topic',
      description: 'quantum_chem_bench: HF/CCSD/VQE/SQD/QPE multi-method comparison',
      content: `# Quantum Chemistry Benchmark Platform

**Package**: \`quantum_chem_bench\`  
**Path**: \`materials/software/qc-x-chem/quantum_chem_bench/\`

## Architecture

\`\`\`
core/               # runner.py, config.py, registry.py
molecule/           # builder.py, hamiltonian.py
classical_solvers/  # hf, mp2, ccsd, cisd, fci solvers
quantum_solvers/    # vqe, sqd, adapt_vqe, qpe, qse solvers
analysis/           # benchmark.py, pes_scanner.py
error_mitigation/   # zne.py (zero-noise extrapolation)
reproductions/      # two paper reproductions (see below)
configs/            # h2_sto3g.yaml, lih_sto3g.yaml, n2_631g.yaml
examples/           # 3 benchmark notebooks
tests/              # 5 test modules
\`\`\`

## Supported Methods

| Type | Methods |
|------|---------|
| Classical | HF, MP2, CCSD, CISD, FCI |
| Quantum | VQE, ADAPT-VQE, SQD, QPE, QSE |
| Error mitigation | ZNE (zero-noise extrapolation) |

## YAML Config Example

\`\`\`yaml
# configs/h2_sto3g.yaml
molecule: h2
basis: sto-3g
methods: [hf, vqe, sqd]
\`\`\``,
      tags: ['benchmark', 'VQE', 'QPE', 'SQD', 'HF', 'CCSD', 'FCI', 'ZNE'],
      links: [],
    },
    {
      id: 'sw-repro',
      label: 'Reproductions',
      type: 'topic',
      description: 'SQD Nature Chem. 2024 and HF Sycamore Science 2020 reproductions',
      content: `# Paper Reproductions

**Path**: \`materials/software/qc-x-chem/quantum_chem_bench/reproductions/\`

## SQD — Nature Chemistry 2024

\`reproductions/sqd_nat_chem_2024/\`

Reproduction of the IBM SQD paper demonstrating sample-based quantum diagonalization on real quantum hardware.

- \`run.py\` — main reproduction script
- \`README.md\` — setup and expected results

## HF Sycamore — Science 2020

\`reproductions/hf_sycamore_science2020/\`

Reproduction of the Google Hartree-Fock calculation on Sycamore quantum processor.

- \`run.py\` — main reproduction script
- \`README.md\` — methodology and results

## Purpose

- Validate \`quantum_chem_bench\` against published results
- Understand hardware-level quantum chemistry workflow
- Build intuition for noise effects and error mitigation`,
      tags: ['SQD', 'Nature Chemistry', 'Sycamore', 'HF', 'reproduction', 'Science 2020'],
      links: [
        { label: 'SQD Paper (Nature Chem. 2024)', url: 'https://www.nature.com/articles/s41557-024-01578-z' },
      ],
    },

    /* ── 3-2  Soft. Eng. / QC×ML ────────────────────────────── */
    {
      id: 'sw-qc-ml',
      label: 'QC×Machine Learning',
      type: 'category',
      description: 'QML toolchain, quantum data pipeline, kernel benchmarks',
      content: `# Software Engineering: QC × Machine Learning

**Path**: \`materials/software/qc-x-ml/\`

## Engineering Directions

### QML Toolchain
- **PennyLane** — quantum kernel methods, VQC implementation
- **Qiskit Machine Learning** — QSVC, VQC
- Quantum kernel matrix computation and visualization

### Quantum Data Processing Pipeline
- Bitstring preprocessing and valid configuration filtering
- Statistical feature extraction (configuration entropy, top-K frequency)
- Subspace dimension and convergence rate tracking
- AI-assisted configuration recovery

### Benchmark Framework
- Quantum kernel vs. classical RBF kernel
- QML molecular property prediction vs. classical GNN
- SQD quantum features vs. pure classical features`,
      tags: ['QML', 'PennyLane', 'quantum kernel', 'quantum data', 'bitstring'],
      links: [],
    },
    {
      id: 'sw-qml-pipeline',
      label: 'Quantum Data Eng.',
      type: 'leaf',
      description: 'Bitstring processing, feature engineering, AI-assisted recovery pipeline',
      content: `# Quantum Data Engineering

## Full Pipeline

\`\`\`
Quantum circuit sampling → bitstring set
  ↓
Valid configuration filtering (particle number, spin symmetry)
  ↓
Configuration recovery (rule-based + ML-assisted)
  ↓
Subspace construction → classical diagonalization
         ↓
Statistical feature extraction → classical ML input
\`\`\`

## Key Modules

### Feature Extraction
- High-frequency configuration fraction (Top-K bitstrings)
- Configuration entropy: $H = -\\sum_i p_i \\log p_i$
- Subspace dimension convergence curve
- Temporal sampling distribution differences (SKQD time evolution)

### AI-Assisted Recovery
- Rule filtering + machine learning hybrid
- Learn noisy → valid configuration mapping
- Improves SQD engineering usability

## Deliverables

- Bitstring data processing template
- SQD workflow demo
- Quantum features vs. classical features comparison`,
      tags: ['quantum data', 'bitstring', 'SQD', 'AI', 'feature engineering'],
      links: [],
    },

    /* ── 3-3  Soft. Eng. / QC×Mol.Dynamics ─────────────────── */
    {
      id: 'sw-qc-md',
      label: 'QC×Mol. Dynamics',
      type: 'category',
      description: 'Quantum-accuracy PES generation, NNP augmentation, QM/MM interface',
      content: `# Software Engineering: QC × Molecular Dynamics

**Path**: \`materials/software/qc-x-md/\`

## Engineering Directions

### Quantum-Accuracy PES Data Generation
- Identify DFT-weak configurations (strongly-correlated, transition states)
- Batch-call VQE/SQD for critical points
- PES data storage and formatting (ASE / extended XYZ)

### NNP Training Data Augmentation
- Integrate quantum data points into DeepMD-kit / NequIP / MACE training
- Active learning loop: model uncertainty regions → quantum computing supplement
- Comparative experiments: pure DFT vs. quantum-augmented data

### QM/MM Interface Framework
- PySCF (QM region) + OpenMM/LAMMPS (MM region)
- Quantum computer as QM solver (future direction)
- Active-site embedding → quantum computation → force feedback`,
      tags: ['MD', 'NNP', 'PES', 'DeepMD', 'QM/MM', 'ReaxFF'],
      links: [],
    },
    {
      id: 'sw-nnp-workflow',
      label: 'NNP Training Tools',
      type: 'leaf',
      description: 'Quantum-accuracy data points → NNP training set augmentation workflow',
      content: `# NNP Training Data Augmentation Workflow

## Workflow Design

\`\`\`
Classical MD initial sampling
  ↓
Uncertainty analysis (query-by-committee)
  ↓
Identify configurations needing high-accuracy calculation
  ↓
VQE/SQD: quantum-accuracy energy / force
  ↓
Incrementally update NNP training set
  ↓
Retrain → iterate until convergence
\`\`\`

## Target Scenarios

| Scenario | Quantum contribution |
|----------|---------------------|
| Near transition states | DFT error large; quantum accuracy determines mechanism |
| Spin-state crossing | Multi-reference effects; quantum methods have natural advantage |
| Strongly-correlated active sites | Embedding + VQE/SQD |
| Bond breaking/forming | Strong correlation + near-degeneracy, DFT fails |

## Tool Integration

- **ASE** — atomic simulation environment interface
- **PySCF + qiskit-addon-sqd** — quantum computing module
- **DeepMD-kit / MACE** — NNP training framework
- Quantum-augmented vs. pure DFT comparison template`,
      tags: ['NNP', 'PES', 'active learning', 'DeepMD', 'VQE', 'SQD'],
      links: [],
    },
  ],

  /* ─── EDGES ─────────────────────────────────────────────── */
  edges: [
    /* Root → three modules */
    { source: 'root', target: 'learning' },
    { source: 'root', target: 'work-plan' },
    { source: 'root', target: 'software' },

    /* Learning → 6 branches */
    { source: 'learning', target: 'learn-classical-chem' },
    { source: 'learning', target: 'learn-qc' },
    { source: 'learning', target: 'learn-ml' },
    { source: 'learning', target: 'learn-qc-chem' },
    { source: 'learning', target: 'learn-qc-ml' },
    { source: 'learning', target: 'learn-qc-md' },

    /* Learning → leaves */
    { source: 'learn-qc',      target: 'phase1' },
    { source: 'learn-qc-chem', target: 'qc-vqe' },
    { source: 'learn-qc-chem', target: 'qc-sqd' },
    { source: 'learn-qc-chem', target: 'qc-adaptvqe' },
    { source: 'learn-qc-chem', target: 'literature' },
    { source: 'literature',    target: 'lit-revmodphys' },
    { source: 'literature',    target: 'lit-qcage' },
    { source: 'literature',    target: 'lit-towards' },
    { source: 'literature',    target: 'lit-2508' },
    { source: 'literature',    target: 'lit-method-map' },
    { source: 'learn-qc-chem', target: 'qc-frontier' },

    /* Research Plans → 2 branches */
    { source: 'work-plan', target: 'wp-qc-chem' },
    { source: 'work-plan', target: 'wp-qc-ml' },

    /* Research Plans → topics */
    { source: 'wp-qc-chem', target: 'wp-fewqubit' },
    { source: 'wp-qc-chem', target: 'wp-advantage' },
    { source: 'wp-qc-chem', target: 'wp-chemistry' },
    { source: 'wp-qc-ml',   target: 'wp-qdata' },
    { source: 'wp-qc-ml',   target: 'wp-qml' },

    /* Software → 3 branches */
    { source: 'software', target: 'sw-qc-chem' },
    { source: 'software', target: 'sw-qc-ml' },
    { source: 'software', target: 'sw-qc-md' },

    /* Software → topics/leaves */
    { source: 'sw-qc-chem', target: 'sw-pipeline' },
    { source: 'sw-qc-chem', target: 'sw-bench' },
    { source: 'sw-qc-chem', target: 'sw-repro' },
    { source: 'sw-qc-ml',   target: 'sw-qml-pipeline' },
    { source: 'sw-qc-md',   target: 'sw-nnp-workflow' },

    /* Cross-module links (dashed) */
    { source: 'wp-qc-chem',   target: 'learn-qc-chem',      type: 'cross' },
    { source: 'wp-qc-ml',     target: 'learn-qc-ml',         type: 'cross' },
    { source: 'sw-qc-chem',   target: 'wp-qc-chem',          type: 'cross' },
    { source: 'sw-qc-ml',     target: 'wp-qc-ml',            type: 'cross' },
    { source: 'sw-qc-md',     target: 'learn-qc-md',         type: 'cross' },
    { source: 'wp-qdata',     target: 'qc-sqd',              type: 'cross' },
    { source: 'wp-chemistry',  target: 'sw-pipeline',         type: 'cross' },
    { source: 'sw-bench',     target: 'sw-repro',             type: 'cross' },
    { source: 'learn-ml',     target: 'learn-qc-ml',          type: 'cross' },
    { source: 'learn-classical-chem', target: 'learn-qc-chem', type: 'cross' },
    { source: 'sw-nnp-workflow', target: 'learn-ml',          type: 'cross' },
    { source: 'lit-towards',  target: 'wp-advantage',         type: 'cross' },
    { source: 'lit-method-map', target: 'wp-advantage',       type: 'cross' },
    { source: 'qc-frontier',   target: 'lit-method-map',      type: 'cross' },
  ],
}

export const NODE_TYPE_CONFIG = {
  root:     { color: '#00d4ff', radius: 36, glowColor: 'rgba(0,212,255,0.6)' },
  module:   { color: '#8b5cf6', radius: 28, glowColor: 'rgba(139,92,246,0.5)' },
  category: { color: '#10b981', radius: 22, glowColor: 'rgba(16,185,129,0.4)' },
  topic:    { color: '#f97316', radius: 16, glowColor: 'rgba(249,115,22,0.35)' },
  leaf:     { color: '#f472b6', radius: 12, glowColor: 'rgba(244,114,182,0.3)' },
}
