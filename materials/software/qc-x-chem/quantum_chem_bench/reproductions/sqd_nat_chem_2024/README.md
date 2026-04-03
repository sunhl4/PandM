# SQD Reproduction — Robledo-Moreno et al., Nature Chemistry 2024

## Paper

**"Chemistry beyond exact solutions on a quantum-centric supercomputer"**  
Javier Robledo-Moreno, Mario Motta, Holger Haas, Ali Javadi-Abhari,
Petar Jurcevic, William Kirby, Simon Martiel, Kunal Sharma,
Sankalp Sharma, Toby Shieh, Iskandar Sitdikov, Rui-Yang Sun,
Kevin J. Sung, Maika Takita, Minh C. Tran, Nora Weaver, Zlatko K. Minev  
*Nature Chemistry* (2024). DOI: 10.1038/s41557-024-01500-5

## What this reproduction demonstrates

1. **SQD iterative convergence**: Energy vs iteration count converges toward
   FCI for H₂ and LiH (small proxies for the paper's Fe-S clusters).
2. **SQD vs classical methods**: Side-by-side comparison of SQD energy
   with HF, CISD, CCSD, and FCI.
3. **Shot scaling**: How SQD accuracy improves with more samples.

## Algorithm summary

```
Quantum Sampling                 Classical CI
─────────────────                ─────────────
Parameterised circuit   ──→   Sample bit-strings
                               ↓
                         Configuration recovery
                         (enforce correct N_e)
                               ↓
                         Selected CI matrix
                         diagonalisation
                               ↓
                         Ground-state energy E
                               ↓
                         Update circuit params
                         ← (self-consistent) ←
```

## Key differences from the paper

- The paper uses a real IBM quantum processor (127-qubit Eagle) on FeMo
  cofactor and Fe-S clusters (up to 77 spatial orbitals).
- This reproduction uses a **statevector simulator** on **H₂ / LiH**
  (2–4 spatial orbitals) for accessibility, demonstrating the same
  algorithmic behaviour at small scale.

## Running

```bash
python reproductions/sqd_nat_chem_2024/run.py
```

Results are saved to `reproductions/sqd_nat_chem_2024/results/`.
