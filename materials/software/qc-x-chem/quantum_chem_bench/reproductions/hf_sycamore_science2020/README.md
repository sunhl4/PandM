# Reproduction: HF on a Superconducting Qubit Quantum Computer (Science 2020)

## Paper

**"Hartree-Fock on a superconducting qubit quantum computer"**  
Frank Arute, Kunal Arya, Ryan Babbush, …, John M. Martinis  
*Science* 369, 1084–1089 (2020). DOI: 10.1126/science.abd3880

## What this reproduction demonstrates

1. **Error-mitigated VQE for H₂ PEC**: Ideal vs noisy (Aer depolarizing noise
   model) vs ZNE-mitigated energies across H-H bond distances.
2. **Diazene isomerization**: Energy difference between *cis* and *trans*
   diazene using VQE + ZNE on a simulated noisy device.
3. **ZNE extrapolation curves**: ⟨H⟩ vs noise scale factor λ → 0 plot.

## Algorithm summary

The paper uses a Givens-rotation ansatz (specific to paired-electron systems),
runs on Google Sycamore (53 qubits), and applies:
- **Symmetry verification** (particle-number + Z₂ parity checks)
- **N-representability corrections** to the 1-RDM
- **ZNE via gate folding** (noise amplification by factors 1, 3, 5)

This reproduction uses:
- **UCCSD ansatz** (chemically equivalent for these small systems)
- **Qiskit Aer depolarizing noise model** (proxy for Sycamore noise)
- **ZNE Richardson extrapolation** from `quantum_chem_bench.error_mitigation.zne`

## Key differences from the paper

- Paper: real superconducting device with gate-specific calibrated noise.
- Reproduction: Aer simulator with uniform depolarizing noise p = 0.5–2%.
- Paper: Givens-rotation circuit with 72 two-qubit gates for 12-qubit system.
- Reproduction: UCCSD circuit (smaller, exact same physics for H₂/diazene).

## Running

```bash
python reproductions/hf_sycamore_science2020/run.py
python reproductions/hf_sycamore_science2020/run.py --molecule diazene
python reproductions/hf_sycamore_science2020/run.py --scale-factors 1 3 5 7
```

Results saved to `reproductions/hf_sycamore_science2020/results/`.
