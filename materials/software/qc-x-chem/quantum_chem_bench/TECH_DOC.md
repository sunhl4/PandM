# quantum_chem_bench — Technical Reference

This document supplements [README.md](README.md) with **YAML schema**, **solver constructor parameters**, and **implementation notes** for contributors.

---

## 1. Data model

### `MolSpec` (`core/interfaces.py`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `geometry` | `str` | required | PySCF atom string |
| `basis` | `str` | required | Basis set name |
| `charge` | `int` | `0` | Molecular charge |
| `spin` | `int` | `0` | \(2S\) (0 singlet, 1 doublet, …) |
| `n_active_electrons` | `(int,int)` \| `None` | `None` | `(n_alpha, n_beta)` in active space |
| `n_active_orbitals` | `int` \| `None` | `None` | Number of spatial orbitals in active space |
| `mapper_type` | `str` | `"parity"` | `"jw"`, `"parity"`, `"bk"` |
| `z2symmetry_reduction` | `bool` | `True` | Passed to quantum paths / `HamiltonianBuilder` |
| `density_fit` | `bool` | `False` | PySCF mean-field RI/J,K (`mf.density_fit`) for large bases |
| `auxbasis` | `str` \| `None` | `None` | Auxiliary basis for density fitting; `None` uses PySCF default |

### `MethodResult`

| Field | Description |
|-------|-------------|
| `method_name` | Display name |
| `energy` | Total electronic energy (Ha) |
| `corr_energy` | Correlation energy vs HF (may be recomputed in `BenchRunner`) |
| `converged` | Solver convergence flag |
| `n_qubits` | `None` for classical; qubit count for quantum |
| `wall_time` | Seconds |
| `extra` | Solver-specific dict |

### `BenchResult`

Aggregates `MethodResult` entries; `hf_energy` / `fci_energy` filled when those methods ran. `summary_table()` returns rows with optional error vs FCI in mHa.

---

## 2. YAML configuration

### Validation

`BenchConfig.from_yaml` / `from_dict` calls **`validate_bench_config`**:

- Non-empty `molecule.geometry` and `molecule.basis`
- `spin` (2S) ≥ 0
- `mapper.type` ∈ `jw`, `parity`, `bk`
- If `n_active_electrons` is set, `n_active_orbitals` must also be set (and vice versa)
- `n_alpha`, `n_beta` ≥ 0 and `n_alpha + n_beta` ≤ `2 * n_active_orbitals`

### Top-level keys

```yaml
name: optional_label

molecule:
  geometry: "..."
  basis: sto-3g
  charge: 0
  spin: 0
  n_active_electrons: [1, 1]   # optional
  n_active_orbitals: 2         # optional

solvers:
  classical: [hf, mp2, cisd, ccsd, fci]
  quantum:   [vqe_uccsd, adapt_vqe, sqd]

mapper:
  type: parity                   # jw | parity | bk
  z2symmetry_reduction: true

solver_options:
  vqe_uccsd:
    optimizer: slsqp
    max_iter: 300
  # ...
```

**Note:** `solver_options` keys **must match** the solver registry name (e.g. `vqe_uccsd`, not a display name).

### `BenchRunner` behavior

- Builds each solver via `registry.build(name, category="solver", **solver_options[name], **mapper_defaults)`.
- Mapper defaults from `config.mapper` are merged into kwargs for quantum solvers (`mapper_type`, `z2symmetry_reduction`).
- Unknown registry names log a warning and are skipped.

---

## 3. Solver parameters (constructor kwargs)

Values below are defaults or common options; see source files for full lists.

### Classical

| Name | Module | Notable kwargs |
|------|--------|----------------|
| `hf` | `hf_solver.py` | — |
| `mp2` | `mp2_solver.py` | — |
| `cisd` | `cisd_solver.py` | — |
| `ccsd` | `ccsd_solver.py` | — |
| `ccsd_t` | `ccsd_solver.py` | — |
| `fci` | `fci_solver.py` | `nroots` (default 1) |

### Quantum

| Name | Module | Notable kwargs |
|------|--------|----------------|
| `vqe_uccsd` | `vqe_solver.py` | `optimizer`, `max_iter`, `shots`, `seed`, `mapper_type`, `z2symmetry_reduction`, `k`, `reps` (base class) |
| `vqe_hea` | `vqe_solver.py` | `reps` (HEA depth), same optimizer fields |
| `vqe_kupccgsd` / `vqe_uccsd_stack` | `vqe_solver.py` | `k` (GSD stack depth); `vqe_uccsd_stack` is an alias of `vqe_kupccgsd` |
| `adapt_vqe` | `adapt_vqe_solver.py` | `max_iter`, `gradient_threshold`, `optimizer`, `vqe_max_iter`, `seed`, mapper flags |
| `qpe` | `qpe_solver.py` | `num_time_slices`, `num_iterations`, `evolution_time`, `seed`, mapper flags |
| `qpe_full` | `qpe_solver.py` | Same as `qpe`; extra resource fields in `extra` |
| `sqd` | `sqd_solver.py` | `shots`, `iterations`, `ansatz`, `optimizer`, `max_iter`, `seed`, mapper flags |
| `qse` | `qse_solver.py` | `expansion_order`, `vqe_optimizer`, `vqe_max_iter`, `seed`, mapper flags |

All quantum solvers accept optional **`seed`** (int or omitted): sets NumPy / Qiskit `algorithm_globals` for reproducible optimization and sampling where applicable.

---

## 4. Molecule pipeline

1. **`MoleculeBuilder.build(MolSpec)`** → `MolIntegrals`: RHF/ROHF, active orbital slice, `h1e`, `h2e`, `e_core`, `nelec`, `norb`.
2. **`HamiltonianBuilder.build(MolIntegrals)`** → `(SparsePauliOp, nelec, norb)` for quantum solvers.
3. Classical solvers that need PySCF may call PySCF APIs directly or reuse `MoleculeBuilder` / FCI helpers as in `fci_solver.py`.

**Frozen core:** Implemented inside `MoleculeBuilder` when a smaller active space is requested; `e_core` includes nuclear repulsion and frozen-core contributions.

---

## 5. Error mitigation (`error_mitigation/zne.py`)

- `extrapolate_zne(scale_factors, expectations, method="richardson"|"polynomial")`
- `fold_gates(circuit, scale_factor)` — global folding; used by `ZNEWrapper`
- `ZNEWrapper(energy_fn, scale_factors, extrapolation)` — calls `energy_fn` on folded circuits

Reproduction scripts use these for Science 2020–style demos; noise model details are in `reproductions/hf_sycamore_science2020/run.py`.

---

## 6. Testing markers

| Marker | Meaning |
|--------|---------|
| `requires_pyscf` | Skip if `pyscf` import fails |
| `requires_qiskit_nature` | Skip if `qiskit_nature` import fails |
| `requires_sqd` | Reserved for SQD-specific tests |
| `slow` | Long VQE/QSE tests; skip if `CI_FAST_NB` set |

---

## 7. Version

Package version is defined in `quantum_chem_bench/__init__.py` (`__version__`).
