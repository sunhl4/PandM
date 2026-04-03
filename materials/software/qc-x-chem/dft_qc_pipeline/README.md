# dft_qc_pipeline

**DFT + Local Quantum Embedding Pipeline**

A modular, plug-and-play Python framework that combines classical DFT/HF
with quantum-computing solvers (VQE, ADAPT-VQE, SQD, NumPy-FCI) via
density-based embedding methods (SimpleCAS, DMET, AVAS, Projector).

---

## Architecture

```
YAML config
    ↓
Pipeline Orchestrator
    ├── Layer 1  ClassicalBackend   (PySCF HF / DFT)
    ├── Layer 2  EmbeddingMethod    (SimpleCAS | DMET | AVAS | Projector)
    ├── Layer 3  FragmentRegion     (atom/orbital selection)
    ├── Layer 4  Hamiltonian        (localiser → FermionicOp → qubit mapper)
    ├── Layer 5  QuantumSolver      (NumPy | VQE | ADAPT-VQE | SQD)
    └── Layer 6  Post-processing    (RDM | Benchmark | ML export | inter-fragment 估计)
```

Every layer is **registry-driven**: swap methods by changing one line in a
YAML file or `config.solver.type = "sqd"` in Python.

---

## Documentation

工作区根目录 **[`../README.md`](../README.md)**（`LearningPlan`）说明本包与 `learning_materials` 的并列关系、可编辑安装与 CI / Notebook 冒烟（含 Windows PySCF 提示）。

**姊妹包（独立）**：全分子哈密顿量上的量子–经典方法对比与论文复现见 **[`../quantum_chem_bench/README.md`](../quantum_chem_bench/README.md)**（HF～FCI、VQE/QPE/SQD 等，无嵌入层）。

| 文档 | 内容 |
|------|------|
| [TECH_DOC.md](TECH_DOC.md) | 接口说明、参数参考、使用场景、插件扩展指南、FAQ |
| [DEV_MEMORY.md](DEV_MEMORY.md) | 架构决策（ADR）、技术债 §4、路线图、**变更日志 §7**（与版本演进同步） |
| [OPTIMIZATION.md](OPTIMIZATION.md) | 审查备忘：可执行优化项 + 文首**已落地摘要**（与 §7 交叉引用） |

**论文级能量与 DMET 诊断**：`Pipeline.run()` 后查看 **`result.extra["energy_corrections"]`**（字段表见 [TECH_DOC §11a](TECH_DOC.md#11a-energy_corrections-字段说明)）。

---

## Quick Start

```bash
# Editable install (from the LearningPlan directory that contains pyproject.toml)
pip install -e .

# Or minimal deps only (on top of existing qc_chem environment)
pip install pyyaml
```

```python
# If not pip-installed, add the parent of this package to sys.path:
# import sys; sys.path.insert(0, '/path/to/LearningPlan')

from dft_qc_pipeline import Pipeline, PipelineConfig

# 1. Load config
config = PipelineConfig.from_yaml('dft_qc_pipeline/configs/h2_vqe.yaml')

# 2. Run
result = Pipeline(config).run()
print(f"Fragment energy: {result.total_energy:.8f} Ha")
# result.extra['total_energy_note'] explains when total_energy is a sum of
# fragment energies vs backend-only reference (see TECH_DOC / pipeline extra).

# 3. Swap solver (no other changes)
config.solver.type = 'vqe'
config.solver.ansatz = 'uccsd'
result_vqe = Pipeline(config).run()
```

---

## Module Overview

| Module | Description |
|--------|-------------|
| `core/interfaces.py` | All ABC definitions (`ClassicalBackend`, `EmbeddingMethod`, `QuantumSolver`, …) |
| `core/registry.py` | `@registry.register` decorator + `registry.build(cfg, category)` |
| `core/config.py` | `PipelineConfig.from_yaml()` + all `*Config` dataclasses |
| `core/pipeline.py` | `Pipeline.run()` orchestrator with DMET loop；`extra["energy_corrections"]` 汇总诊断能量项 |
| `classical_backends/pyscf_backend.py` | PySCF HF/DFT → `BackendResult`（可选 `density_fit` 加速 SCF） |
| `classical_backends/toy_backend.py` | 占位后端（单位 MO、零 HF），配合模型嵌入 |
| `embedding/simple_cas.py` | Direct CAS extraction (no embedding, baseline) |
| `embedding/dmet.py` | Schmidt bath + 1-RDM self-consistency loop |
| `embedding/avas.py` | Atomic Valence Active Space (Sayfutyarova 2017) |
| `embedding/projector.py` | Manby–Werner level-shift projection |
| `embedding/hubbard.py` | 一维单带 Hubbard（需 `toy` 后端） |
| `hamiltonian/fragment_region.py` | `FragmentRegion` dataclass |
| `hamiltonian/localizer.py` | Boys / Pipek-Mezey / IAO orbital localisation |
| `hamiltonian/builder.py` | `h1e, g2e → FermionicOp` with frozen-core folding |
| `hamiltonian/mappers.py` | JW / Parity / BK → `SparsePauliOp` |
| `quantum_solvers/numpy_solver.py` | Exact diagonalisation (FCI reference) |
| `quantum_solvers/vqe_solver.py` | UCCSD / HEA / k-UpCCGSD VQE |
| `quantum_solvers/adapt_vqe_solver.py` | ADAPT-VQE with fermionic-SD pool |
| `quantum_solvers/sqd_solver.py` | `qiskit-addon-sqd` sample-based diagonalisation |
| `postprocessing/rdm_extractor.py` | 1-RDM extraction, AO transform, population |
| `postprocessing/benchmark.py` | Multi-solver comparison table |
| `postprocessing/ml_export.py` | PES / 结果 JSONL、CSV 导出（含关键 `energy_corrections`） |
| `postprocessing/inter_fragment_estimate.py` | Mulliken 电荷与跨 region 经典 **q_i q_j / R_ij**（供 Pipeline / DMET） |

---

## Adding a Custom Solver

```python
from dft_qc_pipeline.core.interfaces import QuantumSolver, SolverResult
from dft_qc_pipeline.core.registry import registry

@registry.register("my_solver", category="solver")
class MySolver(QuantumSolver):
    def solve(self, hamiltonian, num_particles, num_spatial_orbitals):
        ...
        return SolverResult(energy=..., rdm1=..., rdm2=None)
```

Then in YAML:
```yaml
solver:
  type: my_solver
```

---

## Example Notebooks

| Notebook | What it shows |
|----------|---------------|
| `examples/01_H2_minimal.ipynb` | H₂ PEC, NumPy FCI vs VQE, solver swap in one line |
| `examples/02_N2_multisolver.ipynb` | Multi-solver benchmark, UCCSD vs HEA, dissociation curve |
| `examples/03_FeN4_DMET_SQD.ipynb` | Fe-N4 catalyst, DMET+SQD, spin states, embedding comparison |

示例 Notebook 在仓库根目录（含 `pyproject.toml`）下运行时，会自动把该根目录加入 `sys.path`。设置环境变量 **`CI_FAST_NB=1`** 可缩短 `01` / `02` / `03` 中的扫描、自旋循环、VQE 步数、SQD/DMET 负担及多嵌入对比；GitHub Actions 在 **`pytest`** 之后对三份示例 Notebook 运行 **`pytest --nbmake`**（需 **PySCF**，见 workflow）。

**单元测试**：`pytest dft_qc_pipeline/tests`。带 **`@pytest.mark.requires_pyscf`** / **`requires_qiskit_nature`** 的用例（含 **`test_energy_regression.py`** 中的参考能量）在无对应依赖时自动 skip；装全 `[dev,qc]` 与 PySCF 后可在本地或 CI 上跑通。

**论文中如何写能量**：`result.total_energy` 是多片段求解器能量之和，一般**不等于**全体系 FCI/CCSD 总能量。请使用 `result.extra["energy_corrections"]`：`backend_reference_energy_ha` / `sum_fragment_energies_ha` / `delta_backend_minus_fragments_ha` 为**诊断**；**DMET** 另有 **`dmet_correlation_potential_ha`**（各片段末次 **μ·ΔN** 之和，自洽时趋于 0）与 **`dmet_inter_fragment_ha`**（多 region 时 **Mulliken 点电荷** 经典 **q_i q_j / R_ij**，非全量子浴项）。模型名见 `dmet_*_model` 与 `notes`。详见 **TECH_DOC**。

---

## Literature Basis

| Method | Reference |
|--------|-----------|
| DMET | Knizia & Chan, PRL 109, 186404 (2012) |
| DMET + VQE/SQD | Shajan et al. arXiv:2411.09861; Patra et al. arXiv:2511.22158 |
| Projector embedding | Manby et al., JCTC 8, 2564 (2012) |
| Periodic DFT + VQE | Battaglia et al., npj CM (2024) arXiv:2404.18737 |
| AVAS | Sayfutyarova et al., JCTC 13, 4063 (2017) |
| Pt ORR + ADAPT-VQE | Di Paola et al., npj CM (2024) |
| ADAPT-VQE | Grimsley et al., Nat. Commun. 10, 3007 (2019) |
| SQD | Robledo-Moreno et al., Nat. Chem. (2024); IBM qiskit-addon-sqd |

---

## Dependencies

```
qiskit>=1.3.1
qiskit-nature>=0.7.2
qiskit-algorithms
qiskit-addon-sqd>=0.8.0   # optional; falls back to NumPy if absent
pyscf>=2.6.2
scipy numpy matplotlib pandas
pyyaml
```

All of the above except `pyyaml` are already in the `qc_chem` conda environment
(see [`learning_materials/README.md`](../learning_materials/README.md)).
