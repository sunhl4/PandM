# quantum_chem_bench

**量子化学算法测试平台** — 在分子体系上对比经典电子结构方法与基于 Qiskit 的量子算法，采用注册表驱动的可插拔求解器接口。

---

## 目录

- [定位与关系](#定位与关系)
- [架构概览](#架构概览)
- [环境安装](#环境安装)
- [快速开始](#快速开始)
- [YAML 配置](#yaml-配置)
- [求解器注册名一览](#求解器注册名一览)
- [自定义求解器](#自定义求解器)
- [论文复现实验](#论文复现实验)
- [示例 Notebook](#示例-notebook)
- [模块说明](#模块说明)
- [测试](#测试)
- [常见问题](#常见问题)
- [参考文献](#参考文献)
- [英文摘要](#英文摘要-english-summary)

---

## 定位与关系

本包与 [`dft_qc_pipeline`](../dft_qc_pipeline/README.md) **平行独立**，**互不 import**，可单独安装与测试。

| 维度 | `dft_qc_pipeline` | `quantum_chem_bench` |
|------|-------------------|----------------------|
| 目标 | DFT/HF + 嵌入（DMET/AVAS 等）→ 片段哈密顿量 → 量子求解器 | 给定分子与活跃空间 → **全方法**能量与资源对比 |
| 经典端 | 主要提供 HF/DFT 与积分 | **HF / MP2 / CISD / CCSD / CCSD(T) / FCI**（PySCF） |
| 量子端 | VQE / ADAPT-VQE / SQD 等 | 同上 + **QPE / QSE** 等 |
| 典型用途 | 大体系、嵌入精度 | 算法正确性、与 FCI/CCSD 对照、论文复现 |

---

## 架构概览

```
YAML 配置 (BenchConfig)
        ↓
   BenchRunner.run()
        ├── 经典求解器（PySCF）：hf → mp2 → cisd → ccsd → fci …
        ├── 量子求解器（Qiskit）：vqe_* → adapt_vqe → qpe → sqd → qse …
        └── 结果聚合：BenchResult（含 HF/FCI 参考、各方法能量与耗时）

分子与哈密顿量：molecule/builder.py（积分） + molecule/hamiltonian.py（FermionicOp → SparsePauliOp）
```

所有求解器通过 **`@registry.register(..., category="solver")`** 注册，在 YAML 或 Python 中**按名字切换**，无需改框架代码。

---

## 环境安装

**Python** ≥ 3.10。

在 **`LearningPlan`** 根目录（含 `pyproject.toml`）执行：

```bash
# 最小依赖 + 量子基准常用库（Qiskit Nature 等）
pip install -e ".[bench]"

# 完整：PySCF + Aer + qiskit-addon-sqd（经典计算 + 噪声模拟 + SQD）
pip install -e ".[full]"
```

单独安装 PySCF（经典求解器与积分必需）：

```bash
pip install "pyscf>=2.6"
```

**Windows**：若 `pip install pyscf` 失败，建议使用 **Conda** 安装 `pyscf`，或参见 [`learning_materials/README.md`](../learning_materials/README.md) 中的环境说明。

可选依赖：

- `qiskit-aer`：噪声模型（`reproductions/hf_sycamore_science2020`）
- `qiskit-addon-sqd`：SQD 求解器与第一篇复现脚本
- `matplotlib` / `pandas`：图表与表格

---

## 快速开始

```python
import quantum_chem_bench as qcb

# 1. 从 YAML 加载（路径相对于当前工作目录）
config = qcb.BenchConfig.from_yaml("quantum_chem_bench/configs/h2_sto3g.yaml")

# 2. 运行配置中列出的全部求解器
bench = qcb.BenchRunner(config).run()

# 3. 打印汇总表
qcb.BenchRunner.print_summary(bench)

# 4. 相对 FCI 的能量误差柱状图（需结果中含 fci）
plotter = qcb.BenchmarkPlotter()
fig = plotter.energy_bar(bench, reference="fci")
```

**只跑部分方法**（一行改配置）：

```python
config.solvers.classical = ["hf", "fci"]
config.solvers.quantum = ["vqe_uccsd"]
bench = qcb.BenchRunner(config).run()
```

**命令行**：从 `LearningPlan` 根目录将 `PYTHONPATH` 设为当前目录，或先 `pip install -e .`，再运行复现脚本（见下文）。

---

## YAML 配置

顶层字段说明（与 `core/config.py` 一致）：

| 字段 | 说明 |
|------|------|
| `name` | 可选，本次基准的人类可读名称 |
| `molecule` | 分子：几何、基组、电荷、自旋、活跃电子与轨道 |
| `solvers` | `classical` / `quantum` 各为求解器名字符串列表 |
| `mapper` | 量子侧映射：`type` 为 `jw` / `parity` / `bk`，`z2symmetry_reduction` 布尔 |
| `solver_options` | 嵌套字典：**键 = 求解器注册名**，值为传给该求解器构造函数的 `**kwargs` |

**`molecule` 子字段：**

| 键 | 含义 |
|----|------|
| `geometry` | PySCF 风格几何字符串，如 `"H 0 0 0; H 0 0 0.735"` |
| `basis` | 基组名，如 `sto-3g`、`6-31g` |
| `charge` / `spin` | 总电荷与自旋多重度 \(2S\)（0 为单线态） |
| `n_active_electrons` | `[n_alpha, n_beta]`，省略则使用全空间（由 `MoleculeBuilder` 决定） |
| `n_active_orbitals` | 活跃空间空间轨道数，与上者配合使用 |
| `density_fit` / `auxbasis` | 可选；大基组时开启 PySCF 密度拟合（RI）及辅助基 |

**`solver_options` 示例**（键名须与注册名一致）：

```yaml
solver_options:
  vqe_uccsd:
    optimizer: slsqp
    max_iter: 300
    seed: 42
  adapt_vqe:
    max_iter: 40
    gradient_threshold: 1.0e-3
    vqe_max_iter: 200
    seed: 42
  sqd:
    shots: 10000
    iterations: 10
    ansatz: hea
    seed: 42
```

量子求解器均可选传入 **`seed`**，便于复现 VQE/采样路径（见 TECH_DOC）。

更细的参数说明见 **[TECH_DOC.md](TECH_DOC.md)**。

---

## 求解器注册名一览

### 经典（PySCF）

| 注册名 | 方法 | 备注 |
|--------|------|------|
| `hf` | Hartree-Fock | 参考态，相关能相对 HF 为 0 |
| `mp2` | MP2 | 二阶微扰 |
| `cisd` | CISD | 截断 CI |
| `ccsd` | CCSD | 耦合簇 SD |
| `ccsd_t` | CCSD(T) | 含微扰三重激发 |
| `fci` | FCI | 给定活跃空间下精确对角化，作金标准 |

### 量子（Qiskit / 混合）

| 注册名 | 方法 | 备注 |
|--------|------|------|
| `vqe_uccsd` | VQE + UCCSD | |
| `vqe_hea` | VQE + HEA | `reps` 等见 TECH_DOC |
| `vqe_kupccgsd` | VQE + k-UpCCGSD | `k` 为重复层数 |
| `adapt_vqe` | ADAPT-VQE | 依赖 `qiskit_algorithms.AdaptVQE`（若不可用会回退说明见实现） |
| `qpe` / `qpe_full` | 理想 QPE 相关 | 当前实现以精确对角化给出无噪声极限能量 |
| `sqd` | SQD | 需 `qiskit-addon-sqd`，否则可能回退 FCI |
| `qse` | QSE | 子空间展开，内含 VQE 参考态 |

---

## 自定义求解器

```python
from quantum_chem_bench.core.interfaces import BaseSolver, MethodResult, MolSpec
from quantum_chem_bench.core.registry import registry

@registry.register("my_solver", category="solver")
class MySolver(BaseSolver):
    def solve(self, mol_spec: MolSpec) -> MethodResult:
        t0 = self._start_timer()
        energy = 0.0  # 你的计算
        return MethodResult(
            method_name="my_solver",
            energy=energy,
            corr_energy=0.0,
            converged=True,
            n_qubits=None,  # 经典方法填 None
            wall_time=self._elapsed(t0),
            extra={},
        )
```

使用前需 **`import` 你的模块**以触发注册，或在包内 `__init__.py` 中导入。

YAML 中：

```yaml
solvers:
  quantum: [my_solver]
```

---

## 论文复现实验

### 1. SQD（Nature Chemistry 2024）

- 目录：`reproductions/sqd_nat_chem_2024/`
- 说明：[README](reproductions/sqd_nat_chem_2024/README.md)
- 示例：`python reproductions/sqd_nat_chem_2024/run.py --iterations 12`
- 可选：`--shot-scaling` 做 shot 缩放研究

### 2. 误差缓解 VQE / H₂ PEC / diazene（Science 2020, science.abd3880）

- 目录：`reproductions/hf_sycamore_science2020/`
- 说明：[README](reproductions/hf_sycamore_science2020/README.md)
- 示例：`python reproductions/hf_sycamore_science2020/run.py --molecule h2`
- Diazene：`--molecule diazene`（需 PySCF、Qiskit 等）

结果图默认写入各 `reproductions/.../results/`。

---

## 示例 Notebook

| 文件 | 内容 |
|------|------|
| [examples/01_H2_all_methods.ipynb](examples/01_H2_all_methods.ipynb) | H₂ 多方法对比、注册表查询 |
| [examples/02_LiH_benchmark.ipynb](examples/02_LiH_benchmark.ipynb) | LiH：CCSD / VQE / FCI |
| [examples/03_N2_dissociation.ipynb](examples/03_N2_dissociation.ipynb) | N₂ 解离曲线与误差 |

Notebook 中可设环境变量 **`CI_FAST_NB=1`** 缩短运行时间（仅保留少量求解器与扫描点）。

---

## 模块说明

| 路径 | 作用 |
|------|------|
| `core/interfaces.py` | `MolSpec`、`MethodResult`、`BenchResult`、`MolIntegrals`、`BaseSolver` |
| `core/registry.py` | 注册表 `registry` |
| `core/config.py` | `BenchConfig.from_yaml` / `from_dict` |
| `core/runner.py` | `BenchRunner` |
| `molecule/builder.py` | PySCF 积分与活跃空间 |
| `molecule/hamiltonian.py` | 费米子算符 → `SparsePauliOp` |
| `classical_solvers/` | 经典求解器实现 |
| `quantum_solvers/` | 量子求解器实现 |
| `error_mitigation/zne.py` | ZNE 外推与门折叠辅助 |
| `analysis/benchmark.py` | 表格、误差图、ZNE 图 |
| `analysis/pes_scanner.py` | 键长扫描 |

接口与参数细节见 **[TECH_DOC.md](TECH_DOC.md)**。

---

## 工程维护与记忆

- **变更记录**：与 `dft_qc_pipeline` 同仓库时，重要行为变更（新求解器注册名、YAML 校验、`MolSpec` 字段等）建议在工作区 **`dft_qc_pipeline/DEV_MEMORY.md` §7** 或本包 **README / TECH_DOC** 中留痕，便于复现与 CI 对照。
- **近期工程对齐（2026-03）**：全量子注册求解器支持可选 **`seed`**（`solver_options` 中按注册名传入）；`MolSpec` 支持 **`density_fit` / `auxbasis`**；`vqe_uccsd_stack` 为 `vqe_kupccgsd` 别名。详见 **TECH_DOC** 数据模型与求解器参数表。

---

## 测试

在 `LearningPlan` 根目录：

```bash
pytest quantum_chem_bench/tests -v
pytest quantum_chem_bench/tests -v -m "not slow"
```

未安装 PySCF / Qiskit Nature 时，带 `requires_pyscf` / `requires_qiskit_nature` 的用例会自动跳过，属预期行为。

---

## 常见问题

**Q：为什么 FCI 很慢或内存爆？**  
A：FCI 复杂度随活跃空间指数增长；请缩小 `n_active_orbitals` 或只用 `fci` 作小体系参考。

**Q：`BenchRunner` 报某求解器未注册？**  
A：需先 `import quantum_chem_bench.classical_solvers` 与 `quantum_chem_bench.quantum_solvers`（或 `import quantum_chem_bench` 包本身），以执行注册。

**Q：量子结果与 FCI 差很多？**  
A：检查活跃空间是否与经典方法一致、VQE 迭代次数、`mapper` 是否与哈密顿量构建一致；噪声实验请用 `qiskit-aer` 与复现脚本中的说明。

**Q：与 `dft_qc_pipeline` 会冲突吗？**  
A：不会。二者包名不同，仅共享可选依赖版本建议；可同时 `pip install -e .`。

---

## 参考文献

- VQE：Peruzzo *et al.*, Nat. Commun. 2014  
- ADAPT-VQE：Grimsley *et al.*, Nat. Commun. 2019  
- SQD：Robledo-Moreno *et al.*, Nat. Chem. 2024  
- HF on Sycamore + 误差缓解：Arute *et al.*, Science 2020 ([DOI: 10.1126/science.abd3880](https://doi.org/10.1126/science.abd3880))  
- 量子化学算法综述：McArdle *et al.*, Rev. Mod. Phys. 92, 015003 (2020)

---

## English summary (English Summary)

`quantum_chem_bench` is a **registry-based benchmark platform** for molecular electronic structure: classical methods (HF through FCI via PySCF) and quantum methods (VQE variants, ADAPT-VQE, QPE, SQD, QSE via Qiskit) share a common `BaseSolver.solve(MolSpec) -> MethodResult` API. Configuration is YAML-driven (`BenchConfig`). Two bundled reproductions cover **SQD (Nat. Chem. 2024)** and **error-mitigated VQE-style experiments (Science 2020, abd3880)**. See **[TECH_DOC.md](TECH_DOC.md)** for YAML keys and solver kwargs.
