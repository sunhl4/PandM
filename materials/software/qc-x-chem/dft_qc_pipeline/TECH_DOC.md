# dft_qc_pipeline 技术文档

> 版本：0.3.1 | 最后更新：2026-04-01

---

## 目录

1. [概述与定位](#1-概述与定位)
2. [依赖与安装](#2-依赖与安装)
3. [工程目录一览](#3-工程目录一览)
4. [数据流与模块接口](#4-数据流与模块接口)
5. [Layer 1 – 经典化学后端](#5-layer-1--经典化学后端)
6. [Layer 2 – 嵌入方法](#6-layer-2--嵌入方法)
7. [Layer 3 – 片段区域定义](#7-layer-3--片段区域定义)
8. [Layer 4 – 哈密顿量构建](#8-layer-4--哈密顿量构建)
9. [Layer 5 – 量子求解器](#9-layer-5--量子求解器)
10. [Layer 6 – 后处理](#10-layer-6--后处理)
11. [核心层：注册表 / 配置 / 编排器](#11-核心层注册表--配置--编排器)
11a. [`energy_corrections` 字段说明](#11a-energy_corrections-字段说明)
12. [YAML 配置参考](#12-yaml-配置参考)
13. [示例使用场景](#13-示例使用场景)
14. [插件扩展指南](#14-插件扩展指南)
15. [常见问题与调试](#15-常见问题与调试)
16. [文献依据](#16-文献依据)

---

## 1. 概述与定位

`dft_qc_pipeline` 是一个 **模块化、可插拔的 DFT + 局部量子嵌入计算管线**，以 PySCF（经典端）和 Qiskit（量子端）为双核心，通过统一抽象接口连接六个可独立替换的计算层：

```
经典化学后端 → 嵌入方法 → 片段 Hamiltonian → 量子映射 → 量子求解器 → 后处理
```

**设计目标**：

- 支持 VQE / ADAPT-VQE / SQD / NumPy-FCI 的 **一行切换**
- 支持 SimpleCAS / DMET / AVAS / Projector / Hubbard（模型）嵌入策略
- 支持 HF / DFT 经典后端的无缝替换
- 内置 DMET 1-RDM 自洽循环
- YAML 驱动，不改代码即可切换方法

---

## 2. 依赖与安装

在已有 `qc_chem` Conda 环境的基础上：

```bash
pip install pyyaml
```

| 依赖 | 版本要求 | 用途 |
|------|---------|------|
| `qiskit` | ≥ 1.3.1 | 量子线路与算符 |
| `qiskit-nature` | ≥ 0.7.2 | 二次量子化 / UCCSD / Mapper |
| `qiskit-algorithms` | 最新 | VQE / NumPyMinimumEigensolver |
| `qiskit-addon-sqd` | ≥ 0.8.0 | SQD 采样对角化（可选，缺失则回退 NumPy） |
| `pyscf` | ≥ 2.6.2 | HF / DFT / CASSCF / AVAS / 积分 |
| `scipy numpy matplotlib pandas` | 标准 | 数值 / 可视化 |
| `pyyaml` | 任意 | YAML 配置解析 |

**导入路径**：将 **`LearningPlan/`**（工作区根，与 `dft_qc_pipeline` 同级）加入 `sys.path`，然后：

```python
import sys; sys.path.insert(0, '/path/to/LearningPlan')
import dft_qc_pipeline
```

---

## 3. 工程目录一览

```
dft_qc_pipeline/
├── __init__.py                     ← 顶级包；导入触发所有插件注册
├── README.md                       ← 快速上手
├── TECH_DOC.md                     ← 本文件
├── DEV_MEMORY.md                   ← 开发决策记忆文档
│
├── core/
│   ├── interfaces.py               ← 所有 ABC + 数据容器 dataclass
│   ├── registry.py                 ← 插件注册表（全局单例 registry）
│   ├── config.py                   ← PipelineConfig + *Config dataclass + YAML 解析
│   └── pipeline.py                 ← Pipeline.run() 主编排器
│
├── classical_backends/
│   ├── base.py                     ← ClassicalBackend ABC（re-export）
│   ├── pyscf_backend.py            ← PySCFBackend（注册名："pyscf"）
│   └── toy_backend.py              ← ToyBackend（注册名："toy"，模型 Hamiltonian 用）
│
├── embedding/
│   ├── base.py                     ← EmbeddingMethod ABC（re-export）
│   ├── simple_cas.py               ← SimpleCASEmbedding（注册名："simple_cas"）
│   ├── dmet.py                     ← DMETEmbedding（注册名："dmet"）
│   ├── avas.py                     ← AVASEmbedding（注册名："avas"）
│   ├── projector.py                ← ProjectorEmbedding（注册名："projector"）
│   └── hubbard.py                  ← HubbardEmbedding（注册名："hubbard"）
│
├── hamiltonian/
│   ├── fragment_region.py          ← FragmentRegion dataclass
│   ├── localizer.py                ← Boys / PM / IAO 轨道局部化
│   ├── builder.py                  ← h1e + g2e → FermionicOp（冻核折叠）
│   └── mappers.py                  ← JW / Parity / BK → SparsePauliOp
│
├── quantum_solvers/
│   ├── base.py                     ← QuantumSolver ABC（re-export）
│   ├── numpy_solver.py             ← NumPySolver（注册名："numpy"）
│   ├── vqe_solver.py               ← VQESolver（注册名："vqe"）
│   ├── adapt_vqe_solver.py         ← ADAPTVQESolver（注册名："adapt_vqe"）
│   └── sqd_solver.py               ← SQDSolver（注册名："sqd"）
│
├── postprocessing/
│   ├── rdm_extractor.py            ← 1-RDM 提取 / AO 变换 / 原子布居
│   ├── benchmark.py                ← BenchmarkCollector（多求解器对比表）
│   ├── ml_export.py                ← PES / 结果导出 JSONL、CSV（含 energy_corrections 关键项）
│   └── inter_fragment_estimate.py  ← Mulliken 电荷 + 跨 region 经典 q_i q_j / R_ij（DMET 诊断）
│
├── configs/
│   ├── h2_vqe.yaml                 ← H₂ HF → SimpleCAS → NumPy FCI
│   ├── hubbard_2site_numpy.yaml    ← toy + Hubbard + NumPy（无 PySCF）
│   ├── n2_compare.yaml             ← N₂ 多求解器基准模式
│   └── fen4_dmet_sqd.yaml          ← Fe-N4 DFT → DMET → SQD
│
└── examples/
    ├── 01_H2_minimal.ipynb         ← H₂ PEC，求解器切换演示
    ├── 02_N2_multisolver.ipynb     ← BenchmarkCollector，多方法对比
    └── 03_FeN4_DMET_SQD.ipynb      ← 催化活性位，自旋态，嵌入方法对比
```

---

## 4. 数据流与模块接口

```
PipelineConfig (YAML / dataclass)
        │
        ▼
Pipeline.run()
        │
        ├─[1]─► ClassicalBackend.run(geometry, basis, **kw)
        │              └─► BackendResult
        │                    .mol          PySCF Mole
        │                    .energy_hf    float
        │                    .mo_coeff     (nao, nmo)
        │                    .mo_occ       (nmo,)
        │                    .h1e_ao       (nao, nao)
        │                    .h2e_ao       (nao,)*4 or None
        │                    .nelec        (n_α, n_β)
        │                    .mf           PySCF mean-field obj
        │
        ├─[2]─► EmbeddingMethod.embed(BackendResult, FragmentRegion)
        │              └─► EmbeddedHamiltonian
        │                    .h1e          (norb, norb)   MO 基
        │                    .h2e          (norb,)*4
        │                    .nelec        (n_α, n_β)
        │                    .norb         int
        │                    .e_core       float
        │
        ├─[4]─► mapper.map(EmbeddedHamiltonian)
        │              └─► (SparsePauliOp, num_particles, num_spatial_orbitals)
        │
        ├─[5]─► QuantumSolver.solve(SparsePauliOp, ...)
        │              └─► SolverResult
        │                    .energy       float
        │                    .rdm1         (norb, norb) or None
        │                    .converged    bool
        │                    .extra        dict
        │
        └─[DMET 自洽]─► EmbeddingMethod.update_from_rdm(rdm1) → bool
```

所有接口均定义于 `core/interfaces.py`，各层实现不直接相互依赖。

---

## 5. Layer 1 – 经典化学后端

### `PySCFBackend`（`classical_backends/pyscf_backend.py`）

**注册名**：`"pyscf"`

**构造参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `method` | str | `"hf"` | `"hf"` 或 `"dft"` |
| `xc` | str | `"pbe"` | DFT 泛函（仅 method=dft 有效） |
| `charge` | int | 0 | 分子总电荷 |
| `spin` | int | 0 | 2S（未成对电子数） |
| `verbose` | int | 0 | PySCF 输出等级 |
| `conv_tol` | float | 1e-9 | SCF 收敛阈值 |
| `max_cycle` | int | 100 | 最大 SCF 迭代数 |
| `density_fit` | bool | False | 为 True 时对 J/K 使用 **密度拟合（RI）** 加速 SCF；与精确四中心积分略有数值差 |
| `auxbasis` | str / None | None | DF 辅助基（如 `weigend`）；None 时由 PySCF 按主基自动选 |

**返回**：`BackendResult`（见第 4 节数据流表）

**注意**：
- `spin ≠ 0` 时自动切换到 `ROHF` / `ROKS`
- `BackendResult.scf_converged` 反映 `mf.converged`；`mol.nelectron` 与 `n_alpha+n_beta` 不一致时打 **WARNING**
- `nao > 50` 时跳过全 2e AO 积分储存；大体系可在 YAML 设 **`density_fit: true`** 加速 SCF，下游 `ao2mo.kernel` 仍对活性 MO 子集变换
- 通过 `mf` 字段可直接访问 PySCF 平均场对象，用于嵌入层的 `get_fock()` 等调用

**扩展**：若需 CASSCF 初始猜测，可在 `BackendResult.extra` 里存放 CASSCF 对象，下游嵌入层读取。

---

### `ToyBackend`（`classical_backends/toy_backend.py`）

**注册名**：`"toy"`

占位后端：单位 MO 系数、零 HF 能量、无真实 AO 积分；供 **Hubbard** 等由嵌入层直接构造 `h1e/h2e` 的模型使用。

**构造参数**：`norb`（int）须与片段活性轨道数一致。YAML 中可写 `backend.norb`，并与 `geometry`/`basis` 等字段一并传入（`ToyBackend.run` 忽略几何字符串内容）。

---

## 6. Layer 2 – 嵌入方法

所有嵌入方法继承 `EmbeddingMethod` ABC（`core/interfaces.py`），实现两个方法：

- `embed(backend_result, region) → EmbeddedHamiltonian`
- `update_from_rdm(rdm1) → bool`（DMET 自洽用；one-shot 方法直接返回 True）

### 6.1 SimpleCASEmbedding（`"simple_cas"`）

无嵌入基线。直接在局部化 MO 上选活性空间，冻核折叠后输出片段 Hamiltonian。

**适用场景**：快速测试 / FCI 参考值 / 小分子

**流程**：
1. 对占据 MO 做 Boys/PM/IAO 局部化
2. 按原子 Mulliken 布居选出 `norb` 个 MO（或直接使用 `orbital_indices`）
3. 冻核折叠 → `EmbeddedHamiltonian`

**参数**：`max_iter`、`conv_tol` 不生效（为 API 对称性保留）

---

### 6.2 DMETEmbedding（`"dmet"`）

Schmidt 分解浴轨道 + 1-RDM 自洽循环。

**理论参考**：Knizia & Chan, PRL 109, 186404 (2012)

**构造参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_iter` | int | 20 | 最大自洽迭代数 |
| `conv_tol` | float | 1e-5 | 收敛条件：max\|ΔRDM\| |
| `mu_init` | float | 0.0 | 初始化学势 μ |
| `mu_step` | float | 0.05 | 化学势梯度步长 |
| `bath_threshold` | float | 1e-6 | Schmidt 奇异值截断阈值 |
| `mu_update` | str | `gradient` | 化学势更新：`gradient`（Δμ∝布居差）、`damped`（`tanh` 阻尼）、`bisection`（对 (μ, 布居误差) 历史做 regula falsi）、`bisection_bracket`（在历史中取误差异号且 **μ 不同** 的最小区间，取中点；见 `embedding/dmet.py`） |
| `mu_max_abs` | float / None | None | 每次更新后限制 **\|μ\|**（可选，抑制发散） |

**DMET 自洽循环**（由 `Pipeline.run()` 驱动）：

```
embed() → 量子求解 → update_from_rdm()
                           ↓ (若未收敛)
                    更新 μ → embed() → ...
```

**注意**：`region.atom_indices` 必须非空（DMET 需原子级片段定义）

---

### 6.3 AVASEmbedding（`"avas"`）

原子价活性空间（Sayfutyarova 2017, JCTC 13, 4063）。

调用 `pyscf.mcscf.avas` 自动投影价 d/p 轨道到活性空间。

**构造参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `threshold` | float | 0.2 | AVAS 投影阈值 |
| `canonicalize` | bool | True | 是否规范化活性轨道 |

**AO 标签选择逻辑**：
- 如 `atom_indices` 非空 → 为每个原子 `iat` 生成 `"{sym} 3d"` 和 `"{sym} 4s"` 标签
- 否则自动选非 HCNOF 重原子 d 轨道

---

### 6.4 ProjectorEmbedding（`"projector"`）

Manby-Werner 投影嵌入，添加 μ·P_B 到片段 1e Hamiltonian。

**理论参考**：Manby et al., JCTC 8, 2564 (2012)

**构造参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mu` | float | 1e6 | 投影能级移位（大值 = 硬投影） |

**关键公式**：

```
h_eff = h_core + v_emb + μ·P_B

P_B = S·C_env·C_env†·S     (环境空间投影算符)
v_emb = F_full - h_core     (全系统 Fock - 单电子核 Hamilton)
```

**DFT 警告**：若经典端为 **Kohn–Sham DFT**（`method: dft`），上述 `v_emb` 与严格 WF-in-DFT 嵌入势相比**缺少显式 XC-kernel 修正**；片段能量为近似。运行时在日志中打 **WARNING**（见 `embedding/projector.py`）。

---

### 6.5 HubbardEmbedding（`"hubbard"`）

一维单带 Hubbard 模型，**不依赖 PySCF 积分**；需配合 **`backend.type: toy`**。`h1e` 为最近邻 **`-t`**，`h2e` 的 chemist 记号下 **`(ii|ii)=U`**。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `t` | float | 1.0 | 最近邻跃迁强度（矩阵元为 `-t`） |
| `U` | float | 4.0 | 在位排斥 |
| `n_sites` | int | 2 | 格点数，须与 `region.norb` 一致 |
| `periodic` | bool | False | `n_sites>2` 时增加首尾跃迁 |

参考配置：`configs/hubbard_2site_numpy.yaml`。多 region + **`parallel_regions`** 的线程池冒烟路径见 `tests/test_hubbard.py::test_parallel_regions_two_hubbard_fragments_smoke`（无 PySCF）。

---

## 7. Layer 3 – 片段区域定义

### `FragmentRegion`（`hamiltonian/fragment_region.py`）

```python
@dataclass
class FragmentRegion:
    name: str                    # 片段标签（用于结果字典键）
    atom_indices: list[int]      # 0-based 原子索引（PySCF Mole 顺序）
    orbital_indices: list[int]   # 直接指定 MO 索引（与 atom_indices 二选一）
    nelec: int                   # 片段活性空间电子数
    norb: int                    # 片段活性轨道数
    localization: str            # boys | pm | iao | none
```

**属性**：
- `n_alpha` = `(nelec + 1) // 2`
- `n_beta` = `nelec // 2`

**验证**：`FragmentRegion.validate()` 检查 `nelec ≤ 2·norb` 等条件

---

## 8. Layer 4 – 哈密顿量构建

### 8.1 轨道局部化（`hamiltonian/localizer.py`）

```python
C_loc = localize_orbitals(mf, scheme="iao")
```

支持 `"boys"` / `"pm"` / `"iao"` / `"none"`。IAO 后自动 Löwdin 正交化。

### 8.2 冻核折叠（`hamiltonian/builder.py`）

```python
emb_H = build_fragment_hamiltonian(
    backend_result, active_mo_indices, nelec_frag, e_core_correction, region_name
)
```

**核心逻辑**：
- 从 `active_mo_indices` 提取活性 MO 块
- 冻核 MO（所有占据 MO 中不在活性块内的）贡献 Fock 矩阵修正 `h1e_eff`
- 核能量 `e_core` = 核–核排斥 + 冻核轨道能量修正
- 2e 积分通过 `pyscf.ao2mo` 变换到活性 MO 基

### 8.3 Fermionic → Qubit（`hamiltonian/mappers.py`）

```python
mapper = build_mapper(MapperConfig(type="parity"))
H_qubit, num_particles, num_spatial_orbitals = mapper.map(emb_H)
```

`map()` 内部调用 `hamiltonian_to_fermionic_op()` 先生成 `FermionicOp`，
再调用 Qiskit Nature mapper 产生 `SparsePauliOp`。

---

## 9. Layer 5 – 量子求解器

所有求解器实现 `QuantumSolver.solve(hamiltonian, num_particles, num_spatial_orbitals) → SolverResult`。

### 9.1 NumPySolver（`"numpy"`）

- 使用 `qiskit_algorithms.NumPyMinimumEigensolver` 精确对角化
- 适用范围：≤ 12 qubits（更大体系内存/时间指数增长）
- `compute_rdm=True` 时通过 JW 映射期望值计算 1-RDM

### 9.2 VQESolver（`"vqe"`）

**构造参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `ansatz` | `"uccsd"` | `uccsd` / `hea` / `kupccgsd` |
| `optimizer` | `"cobyla"` | `cobyla` / `slsqp` / `l_bfgs_b` / `spsa` / `adam` |
| `max_iter` | 300 | 优化器最大迭代数 |
| `shots` | None | None = StatevectorEstimator（精确）；整数 = AerSimulator |
| `reps` | 2 | HEA 层数 / k-UpCCGSD 重复数 |

**ansatz 选择指南**：
- `uccsd`：化学精度，参数数 ∝ norb⁴，推荐小活性空间（norb ≤ 8）
- `hea`：深度可控，化学精度不保证，适合噪声设备测试
- `kupccgsd`：当前实现为重复 UCCSD 层，是 k-UpCCGSD 的近似

### 9.3 ADAPTVQESolver（`"adapt_vqe"`）

**构造参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `gradient_threshold` | 1e-3 | 停止条件：max \|∂⟨H⟩/∂θ\| |
| `max_adapt_iter` | 20 | 最大算符添加轮数 |
| `pool` | `"fermionic_sd"` | `fermionic_sd` / `qubit_commutator` |

**算子池**：
- `fermionic_sd`：单/双激发 UCC 生成元 → JW 映射（与 `ADAPT-VQE.md` 一致）
- `qubit_commutator`：`[H, Y_k]` 通勤子

### 9.4 SQDSolver（`"sqd"`）

**构造参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `sqd_iterations` | 10 | SQD 外层迭代数 |
| `sqd_shots` | 10000 | 每次迭代采样数 |
| `ansatz` | `"hea"` | 采样线路 ansatz |
| `n_batches` | 10 | bitstring 批次数 |

**工作流**：
1. VQE 预优化得到合理参数
2. 循环：采样 → `bitstring_matrix_to_ci_strs` 恢复组态 → 投影子空间对角化
3. 取最低能量为结果

**回退**：`qiskit-addon-sqd` 未安装时自动回退到 `NumPySolver`

**1-RDM（v0.1.2+）**：在 SQD 采样子空间的 Slater 行列式基上按  
γ_pq = Σ_{IJ} c_I^* c_J ⟨D_I|E_pq|D_J⟩ 解析计算（`quantum_solvers/ci_subspace_rdm.py`），含单激发 off-diagonal；`SolverResult.extra["rdm1_model"]` 标明仍受**子空间截断**限制。若构造失败则回退到旧的单主组态对角近似。

---

## 10. Layer 6 – 后处理

### `rdm_extractor.py`

```python
from dft_qc_pipeline.postprocessing import extract_rdm1, rdm1_to_ao, fragment_population, print_rdm1_summary

rdm1 = extract_rdm1(solver_result)
dm_ao = rdm1_to_ao(rdm1, C_act)    # 转换回 AO 基
pop = fragment_population(rdm1, C_act, mol, atom_indices=[0])
print_rdm1_summary(rdm1, "Fe 3d 1-RDM")
```

### `benchmark.py`

```python
from dft_qc_pipeline.postprocessing.benchmark import BenchmarkCollector

bc = BenchmarkCollector(
    solver_names=["numpy", "vqe", "adapt_vqe"],
    solver_kwargs={"vqe": {"ansatz": "uccsd", "max_iter": 300}},
)
result = bc.run(emb_H, mapper)
print(result.summary_table())
```

输出格式：

```
Benchmark: region 'N2_active'
Solver               Energy (Ha)  ΔE vs FCI (mHa)  Converged
──────────────────────────────────────────────────────────────
numpy          -107.5480823481           0.0000       True
vqe            -107.5480812345           0.1014       True
adapt_vqe      -107.5480819012           0.0435       True
```

### `ml_export.py`

将 `PipelineResult` 转为扁平记录并写出 **JSONL** / **CSV**（例如扫描键长时的能量曲线）：

```python
from dft_qc_pipeline.postprocessing import pipeline_result_to_record, write_pes_jsonl

rec = pipeline_result_to_record(
    label="pt_001",
    result=result,
    metadata={"R": 0.74, "solver": "numpy"},
)
write_pes_jsonl("pes.jsonl", [rec])
```

**导出规则**：若 `energy_corrections` 含 `backend_reference_energy_ha` / `sum_fragment_energies_ha` / `delta_backend_minus_fragments_ha` 中任一键，则记录这三项；**否则**由 `backend_result` 与片段能量自动推导。若 DMET 相关项 **`dmet_correlation_potential_ha`**、**`dmet_inter_fragment_ha`** 非空，亦写入记录（便于 ML / 元数据流水线）。

### `inter_fragment_estimate.py`

供 **`Pipeline`** 内部调用：`mulliken_net_charges_per_atom`、`inter_fragment_point_charge_from_backend`（见 **§11a**）。若需在脚本中复现 Mulliken inter 估计，可：

```python
from dft_qc_pipeline.postprocessing import inter_fragment_point_charge_from_backend
```

---

## 11. 核心层：注册表 / 配置 / 编排器

### 11.1 插件注册表（`core/registry.py`）

全局单例 `registry` 提供：

```python
from dft_qc_pipeline.core.registry import registry

# 注册（装饰器方式）
@registry.register("my_backend", category="backend")
class MyBackend(ClassicalBackend): ...

# 查询可用名称
registry.list(category="solver")  # ["numpy", "vqe", "sqd", "adapt_vqe"]

# 从 config dict 构建实例
solver = registry.build({"type": "vqe", "ansatz": "uccsd"}, category="solver")
```

**类别**：`"backend"` / `"embedding"` / `"solver"`

### 11.2 配置（`core/config.py`）

```python
@dataclass
class PipelineConfig:
    backend: BackendConfig
    regions: list[RegionConfig]
    embedding: EmbeddingConfig
    mapper: MapperConfig
    solver: SolverConfig
    benchmark_mode: bool = False
    benchmark_solvers: list[str] = []
    parallel_regions: bool = False
    max_parallel_workers: int | None = None  # None → min(32, len(regions))
    include_inter_fragment_point_charge: bool = True  # DMET + 多 region：Mulliken 点电荷 inter
```

加载方式：

```python
# 从 YAML 文件
config = PipelineConfig.from_yaml("configs/h2_vqe.yaml")

# 从 dict
config = PipelineConfig.from_dict({...})

# 直接构造
config = PipelineConfig()
config.solver.type = "vqe"
```

### 11.3 Pipeline 编排器（`core/pipeline.py`）

`Pipeline.run()` 执行顺序：

1. 调用 `validate_pipeline_config(config)`；用 `merge_registry_kwargs` + `registry.build` 实例化 **backend / mapper / 主 solver**（各节 `extra` 与显式字段合并，**显式字段优先**）。
2. 调用 `backend.run(geometry, basis, method, ...)`
3. 对每个 `region`：**新建**一个 `EmbeddingMethod` 实例（避免多片段 DMET 状态串扰），再进入自洽循环（`max_iter` 次）：
   - `embedding.embed(backend_result, region)` → `EmbeddedHamiltonian`
   - `mapper.map(emb_H)` → `(SparsePauliOp, ...)`
   - `solver.solve(H_qubit, ...)` → `SolverResult`
   - `embedding.update_from_rdm(rdm1)` → `bool`（是否收敛）
   - 若 `parallel_regions=True` 且 `len(regions) > 1` 且 **未** 开启 `benchmark_mode`，各 region 在 `ThreadPoolExecutor` 中并行；每 region **新建** `solver`（避免多线程共享 VQE/SQD 可变状态），`backend_result` / `mapper` 只读共享。与 `benchmark_mode` 互斥（校验报错）。PySCF `mf`/`mol` 并非为跨线程并发设计，并行属**实验性**选项。
4. 若 `benchmark_mode` 开启，对同一 `emb_H` **只** `mapper.map` 一次，再对各 benchmark 求解器 `solve`
5. 返回 `PipelineResult`；`result.total_energy` 在至少有一个片段成功时为**各片段求解器能量之和**；若**没有任何片段成功**则为 `backend_result.energy_hf`（参考值）。`result.extra["total_energy_note"]` 说明上述两种情形及**未含**片段间耦合与全体系 DMET 能量修正。
6. `result.extra["energy_corrections"]`：见下文 **[§11a](#11a-energy_corrections-字段说明)**。

---

### 11a. `energy_corrections` 字段说明

`Pipeline.run()` 在 `PipelineResult.extra["energy_corrections"]` 中返回 **dict**（键可能为 `None` 的标量字段在 JSON 导出时需注意）。用途：**论文/报告**中区分「mean-field 参考」「片段求解能量」「DMET 自洽残差」「经典 inter 估计」。

| 键 | 类型 | 何时有值 | 含义 |
|----|------|----------|------|
| `notes` | str | 始终 | 人类可读说明：各量的**物理边界**（何者为诊断、何者非全量子 DMET） |
| `backend_reference_energy_ha` | float | 始终 | 全体系 **HF/DFT** 总能量（`backend_result.energy_hf`） |
| `sum_fragment_energies_ha` | float / None | 有成功片段时 | 各 `SolverResult.energy` 之和 |
| `delta_backend_minus_fragments_ha` | float / None | 同上 | `backend_reference − sum_fragment`，**簿记差** |
| `fragment_energies_by_region_ha` | dict | 有片段时 | `region_name → energy` |
| `dmet_mu_by_region_ha` | dict / None | `embedding: dmet` | 各 region 末次嵌入 **μ**（来自末次 `EmbeddedHamiltonian.extra`） |
| `dmet_mu_times_deltaN_by_region_ha` | dict / None | `embedding: dmet` | 各 region 末次 **μ·ΔN**（ΔN = 片段布居 MF−solver） |
| `dmet_correlation_potential_ha` | float / None | `embedding: dmet` | 上表各 region **μ·ΔN** 之和；自洽好时 → **0** |
| `dmet_correlation_potential_model` | str / None | 有上值时 | 固定为 `sum_over_regions_mu_times_deltaN_electrons` |
| `dmet_inter_fragment_ha` | float / None | **dmet** 且 **多 region** 且 `include_inter_fragment_point_charge` | **Mulliken 净电荷** 跨 region 的 **Σ q_i q_j / R_ij**（Ha，Bohr） |
| `dmet_inter_fragment_model` | str / None | 有上值时 | `mulliken_point_charge_between_region_atom_groups` |

**非本 dict 承诺的内容**：全量子多片段 **DMET 浴** 能量分解、文献中其它 correlation-potential 定义；若需严格对标某篇论文，请在正文引用 **`dmet_*_model`** 与 **`notes`**。

---

## 12. YAML 配置参考

### 完整字段说明

```yaml
# ──── 经典后端 ────────────────────────────────────────────────────────
backend:
  type: pyscf             # [str] pyscf | toy（模型 Hamiltonian 用 toy）
  geometry: "H 0 0 0; H 0 0 0.735"  # [str] PySCF 原子规格字符串
  basis: sto-3g           # [str] 基组
  method: hf              # [str] hf | dft
  xc: pbe                 # [str] DFT 泛函（method=dft 时生效）
  charge: 0               # [int]
  spin: 0                 # [int] 2S
  density_fit: false      # [bool] PySCF：RI-J/K 加速 SCF（大基组推荐）
  # auxbasis: weigend       # [str] 可选；DF 辅助基

# ──── 片段区域（可以有多个 region）─────────────────────────────────────
regions:
  - name: fragment_A
    atom_indices: [0, 1]  # [list[int]] 0-based 原子序号
    # orbital_indices: [] # 若指定则绕过 atom_indices
    nelec: 2              # [int] 活性空间电子数
    norb: 2               # [int] 活性空间轨道数
    localization: iao     # [str] boys | pm | iao | none

# ──── 嵌入方法 ──────────────────────────────────────────────────────
embedding:
  type: simple_cas        # [str] simple_cas | dmet | avas | projector | hubbard
  max_iter: 1             # [int] DMET 最大自洽迭代数（non-DMET 忽略）
  conv_tol: 1.0e-6        # [float] 收敛阈值
  mu_update: gradient     # [str] DMET：gradient | damped | bisection | bisection_bracket
  # mu_max_abs: 5.0       # [float] DMET：可选，限制 |μ|
  # ao_labels: ["Fe 3d", "Fe 4s"]  # AVAS：显式 AO 模式，覆盖 3d/4s 启发式
  # 其余嵌入专属键可写在同节，解析进 extra 后并入构造参数

# ──── Qubit 映射 ────────────────────────────────────────────────────
mapper:
  type: parity            # [str] jw | parity | bk
  z2symmetry_reduction: true  # [bool] 粒子数对称性降维（Parity 专用）

# ──── 量子求解器 ─────────────────────────────────────────────────────
solver:
  type: numpy             # [str] numpy | vqe | sqd | adapt_vqe
  seed: null              # [int|null] VQE/SQD/ADAPT-VQE 可复现用 RNG 种子
  # VQE 参数
  ansatz: uccsd           # [str] uccsd | hea | uccsd_stack | kupccgsd（后者为 uccsd_stack 别名）
  optimizer: cobyla       # [str] cobyla | slsqp | l_bfgs_b | spsa | adam
  max_iter: 300           # [int] 优化器迭代数
  shots: null             # [int|null] null = 精确模拟
  reps: 2                 # [int] HEA 层数
  # SQD 参数
  sqd_shots: 10000        # [int]
  sqd_iterations: 10      # [int]

# ──── 基准模式（可选）────────────────────────────────────────────────
benchmark_mode: false
benchmark_solvers: []     # [list[str]] 如 [numpy, vqe, adapt_vqe]

# ──── 多 region 并行（可选，实验性）──────────────────────────────────
parallel_regions: false   # [bool] 与 benchmark_mode 互斥
# max_parallel_workers: 4  # [int|null] 默认 min(32, len(regions))
# include_inter_fragment_point_charge: true  # DMET 多 region 时 Mulliken 经典 inter
```

---

## 13. 示例使用场景

### 场景 A：H₂ 键解离曲线（NumPy FCI）

```python
import numpy as np
from dft_qc_pipeline import Pipeline, PipelineConfig

distances = np.linspace(0.5, 3.0, 20)
energies = []

for d in distances:
    cfg = PipelineConfig.from_yaml('configs/h2_vqe.yaml')
    cfg.backend.geometry = f'H 0 0 0; H 0 0 {d:.4f}'
    result = Pipeline(cfg).run()
    energies.append(result.total_energy)
```

### 场景 B：求解器切换（一行代码）

```python
cfg = PipelineConfig.from_yaml('configs/n2_compare.yaml')
for solver_type in ['numpy', 'vqe', 'adapt_vqe', 'sqd']:
    cfg.solver.type = solver_type
    r = Pipeline(cfg).run()
    print(f'{solver_type}: {r.total_energy:.8f} Ha')
```

### 场景 C：注册自定义经典后端

```python
from dft_qc_pipeline.core.interfaces import ClassicalBackend, BackendResult
from dft_qc_pipeline.core.registry import registry

@registry.register("orca", category="backend")
class ORCABackend(ClassicalBackend):
    def run(self, geometry, basis, **kw) -> BackendResult:
        # 调用 ORCA，解析输出，构建 BackendResult
        ...
```

### 场景 D：DMET + SQD Fe-N4 催化

```python
cfg = PipelineConfig.from_yaml('configs/fen4_dmet_sqd.yaml')
result = Pipeline(cfg).run()

frag = result.fragment_results['Fe_d_active']
print(f'DMET+SQD fragment energy: {frag.energy:.8f} Ha')
```

---

## 14. 插件扩展指南

### 添加新求解器

```python
# 1. 继承 QuantumSolver
from dft_qc_pipeline.core.interfaces import QuantumSolver, SolverResult
from dft_qc_pipeline.core.registry import registry

@registry.register("my_solver", category="solver")
class MySolver(QuantumSolver):
    def __init__(self, my_param=1.0, **kw):
        self.my_param = my_param

    def solve(self, hamiltonian, num_particles, num_spatial_orbitals) -> SolverResult:
        energy = ...   # 你的求解逻辑
        return SolverResult(energy=energy, rdm1=None, rdm2=None)

# 2. YAML 中使用
# solver:
#   type: my_solver
#   my_param: 2.0
```

### 添加新嵌入方法

```python
from dft_qc_pipeline.core.interfaces import EmbeddingMethod, EmbeddedHamiltonian
from dft_qc_pipeline.core.registry import registry

@registry.register("my_embedding", category="embedding")
class MyEmbedding(EmbeddingMethod):
    def embed(self, backend_result, region) -> EmbeddedHamiltonian:
        ...
```

### 添加新经典后端（如 VASP stub）

```python
from dft_qc_pipeline.core.interfaces import ClassicalBackend, BackendResult
from dft_qc_pipeline.core.registry import registry

@registry.register("vasp_stub", category="backend")
class VASPStubBackend(ClassicalBackend):
    def run(self, geometry, basis, **kw) -> BackendResult:
        # 解析 VASP WAVECAR → mo_coeff 等
        ...
```

---

## 15. 常见问题与调试

### Q1: `ImportError: PySCF is required`

PySCF 在原生 Windows 需手动编译，推荐在 WSL2 下运行：

```bash
conda activate qc_chem
./install_pyscf_wsl.sh
```

### Q2: `KeyError: No 'my_solver' registered in category 'solver'`

确保自定义求解器模块在调用前已被 import（注册由 `@registry.register` 装饰器在 import 时触发）。

### Q3: DMET 不收敛

- 减小 `mu_step`（如 0.01）
- 增大 `max_iter`（如 50）
- 检查 `bath_threshold`（若过高会截断重要浴轨道）

### Q4: SQD 回退到 NumPy

```bash
pip install qiskit-addon-sqd
```

检查版本 ≥ 0.8.0：`pip show qiskit-addon-sqd`

### Q5: ADAPT-VQE 梯度始终不降

- 增大 `max_adapt_iter`
- 降低 `gradient_threshold`（如 1e-4）
- 检查算子池是否与体系对称性匹配

---

## 16. 文献依据

| 模块 | 理论参考 |
|------|---------|
| `DMETEmbedding` | Knizia & Chan, PRL 109, 186404 (2012) · DOI: 10.1103/PhysRevLett.109.186404 |
| `DMETEmbedding` | Shajan et al., arXiv:2411.09861 (DMET+SQD，IBM Eagle R3) |
| `AVASEmbedding` | Sayfutyarova et al., JCTC 13, 4063 (2017) · DOI: 10.1021/acs.jctc.7b00128 |
| `ProjectorEmbedding` | Manby et al., JCTC 8, 2564 (2012) · DOI: 10.1021/ct300544e |
| `ProjectorEmbedding` | Battaglia et al., npj CM (2024) arXiv:2404.18737 |
| `SQDSolver` | Robledo-Moreno et al., Nat. Chem. (2024) arXiv:2405.05068 |
| `ADAPTVQESolver` | Grimsley et al., Nat. Commun. 10, 3007 (2019) · DOI: 10.1038/s41467-019-10957-2 |
| `VQESolver` | Peruzzo et al., Nat. Commun. 5, 4213 (2014) |
| `VQESolver` UCCSD | Anand et al., Chem. Soc. Rev. 51, 1759 (2022) |
| Fe-N4 催化应用 | Di Paola et al., npj CM (2024) – Pt ORR ADAPT-VQE |
| 周期体系嵌入 | Selisko et al., npj CM (2025) arXiv:2404.09527 |

---

*本文档随工程代码同步维护；版本变更记录见 `DEV_MEMORY.md`。*
