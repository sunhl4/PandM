# dft_qc_pipeline 开发记忆文档

> **用途**：记录开发过程中的关键决策、设计取舍、遇到的坑与解决方案，以及未来维护的方向。  
> **维护约定**：每次有实质性变更时在对应节新增条目，格式为 `### [日期] 描述`。

---

## 目录

1. [项目起源与动机](#1-项目起源与动机)
2. [架构决策记录（ADR）](#2-架构决策记录adr)
3. [各模块关键实现决策](#3-各模块关键实现决策)
4. [已知限制与技术债](#4-已知限制与技术债)
5. [未来演进路线图](#5-未来演进路线图)
6. [文献调研笔记（截至 2026-04-01）](#6-文献调研笔记截至-2026-04-01)
7. [变更日志](#7-变更日志)

---

## 1. 项目起源与动机

### 2026-04-01 初始建立

**背景**：仓库 `LearningPlan/learning_materials/` 内已有 Phase1–4 的量子计算化学学习材料，但存在以下工程空白：

1. VQE、SQD、NumPy 三种求解器分散在不同 Notebook，无统一接口，不能对同一问题互换
2. 活性空间的选取全靠手工填 `(nelec, norb)` 数字，无原子级的自动化路径
3. DMET 等嵌入方法在 `SQD.md` 中有文字描述，但无可运行代码
4. 经典端（PySCF DFT/HF）与量子端（Qiskit）之间没有明确的数据接口层

**触发文献**（截至 2026-04-01 调研）：
- Shajan et al. arXiv:2411.09861（DMET + SQD IBM Eagle）——说明 DMET + 量子采样对角化已进入实验验证阶段
- Battaglia et al. npj CM 2024（周期 DFT + VQE 嵌入）——说明投影嵌入也有 Qiskit 生态实现路径
- Di Paola et al. npj CM 2024（DFT → AVAS → ADAPT-VQE Pt 催化）——与 Phase4 催化 Notebook 直接对应

**结论**：有充分文献依据建立一个统一 pipeline，填补上述工程空白。

---

## 2. 架构决策记录（ADR）

### ADR-001：为什么选择六层架构而非两层（经典 + 量子）

**考量**：两层架构（Backend → Solver）最简单，但嵌入方法和哈密顿量构建是非常复杂且独立可替换的功能，把它们合并到任一侧会导致层内逻辑过重。

**决策**：拆为六层：Backend → Embedding → FragmentRegion → Hamiltonian → Solver → Postprocessing。

**代价**：接口层增加，初次理解成本略高。

---

### ADR-002：为什么使用注册表（registry）而非工厂函数或直接 import

**考量**：用户需要能在不改动核心代码的情况下注册自定义 backend / embedding / solver。工厂函数需要修改工厂内部逻辑；直接 import 无法做到配置驱动。

**决策**：全局单例注册表 `registry`，装饰器 `@registry.register(name, category)` 在 import 时自动注册。`pipeline.py` 通过 `registry.build(cfg_dict, category)` 实例化。

**注意**：注册触发依赖 import。顶层 `__init__.py` 强制 import 所有子模块，确保内置实现全部注册。用户自定义扩展需在 `Pipeline.run()` 之前手动 import 相应模块。

---

### ADR-003：为什么 EmbeddingMethod 既负责 embed() 又负责 update_from_rdm()

**考量**：DMET 的自洽循环状态（μ 值、上次 RDM 等）与嵌入方法强耦合；如果把 `update_from_rdm` 放到 `Pipeline` 或单独的 `SCFUpdater`，状态管理会更复杂。

**决策**：`EmbeddingMethod` 持有自洽状态，`update_from_rdm()` 改变内部 μ 并返回是否收敛。`Pipeline.run()` 只负责驱动循环（`while not converged`）。

**代价**：`EmbeddingMethod` 实例是有状态的（stateful），**同一实例**不能安全地并行用于多个片段。**v0.1.1+**：`Pipeline._process_region` 对每个 `region` 调用 `registry.build` **新建** embedding；`parallel_regions=True` 时另为每 region 新建 **solver**（线程并行），与 ADR 一致。

---

### ADR-004：为什么 SQDSolver 在 qiskit-addon-sqd 缺失时回退到 NumPy 而非抛出异常

**考量**：`qiskit-addon-sqd` 在 Windows 原生环境安装有时有问题；抛出异常会中断整个 pipeline；NumPy 回退能让用户至少看到结果并比对。

**决策**：回退 + WARNING 日志。用户看到警告后可以 `pip install qiskit-addon-sqd` 再运行。

---

### ADR-005：YAML `extra` 字段的设计

**考量**：不同后端/嵌入方法有各自的专属参数，若全放到顶层 `*Config` 会造成字段爆炸。

**决策**：每个 `*Config` dataclass 都有 `extra: dict` 字段。YAML 中未被已知字段名匹配的键自动收入 `extra`，由具体实现类通过 `**kwargs` 或 `self.extra.get(...)` 读取。

---

### ADR-006：h2e 积分存储的 nao ≤ 50 阈值

**考量**：`pyscf.ao2mo.kernel` 全量 (nao,nao,nao,nao) 4 维张量对于 nao=100 需要约 800 MB。对大体系应使用 DF 积分（density fitting）。

**决策**：`nao > 50` 时跳过全量 `h2e_ao` 存储，`BackendResult.h2e_ao = None`，并打 INFO 日志提示用户使用 DF。`builder.py` 的 `ao2mo.kernel` 直接作用于活性 MO 列，绕过全量 AO 积分，因此不受此限制影响。

---

### ADR-007：`energy_corrections` 与文献能量项的边界

**考量**：`total_energy` 为片段求解能量之和时，与全体系 HF/DFT 或文献中「DMET 总能量」不可直接等同；若不在结果中显式区分，论文与复现容易混淆。

**决策**：在 `PipelineResult.extra["energy_corrections"]` 中集中返回 **诊断量**（mean-field 参考、片段和、差值、DMET **μ**、**μ·ΔN**、可选 **Mulliken 点电荷 inter**），并用 `*_model` 字符串与 `notes` 标明**模型名**与**非全量子 DMET 浴**等边界。文献中的全量子 inter-fragment / correlation-potential 若未实现，保持为文档说明而非伪造数值。

**代价**：调用方需阅读 `TECH_DOC` §11a 才能正确写进论文「我们报告的是…」。

---

## 3. 各模块关键实现决策

### 3.1 `DMETEmbedding`

**Schmidt 分解策略**：对占据 MO 块的片段 AO 行做 SVD，取奇异值 > `bath_threshold` 的向量为浴轨道。这是标准分子 DMET 的实现（Wouters 2016 JCTC）。

**化学势更新**：YAML / `EmbeddingConfig` 支持 **`mu_update`**：`gradient`、`damped`（tanh 阻尼）、`bisection`（regula falsi on (μ, pop_error)）、`bisection_bracket`（误差异号的最小区间取中点；v0.2.4 起忽略退化「同 μ」括号）。另 **`mu_max_abs`** 限制 \|μ\|。每次 `update_from_rdm` 记录 **μ·ΔN**（`embed().extra["dmet_mu_times_deltaN_ha"]`），`Pipeline` 汇总进 **`energy_corrections`**（见 **ADR-007**、**TECH_DOC §11a**）。**已知限制**：强关联片段下任一模仍可能难收敛，需调 `mu_step` / `max_iter` 或换 `mu_update`。

**浴轨道正交化**：使用 Löwdin 正交化而非 Gram-Schmidt，避免数值不稳定性。

---

### 3.2 `AVASEmbedding`

**AO 标签生成逻辑**：当前对过渡金属原子自动生成 `"{sym} 3d"` 和 `"{sym} 4s"` 标签。对于第二/三系列过渡金属应改为 `"4d"`/`"4s"` 等。这是一个已知的粗糙近似，实际使用建议通过 `FragmentRegion.extra` 传入自定义标签列表（未来功能）。

---

### 3.3 `ProjectorEmbedding`

**投影能级 μ 的选择**：μ = 1e6 Ha（硬投影）是 Manby-Werner 论文推荐值，使环境轨道完全排斥于片段空间。较小的 μ 值（如 1e3）可产生"软投影"，保留部分片段–环境杂化，但收敛更难控制。

**v_emb 的构造**：当前用 `v_emb = F_full - h_core`（Fock 矩阵减去单电子 Hamilton）近似。精确的 WF-in-DFT 需要减去 DFT 泛函导数，当前实现是 HF 精确、DFT 近似的。

---

### 3.4 `VQESolver` 的 `kupccgsd`

当前 `kupccgsd` 实现为重复 `reps` 次的 UCCSD，**不是**真正的 k-UpCCGSD（Lee et al., JCTC 2019）。真正实现需要 `EvolvedOperatorAnsatz` 和广义单/双激发算符池，Qiskit Nature 0.7.x 的支持不完整。这个局限已在代码注释和 TECH_DOC 中标注。

---

### 3.5 `ADAPTVQESolver` 的内层 VQE

每次添加算符后做完整 VQE 重优化（所有参数，包括之前的参数）。这与 Grimsley 2019 原版一致，但计算量大。已知优化：`warm-start`（用上一次最优参数作初始值，`initial_point` 追加一个 0）已实现。

---

### 3.6 `builder.py` 的 FermionicOp 构造

`hamiltonian_to_fermionic_op()` 手动构造 `FermionicOp` 的 key-value dict，而非通过 `ElectronicIntegrals` / `ElectronicEnergy` 路径。原因：直接从 `h1e`/`h2e` numpy array 到 `FermionicOp` 的路径更通用，不依赖 `ElectronicStructureProblem` 对象，方便嵌入层传入任意 numpy 积分。

**注意**：Qiskit Nature 0.7.x 中 `FermionicOp` 的 key 格式为 `"+_p -_q"` (spinless index)，构造参数为 `num_spin_orbitals=`（旧版名 `num_spin_orbs` 已弃用）；spin-orbital 索引为 `0..2*norb-1`（前 norb 为 alpha，后 norb 为 beta）。这与 JW mapper 的约定一致。

---

## 4. 已知限制与技术债

| ID | 模块 | 描述 | 优先级 | 建议修复 |
|----|------|------|--------|---------|
| L1 | `DMETEmbedding` | ~~同一 `embedding` 实例用于所有片段；有状态冲突风险~~ **v0.1.1 已修复** | — | `Pipeline` 对每个 region 新建 embedding |
| L2 | `DMETEmbedding` | 虽有 `bisection` / `bisection_bracket` / `damped`，强关联体系仍可能振荡或慢收敛 | 中 | DIIS、更稳健括号策略或与文献全 DMET 循环对齐 |
| L3 | `VQESolver.kupccgsd` | 是 UCCSD 重复层，非真正 k-UpCCGSD | 低 | 用 `EvolvedOperatorAnsatz` + 广义 SD 池实现 |
| L4 | `ProjectorEmbedding` | DFT 体系的 v_emb 近似（缺少 XC 导数修正） | 中 | v0.2.3：日志 WARNING + TECH_DOC；长期可接 `numint` XC kernel |
| L5 | `AVASEmbedding` | AO 标签自动生成对二/三系列过渡金属不准确 | 低 | 通过 `extra.ao_labels` 支持用户传入标签 |
| L6 | `SQDSolver` | ~~1-RDM 仅用主 CI 对角近似~~ **v0.1.2**：子空间 Slater 基下 **γ_pq=⟨E_pq⟩**（含单激发 off-diagonal）；子空间截断误差仍在 | 低 | 与 FCI 全空间对比时的外推 / 更大子空间 |
| L7 | `builder.py` | `nao > 50` 时不计算 `h2e_ao`，但 `ao2mo.kernel` 仍逐核重算 | 低 | 缓存 AO 积分或切换 DF 积分 |
| L8 | `Pipeline` | ~~未支持多 region 并行~~ **v0.2.4+**：`parallel_regions` + `ThreadPoolExecutor`（每 region 独立 `solver`；与 `benchmark_mode` 互斥；**PySCF 对象跨线程不保证安全**，属实验选项） | 低 | 长期可考虑多进程隔离 + 每进程独立 PySCF |
| L9 | `Pipeline` | `total_energy` 仍为片段能量之和；与全体系 / 文献 DMET 总能量不等价 | 高 | v0.3.1：`energy_corrections` 含 **Σμ·ΔN**、**Mulliken 经典 inter**（`include_inter_fragment_point_charge`）；**全量子**浴与文献 correlation potential 仍缺 |
| L10 | `ADAPTVQESolver` | ~~`pool_ops` 的 `exp_i()` 假设 Qiskit SparsePauliOp 有此方法（实际无）~~ **已修复 v0.1.0** | ~~高~~ 已解决 | 改用 `PauliEvolutionGate` + `LieTrotter` |

**L10 修复说明（已于 v0.1.0 修复）**：

原代码 `pool_ops[best_op_idx].exp_i()` 调用了 `SparsePauliOp` 上不存在的方法。
已改为：

```python
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter

evo_gate = PauliEvolutionGate(op, time=theta, synthesis=LieTrotter(reps=1))
```

当 `PauliEvolutionGate` 也失败时（旧版 Qiskit），降级为追加 identity 并记录警告。

---

## 5. 未来演进路线图

### 已交付里程碑（0.2.x–0.3.x，与版本号并行维护）

- [x] **修复 L10**：ADAPT-VQE 算符指数化用 `PauliEvolutionGate`（已完成）
- [x] **修复 L1**：`Pipeline` 为每个 region 创建独立 embedding 实例（v0.1.1）
- [x] **SQD 1-RDM**：子空间 Slater 行列式基上解析 1-RDM（`ci_subspace_rdm.py`，v0.1.2）
- [x] **单元测试**：`config`、`registry`、`builder`、`localizer`、`ci_subspace_rdm`、示例 YAML（v0.1.2）
- [x] **能量回归**：`test_energy_regression.py`（`@requires_pyscf` / `requires_qiskit_nature`）；Hubbard 等见既有 `test_hubbard.py`
- [ ] **`configs/` 补充**：LiH、N₂ 多键长、Hubbard 2-site 标准测试集（YAML 已部分具备，可再整理为「官方基准集」目录说明）

### 中期（计划，约 0.4.x+）

- [ ] **周期体系 backend**（PeriodicBackend stub）：从 ASE / Wannier90 读入 Wannier 化基函数
- [ ] **DMFT 杂质求解器接口**：参考 Selisko 2025（arXiv:2404.09527）的 DFT + DMFT 流程
- [ ] **真正 k-UpCCGSD**：用 `EvolvedOperatorAnsatz` + GSD 池实现（参考 Lee JCTC 2019）
- [ ] **qEOM（量子方程运动）求激发态**：在 VQE/SQD 得到基态后用 qEOM 算光学跃迁（参考 Battaglia 2024）
- [ ] **噪声缓解（ZNE / M3）**：在 SQDSolver / VQESolver 中可选开启，与 IBM 真机工作流对接

### 长期（1.0.0）

- [ ] **ByteQC 风格的 BNO 截断**：用 MP2 自然轨道截断控制浴的精度（参考 Huang 2025 Nat. Commun.）
- [ ] **与 Phase4 Notebook 融合**：`03_FeN4_DMET_SQD.ipynb` 直接成为 `learning_materials/Phase4_Applications/01_非均相催化量子计算.ipynb` 的升级版
- [ ] **QML 数据导出接口**：把 SQD / VQE 生成的高精度 PES 点直接导出为 NequIP / MACE 训练格式（接 `02_量子计算辅助力场开发.ipynb`）
- [x] **CI/CD 测试（基础）**：`.github/workflows/pytest.yml` + `pip install -e ".[dev,qc]"` + pytest（v0.1.2）；能量基准可后续加 `@requires_pyscf` 慢测

---

## 6. 文献调研笔记（截至 2026-04-01）

### 6.1 DMET 相关文献

**Knizia & Chan (PRL 2012)**：
- 建立了分子 DMET 的理论框架（Schmidt 分解 → 浴轨道 → 小哈密顿量 → 高水平求解 → RDM 匹配）
- 关键：浴轨道数 = 片段轨道数（精确 Schmidt 截断），impurity+bath 系统的大小是可控的

**Shajan et al. (arXiv:2411.09861)**：
- **DMET + SQD** on IBM Eagle R3（H18 链、环己烷）
- 系统大小：~89 量子比特的全问题被分解为 27–32 比特的片段
- **化学精度**：DMET 分解后 SQD 在片段上达到 FCI 级精度
- **与本工程的关联**：`DMETEmbedding` + `SQDSolver` 的直接文献依据

**Patra et al. (arXiv:2511.22158)**：
- DMET + SQD vs DMET-FCI 对配体类分子的系统基准
- 验证 SQD 在多个 DMET 片段上均接近 FCI 基准

---

### 6.2 投影嵌入 / WF-in-DFT

**Manby et al. (JCTC 2012)**：
- WF-in-DFT 理论：μ·P_B 级移使环境轨道不可见于片段求解器
- 适用于分子晶体、表面缺陷等大周期体系

**Battaglia et al. (arXiv:2404.18737, npj CM 2024)**：
- 在周期体系中用范围分离 DFT + 投影嵌入 + VQE/qEOM
- **证明**：MgO 中性氧空位的光发射峰位，与实验的误差 < 0.1 eV
- **对本工程的启示**：`ProjectorEmbedding` 的 v_emb 公式来自此工作；qEOM 是未来路线图方向

---

### 6.3 AVAS

**Sayfutyarova et al. (JCTC 2017)**：
- 提出了通过原子价轨道的投影占据数来自动选活性空间的方法
- 无需手动调 `nelec/norb`，适合过渡金属 d 轨道

**Di Paola et al. (npj CM 2024)**：
- DFT(QE) → AVAS（活性 d 轨道）→ ADAPT-VQE (H1 量子计算机) + NEVPT2
- **Pt ORR 催化**：活性空间 (6e, 6o)，H1 上 8 量子比特运行
- **与本工程的关联**：`03_FeN4_DMET_SQD.ipynb` 的分析范式直接来自此文

---

### 6.4 SQD（采样量子对角化）

**Robledo-Moreno et al. (Nat. Chem. 2024, arXiv:2405.05068)**：
- 量子端采样 bitstring → 组态恢复 → 子空间 Rayleigh-Ritz → 迭代
- 已有 `qiskit-addon-sqd` 官方实现

**与 SKQD（arXiv:2508.02578）的区别**：
- SQD：采样 + 子空间对角化，理论收敛不保证
- SKQD：Krylov 时间演化 + 采样，有可证明收敛性
- 本工程当前实现 SQD；SKQD 可作为下一版 `SQDSolver` 的替换后端

---

### 6.5 已识别的文献空白（对应未来方向）

| 空白 | 代表文献 | 对应路线图 |
|------|---------|-----------|
| 真正 k-UpCCGSD 实现 | Lee et al. JCTC 2019 | VQESolver 0.3.0 |
| qEOM 激发态 | Ollitrault et al. CRS 2020 | qEOM 0.3.0 |
| BNO 截断浴（SIE） | Huang et al. Nat. Commun. 2025 | 长期 1.0.0 |
| DFT+DMFT 量子杂质 | Selisko et al. npj CM 2025 | 中期 0.3.0 |
| 批量 ADAPT-VQE | Commun. Phys. 5, 257 (2022) | ADAPTVQESolver 0.2.0 |

---

## 7. 变更日志

### v0.3.2 – 2026-03-31 工程文档与回归：`ml_export`、并行冒烟、`quantum_chem_bench` seed

- **`postprocessing/ml_export.py`**：`pipeline_result_to_record` 在 `energy_corrections` **无**数值键（或仅有 `notes` 等占位）时，用 `backend_result.energy_hf` 与片段能量**自动推导** `backend_reference_energy_ha` / `sum_fragment_energies_ha` / `delta_backend_minus_fragments_ha`；与显式字典分支的判定见 **TECH_DOC** 中「`ml_export.py`」小节。
- **测试**：`tests/test_ml_export.py` — 显式 `energy_corrections` 全键、`extra` 无 `energy_corrections`、仅占位 `notes` 三种路径。
- **并行**：`tests/test_hubbard.py::test_parallel_regions_two_hubbard_fragments_smoke` — `toy` + 双 Hubbard region + `parallel_regions`（无 PySCF）；**TECH_DOC §6.5** 指向该用例。
- **`quantum_chem_bench`（工作区）**：`adapt_vqe` / `sqd` / `qse` / `qpe` / `qpe_full` 与既有 VQE 族统一 **`seed`** + `apply_solver_seed`；SQD 采样/resample 使用实例 RNG；**README / TECH_DOC** 已补充；`tests/test_core.py` 注册表 `seed` 注入测。

### v0.3.1 – 2026-03-31 DMET μ·ΔN 与 Mulliken inter-fragment 估计

- **`DMETEmbedding`**：每次 `update_from_rdm` 记录 **μ·ΔN**（Ha）→ `embed().extra["dmet_mu_times_deltaN_ha"]`；Pipeline 汇总为 **`dmet_correlation_potential_ha`** 与 **`dmet_mu_times_deltaN_by_region_ha`**。
- **`postprocessing/inter_fragment_estimate.py`**：Mulliken 净电荷 + 跨 region **q_i q_j / R_ij**（Bohr/Ha）→ **`dmet_inter_fragment_ha`**（`embedding: dmet`、**`len(regions)>1`**、**`include_inter_fragment_point_charge`**）。
- **`PipelineConfig.include_inter_fragment_point_charge`**；**`ml_export`** 可选写出 DMET 两项。
- **文档**：**`TECH_DOC`** §3 目录树、§10 `inter_fragment_estimate`、`§11a` 字段表；**`DEV_MEMORY`** **ADR-007**、**ADR-003** 更正、§3.1 DMET、§4 **L2/L8/L9**、§5 路线图标题；**`OPTIMIZATION`** 文首与 §2；**`README`**（本包与 **`LearningPlan/README.md`**）交叉指引 **`energy_corrections`**。

### v0.3.0 – 2026-03-31 L9 能量诊断、DMET \|μ\| 限幅、builder 大体系提示

- **`PipelineResult.extra["energy_corrections"]`**：`backend_reference_energy_ha`、`sum_fragment_energies_ha`、`delta_backend_minus_fragments_ha`、`fragment_energies_by_region_ha`、`dmet_mu_by_region_ha`（DMET）；文献项仍为 `None`；`notes` 区分诊断与文献修正。
- **`_process_region`**：返回末次 `EmbeddedHamiltonian` 以收集 DMET **μ**。
- **DMET**：`embedding.mu_max_abs`（YAML / `EmbeddingConfig`）；每次更新后截断 **\|μ\|**。
- **`build_fragment_hamiltonian`**：`nao>50` 时 INFO（活性子空间 `ao2mo`）。
- **`ml_export.pipeline_result_to_record`**：写出关键诊断标量。
- **文档**：`README` 论文能量表述、`TECH_DOC`、`OPTIMIZATION` 文首、`DEV_MEMORY` L9 行。

### v0.2.4 – 2026-03-31 DMET `bisection_bracket`、多 region 并行与 bench 对齐

- **`DMETEmbedding.mu_update == bisection_bracket`**：在历史 `(μ, err)` 中取误差异号且 **μ 不同** 的最小区间后取中点；忽略 **同 μ** 的退化括号对（否则追加当前点后 span=0 会错误地把 μ 钉在原值）。
- **`PipelineConfig`**：`parallel_regions`、`max_parallel_workers`；`validate_pipeline_config` 禁止与 `benchmark_mode` 同时为真；`Pipeline.run` 在并行路径下每 region 独立 `solver` + `ThreadPoolExecutor`（见 **TECH_DOC §11.3**）。
- **`quantum_chem_bench`**：去重 `vqe_uccsd_stack` 注册；`test_quantum_solvers` 期望列表含该名。
- **测试**：`test_dmet_mu_update.py`（无 PySCF）；`test_energy_regression.test_merge_registry_builds_dmet_with_bisection_bracket`；`test_config` 已覆盖校验。

### v0.2.3 – 2026-03-31 PySCF 密度拟合与 Projector/DFT 提示

- **`BackendConfig` / `PySCFBackend`**：`density_fit`、`auxbasis`；`mf.density_fit()` 加速大基组 SCF；`Pipeline.run` 传入后端；`nao>50` 日志指向 `density_fit`。
- **`ProjectorEmbedding`**：若平均场为 `pyscf.dft.*`，打 **WARNING** 说明 `v_emb = F - h_core` 对 KS-DFT 为近似（无 XC-kernel 修正）；**TECH_DOC §6.4** 补充 DFT 段。
- **测试**：`test_config` 合并 `density_fit`；`test_energy_regression` 中 H₂ DF 能量与参考接近。

### v0.2.2 – 2026-03-31 能量回归与 DMET 文档

- **测试**：`tests/test_energy_regression.py` — H₂ STO-3G RHF 参考能量（`assert_allclose`）、`h2_vqe.yaml` 整管线片段能量带；`merge_registry_kwargs` + `registry.build` 验证 **DMET** 接收 `mu_update` / `mu_step`。
- **TECH_DOC**：DMET 参数表补充 **`mu_update`**（`gradient` / `damped` / `bisection`）。

### v0.2.1 – 2026-03-31 Pipeline 小优化与文档

- **`Pipeline.run`**：benchmark 模式复用最后一次 `mapper.map` 的 `(H_qubit, n_particles, n_orbs)`，避免对同一最终 `emb_H` 重复映射。
- **`total_energy` 语义**：有片段结果时为片段能量之和；**无**成功片段时退回 `backend_result.energy_hf`；`extra["total_energy_note"]` 同步说明。
- **示例 Notebook**：为所有 cell 补全 **`id`**（满足 nbformat 要求，减少 nbmake 警告）。
- **工作区 `LearningPlan/README.md`**：与 `dft_qc_pipeline` 文档交叉指引、CI / nbmake / Windows PySCF 说明。

### v0.2.0 – 2026-03-31 模型嵌入、导出与 CI Notebook

- **`ToyBackend` / `HubbardEmbedding`**：无 PySCF 的一维 Hubbard 路径；`configs/hubbard_2site_numpy.yaml`；测试 `test_hubbard.py`。
- **`classical_backends` 包 `__init__.py`**：显式 import `pyscf_backend` 与 `toy_backend`，保证 `import dft_qc_pipeline` 时 **`pyscf` 与 `toy` 均已注册**。
- **`postprocessing/ml_export.py`**：`pipeline_result_to_record`、`write_pes_jsonl`、`write_pes_csv`；`test_ml_export.py`。
- **`PipelineResult.extra["energy_corrections"]`**：DMET 文献向能量分解占位字段（当前未数值计算）。
- **示例 Notebook**：`01`/`02`/`03` 统一从仓库根引导 `sys.path` 与 `_CFG_DIR`；`CI_FAST_NB` 缩短距离扫描与 Fe-N4 自旋列表。
- **CI**：`pytest.yml` 在 `[dev,qc]` + PySCF 之后 **`pytest --nbmake`** 依次跑 `01_H2_minimal.ipynb`、`02_N2_multisolver.ipynb`、`03_FeN4_DMET_SQD.ipynb`（均设 `CI_FAST_NB=1`）；`[dev]` 含 `matplotlib`。
- **文档**：`README` / `TECH_DOC` 同步上述能力；版本号 `0.2.0`。

### v0.1.3 – 2026-03-31 PySCF 集成测试与 MO 索引修正

- **`get_atom_orbital_indices`**：返回 **全局 MO 列指标**（`occ_positions[...]`），并 `min(n_orbs, n_occ)` 防止越界；此前误用占域内 0…nocc-1 指标，与 `build_fragment_hamiltonian` 的 `C[:, idx]` 语义不一致。
- **配置**：`lih_sto3g_numpy.yaml`（2e/2o）、`h2_sto3g_dft_pbe.yaml`（DFT/PBE 路径）。
- **测试**：`test_pyscf_integration.py`（`@requires_pyscf` / `@requires_qiskit_nature`）；`conftest` 在无 PySCF 或无时自动 skip。
- **CI**：workflow 中 `pip install pyscf` 后跑全套 pytest（含集成测）。

### v0.1.2 – 2026-03-31 SQD 1-RDM 与测试/CI

- **SQD**：`quantum_solvers/ci_subspace_rdm.py` 在采样子空间的 Slater 基上计算完整空间 **γ_pq**（对角 + 单激发 off-diagonal）；`SolverResult.extra["rdm1_model"]` 更新说明。
- **测试**：`test_ci_subspace_rdm.py`、`test_localizer.py`、`test_pipeline_config_yaml.py`（三份 `configs/*.yaml`）。
- **CI**：`LearningPlan/.github/workflows/pytest.yml`（Python 3.10/3.11，`[dev,qc]`）。
- **开发依赖**：`nbmake` 加入 `[dev]`（可选 `pytest --nbmake …` 做 Notebook 冒烟）。

### v0.1.1 – 2026-03-31 工程优化与兼容性

**变更摘要**：

- `Pipeline`：每个 `region` 独立 `EmbeddingMethod` 实例；benchmark 模式缓存 `mapper.map`；`PipelineResult.extra["total_energy_note"]` 标明总能量近似性。
- `merge_registry_kwargs`：`backend` / `embedding` / `solver` 的 YAML `extra` 与显式字段合并后传入构造器（显式字段覆盖 `extra`）。
- `validate_pipeline_config`：`regions`、`embedding`、`mu_update`、基础自洽性检查；`Pipeline.run()` 入口调用。
- `EmbeddingConfig`：`mu_update`、`ao_labels`；`DMETEmbedding` 支持 `damped` / `bisection` μ 更新；`AVASEmbedding` 支持显式 `ao_labels`。
- `SolverConfig.seed`：`VQESolver` / `SQDSolver` / `ADAPTVQESolver`；`uccsd_stack` ansatz 名与 `kupccgsd` 别名 + 警告。
- `BackendResult.scf_converged`；PySCF 电子数一致性 WARNING。
- `SQDSolver`：`SolverResult.extra["rdm1_model"]` 说明 1-RDM 模型（v0.1.2 起为子空间 Slater 精确 ⟨E_pq⟩）。
- Qiskit Nature：`FermionicOp(..., num_spin_orbitals=...)` 全库替换。
- 可编辑安装：`LearningPlan/pyproject.toml`；`dft_qc_pipeline/tests/` pytest 最小集。

### 2026-03-31 工作区目录调整（非版本号发布）

- `LearningPlan/learning_materials/`：集中存放 Phase1–4 Notebook、讲稿、`SQD.md` / `ADAPT-VQE.md`、文献补充、PySCF 安装脚本与 `.pyscf_win_build` 等学习资料。
- `LearningPlan/dft_qc_pipeline/`：仅保留管线软件与 `examples/`、`configs/`、工程文档。
- 工作区根新增 [`README.md`](../README.md) 作为并列索引；环境说明仍以 `learning_materials/README.md` 为准。

### v0.1.0 – 2026-04-01 初始版本

**新增**：

- `core/`: `interfaces.py`、`registry.py`、`config.py`、`pipeline.py` 完整实现
- `classical_backends/pyscf_backend.py`：PySCF HF/DFT 后端，注册名 `"pyscf"`
- `embedding/`: `simple_cas`、`dmet`、`avas`、`projector` 四种嵌入方法
- `hamiltonian/`: `fragment_region`、`localizer`（Boys/PM/IAO）、`builder`（冻核折叠）、`mappers`（JW/Parity/BK）
- `quantum_solvers/`: `numpy`、`vqe`（UCCSD/HEA/k-UpCCGSD）、`adapt_vqe`（费米子 SD 池）、`sqd`
- `postprocessing/`: `rdm_extractor`、`benchmark`
- `configs/`: `h2_vqe.yaml`、`n2_compare.yaml`、`fen4_dmet_sqd.yaml`
- `examples/`: `01_H2_minimal.ipynb`、`02_N2_multisolver.ipynb`、`03_FeN4_DMET_SQD.ipynb`
- `README.md`、`TECH_DOC.md`（本文件姐妹文档）、`DEV_MEMORY.md`（本文件）

**已知问题**：见 §4 技术债表（L1–L9 等）；L10（ADAPT-VQE `exp_i()`）已在后续提交中修复。

---

*下次维护时在此节添加新版本条目，格式：`### v0.x.y – YYYY-MM-DD 标题`*
