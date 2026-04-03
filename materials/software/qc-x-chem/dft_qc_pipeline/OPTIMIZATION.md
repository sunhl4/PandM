# dft_qc_pipeline 优化建议（审查备忘）

> 目的：在 `DEV_MEMORY.md` §4 技术债（L1–L9）之外，从**工程化、物理正确性、性能与可测试性**角度列出可执行改进项。优先级为建议性质，实施时按你的发布节奏取舍。

**v0.3.1（摘要）**：`energy_corrections` 增加 **Σμ·ΔN**（`dmet_correlation_potential_ha`）、多 region **Mulliken 点电荷**经典 inter（`dmet_inter_fragment_ha`，可由 **`include_inter_fragment_point_charge`** 关闭）；`ml_export` 可写出上述 DMET 项。设计边界见 **`DEV_MEMORY` ADR-007**、**`TECH_DOC` §11a**。

**v0.3.0 起（摘要）**：`energy_corrections` 含全体系 mean-field 与片段能量**诊断**、DMET **μ** 按 region；`embedding.mu_max_abs`；`build_fragment_hamiltonian` 在大 **nao** 时 INFO；`ml_export` 写出关键诊断字段。**v0.3.2**：`ml_export` 对无数值键 / 仅占位 `notes` 的 `energy_corrections` 走**自动推导**；Hubbard 双 region **`parallel_regions` 冒烟测**（无 PySCF）；工作区 **`quantum_chem_bench`** 全量子注册求解器统一 **`seed`**。此前版本已落地：每 region 独立 embedding、benchmark 复用 `map`、`density_fit`、Projector/DFT 警告、能量回归测、`parallel_regions`（与 benchmark 互斥）、DMET `bisection_bracket` 等 — 详见 `DEV_MEMORY` §7。

**v0.1.1 已落地（对照本文）**：§1 部分（`pyproject.toml`、`__version__`）、§2 每片段 embedding、总能量说明、`benchmark` 单次 `map`；§3 `scf_converged`、电子数 WARNING；§4 `mu_update` / `ao_labels`、Projector DFT 文档句；§5 `seed`、`uccsd_stack`；§6 配置校验与 `extra` 合并策略；§7 最小 pytest；§8 日志 `region=` / `solver=`；§9 `SupportsEmbeddedMap` Protocol。

**v0.1.2 补充**：SQD 子空间 **完整空间 1-RDM**（`ci_subspace_rdm.py`）、扩展单元测试、GitHub Actions pytest、`nbmake` 可选 Notebook 冒烟。

**v0.3+ 补充（工作区级）**：CI 触发路径包含 `quantum_chem_bench/**`，并在同一 workflow 中跑 `quantum_chem_bench/tests`（需 PySCF 的用例在 CI 中可执行）。`validate_pipeline_config` 增加 **region 名称唯一性** 校验。`quantum_chem_bench` 增加 `validate_bench_config`（mapper / 活跃空间一致性）及 **HF vs MoleculeBuilder** 能量一致性测试。

---

## 1. 打包与导入路径

| 建议 | 说明 |
|------|------|
| **提供 `pyproject.toml` + 可编辑安装** | `pip install -e ./dft_qc_pipeline`（或把包提到仓库根）后，用户无需再手写 `sys.path.insert`。Notebook 示例可改为「先 `pip install -e`」。 |
| **`__version__` 与变更日志对齐** | 在 `dft_qc_pipeline/__init__.py` 暴露版本号，与 `DEV_MEMORY` 变更日志一致，便于复现实验。 |

---

## 2. 编排与多片段（`Pipeline`）

| 建议 | 说明 |
|------|------|
| **每片段独立 `EmbeddingMethod` 实例** | **L1 已于 v0.1.1 修复**：`Pipeline` 每 region `registry.build` 新 embedding；与下文并行路径一致。 |
| **总能量定义与文档一致** | **L9（部分落地 v0.3.0–v0.3.1）**：`total_energy` 仍为片段和；**`energy_corrections`** 提供 mean-field 参考、差值、**Σμ·ΔN**、可选 **Mulliken 经典 inter** 及 `notes` / `*_model`。全量子 DMET 浴与文献 correlation potential 仍为技术债 — 见 **`TECH_DOC` §11a**。 |
| **Benchmark 模式避免重复 `mapper.map`** | **v0.2.1**：`Pipeline.run` 在 benchmark 分支复用最后一次 `mapper.map` 输出，不再对同一 `emb_H` 二次映射。 |

---

## 3. 经典后端（`PySCFBackend`）

| 建议 | 说明 |
|------|------|
| **开壳自旋与电子数** | `nelec` 从 `mo_occ` 推断：在 **UHF/非对称占据** 或复杂 ROHF 场景下建议与 `mol.nelectron`、用户 `charge/spin` 做一致性校验并记录 WARNING。 |
| **SCF 未收敛** | 已打 warning；可选择在 `BackendResult` 中增加 `scf_converged: bool`，供上层决定是否中止或降级。 |
| **大体系积分策略** | `nao > 50` 跳过全量 `h2e_ao` 已文档化；长期可接 **density fitting** 或 **Cholesky**，与 `builder` 的 `ao2mo` 路径统一，避免大体系静默踩坑。 |

---

## 4. 嵌入与哈密顿量

| 建议 | 说明 |
|------|------|
| **DMET 化学势更新** | **L2**：已实现 `gradient` / `damped` / `bisection` / `bisection_bracket`；强关联仍可能难收敛。可继续：DIIS、`mu_step`/`max_iter` 校验范围、与文献循环对齐。 |
| **Projector / DFT** | **L4**：在文档中强调当前 `v_emb` 对 DFT 为近似；路线图中的 XC kernel 修正可渐进实现。 |
| **AVAS 金属价层** | **L5**：二、三过渡系默认 `3d/4s` 标签易错；优先实现 `extra.ao_labels` 并在 `TECH_DOC` 给示例。 |

---

## 5. 量子求解器

| 建议 | 说明 |
|------|------|
| **SQD 1-RDM** | **L6**：与文献级 DMET 循环对齐时，需更完整的子空间 RDM；可在 `SolverResult.extra` 中注明当前 RDM 近似级别。 |
| **VQE `kupccgsd`** | **L3**：名称与真实 k-UpCCGSD 不一致，建议在 YAML/API 中标记 `experimental` 或改名 `uccsd_stack`，减少论文对标误解。 |
| **随机种子与可复现** | **`dft_qc_pipeline`**：`SolverConfig.seed` 已用于 VQE/SQD/ADAPT-VQE 等（见 `config.py`）。**`quantum_chem_bench`**：全部量子注册求解器（含 `adapt_vqe`、`qpe`/`qpe_full`、`sqd`、`qse` 与 VQE 族）已支持构造参数 **`seed`**，见 **README / TECH_DOC**。 |

---

## 6. 配置与校验

| 建议 | 说明 |
|------|------|
| **YAML 加载后校验** | 对 `regions[].nelec/norb` 与 mapper 粒子数、`spin` 与开壳设定做简单一致性检查，失败时给出明确错误信息。 |
| **缺失字段默认值** | `extra` 字典深合并策略（若未来支持多文件配置）建议写清，避免静默覆盖。 |

---

## 7. 测试与 CI

| 建议 | 说明 |
|------|------|
| **pytest 最小集** | `builder`、`mappers`、`registry`、`config.from_yaml`、小规模 H₂ 整管线索引（可用 STO-3G + `numpy` solver）——与 `DEV_MEMORY` §5 路线图一致。**`ml_export`**：`test_ml_export.py` 覆盖显式/推导/占位三路径。**`parallel_regions`**：`test_hubbard.py` 中 toy+Hubbard 双 fragment 冒烟（无 PySCF）。 |
| **可选 PySCF 标记** | 用 `@pytest.mark.requires_pyscf` 跳过无 PySCF 的环境（如部分 CI 容器），保证纯量子层仍可测。 |
| **Notebook 冒烟** | `nbmake` 或 `papermill` 仅跑 `01_H2_minimal`，防止 API 断裂。 |

---

## 8. 日志与观测性

| 建议 | 说明 |
|------|------|
| **结构化日志字段** | 在关键步骤打 `region`、`solver`、`iteration`，便于从日志重建一次运行配置。 |
| **日志级别** | 默认 INFO；DMET 内层循环细节用 DEBUG，避免生产日志膨胀。 |

---

## 9. 类型与静态检查

| 建议 | 说明 |
|------|------|
| **`mypy --strict` 渐进** | 先对 `core/interfaces.py`、`config.py` 收紧，再扩到 solvers。 |
| **协议类 `Protocol`** | 对 `mapper` 等 duck-typed 对象可用 `typing.Protocol` 标注，改善 IDE 体验。 |

---

## 10. 与 `learning_materials` 的衔接

| 建议 | 说明 |
|------|------|
| **示例 Notebook 路径** | `examples/*.ipynb` 中 `sys.path` 说明改为「指向 `LearningPlan` 根目录」；与根 [`README.md`](../README.md) 一致。 |
| **催化叙事** | 将 `03_FeN4_DMET_SQD.ipynb` 与 Phase4 催化 Notebook 交叉链接（双向），降低初学者迷路成本。 |

---

*审查基准：2026-04-01 代码快照；若实现其中某项，建议在 `DEV_MEMORY.md` §7 记一笔并更新 §4 表格状态。*
