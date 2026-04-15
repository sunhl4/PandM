# 领域专家路线与工程总纲（长期维护）

本仓库定位为：**以量子计算在计算化学与分子模拟中的应用为主线**，**以量子计算与机器学习的结合为副线**的个人/小团队 **长期知识库 + 可运行代码基地**。目标不仅是「学完教程」，而是持续完成：**文献搜集与阅读 → 系统化知识沉淀 → 复现文献算法 → 提出并实现新想法**。

---

## 1. 领域定位（主 / 副线）

| 优先级 | 方向 | 在本仓库中的落点 |
|--------|------|------------------|
| **主线** | 量子计算 × **计算化学 / 电子结构 / 分子模拟**（VQE、映射、哈密顿量、势能面、动力学接口、噪声与资源估计等） | `quantum_chemistry/`、`docs/qc_learn/`、`docs/PHASE0_H2_BASELINE.md`、`benchmarks/`、与分子任务相关的 `zno_pd_qml/`、`gnn_adsorption/`、`docs/QMLFF_INTEGRATION.md` |
| **副线** | 量子计算 × **机器学习**（量子核、VQC、表示与训练动力学、与经典基线对照） | 根目录 `week*.py`、`docs/QML_KERNEL_VS_VQC.md`、根目录综述与 `docs/qml_tutorial_zh_consolidated.md`、`quantum_ml_complete_guide*.md` |

**阅读顺序建议（建立专家心智模型）**

1. **理论与推导**：通读 **[`theory_and_derivations.md`](theory_and_derivations.md)**（单一主稿；源文件见 [`README.md`](README.md) §2）。  
2. **个人问答与备忘**：维护 **[`domain_qa.md`](domain_qa.md)**，与主稿交叉引用。  
3. 可运行闭环：`docs/PHASE0_H2_BASELINE.md` → `quantum_chemistry/tutorials/`  
4. 分子尺度管线：`docs/PHASE2_PIPELINES.md`、外部 `QML-FF`（见 `QMLFF_INTEGRATION.md`）  
5. QML 副线按需：`docs/QML_KERNEL_VS_VQC.md` 与根目录 `week*.py`  

---

## 2. 长期维护循环（建议节奏）

成为领域专家依赖 **可重复的日课/周课**，而不是单次突击。

| 周期 | 动作 | 仓库内工具/文档 |
|------|------|-----------------|
| **每周** | arXiv/期刊扫读 → **精读 1–2 篇** | [`WEEKLY_LITERATURE_WORKFLOW.md`](WEEKLY_LITERATURE_WORKFLOW.md)、[`tools/arxiv_qchem_digest.py`](../tools/arxiv_qchem_digest.py)、精读模板 [`literature/chemistry_materials_reading.md`](literature/chemistry_materials_reading.md) |
| **每周** | 笔记落地（公式假设、与主线关系、可复现实验是否可行） | [`literature_notes/README.md`](literature_notes/README.md) |
| **每月** | 1 页趋势小结：本周精读共性、对下一版实验的影响 | 可放在 `docs/literature_notes/reviews/` 或 Zotero |
| **每季度** | 对照 [`EXTERNAL_ANCHORS.md`](EXTERNAL_ANCHORS.md) 修正符号与引用习惯；更新一次「我正在攻的子问题」陈述 | 与内部报告 [`INTERNAL_REPORT_OUTLINE.md`](INTERNAL_REPORT_OUTLINE.md) 对齐 |
| **持续** | **复现一条**文献算法或 **推进一条**自有假设（见 §4–§5） | `benchmarks/`、`quantum_chemistry/`、Phase2 管线 |

---

## 3. 知识学习：分层资料地图

### 3.1 理论书面课（化学 + 经典/量子算法）

- **唯一主稿（推荐）**：[**theory_and_derivations.md**](theory_and_derivations.md) — 理论介绍与数学推导的**单文件**汇总（`tools/build_theory_master.py` 生成）。  
- **分卷/旧入口**：[unified_chemistry_theory/README.md](unified_chemistry_theory/README.md) 仅作历史分卷说明；日常以主稿为准。  
- 源文件仍分布在 [`qc_learn/`](qc_learn/)、[`quantum_chemistry/docs/`](../quantum_chemistry/docs/)（修改后重跑 `build_theory_master.py`）。  

### 3.2 符号与「对外一致」锚点

- [`EXTERNAL_ANCHORS.md`](EXTERNAL_ANCHORS.md)：教材 + PennyLane/Qiskit 栈二选一为主锚。  
- [`literature/references/`](literature/references/)：教材书目与论文列表（与 Zotero 互补；原 `qc_learn/references/` 已迁入）。  

### 3.3 副线：量子机器学习

- [`QML_KERNEL_VS_VQC.md`](QML_KERNEL_VS_VQC.md)：核方法 vs VQC 及与仓库脚本对应关系。  
- [`QML_GUIDE_INDEX.md`](QML_GUIDE_INDEX.md)：根目录多版本「完整指南」与 Tutorial 中文版导航。  

### 3.4 文献综述类（根目录 Markdown）

- [`literature/qml_literature_complete_2007_2025.md`](literature/qml_literature_complete_2007_2025.md) 等：领域鸟瞰；**精读仍应以单篇论文为单位**（见周节奏文档）。根目录按年拆分文件已并入该综述，避免重复维护。  

---

## 4. 工作流 A：复现文献中的算法

目标：把论文中的 **可计算陈述**（算法、复杂度、关键超参）落到 **本仓库可运行的最小脚本**。

1. **选题**：与当前主线子问题一致（例如某类 VQE、某映射、某小体系基线）。  
2. **规格化**：写清输入（哈密顿量来源、基组、比特数）、输出（能量误差、资源）、与原文图/表的对应关系。  
3. **代码落点**（推荐）：  
   - 小而专的基准 → [`benchmarks/`](../benchmarks/)  
   - 教学式分步实现 → [`quantum_chemistry/tutorials/`](../quantum_chemistry/tutorials/)  
   - 与吸附/表面任务相关 → [`gnn_adsorption/`](../gnn_adsorption/)、[`zno_pd_qml/`](../zno_pd_qml/)  
4. **可重复**：固定随机种子、依赖版本写入 `requirements.txt` 或子目录 `requirements.txt`；在脚本头部注释 **arXiv ID + 复现的图表编号**。  
5. **对照**：至少一个经典或简单量子基线（如 HF、固定-depth VQE），避免只报单次最优。  

工具辅助：Zotero 解析脚本 [`tools/zotero_resolve_arxiv_search_links.py`](../tools/zotero_resolve_arxiv_search_links.py)（输出目录默认 `zotero_import_full_v2/`）。

---

## 5. 工作流 B：提出并实现新算法 / 新假设

目标：在 **明确假设** 下做 **可证伪** 的小步实验，而不是一次性大系统。

1. **问题句**：一句话说明要改进的是哪类误差/哪类标度（例如某类 PES 片段、某噪声模型下的方差）。  
2. **假设与对照**：新想法 vs 最强简单基线（同一数据、同一评价指标）。  
3. **最小实验**：能在一个笔记本或单脚本里跑通；再考虑并入包结构。  
4. **记录**：使用 [`INTERNAL_REPORT_OUTLINE.md`](INTERNAL_REPORT_OUTLINE.md) 的缩略版记录假设、负结果；对外可用 [`SHAREABLE_TECH_NOTE_TEMPLATE.md`](SHAREABLE_TECH_NOTE_TEMPLATE.md)。  
5. **与主线对齐**：若涉及等变势能面或力场，优先对照 [`QMLFF_INTEGRATION.md`](QMLFF_INTEGRATION.md) 中的模块与约定。  

---

## 6. 工程卫生（长期可维护）

- **单一清单**：新增/移动文件时更新 [`PROJECT_FILE_MANIFEST.md`](PROJECT_FILE_MANIFEST.md) 与本文档「资料地图」若有结构性变化。  
- **文档入口**：[`README.md`](README.md) 按主线优先重排；根目录 [`README.md`](../README.md) 指向本页作为 **第一阅读文档**。  
- **大文件与二进制**：PDF、大数据勿提交 Git；笔记仅存链接与要点。  

---

## 7. 你现在可以从哪一步开始

- **若偏理论**：`qc_learn/quantum_chemistry_foundations.md` → `quantum_chemistry/docs/` → `week4_quantum_ml.md`。  
- **若偏动手**：`PHASE0_H2_BASELINE.md` → 跑通 `quantum_chemistry/tutorials/01_complete_h2_vqe.py`。  
- **若偏文献**：`WEEKLY_LITERATURE_WORKFLOW.md` → 建第一篇 `literature_notes/` 精读笔记。  
- **若偏分子任务**：`PHASE2_PIPELINES.md` → 选 `zno_pd_qml` 或 `gnn_adsorption` 一条烟测全链路。  

本页应随你的研究阶段 **每季度小幅修订**（更新子问题、默认阅读顺序、正在复现的论文列表）。
