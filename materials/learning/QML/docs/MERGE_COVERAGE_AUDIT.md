# 文档合并与覆盖范围审计（诚实说明）

**2026 更新**：日常**理论与推导**请以 **[`theory_and_derivations.md`](theory_and_derivations.md)** 为**单一阅读入口**（`tools/build_theory_master.py`）；**问答**以 **[`domain_qa.md`](domain_qa.md)** 为第二核心。下文保留历史说明。

**结论先行**：**不能**声称「工程内所有相近文档都已合并成唯一稿」。本仓库刻意保留 **源文件 + 合订本**、**不同写作目的** 的多份正文，并对剩余重叠在下方逐项说明。

若你希望「再合并一档」，请以本页 **§4 可选下一步** 为清单提需求。

---

## 1. 已完成的合并簇（内容合并或目录归一）

| 簇 | 做法 | 合订/归一后位置 |
|----|------|-----------------|
| 根目录按年 QML 文献 + 完整综述 | 删除按年重复，保留完整综述 | `docs/literature/qml_literature_complete_2007_2025.md` |
| QAOA 长文 | 迁入文献目录 | `docs/literature/qaoa_literature_and_notes.md` |
| 化材精读模板 + 2025 优先清单 | 两文件合一 | `docs/literature/chemistry_materials_reading.md` |
| 贫瘠高原、QNTK、非线性、旋转/纠缠、QRC | 五篇合一（标题升一级） | `docs/notes/qml_training_landscape_compendium.md` |
| 吸附、力场论文列表、单篇示例 | 三篇合一 | `docs/topics/chemistry_qml_application_notes.md` |
| 分子力场综述基础/增强 | 迁入 `docs/topics/`，互指链接更新 | `molecular_ff_survey_*.md` |
| 量子 ML「完整指南」两版本 | 迁入 `docs/guides/` | `docs/guides/quantum_ml_complete_guide*.md` |
| `quantum_chemistry/docs` 01–05 | **不删源**；脚本合订 | `docs/unified_chemistry_theory/vol_package_01_05_theory.md` |
| QC-learn 第 3–4 周 + 项目 | **不删源**；脚本合订 | `docs/unified_chemistry_theory/vol_qc_learn_weeks_3_4_and_project.md` |
| QC-learn `references/` 三列表 | **迁入**文献目录 | `docs/literature/references/*.md` |

---

## 2. 刻意保留的「多份」——不是遗漏

| 情况 | 说明 |
|------|------|
| **合订本 vs 源文件** | `vol_*.md`、`*_compendium.md` 由脚本生成；**编辑以源 `.md` 为准**，再重跑脚本。 |
| **`quantum_chemistry_foundations.md`（约 3k+ 行）** | 未并入 `vol_*`，避免单文件过大；在 [`unified_chemistry_theory/README.md`](unified_chemistry_theory/README.md) 中单独列为传统电子结构主阅读。 |
| **`docs/qml_tutorial_zh_consolidated.md`** | 来源为外部 QML Tutorial 中文版，与 `docs/guides/quantum_ml_complete_guide*.md` **不是同一套结构**，不宜硬合并。 |
| **`quantum_ml_complete_guide.md` 与 `_3900lines_version.md`** | 目录与章节结构不同，[`QML_GUIDE_INDEX.md`](QML_GUIDE_INDEX.md) 已说明「默认读主稿、对照读长稿」。 |
| **`molecular_ff_survey_base` 与 `_enhanced`** | 增强版省略前九章正文；两文件互为补充，合并成单文件会重复或丢失交叉引用语义。 |
| **`qml_literature_complete_2007_2025.md` 与 `literature/references/quantum_ml.md` 等** | 前者为逐年综述型长文，后者为教材/主题论文 **列表**，功能不同。 |
| **Phase/工作流类** | `PHASE0`、`PHASE2`、`INTERNAL_REPORT`、`WEEKLY_*` 等是 **流程与模板**，不与理论讲义合并。 |
| **子项目 README** | `gnn_adsorption/README.md`、`zno_pd_qml/README.md` 保留在包内，便于从代码目录阅读。 |

---

## 3. 主题重叠但未合并（需读统一枢纽）

以下 **内容有交叉**（例如 VQE、映射既出现在 `qc_learn/week4`，又出现在 `quantum_chemistry/docs/01–05`），已通过 **[`unified_chemistry_theory/README.md`](unified_chemistry_theory/README.md)** 的阅读顺序对齐，**未**做成物理上的单文件合并：

- `docs/qc_learn/week4_quantum_ml.md` ↔ `quantum_chemistry/docs/02–03`
- `quantum_chemistry/tutorials/01_complete_h2_vqe_explanation.md` ↔ `docs/PHASE0_H2_BASELINE.md`（教程 vs 基线协议）

---

## 4. 可选下一步（若你希望继续压缩）

1. **将 `QML_GUIDE_INDEX.md` 全文并入 `docs/README.md` 节 B**（减少一个入口文件）。  
2. **将 `INTERNAL_REPORT_OUTLINE` 与 `SHAREABLE_TECH_NOTE_TEMPLATE` 做「对内/对外」对照表**（仍保留两文件，仅增加互链段落）。  
3. **对 `qml_literature_complete_2007_2025.md` 与 `literature/references/quantum_ml.md` 做去重标注**（在文献 README 中用表格说明「综述 vs 列表」分工）。  

以上均需你确认偏好（可读性 vs 文件数量）。

---

## 5. 维护约定

- 新增大段理论或文献时：先定 **canonical 源文件**，再决定是否加入对应 `tools/build_*.py` 合订流程。  
- 本页与 [`PROJECT_FILE_MANIFEST.md`](PROJECT_FILE_MANIFEST.md) 在结构性合并后应同步更新。
