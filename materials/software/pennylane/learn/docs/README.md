# 文档入口（压缩版）

长期维护**两份核心**即可支撑领域专家知识体系；其余为脚本、历史索引与子项目说明。

---

## 1. 必维护（两份）

| 文档 | 作用 |
|------|------|
| **[`theory_and_derivations.md`](theory_and_derivations.md)** | **唯一理论 + 数学推导主稿**（由 `tools/build_theory_master.py` 从源合并生成；**日常改源文件后重跑脚本**）。 |
| **[`domain_qa.md`](domain_qa.md)** | **问答与备忘**：概念、推导疑点、算法、栈、力场、待读论文、对外复述。 |

总纲与节奏（可选重读）：[`DOMAIN_EXPERT_ROADMAP.md`](DOMAIN_EXPERT_ROADMAP.md)。

---

## 2. 主稿的源文件（改这里，再生成主稿）

| 内容块 | 源路径 |
|--------|--------|
| 传统电子结构大稿 | `docs/qc_learn/quantum_chemistry_foundations.md` |
| 二次量子化–激发态 | `quantum_chemistry/docs/01` … `05_*.md` |
| 经典 ML 周次 | `docs/qc_learn/week3_classical_ml.md` |
| 量子算法深度周次 | `docs/qc_learn/week4_quantum_ml.md` |
| 项目与创新思路 | `docs/qc_learn/final_project_ideas.md` |
| QML 训练景观合集 | `docs/notes/qml_training_landscape_compendium.md` |

```bash
python3 tools/build_theory_master.py
```

---

## 3. 其余目录（按需查阅，非日常入口）

| 目录/文件 | 说明 |
|-----------|------|
| [`literature/`](literature/) | 文献综述、QAOA、化材精读模板、教材/论文列表 |
| [`unified_chemistry_theory/`](unified_chemistry_theory/) | 旧「分卷」索引（与主稿重叠；**以 `theory_and_derivations.md` 为准**） |
| [`qc_learn/`](qc_learn/) | 书面课元数据（含 week4 PDF 等） |
| [`PROJECT_FILE_MANIFEST.md`](PROJECT_FILE_MANIFEST.md) | 全路径清单 |
| [`MERGE_COVERAGE_AUDIT.md`](MERGE_COVERAGE_AUDIT.md) | 历史合并审计 |

---

## 4. 工程与管线（非理论正文）

| 文档 | 用途 |
|------|------|
| [`PHASE0_H2_BASELINE.md`](PHASE0_H2_BASELINE.md)、[`PHASE2_PIPELINES.md`](PHASE2_PIPELINES.md) | 基线与烟测 |
| [`QMLFF_INTEGRATION.md`](QMLFF_INTEGRATION.md)、[`QML_KERNEL_VS_VQC.md`](QML_KERNEL_VS_VQC.md) | 与外部仓库/方法对照 |
| [`WEEKLY_LITERATURE_WORKFLOW.md`](WEEKLY_LITERATURE_WORKFLOW.md) | 每周文献节奏 |
