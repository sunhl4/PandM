# 量子化学 4 周学习计划（书面课程）

> **日常通读理论**：请以 **[`../theory_and_derivations.md`](../theory_and_derivations.md)** 为唯一主稿（含本目录 `quantum_chemistry_foundations.md`、`week3`/`week4` 等合并生成）。修改正文请在源 `.md` 中编辑后运行 `python3 tools/build_theory_master.py`。

> **来源**：原独立仓库 `/Users/shl/nvidia/QC-learn` 已于 **2026-03-29** 并入本仓库，canonical 路径为 **`docs/qc_learn/`**。根目录 `week1`–`week5` 的 **Python 演示**与此处「周」主题相关但不一一对应。

> **旧分卷索引**（可选）：[`../unified_chemistry_theory/README.md`](../unified_chemistry_theory/README.md)。

## 计划概述

本计划面向有 DFT 实践背景的量子化学研究者，系统复习量子化学数学与近似层次，并衔接经典 ML、量子 ML；目标之一是能够**提出并分析**用机器学习方法近似求解薛定谔方程的新思路。

## 学习目标

完成本计划后，学习者应能：

1. 理解薛定谔方程的数学结构与求解难点  
2. 掌握传统量子化学方法的数学原理与近似思想  
3. 理解经典 ML 与量子 ML 在量子化学中的理论位置  
4. 能提出新的近似思路并讨论其数学性质与可行性  
5. 具备阅读前沿理论论文的基础  

## 学习结构（与仓库内其他材料对齐）

| 周次 | 本目录文件 | 内容重点 | 与本仓库其他位置的衔接 |
|------|------------|----------|-------------------------|
| 第 1–2 周 | [`quantum_chemistry_foundations.md`](quantum_chemistry_foundations.md) | 希尔伯特空间、TISE、变分原理、HF、后 HF、DFT、基组、误差 | 推导型补充：[`quantum_chemistry/docs/`](../quantum_chemistry/docs/)（二次量子化、映射、VQE、ansatz） |
| 第 3 周 | [`week3_classical_ml.md`](week3_classical_ml.md) | NNQS、VMC+ML、深度学习电子结构 | 实践与力场管线：根目录 `week*` 脚本、`gnn_adsorption/` |
| 第 4 周 | [`week4_quantum_ml.md`](week4_quantum_ml.md)、[`week4_quantum_ml.pdf`](week4_quantum_ml.pdf) | 费米子–qubit 映射、VQE、QPE、ADAPT-VQE、噪声与容错展望 | 代码：[`quantum_chemistry/tutorials/`](../quantum_chemistry/tutorials/)、[`docs/PHASE0_H2_BASELINE.md`](../PHASE0_H2_BASELINE.md) |
| 项目 | [`final_project_ideas.md`](final_project_ideas.md) | 新方法分类与创新方向小结 | 与 [`docs/INTERNAL_REPORT_OUTLINE.md`](../INTERNAL_REPORT_OUTLINE.md) 可对照使用 |

> **说明**：原 QC-learn `README` 中曾列出 `week1_foundations.md`、`week2_traditional_methods.md`，仓库中未单独拆分；**第 1–2 周内容已合并进** `quantum_chemistry_foundations.md`。

## 文件结构

```
docs/qc_learn/
├── README.md                          # 本总览（已更新链接与对齐说明）
├── quantum_chemistry_foundations.md   # 第 1–2 周：理论基础与传统方法（合并稿）
├── week3_classical_ml.md              # 第 3 周：经典 ML
├── week4_quantum_ml.md                # 第 4 周：量子计算/量子 ML 深度稿（Markdown）
├── week4_quantum_ml.pdf               # 第 4 周：同上 PDF 导出
├── final_project_ideas.md             # 最终项目与方法框架思路
└── references/
    └── README.md                      # 占位：正文已迁至 [`../literature/references/`](../literature/references/)
```

## 学习方法建议

### 每周深度

- **数学推导**：重要结论尽量跟完推导，再记物理图像。  
- **方法比较**：对照近似假设、复杂度与误差来源。  
- **前沿**：结合 [`../literature/references/`](../literature/references/) 中选读，与 [`docs/WEEKLY_LITERATURE_WORKFLOW.md`](../WEEKLY_LITERATURE_WORKFLOW.md) 节奏一致。

### 符号与引用锚点

对外写作或内部报告中的符号约定见 [`docs/EXTERNAL_ANCHORS.md`](../EXTERNAL_ANCHORS.md)；更长的教材书目见本目录 [`references/textbooks.md`](references/textbooks.md)。

## 学习进度跟踪

- [ ] 第 1–2 周：`quantum_chemistry_foundations.md`  
- [ ] 第 3 周：`week3_classical_ml.md`  
- [ ] 第 4 周：`week4_quantum_ml.md`（及可选 PDF）  
- [ ] 最终项目思路：`final_project_ideas.md`  

## 参考文献

分主题列表见 [`../literature/references/README.md`](../literature/references/README.md)。
