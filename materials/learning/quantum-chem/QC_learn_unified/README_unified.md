# 统一阅读入口（QC-learn + learning-ms）

原材料曾分属两处：

1. **`../classical-chem/QC-learn/`** — 学习计划 README、量子化学基础大稿、第3周经典 ML、`references/` 子目录。
2. **`../learning-ms/`** — 第4周量子计算讲义、`final_project_ideas.md`、以及若干 Jupyter 练习本。

## 合订本（本目录）

仅保留一份全文合订，避免与主稿重复维护：

| 文件 | 内容 |
|------|------|
| [`QC_learn_unified_compendium.md`](QC_learn_unified_compendium.md) | 全部相关 Markdown **全文**按周次与文献顺序拼接 |

Notebook 仍在 `../learning-ms/*.ipynb`，未嵌入 Markdown。

若需改稿，请修改 **QC-learn** 或 **learning-ms** 中的源文件，再运行仓库内 `build_qc_learn_unified.py` 更新合订本。
