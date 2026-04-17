"""Regenerate QC_learn_unified/*.md from classical-chem/QC-learn + quantum-chem/learning-ms.

Run from repo root or any cwd:
  python PandM/materials/learning/quantum-chem/build_qc_learn_unified.py
"""
from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    qc = root.parent / "classical-chem" / "QC-learn"
    ms = root / "learning-ms"
    out = root / "QC_learn_unified"
    out.mkdir(parents=True, exist_ok=True)

    def read(p: Path) -> str:
        return p.read_text(encoding="utf-8")

    def sep(title: str) -> str:
        return f"\n\n---\n\n# 【合订分隔】{title}\n\n---\n\n"

    parts: list[str] = []
    parts.append(
        "# QC-learn（classical-chem/QC-learn）与 learning-ms（quantum-chem/learning-ms）合订本\n\n"
        "> **生成说明**：本文件由 `build_qc_learn_unified.py` 将下列源文件按教学顺序全文拼接，"
        "**未对正文做删节或摘要**。原相对路径引用（如 `quantum_chemistry_foundations.md`）仍指向 "
        "QC-learn 目录内文件名，阅读时请以本合订本中对应章节为准。\n>\n"
        "> **源目录**：\n"
        "> - `PandM/materials/learning/classical-chem/QC-learn/`\n"
        "> - `PandM/materials/learning/quantum-chem/learning-ms/`（Markdown 与配套 Notebook 见文末索引）\n\n"
    )

    order: list[tuple[str, Path]] = [
        ("学习计划总览（QC-learn/README.md）", qc / "README.md"),
        ("量子化学理论基础（QC-learn/quantum_chemistry_foundations.md）", qc / "quantum_chemistry_foundations.md"),
        ("第3周 经典机器学习（QC-learn/week3_classical_ml.md）", qc / "week3_classical_ml.md"),
        ("第4周 量子计算方法（learning-ms/week4_quantum_ml.md）", ms / "week4_quantum_ml.md"),
        ("最终项目思路（learning-ms/final_project_ideas.md）", ms / "final_project_ideas.md"),
        ("参考文献：经典教材（QC-learn/references/textbooks.md）", qc / "references" / "textbooks.md"),
        ("参考文献：ML+量子化学（QC-learn/references/ml_quantum_chemistry.md）", qc / "references" / "ml_quantum_chemistry.md"),
        ("参考文献：量子ML（QC-learn/references/quantum_ml.md）", qc / "references" / "quantum_ml.md"),
    ]

    for title, path in order:
        if not path.is_file():
            raise FileNotFoundError(path)
        parts.append(sep(title))
        parts.append(read(path))

    parts.append(sep("learning-ms 目录中的 Jupyter 实践文件（索引，非 ipynb 全文）"))
    parts.append(
        "以下文件仍在原目录，便于直接运行；本合订本未将 `.ipynb` JSON 嵌入，以免破坏 Notebook 工具链。\n\n"
        "| 文件 |\n|------|\n"
        "| `learning-ms/01_VQE原理与实现.ipynb` |\n"
        "| `learning-ms/01_进阶算法综述.ipynb` |\n"
        "| `learning-ms/02_Qiskit_Nature_H2_LiH.ipynb` |\n"
        "| `learning-ms/02_量子优势分析.ipynb` |\n"
    )

    (out / "QC_learn_unified_compendium.md").write_text("".join(parts), encoding="utf-8")

    readme = (
        "# 统一阅读入口（QC-learn + learning-ms）\n\n"
        "原材料曾分属两处：\n\n"
        "1. **`../classical-chem/QC-learn/`** — 学习计划 README、量子化学基础大稿、第3周经典 ML、`references/` 子目录。\n"
        "2. **`../learning-ms/`** — 第4周量子计算讲义、`final_project_ideas.md`、以及若干 Jupyter 练习本。\n\n"
        "## 合订本（本目录）\n\n"
        "仅保留一份全文合订，避免与主稿重复维护：\n\n"
        "| 文件 | 内容 |\n|------|------|\n"
        "| [`QC_learn_unified_compendium.md`](QC_learn_unified_compendium.md) | 全部相关 Markdown **全文**按周次与文献顺序拼接 |\n\n"
        "Notebook 仍在 `../learning-ms/*.ipynb`，未嵌入 Markdown。\n\n"
        "若需改稿，请修改 **QC-learn** 或 **learning-ms** 中的源文件，再运行仓库内 `build_qc_learn_unified.py` 更新合订本。\n"
    )
    (out / "README_unified.md").write_text(readme, encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
