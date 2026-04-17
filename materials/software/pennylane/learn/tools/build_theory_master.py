#!/usr/bin/env python3
"""
Build a single canonical theory + derivations document from repository sources.

Output: docs/theory_and_derivations.md

Edit sources, then: python3 tools/build_theory_master.py
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/theory_and_derivations.md"
MATERIALS = ROOT.parents[2]
QC_LEARN = MATERIALS / "learning" / "classical-chem" / "QC-learn"
QC_MS = MATERIALS / "learning" / "quantum-chem" / "learning-ms"
SW_QC = MATERIALS / "software" / "qc-x-chem" / "quantum_chemistry" / "docs"

# (source_path, display_path_for_table, anchor, title)
PARTS: list[tuple[pathlib.Path, str, str, str]] = [
    (
        QC_LEARN / "quantum_chemistry_foundations.md",
        "learning/classical-chem/QC-learn/quantum_chemistry_foundations.md",
        "part-0",
        "Part 0 — 量子化学基础与传统方法（电子结构、HF、DFT 等）",
    ),
    (
        SW_QC / "01_second_quantization_theory.md",
        "software/qc-x-chem/quantum_chemistry/docs/01_second_quantization_theory.md",
        "part-1",
        "Part 1 — 二次量子化",
    ),
    (
        SW_QC / "02_fermion_qubit_mapping_theory.md",
        "software/qc-x-chem/quantum_chemistry/docs/02_fermion_qubit_mapping_theory.md",
        "part-2",
        "Part 2 — 费米子–量子比特映射",
    ),
    (
        SW_QC / "03_vqe_theory.md",
        "software/qc-x-chem/quantum_chemistry/docs/03_vqe_theory.md",
        "part-3",
        "Part 3 — VQE 理论",
    ),
    (
        SW_QC / "04_ansatz_theory.md",
        "software/qc-x-chem/quantum_chemistry/docs/04_ansatz_theory.md",
        "part-4",
        "Part 4 — Ansatz 设计",
    ),
    (
        SW_QC / "05_excited_states_theory.md",
        "software/qc-x-chem/quantum_chemistry/docs/05_excited_states_theory.md",
        "part-5",
        "Part 5 — 激发态方法",
    ),
    (
        QC_LEARN / "week3_classical_ml.md",
        "learning/classical-chem/QC-learn/week3_classical_ml.md",
        "part-6",
        "Part 6 — 经典机器学习 × 量子化学（NNQS 等）",
    ),
    (
        QC_MS / "week4_quantum_ml.md",
        "learning/quantum-chem/learning-ms/week4_quantum_ml.md",
        "part-7",
        "Part 7 — 量子算法与量子化学（映射、VQE、QPE、ADAPT 等深度稿）",
    ),
    (
        QC_MS / "final_project_ideas.md",
        "learning/quantum-chem/learning-ms/final_project_ideas.md",
        "part-8",
        "Part 8 — 项目思路与创新方向（方法分类框架）",
    ),
    (
        ROOT / "docs/notes/qml_training_landscape_compendium.md",
        "docs/notes/qml_training_landscape_compendium.md",
        "part-9",
        "Part 9 — QML 训练景观（贫瘠高原、QNTK、非线性、QRC 等）",
    ),
]


def bump_headings(text: str) -> str:
    return "\n".join("#" + line if line.startswith("#") else line for line in text.splitlines())


def main() -> None:
    toc_rows = []
    for _src, _disp, anchor, title in PARTS:
        toc_rows.append(f"- [{title}](#{anchor})")

    chunks = [
        "# 量子计算化学 · 理论与数学推导（单一主稿）",
        "",
        "> **维护约定**：本文件由 `tools/build_theory_master.py` **自动生成**。"
        "日常修改请在 **下方列出的源文件** 中进行，然后运行 `python3 tools/build_theory_master.py` 更新本稿。"
        "书面课源稿：`learning/classical-chem/QC-learn/`（基础、第 3 周）与 `learning/quantum-chem/learning-ms/`（第 4 周与项目）；"
        "包内理论仍以 `software/qc-x-chem/quantum_chemistry/docs/` 为准。",
        "",
        "## 源文件清单（按合并顺序）",
        "",
        "| 顺序 | 源路径 |",
        "|------|--------|",
    ]
    for i, (_src, disp, _a, _title) in enumerate(PARTS):
        chunks.append(f"| {i} | `{disp}` |")
    chunks.extend(
        [
            "",
            "## 目录（本文件内跳转）",
            "",
            *toc_rows,
            "",
            "---",
            "",
        ]
    )

    for src_path, _disp, anchor, title in PARTS:
        body = src_path.read_text(encoding="utf-8")
        chunks.append(f'<a id="{anchor}"></a>')
        chunks.append(f"## {title}")
        chunks.append("")
        chunks.append(bump_headings(body))
        chunks.append("")
        chunks.append("---")
        chunks.append("")

    OUT.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
