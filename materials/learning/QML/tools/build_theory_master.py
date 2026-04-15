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

# Order: foundations → package 01–05 → QC-learn week3/4 → QML training landscape compendium
PARTS: list[tuple[str, str, str]] = [
    (
        "docs/qc_learn/quantum_chemistry_foundations.md",
        "part-0",
        "Part 0 — 量子化学基础与传统方法（电子结构、HF、DFT 等）",
    ),
    (
        "quantum_chemistry/docs/01_second_quantization_theory.md",
        "part-1",
        "Part 1 — 二次量子化",
    ),
    (
        "quantum_chemistry/docs/02_fermion_qubit_mapping_theory.md",
        "part-2",
        "Part 2 — 费米子–量子比特映射",
    ),
    (
        "quantum_chemistry/docs/03_vqe_theory.md",
        "part-3",
        "Part 3 — VQE 理论",
    ),
    (
        "quantum_chemistry/docs/04_ansatz_theory.md",
        "part-4",
        "Part 4 — Ansatz 设计",
    ),
    (
        "quantum_chemistry/docs/05_excited_states_theory.md",
        "part-5",
        "Part 5 — 激发态方法",
    ),
    (
        "docs/qc_learn/week3_classical_ml.md",
        "part-6",
        "Part 6 — 经典机器学习 × 量子化学（NNQS 等）",
    ),
    (
        "docs/qc_learn/week4_quantum_ml.md",
        "part-7",
        "Part 7 — 量子算法与量子化学（映射、VQE、QPE、ADAPT 等深度稿）",
    ),
    (
        "docs/qc_learn/final_project_ideas.md",
        "part-8",
        "Part 8 — 项目思路与创新方向（方法分类框架）",
    ),
    (
        "docs/notes/qml_training_landscape_compendium.md",
        "part-9",
        "Part 9 — QML 训练景观（贫瘠高原、QNTK、非线性、QRC 等）",
    ),
]


def bump_headings(text: str) -> str:
    return "\n".join("#" + line if line.startswith("#") else line for line in text.splitlines())


def main() -> None:
    toc_rows = []
    for _rel, anchor, title in PARTS:
        short = title.split("—", 1)[-1].strip()
        toc_rows.append(f"- [{title}](#{anchor})")

    chunks = [
        "# 量子计算化学 · 理论与数学推导（单一主稿）",
        "",
        "> **维护约定**：本文件由 `tools/build_theory_master.py` **自动生成**。"
        "日常修改请在 **下方列出的源文件** 中进行，然后运行 `python3 tools/build_theory_master.py` 更新本稿。"
        "若需与代码对齐，仍以 `quantum_chemistry/docs/` 与 `docs/qc_learn/` 下源文件为准。",
        "",
        "## 源文件清单（按合并顺序）",
        "",
        "| 顺序 | 源路径 |",
        "|------|--------|",
    ]
    for i, (rel, _a, _title) in enumerate(PARTS):
        chunks.append(f"| {i} | `{rel}` |")
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

    for rel, anchor, title in PARTS:
        path = ROOT / rel
        body = path.read_text(encoding="utf-8")
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
