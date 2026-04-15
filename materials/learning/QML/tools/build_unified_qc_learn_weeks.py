#!/usr/bin/env python3
"""Merge docs/qc_learn week3, week4, final_project into one compendium."""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/unified_chemistry_theory/vol_qc_learn_weeks_3_4_and_project.md"

PARTS: list[tuple[str, str, str]] = [
    ("docs/qc_learn/week3_classical_ml.md", "ql-w3", "卷 F — 第 3 周：经典 ML × 量子化学"),
    ("docs/qc_learn/week4_quantum_ml.md", "ql-w4", "卷 G — 第 4 周：量子算法与 VQE 深度稿"),
    ("docs/qc_learn/final_project_ideas.md", "ql-proj", "卷 H — 最终项目与创新方向"),
]


def bump_headings(text: str) -> str:
    return "\n".join("#" + line if line.startswith("#") else line for line in text.splitlines())


def main() -> None:
    chunks = [
        "# QC-learn 合订本：第 3–4 周 + 项目思路",
        "",
        "> **生成**：`tools/build_unified_qc_learn_weeks.py`。**编辑请改源文件**（`docs/qc_learn/week*.md` 等），再重跑本脚本。",
        "",
        "> **说明**：与「电子结构传统理论」大稿见 [`../qc_learn/quantum_chemistry_foundations.md`](../qc_learn/quantum_chemistry_foundations.md)（未并入本卷，避免单文件过大）。",
        "",
        "| 原路径 | 跳转 |",
        "|--------|------|",
        "| `week3_classical_ml.md` | [卷 F](#ql-w3) |",
        "| `week4_quantum_ml.md` | [卷 G](#ql-w4) |",
        "| `final_project_ideas.md` | [卷 H](#ql-proj) |",
        "",
        "---",
        "",
    ]
    for rel, anchor, title in PARTS:
        body = (ROOT / rel).read_text(encoding="utf-8")
        chunks.append(f'<a id="{anchor}"></a>')
        chunks.append(f"## {title}")
        chunks.append("")
        chunks.append(bump_headings(body))
        chunks.append("")
        chunks.append("---")
        chunks.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
