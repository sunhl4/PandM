#!/usr/bin/env python3
"""One-off builder: merge root-level QML theory markdown into docs/notes/ compendium."""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/notes/qml_training_landscape_compendium.md"

PARTS: list[tuple[str, str, str]] = [
    ("Barren_Plateau.md", "part-barren", "Part 1 — 贫瘠高原与可训练性"),
    ("QNTK_UCB_论文解析.md", "part-qntk", "Part 2 — QNTK / UCB 解析"),
    ("quantum_nonlinearity_explained.md", "part-nonlinearity", "Part 3 — 量子机器学习中的非线性"),
    ("rotation_entanglement_order_explained.md", "part-rotation-order", "Part 4 — 旋转门与纠缠门顺序"),
    ("Quantum_Reservoir_Computing.md", "part-qrc", "Part 5 — 量子储备池计算"),
]


def bump_headings(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("#"):
            lines.append("#" + line)
        else:
            lines.append(line)
    return "\n".join(lines)


def main() -> None:
    chunks = [
        "# QML 训练景观与理论基础（合并稿）",
        "",
        "> 由根目录五篇笔记合并（2026-03-30）。原独立 `.md` 已删除；精读可仍按 Part 分段。",
        "",
        "| 原文件 | 跳转 |",
        "|--------|------|",
        "| `Barren_Plateau.md` | [Part 1](#part-barren) |",
        "| `QNTK_UCB_论文解析.md` | [Part 2](#part-qntk) |",
        "| `quantum_nonlinearity_explained.md` | [Part 3](#part-nonlinearity) |",
        "| `rotation_entanglement_order_explained.md` | [Part 4](#part-rotation-order) |",
        "| `Quantum_Reservoir_Computing.md` | [Part 5](#part-qrc) |",
        "",
        "---",
        "",
    ]
    for fname, anchor, title in PARTS:
        path = ROOT / fname
        body = path.read_text(encoding="utf-8")
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
