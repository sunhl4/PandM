#!/usr/bin/env python3
"""Merge adsorption + molecular FF paper list + example paper analysis into one doc."""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/topics/chemistry_qml_application_notes.md"

PARTS: list[tuple[str, str, str]] = [
    ("adsorption_energy_qml_vs_ridge_notes.md", "part-adsorption", "Part A — 吸附能：QML 与 Ridge 基线"),
    ("molecular_force_field_qml_papers.md", "part-ff-papers", "Part B — 分子力场 × QML 论文列表"),
    ("example_paper_analysis_2501.04264.md", "part-example-2501", "Part C — 单篇精读示例（arXiv:2501.04264）"),
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
        "# 化学 / 分子模拟 × QML 应用笔记（合并稿）",
        "",
        "> 合并自根目录三篇笔记（2026-03-30）。",
        "",
        "| 原文件 | 跳转 |",
        "|--------|------|",
        "| `adsorption_energy_qml_vs_ridge_notes.md` | [Part A](#part-adsorption) |",
        "| `molecular_force_field_qml_papers.md` | [Part B](#part-ff-papers) |",
        "| `example_paper_analysis_2501.04264.md` | [Part C](#part-example-2501) |",
        "",
        "---",
        "",
    ]
    for fname, anchor, title in PARTS:
        body = (ROOT / fname).read_text(encoding="utf-8")
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
