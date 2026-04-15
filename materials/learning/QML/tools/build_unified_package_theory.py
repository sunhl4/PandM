#!/usr/bin/env python3
"""Merge quantum_chemistry/docs/01–05 into one navigable compendium."""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/unified_chemistry_theory/vol_package_01_05_theory.md"

PARTS: list[tuple[str, str, str]] = [
    (
        "quantum_chemistry/docs/01_second_quantization_theory.md",
        "pkg-01",
        "卷 A — 二次量子化（原 01）",
    ),
    (
        "quantum_chemistry/docs/02_fermion_qubit_mapping_theory.md",
        "pkg-02",
        "卷 B — 费米子–量子比特映射（原 02）",
    ),
    (
        "quantum_chemistry/docs/03_vqe_theory.md",
        "pkg-03",
        "卷 C — VQE 理论（原 03）",
    ),
    (
        "quantum_chemistry/docs/04_ansatz_theory.md",
        "pkg-04",
        "卷 D — Ansatz 设计（原 04）",
    ),
    (
        "quantum_chemistry/docs/05_excited_states_theory.md",
        "pkg-05",
        "卷 E — 激发态方法（原 05）",
    ),
]


def bump_headings(text: str) -> str:
    return "\n".join("#" + line if line.startswith("#") else line for line in text.splitlines())


def main() -> None:
    chunks = [
        "# 量子化学包理论合订本：`quantum_chemistry/docs` 第 01–05 章",
        "",
        "> **生成**：`tools/build_unified_package_theory.py`。**编辑请改源文件**（`quantum_chemistry/docs/0*.md`），再重跑本脚本。",
        "",
        "| 原路径 | 跳转 |",
        "|--------|------|",
        "| `01_second_quantization_theory.md` | [卷 A](#pkg-01) |",
        "| `02_fermion_qubit_mapping_theory.md` | [卷 B](#pkg-02) |",
        "| `03_vqe_theory.md` | [卷 C](#pkg-03) |",
        "| `04_ansatz_theory.md` | [卷 D](#pkg-04) |",
        "| `05_excited_states_theory.md` | [卷 E](#pkg-05) |",
        "",
        "代码对应见 [`quantum_chemistry/docs/README.md`](../../quantum_chemistry/docs/README.md)。",
        "",
        "---",
        "",
    ]
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
