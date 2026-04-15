"""Regenerate 学习问答记录.html from 学习问答记录.md (UTF-8 + MathJax 3).

Run from this directory:
  python regen_qa_html.py

Requires: pandoc on PATH.

Options baked in:
  - zh-CN lang + proper HTML title
  - Table of contents (depth 3) for long-document navigation
  - Extra styles in qa-html-header.html (tables, TOC, CJK-friendly font stack)
"""
import pathlib
import subprocess

here = pathlib.Path(__file__).resolve().parent
md = here / "学习问答记录.md"
html = here / "学习问答记录.html"
header = here / "qa-html-header.html"
text = md.read_text(encoding="utf-8")

cmd = [
    "pandoc",
    "-f",
    "markdown",
    "-t",
    "html5",
    "--standalone",
    "--mathjax",
    "-M",
    "title=量子计算化学学习 · 问答记录",
    "-V",
    "lang=zh-CN",
    "--toc",
    "--toc-depth=3",
    "-o",
    str(html),
]
if header.is_file():
    cmd.extend(["--include-in-header", str(header)])
cmd.append("-")

subprocess.run(cmd, input=text, text=True, encoding="utf-8", check=True)
print("Wrote:", html)
