"""Regenerate 学习问答记录.html from 学习问答记录.md (UTF-8 + MathJax).

Run after editing the markdown file:
  python regen_qa_html.py
Requires: pandoc on PATH.
"""
import pathlib
import subprocess

here = pathlib.Path(__file__).resolve().parent
md = here / "学习问答记录.md"
html = here / "学习问答记录.html"
subprocess.run(
    [
        "pandoc",
        "-f",
        "markdown",
        "-t",
        "html5",
        "--standalone",
        "--mathjax",
        "-o",
        str(html),
    ],
    input=md.read_bytes(),
    check=True,
)
print("Wrote:", html)
