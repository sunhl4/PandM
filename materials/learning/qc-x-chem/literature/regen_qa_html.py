"""Regenerate 学习问答记录.html from 学习问答记录.md (UTF-8 + MathJax).

Run from this directory:
  python regen_qa_html.py
Requires: pandoc on PATH.
"""
import pathlib
import subprocess

here = pathlib.Path(__file__).resolve().parent
md = here / "学习问答记录.md"
html = here / "学习问答记录.html"
text = md.read_text(encoding="utf-8")
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
        "-",
    ],
    input=text,
    text=True,
    encoding="utf-8",
    check=True,
)
print("Wrote:", html)
