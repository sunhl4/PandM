"""Install PySCF on native Windows by patching upstream build files.

Patches applied
---------------
1. **setup.py**: replace ``f'-S{src_dir}'`` with separate ``-S`` / path (avoids ``-SC:\\...``).
2. **CMakeLists.txt**: remove the long Apache ``#`` header (or an old ``#[[ ]]`` block) and
   insert a minimal ASCII stub (no semicolons or double-quotes in that line).
3. **CMakeLists.txt**: move ``) # trailing comments`` onto the next line — CMake on Windows
   can mis-tokenize some of these and report a bogus parse error at line 1.
4. Normalize curly quotes to ASCII in all CMake files.
5. Rewrite ``include(`` to ``INCLUDE(`` at line start (CMake 3.31 + Windows: lowercase
   ``include`` after ``cmake_minimum_required`` can break the lexer).
6. Remove spaces after ``cmake_minimum_required`` / ``project`` before ``(`` (same toolchain quirk).
7. Remove ``set(CMAKE_VERBOSE_MAKEFILE OFF)`` entirely (and any commented copy containing
   ``OFF``): on Windows, CMake mis-parses this pattern and may report a bogus error at line 1.
8. In ``pyscf/lib/CMakeLists.txt`` only, move ``INCLUDE(CheckSymbolExists)`` to **after** the first
   ``if(NOT CMAKE_BUILD_TYPE)...endif()`` block (keeps the top of the file closer to upstream order).

**CMake on native Windows** (several 3.27–3.31 builds were checked): ``cmake -S`` can hit bogus
``Parse error`` / ``Expected a command name`` on valid ``CMakeLists.txt`` (often reported at line 1).
Do **not** bulk-replace ``endif()`` with ``endif(1)`` — that can make parsing worse after a few
replacements. If configure still fails after patching, use ``--no-verify`` to skip the pre-check and
try ``pip install``, or build under **WSL2** / Linux.

Requires **Visual Studio Build Tools 2022** (workload *Desktop development with C++*) so ``cl`` / NMake
are on PATH—install alone is not enough; open **x64 Native Tools Command Prompt for VS** (or run
``vcvars64.bat``) before ``pip``/this script. Even then, some CMake 3.31 + NMake combinations choke on
PySCF’s ``if/else`` blocks; if configure still fails, use WSL/Linux.

Usage (conda env ``qc_chem``), from ``LearningPlan/learning_materials``::

    python install_pyscf_windows.py
    python install_pyscf_windows.py 2.12.1
    python install_pyscf_windows.py --no-verify
    python install_pyscf_windows.py 2.12.1 --no-verify
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile

HERE = pathlib.Path(__file__).resolve().parent
WORK_ROOT = HERE / ".pyscf_win_build"
OLD = r"cmd = ['cmake', f'-S{src_dir}', f'-B{self.build_temp}']"
NEW = "cmd = ['cmake', '-S', src_dir, '-B', self.build_temp]"

# No ``;`` ``"`` or parentheses — CMake Windows lexer is picky inside ``#`` lines.
_LICENSE_STUB = (
    "# PySCF C extension build. License Apache 2.0. Full text in sdist and upstream repo.\n\n"
)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _normalize_curly_quotes(text: str) -> str:
    return (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def replace_pyscf_cmake_license_header(text: str) -> str | None:
    t = _normalize_newlines(text)
    if t.startswith(_LICENSE_STUB):
        return None
    if t.startswith("# PySCF C extension"):
        lines = t.splitlines(True)
        j = 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        return _LICENSE_STUB + "".join(lines[j:])
    if t.startswith("#[["):
        for marker in ("]]\n\n", "]]\n"):
            end = t.find(marker)
            if end != -1:
                return _LICENSE_STUB + t[end + len(marker) :].lstrip("\n")
        return None
    if not t.startswith("# Copyright 2014-2018 The PySCF Developers"):
        return None
    lines = t.splitlines(True)
    cut = None
    for i, line in enumerate(lines):
        if "# limitations under the License." in line:
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            cut = j
            break
    if cut is None:
        return None
    return _LICENSE_STUB + "".join(lines[cut:])


def _uppercase_include_command(text: str) -> str:
    """Turn ``include(`` into ``INCLUDE(`` when it is the CMake command at line start."""
    return re.sub(r"^(\s*)include\s*\(", r"\1INCLUDE(", text, flags=re.MULTILINE)


def _fix_cmake_minimum_and_project_spacing(text: str) -> str:
    text = re.sub(
        r"\bcmake_minimum_required\s+\(",
        "cmake_minimum_required(",
        text,
    )
    text = re.sub(r"\bproject\s+\(", "project(", text)
    return text


def _split_rpath_ifelseif_for_windows(text: str) -> str:
    """Replace ``if(WIN32) elseif(APPLE) else() endif()`` with three plain ``if`` blocks.

    Some Windows ``cmake -S`` builds mis-parse the elseif/else chain after
    ``cmake_minimum_required`` (bogus error at line 1).  Semantics unchanged.
    """

    pattern = (
        r"(?ms)^# See also https://gitlab\.kitware\.com/cmake/community/wikis/doc/cmake/RPATH-handling\s*\n"
        r"if \(WIN32\)\s*\n"
        r"  #\?\s*\n"
        r"elseif \(APPLE\)\s*\n"
        r"(?P<apple>.*?)"
        r"else \(\s*\)\s*\n"
        r"(?P<other>.*?)"
        r"endif \(\s*\)\s*\n"
    )

    def repl(m: re.Match[str]) -> str:
        apple = m.group("apple")
        other = m.group("other")
        return (
            "# See also https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling\n"
            "# Split elseif chain for Windows CMake (parse quirk with cmake -S).\n"
            "if(WIN32)\n"
            "  #?\n"
            "endif()\n"
            "if(APPLE AND NOT WIN32)\n"
            f"{apple}"
            "endif()\n"
            "if(NOT APPLE AND NOT WIN32)\n"
            f"{other}"
            "endif()\n"
        )

    new_text, n = re.subn(pattern, repl, text, count=1)
    return new_text if n else text


def _reorder_include_check_symbol_after_build_type(text: str) -> str:
    """Move INCLUDE(CheckSymbolExists) below the first CMAKE_BUILD_TYPE if/endif block."""

    def repl(m: re.Match[str]) -> str:
        return m.group(1) + "\nINCLUDE(CheckSymbolExists)\n\n"

    return re.sub(
        r"(?ms)^INCLUDE\(CheckSymbolExists\)\s*\n\s*\n"
        r"((?:if\s*\(\s*NOT\s+CMAKE_BUILD_TYPE\s*\).*?^endif\s*\([^)]*\)\s*\n))",
        repl,
        text,
        count=1,
    )


def _remove_verbose_makefile_off(text: str) -> str:
    """Drop ``set(CMAKE_VERBOSE_MAKEFILE OFF)`` — even commenting it out can break Windows CMake."""
    text = re.sub(
        r"^\s*set\(CMAKE_VERBOSE_MAKEFILE\s+OFF\)\s*\n",
        "",
        text,
        flags=re.MULTILINE,
    )
    return re.sub(
        r"^\s*#\s*set\(CMAKE_VERBOSE_MAKEFILE\s+OFF\).*\n",
        "",
        text,
        flags=re.MULTILINE,
    )


def _split_dcmake_trailing_hash(text: str) -> str:
    """Split ``-DVAR=value # comment`` (libxc ExternalProject line) for Windows CMake."""
    return re.sub(
        r"^(\s+)(-DCMAKE_POLICY_VERSION_MINIMUM=3\.5)\s+(#.*)$",
        r"\1\2\n\1    \3",
        text,
        flags=re.MULTILINE,
    )


def _split_inline_close_paren_comments(text: str) -> str:
    """Turn ``foo) # bar`` into ``foo\\n<indent># bar`` when safe (no ``\"`` before ``) #``)."""
    out: list[str] = []
    for line in text.splitlines(True):
        raw = line
        line_nl = ""
        if raw.endswith("\n"):
            line_nl = "\n"
            raw = raw[:-1]
        idx = raw.find(") #")
        if idx == -1:
            out.append(line)
            continue
        before = raw[: idx + 1]
        after_hash = raw[idx + 1 :].lstrip()  # '# ...'
        if '"' in before:
            out.append(line)
            continue
        indent = len(raw) - len(raw.lstrip())
        sp = raw[:indent]
        out.append(before + line_nl)
        out.append(f"{sp}  {after_hash}{line_nl}")
    return "".join(out)


def patch_pyscf_cmake_lists(extracted: pathlib.Path) -> int:
    lib_root = (extracted / "pyscf" / "lib" / "CMakeLists.txt").resolve()
    n = 0
    for path in extracted.rglob("CMakeLists.txt"):
        orig = path.read_text(encoding="utf-8")
        orig_n = _normalize_newlines(orig)
        lic = replace_pyscf_cmake_license_header(orig)
        base = lic if lic is not None else orig_n
        text2 = _remove_verbose_makefile_off(
            _uppercase_include_command(
                _fix_cmake_minimum_and_project_spacing(
                    _split_dcmake_trailing_hash(
                        _split_inline_close_paren_comments(_normalize_curly_quotes(base))
                    )
                )
            )
        )
        if path.resolve() == lib_root:
            text2 = _split_rpath_ifelseif_for_windows(text2)
            text2 = _reorder_include_check_symbol_after_build_type(text2)
        if orig_n != text2:
            path.write_text(text2, encoding="utf-8", newline="\n")
            n += 1
    return n


def patch_setup_py(extracted: pathlib.Path) -> None:
    setup_py = extracted / "setup.py"
    text = setup_py.read_text(encoding="utf-8")
    if NEW in text:
        return
    if OLD not in text:
        m = re.search(r"cmd = \['cmake',[^\n]+\]", text)
        raise SystemExit(
            "Could not find expected cmake cmd line in setup.py: "
            f"{m.group(0) if m else 'no match'}"
        )
    setup_py.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")


def verify_cmake_lib_configure(extracted: pathlib.Path) -> tuple[int, str]:
    """Run cmake on ``pyscf/lib``; return (returncode, stderr tail)."""
    lib = extracted / "pyscf" / "lib"
    if not lib.is_dir():
        return 1, f"missing {lib}"
    bd = WORK_ROOT / "_cmake_verify"
    if bd.exists():
        shutil.rmtree(bd)
    bd.mkdir(parents=True)
    cmake_exe = pathlib.Path(sys.executable).parent / ("cmake.exe" if sys.platform == "win32" else "cmake")
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        cand = pathlib.Path(conda_prefix) / "Library" / "bin" / "cmake.exe"
        if cand.is_file():
            cmake_exe = cand
    if not cmake_exe.is_file():
        cmake_exe = pathlib.Path("cmake")
    cmd = [str(cmake_exe), "-S", str(lib), "-B", str(bd)]
    if pathlib.Path(r"C:\Program Files\Microsoft Visual Studio").exists():
        cmd = [
            str(cmake_exe),
            "-G",
            "Visual Studio 17 2022",
            "-A",
            "x64",
            "-S",
            str(lib),
            "-B",
            str(bd),
        ]
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        encoding="utf-8",
        errors="replace",
    )
    err = (r.stderr or "") + (r.stdout or "")
    # Lexer / parse errors are what we patch for; missing MSVC still yields rc != 0.
    if "Parse error" in err or "Expected a command name, got unquoted argument" in err:
        return 1, err[-2500:]
    return 0, err[-2500:]


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--no-verify"]
    no_verify = "--no-verify" in sys.argv[1:]
    version = args[0] if args else ""
    spec = f"pyscf=={version}" if version else "pyscf"

    if not pathlib.Path(r"C:\Program Files\Microsoft Visual Studio").exists():
        print(
            "Warning: Visual Studio not found under default path. "
            "Build may fail after configure without VS Build Tools (C++).",
            file=sys.stderr,
        )

    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    dl_dir = WORK_ROOT / "download"
    if dl_dir.exists():
        shutil.rmtree(dl_dir)
    dl_dir.mkdir()

    subprocess.run(
        [sys.executable, "-m", "pip", "download", "--no-deps", "-d", str(dl_dir), spec],
        check=True,
    )

    archives = sorted(dl_dir.glob("pyscf-*.tar.gz"))
    if not archives:
        raise SystemExit("No pyscf-*.tar.gz found after pip download.")

    extract_root = WORK_ROOT / "src"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir()

    with tarfile.open(archives[0], "r:gz") as tf:
        if sys.version_info >= (3, 12):
            tf.extractall(extract_root, filter="data")
        else:
            tf.extractall(extract_root)

    candidates = list(extract_root.glob("pyscf-*"))
    if not candidates:
        raise SystemExit(f"No pyscf-* folder under {extract_root}")
    extracted = candidates[0]

    patch_setup_py(extracted)
    n_cm = patch_pyscf_cmake_lists(extracted)
    print(f"Patched {n_cm} CMakeLists.txt file(s).")

    if no_verify:
        print("Skipping CMake configure check (--no-verify).", file=sys.stderr)
    else:
        rc, tail = verify_cmake_lib_configure(extracted)
        if rc != 0:
            print(
                "CMake configure check failed (return code %d). Last output:\n%s\n"
                "Hint: try `python install_pyscf_windows.py --no-verify` or build PySCF under WSL2/Linux."
                % (rc, tail),
                file=sys.stderr,
            )
            raise SystemExit(rc)
        print("CMake configure check passed.")

    r = subprocess.run([sys.executable, "-m", "pip", "install", str(extracted)], check=False)
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
