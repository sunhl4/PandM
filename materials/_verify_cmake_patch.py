import pathlib
import shutil
import subprocess

BAD = '# Licensed under the Apache License, Version 2.0 (the "License");'
OK = "# Licensed under the Apache License, Version 2.0 (the License)"
root = pathlib.Path(__file__).resolve().parent / ".pyscf_win_build" / "src" / "pyscf-2.12.1"
if not root.exists():
    raise SystemExit(f"missing {root}; run install_pyscf_windows.py once to extract")
n = 0
for p in root.rglob("CMakeLists.txt"):
    t = p.read_text(encoding="utf-8")
    if BAD in t:
        p.write_text(t.replace(BAD, OK), encoding="utf-8")
        n += 1
print("patched", n, "CMakeLists.txt")
src = root / "pyscf" / "lib"
bd = pathlib.Path(__file__).resolve().parent / ".pyscf_win_build" / "cmake_verify_build"
shutil.rmtree(bd, ignore_errors=True)
r = subprocess.run(["cmake", "-S", str(src), "-B", str(bd)], capture_output=True, text=True)
print("cmake rc", r.returncode)
print((r.stderr or r.stdout or "")[:1500])
