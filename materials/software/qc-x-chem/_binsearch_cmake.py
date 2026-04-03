"""Binary-search CMakeLists prefix that triggers parse error."""
import re
import shutil
import subprocess
from pathlib import Path

CMAKE = Path(r"C:\Users\Administrator\anaconda3\envs\qc_chem\Scripts\cmake.exe")
_HERE = Path(__file__).resolve().parent
_SRC_ROOT = _HERE / ".pyscf_win_build" / "src"
_candidates = sorted(_SRC_ROOT.glob("pyscf-*/pyscf/lib/CMakeLists.txt"))
SRC = _candidates[-1] if _candidates else _HERE / ".pyscf_win_build" / "src" / "pyscf-2.12.1" / "pyscf" / "lib" / "CMakeLists.txt"
if not SRC.is_file():
    raise SystemExit(
        f"Missing PySCF CMakeLists at {SRC}. Unpack sources under {_SRC_ROOT} "
        "or set SRC in _binsearch_cmake.py."
    )
raw = SRC.read_text(encoding="utf-8")
raw = re.sub(r"\bif\s+\(", "if(", raw)
raw = re.sub(r"\belseif\s+\(", "elseif(", raw)
L = raw.splitlines(True)
td = Path(r"C:\Users\Administrator\AppData\Local\Temp\pyscf_bs")
bd = Path(r"C:\Users\Administrator\AppData\Local\Temp\pyscf_bs_b")


def try_n(n: int) -> bool:
    body = "".join(L[:n])
    td.mkdir(exist_ok=True)
    (td / "CMakeLists.txt").write_text(body, encoding="utf-8", newline="\n")
    shutil.rmtree(bd, ignore_errors=True)
    r = subprocess.run(
        [str(CMAKE), "-S", str(td), "-B", str(bd)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    err = r.stderr or ""
    return "Parse error" in err


lo, hi = 1, len(L)
last_ok = 0
while lo <= hi:
    mid = (lo + hi) // 2
    if try_n(mid):
        hi = mid - 1
    else:
        last_ok = mid
        lo = mid + 1

print("total lines", len(L))
print("last_ok", last_ok)
if last_ok < len(L):
    print("FIRST FAIL LINE", last_ok + 1, repr(L[last_ok][:120]))
