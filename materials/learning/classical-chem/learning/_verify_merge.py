# -*- coding: utf-8 -*-
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def lines_between(text: str, start_pat: str, end_pat: str | None) -> str:
    ls = text.splitlines(keepends=True)
    start_i = None
    for i, line in enumerate(ls):
        if line.startswith(start_pat):
            start_i = i
            break
    if start_i is None:
        raise ValueError(f"start not found: {start_pat!r}")
    if end_pat is None:
        return "".join(ls[start_i:])
    end_i = None
    for j in range(start_i + 1, len(ls)):
        if ls[j].startswith(end_pat):
            end_i = j
            break
    if end_i is None:
        return "".join(ls[start_i:])
    return "".join(ls[start_i:end_i])


def main() -> None:
    f = (ROOT / "quantum_chemistry_foundations.md").read_text(encoding="utf-8")
    c = (ROOT / "经典量子化学方法详解.md").read_text(encoding="utf-8")
    s = (ROOT / "分裂价极化与原子轨道杂化.md").read_text(encoding="utf-8")
    h = (ROOT / "Hartree-Fock理论推导与算法.md").read_text(encoding="utf-8")
    m = (ROOT / "量子化学理论基础与方法_合并本.md").read_text(encoding="utf-8")

    head = f.splitlines(keepends=True)
    goals = "".join(head[0:25])
    sec1 = lines_between(f, "## 1. 量子力学数学基础", "## 2. ")
    sec2 = lines_between(f, "## 2. 时间无关薛定谔方程的数学结构", "## 3. ")
    sec3 = lines_between(f, "## 3. 变分原理（Variational Principle）", "## 4. ")
    sec4 = lines_between(f, "## 4. Hartree-Fock理论的数学推导", "## 5. ")
    sec5 = lines_between(f, "## 5. 电子相关性的数学描述", "## 6. ")
    sec6 = lines_between(f, "## 6. 后Hartree-Fock方法", "## 7. ")
    sec7 = lines_between(f, "## 7. 密度泛函理论", "## 8. ")
    sec8 = lines_between(f, "## 8. 基组展开的数学原理", "## 9. ")
    sec9 = lines_between(f, "## 9. 误差分析和数学性质", "## 10. ")
    sec10 = lines_between(f, "## 10. 理论思想总结", "## 11. ")
    sec11 = lines_between(f, "## 11. 思考题", "## 12. ")
    sec12 = lines_between(f, "## 12. 总结", None)
    recon_f = (
        goals
        + sec1
        + sec2
        + sec3
        + sec4
        + sec5
        + sec6
        + sec7
        + sec8
        + sec9
        + sec10
        + sec11
        + sec12
    )
    ok_f = recon_f == f
    print("foundations byte-equal:", ok_f)
    if not ok_f:
        print("  len f", len(f), "len recon", len(recon_f), "diff", len(f) - len(recon_f))
        for i, (a, b) in enumerate(zip(recon_f, f)):
            if a != b:
                print("  first mismatch index", i)
                break
        else:
            print("  one is prefix of other?", len(recon_f), len(f))

    c_intro = lines_between(c, "# 经典量子化学方法详解", "## §1 ")
    c_s1 = lines_between(c, "## §1 电子结构问题：从薛定谔方程到近似", "## §2 ")
    c_s2 = lines_between(c, "## §2 基组（Basis Sets）", "## §3 ")
    c_s3 = lines_between(c, "## §3 Hartree–Fock（HF）方法", "## §4 ")
    c_s4 = lines_between(c, "## §4 组态相互作用（Configuration Interaction, CI）", "## §5 ")
    c_s5 = lines_between(c, "## §5 多体微扰理论（MBPT / Møller–Plesset）", "## §6 ")
    c_s6 = lines_between(c, "## §6 耦合簇理论（Coupled Cluster, CC）", "## §7 ")
    c_s7 = lines_between(c, "## §7 多参考方法（Multi-Reference Methods）", "## §8 ")
    c_s8 = lines_between(c, "## §8 DFT 理论基础", "## §9 ")
    c_s9 = lines_between(c, "## §9 交换–相关泛函：Jacob's Ladder", "## §10 ")
    c_s10 = lines_between(c, "## §10 DFT 实践指南", "# Part IV")
    c_part4 = lines_between(c, "# Part IV", "## §11 ")
    c_s11 = lines_between(c, "## §11 方法层次与决策", "## §12 ")
    c_s12 = lines_between(c, "## §12 现代发展趋势（2020–2026）", "## 参考文献")
    c_refs = lines_between(c, "## 参考文献", None)
    recon_c = (
        c_intro
        + c_s1
        + c_s2
        + c_s3
        + c_s4
        + c_s5
        + c_s6
        + c_s7
        + c_s8
        + c_s9
        + c_s10
        + c_part4
        + c_s11
        + c_s12
        + c_refs
    )
    ok_c = recon_c == c
    print("classic byte-equal:", ok_c)
    if not ok_c:
        print("  len c", len(c), "len recon", len(recon_c), "diff", len(c) - len(recon_c))

    ok_s = s in m
    ok_h = h in m
    print("split file contiguous in merge:", ok_s)
    print("HF file contiguous in merge:", ok_h)
    if not ok_s:
        print("  split len", len(s), "try normalized trailing nl")
        ok_s = s.rstrip("\n") in m
        print("  after rstrip:", ok_s)
    if not ok_h:
        ok_h = h.rstrip("\n") in m
        print("  HF after rstrip:", ok_h)

    if ok_f and ok_c and ok_s and ok_h:
        print("\nALL CHECKS PASSED: no content omitted from the four sources.")


if __name__ == "__main__":
    main()
