# -*- coding: utf-8 -*-
"""Assemble four markdown sources into one themed document (no content removal)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
paths = {
    "foundations": ROOT / "quantum_chemistry_foundations.md",
    "classic": ROOT / "经典量子化学方法详解.md",
    "split": ROOT / "分裂价极化与原子轨道杂化.md",
    "hf": ROOT / "Hartree-Fock理论推导与算法.md",
}


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
    f = paths["foundations"].read_text(encoding="utf-8")
    c = paths["classic"].read_text(encoding="utf-8")
    s = paths["split"].read_text(encoding="utf-8")
    h = paths["hf"].read_text(encoding="utf-8")

    # foundations: strip duplicate opening paragraph (line 24-25 duplicate of 22)
    # Keep from title through end of section 2
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

    # classic: by anchors
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
    # §10 与 §11 之间的篇标题与分隔线（原 `# Part IV…` 块，不可省略）
    c_part4 = lines_between(c, "# Part IV", "## §11 ")
    c_s11 = lines_between(c, "## §11 方法层次与决策", "## §12 ")
    c_s12 = lines_between(c, "## §12 现代发展趋势（2020–2026）", "## 参考文献")
    c_refs = lines_between(c, "## 参考文献", None)

    head = f.splitlines(keepends=True)
    # 保留 foundations 文首一级标题 + 学习目标 + 目录 + 文首说明段（至第一个 §1 前）
    goals = "".join(head[0:25])

    out: list[str] = []
    out.append("# 量子化学理论基础与方法（合并本）\n\n")
    out.append("2026-04 合并整理 · 按主题归类 **相同/相近内容**，四份原文 **全文保留、未做删节或缩写**。\n\n")
    out.append("---\n\n")
    out.append("## 合并来源与阅读说明\n\n")
    out.append("| 来源文件 | 角色 |\n")
    out.append("|:---|:---|\n")
    out.append("| `quantum_chemistry_foundations.md` | 数学基础、变分、HF/相关/后HF/DFT/基组数学、误差与总结 |\n")
    out.append("| `经典量子化学方法详解.md` | 方法版图、直觉图、CI/MP/CC/MR/DFT 实践与趋势、参考文献 |\n")
    out.append("| `分裂价极化与原子轨道杂化.md` | 分裂价、极化、$s,p,d,f$ 与杂化图示与脚本说明 |\n")
    out.append("| `Hartree-Fock理论推导与算法.md` | HF 从变分到 Roothaan–SCF 的逐步推导与算法 |\n\n")
    out.append(
        "各块之间用水平线分隔；小节标题下 **保留原文件名标注** 的子标题，便于回溯。\n\n"
    )
    out.append("---\n\n")

    # Part A: math + TDSE + 经典 §1 (same theme: Hamiltonian, BO, FCI wall, correlation intro)
    out.append("# A. 量子力学基础、电子结构问题与维数灾难\n\n")
    out.append(goals)
    out.append("\n")
    out.append("### A.1 `quantum_chemistry_foundations.md` — §1 量子力学数学基础\n\n")
    out.append(sec1)
    out.append("\n")
    out.append("### A.2 `quantum_chemistry_foundations.md` — §2 时间无关薛定谔方程的数学结构\n\n")
    out.append(sec2)
    out.append("\n")
    out.append("### A.3 `经典量子化学方法详解.md` — §1 电子结构问题（与上节同一主题链：BO、FCI、相关能分类）\n\n")
    out.append(c_intro)
    out.append(c_s1)

    out.append("\n---\n\n# B. 变分原理\n\n")
    out.append("### B.1 `quantum_chemistry_foundations.md` — §3\n\n")
    out.append(sec3)

    out.append("\n---\n\n# C. 基组：数学原理、层级与分裂价 / 极化 / 杂化\n\n")
    out.append("### C.1 `quantum_chemistry_foundations.md` — §8 基组展开的数学原理\n\n")
    out.append(sec8)
    out.append("\n")
    out.append("### C.2 `经典量子化学方法详解.md` — §2 基组（Basis Sets）\n\n")
    out.append(c_s2)
    out.append("\n")
    out.append("### C.3 `分裂价极化与原子轨道杂化.md` — 全文\n\n")
    out.append(s)

    out.append("\n---\n\n# D. Hartree–Fock：数学推导、方法总览与逐步算法专文\n\n")
    out.append("### D.1 `quantum_chemistry_foundations.md` — §4 Hartree-Fock理论的数学推导\n\n")
    out.append(sec4)
    out.append("\n")
    out.append("### D.2 `经典量子化学方法详解.md` — §3 Hartree–Fock（HF）方法\n\n")
    out.append(c_s3)
    out.append("\n")
    out.append("### D.3 `Hartree-Fock理论推导与算法.md` — 全文\n\n")
    out.append(h)

    out.append("\n---\n\n# E. 电子相关性的数学描述（与 §1.3 呼应）\n\n")
    out.append("### E.1 `quantum_chemistry_foundations.md` — §5\n\n")
    out.append(sec5)

    out.append("\n---\n\n# F. 后 Hartree–Fock：数学框架 + 方法详解（CI / MP / CC / MR）\n\n")
    out.append("### F.1 `quantum_chemistry_foundations.md` — §6\n\n")
    out.append(sec6)
    out.append("\n")
    out.append("### F.2 `经典量子化学方法详解.md` — §4–§7\n\n")
    out.append(c_s4)
    out.append(c_s5)
    out.append(c_s6)
    out.append(c_s7)

    out.append("\n---\n\n# G. 密度泛函理论：数学 + 泛函阶梯与实践\n\n")
    out.append("### G.1 `quantum_chemistry_foundations.md` — §7\n\n")
    out.append(sec7)
    out.append("\n")
    out.append("### G.2 `经典量子化学方法详解.md` — §8–§10（DFT 基础、Jacob 阶梯、实践与 TDDFT）\n\n")
    out.append(c_s8)
    out.append(c_s9)
    out.append(c_s10)
    out.append("\n### G.3 `经典量子化学方法详解.md` — `# Part IV` 篇间标题（§10 与 §11 之间）\n\n")
    out.append(c_part4)

    out.append("\n---\n\n# H. 方法层次、现代趋势、误差分析与总结\n\n")
    out.append("### H.1 `经典量子化学方法详解.md` — §11–§12 与参考文献\n\n")
    out.append(c_s11)
    out.append(c_s12)
    out.append(c_refs)
    out.append("\n")
    out.append("### H.2 `quantum_chemistry_foundations.md` — §9–§12\n\n")
    out.append(sec9)
    out.append(sec10)
    out.append(sec11)
    out.append(sec12)

    out_path = ROOT / "量子化学理论基础与方法_合并本.md"
    out_path.write_text("".join(out), encoding="utf-8")
    print("Wrote", out_path, "lines", len("".join(out).splitlines()))


if __name__ == "__main__":
    main()
