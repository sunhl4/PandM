# 化学与材料 × QML：精读模板与 2025 优先阅读清单

> 合并自原根目录 `chemistry_materials_deep_reading_template.md` 与 `chemistry_materials_2025_priority_reading.txt`（2026-03-30）。

---

## 第一部分：使用方法（SciSpace / 对话式精读）

1. 打开论文 PDF（或 SciSpace / 类似工具）。  
2. 依次复制下面每个问题，粘贴到对话框。  
3. 把回答整理到 Zotero 笔记、`docs/literature_notes/`，或 Obsidian/Notion。  
4. **必做**：下面 **Q1–Q5**；其余按时间选做。

---

## 第二阶段：快速定位（约 5 分钟）

### Q1：三句话总结

```
请用中文回答。用「问题—方法—结论」三句话概括这篇论文，并指出：
1. 最关键的实验/证据在哪一节（section number）？
2. 是否有开源代码/数据？链接是什么？
3. 论文类型：理论/算法/实验验证/综述/应用？
```

### Q2：核心贡献定位

```
请用中文回答。这篇论文的核心贡献是什么？请列出 3 条，每条附上原文证据（英文原句 1–2 句，并标注页码/图表编号）。
```

---

## 第三阶段：方法与可复现性（约 15 分钟）

### Q3：假设与近似

```
列出本文对哈密顿量、基组、映射、噪声模型或数据集的关键假设。哪一条若放松会显著改变结论？
```

### Q4：与基线对比

```
作者对比了哪些基线？公平性如何（相同数据划分、相同计算预算）？若你复现，会补做哪一组对照实验？
```

### Q5：复现清单

```
要复现主结果，最少需要哪些信息（方程编号、超参、随机种子、硬件/模拟器）？文中缺什么会导致无法复现？
```

---

## 第四阶段：化学/材料语义（可选）

### Q6：体系与标度

```
研究的化学体系是什么（分子/表面/固体）？体系大小与当前量子硬件是否现实？作者如何讨论标度？
```

### Q7：误差与不确定度

```
能量/力/性质的误差如何报告？是否有与实验或高精度参考（如 CCSD(T)、DFT）的对照？
```

---

## 第五阶段：批判与迁移（可选）

### Q8：主要局限

```
作者承认的局限是什么？你认为未写明的局限还有什么？
```

### Q9：与你项目的连接

```
若将本文方法用于你当前主线（例如吸附、力场、表面催化），需要改哪几处假设或接口？
```

### Q10：相关工作

```
与最相关的 2–3 篇工作相比，本文的增量是什么？用一句话区分。
```

### Q11：下一步实验

```
若你有一周机时，会做的最小验证实验是什么（输入/输出/成功判据）？
```

---

## 附录：2025 年化学与材料科学应用 QML 论文优先阅读清单

共 32 篇，按主题分组（arXiv ID 用于 Zotero「Add by Identifier」）。

### 1.1 分子能量与量子化学（10 篇）

- 2501.04264 — Li et al. — Hybrid Quantum-Neural Wavefunction  
- 2505.04768 — Fonseca et al. — VQE Applied to Chemistry  
- 2507.08183 — Jones et al. — PQC Learning for Quantum Chemical Applications  
- 2501.14968 — Patel et al. — Quantum Measurement for Quantum Chemistry  
- 2507.20422 — Boy et al. — Encoding molecular structures  
- 2504.07077 — Patil et al. — ML Quantum Error Mitigation for Molecular Energetics  
- 2506.03995 — Carreras et al. — Limitations of Quantum Hardware for Molecular Energy  
- 2511.03726 — Bincoletto et al. — Transferable ML for Quantum Circuit Parameters  
- 2506.04223 — Ralli et al. — Bridging Quantum Chemistry and MaxCut  
- 2509.07460 — Grazioli et al. — Critical point search for excitation energies  

### 1.2 分子结构与几何优化（4 篇）

- 2505.01875 — Hao et al. — Large-scale Molecule Geometry Optimization  
- 2503.21686 — Pan et al. — MolQAE Quantum Autoencoder  
- 2508.02369 — Kamata et al. — Molecular Quantum Transformer  
- 2506.14920 — Linn et al. — Designing lattice proteins with VQA  

### 1.3 药物发现与生物分子（5 篇）

- 2502.18639 — Giraldo et al. — Q2SAR Quantum Multiple Kernel Learning  
- 2503.04239 — Berti et al. — QML in Precision Medicine and Drug Discovery  
- 2508.03446 — Smith & Guven — Bridging Quantum and Classical in Drug Design  
- 2503.09517 — Papalitsas et al. — QAOA for Molecular Docking  
- 2507.08155 — Souza Teixeira et al. — QNN for Protein Binding Affinity  

### 1.4 材料科学（4 篇）

- 2505.19909 — Graña et al. — Materials Discovery with Quantum-Enhanced ML  
- 2507.19276 — Wang et al. — Quantum and Hybrid ML Models for Materials-Science  
- 2505.18519 — Hänseroth & Dreßler — ML Potentials for Hydroxide Transport  
- 2510.14099 — Tandon et al. — Quantum Reservoir Computing for Corrosion Prediction  

### 1.5 量子蒙特卡洛与波函数（5 篇）

- 2511.18552 — Fu et al. — Local Pseudopotential for NN-QMC  
- 2503.02202 — Wu et al. — Hybrid tensor network and NN quantum states  
- 2508.05169 — Xiao & Xiang — Advanced trial wave functions in fermion QMC  
- 2503.15473 — Liu et al. — VQMC for superconducting pairing in La3Ni2O7  
- 2508.10593 — Abdelrahman et al. — Probabilistic Approximate Optimization VMC  

### 1.6 流体力学与物理模拟（4 篇）

- 2508.13651 — Amaral et al. — QML and quantum-inspired methods for CFD  
- 2505.10842 — Li et al. — QML for reduced order modelling of turbulent flows  
- 2503.22590 — Leong et al. — Hybrid Quantum Physics-informed Neural Network  
- 2503.08149 — Lautaro Hickmann et al. — Hybrid quantum tensor networks for aeroelastic  

### 使用建议

1. 在 Zotero 中按主题建子 collection（如「1.1 分子能量与量子化学」）。  
2. 复制 arXiv ID → Zotero「Add by Identifier」。  
3. 精读时每篇使用上文 **Q1–Q5**（必做），其余按时间选做。  
