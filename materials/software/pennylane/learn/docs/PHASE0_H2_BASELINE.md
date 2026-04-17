# Phase 0：H₂ VQE 基线与图表约定

本文件将 **理论文档—代码—参考能量—导出图** 对齐，便于对外展示与复现。

## 固定约定

| 项目 | 值 |
|------|-----|
| 分子 | H₂ |
| 平衡键长（单点演示） | **0.74 Å**（与 `01_complete_h2_vqe.py` 默认一致） |
| 基组 | **STO-3G** |
| 活性空间 | 2 电子 / 2 空间轨道 → **4 自旋轨道 → 4 量子比特** |
| 映射 | Jordan–Wigner（教程内实现） |
| Ansatz | HF 初态 + **双激发** `DoubleExcitation`（单参数） |
| 优化器 | `scipy.optimize.minimize(..., method='COBYLA', maxiter=200)` |
| 参考基态 | 对 qubit Hamiltonian **精确对角化** 的最低本征值；与 PySCF **FCI** 应在数值误差内一致 |

## 参考能量来源

- **有 PySCF**：`get_h2_integrals(R)` 用 PySCF RHF + FCI 给出 `hf_energy`、`fci_energy`。
- **无 PySCF**：使用教程内 **0.74 Å 预存积分** 的 `hf_energy`、`fci_energy`（见 [`01_complete_h2_vqe.py`](../quantum_chemistry/tutorials/01_complete_h2_vqe.py) 中 `from_pyscf: False` 分支）。

## 图表与脚本（同一套超参数）

| 文件 | 内容 |
|------|------|
| [`docs/figures/h2_vqe_convergence.png`](figures/h2_vqe_convergence.png) | 单键长 0.74 Å 下 VQE 能量迭代历史 vs 精确基态 / HF |
| [`docs/figures/h2_pes.png`](figures/h2_pes.png) | 势能面：HF / VQE / FCI；**需 PySCF** 才能扫描多键长；否则为单点 |

**一键重绘（非交互）**：

```bash
cd /Users/shl/nvidia/QML
python -m quantum_chemistry.tutorials.plot_h2_baseline_figures
```

生成前会在 Python 内设置 **`numpy` 随机种子**（若未来 ansatz/优化器引入随机性，仍保证可复现性为团队约定）。

## 理论阅读顺序（仓库内）

1. [`quantum_chemistry/docs/01_second_quantization_theory.md`](../quantum_chemistry/docs/01_second_quantization_theory.md)
2. [`quantum_chemistry/docs/02_fermion_qubit_mapping_theory.md`](../quantum_chemistry/docs/02_fermion_qubit_mapping_theory.md)
3. [`quantum_chemistry/docs/03_vqe_theory.md`](../quantum_chemistry/docs/03_vqe_theory.md)
4. 代码：[`quantum_chemistry/tutorials/01_complete_h2_vqe_explanation.md`](../quantum_chemistry/tutorials/01_complete_h2_vqe_explanation.md) + [`01_complete_h2_vqe.py`](../quantum_chemistry/tutorials/01_complete_h2_vqe.py)

## 精读文献模板

使用 [`literature/chemistry_materials_reading.md`](literature/chemistry_materials_reading.md) 中精读问题；结论须能指回 **章节/图号/原句**。
