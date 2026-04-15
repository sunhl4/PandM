# 量子化学计算理论文档

本目录包含量子化学计算的数学和理论基础，与 `quantum_chemistry/` 中的代码实现相对应。

> **单一理论主稿（阅读入口）**：[`../../docs/theory_and_derivations.md`](../../docs/theory_and_derivations.md)（`python3 tools/build_theory_master.py` 从本目录 01–05 等源合并）。  
> **旧分卷索引**：[`../../docs/unified_chemistry_theory/README.md`](../../docs/unified_chemistry_theory/README.md)。

## 文档结构

### 第一章：二次量子化理论
**文件**：`01_second_quantization_theory.md`

**内容**：
- 1.1 从一次量子化到二次量子化
- 1.2 产生和湮灭算符
- 1.3 反对易关系及其证明
- 1.4 数算符
- 1.5 单粒子算符的二次量子化
- 1.6 双粒子算符的二次量子化
- 1.7 分子电子哈密顿量
- 1.8 激发算符
- 1.9 Wick定理
- 1.10 小结

**核心公式**：
$$\{a_p, a_q^\dagger\} = \delta_{pq}$$
$$\hat{H} = E_{nuc} + \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} g_{pqrs} a_p^\dagger a_q^\dagger a_s a_r$$

---

### 第二章：费米子-量子比特映射理论
**文件**：`02_fermion_qubit_mapping_theory.md`

**内容**：
- 2.1 为什么需要映射？
- 2.2 Jordan-Wigner变换
- 2.3 常用算符的JW变换
- 2.4 JW变换的复杂度分析
- 2.5 Bravyi-Kitaev变换
- 2.6 奇偶映射（Parity Mapping）
- 2.7 映射的数学结构
- 2.8 哈密顿量映射实例
- 2.9 量子电路资源分析
- 2.10 小结

**核心公式**：
$$a_p^\dagger \xrightarrow{JW} \frac{1}{2}(X_p - iY_p) \prod_{j<p} Z_j$$

---

### 第三章：变分量子本征求解器（VQE）理论
**文件**：`03_vqe_theory.md`

**内容**：
- 3.1 变分原理
- 3.2 VQE算法框架
- 3.3 能量测量
- 3.4 参数化量子电路（Ansatz）
- 3.5 梯度计算
- 3.6 优化器
- 3.7 贫瘠高原（Barren Plateaus）
- 3.8 VQE的误差分析
- 3.9 VQE与其他方法的比较
- 3.10 小结

**核心公式**：
$$E_0 \leq E(\boldsymbol{\theta}) = \langle \psi(\boldsymbol{\theta}) | \hat{H} | \psi(\boldsymbol{\theta}) \rangle$$
$$\frac{\partial E}{\partial \theta} = \frac{1}{2}\left[ E(\theta + \frac{\pi}{2}) - E(\theta - \frac{\pi}{2}) \right]$$

---

### 第四章：Ansatz设计理论
**文件**：`04_ansatz_theory.md`

**内容**：
- 4.1 Ansatz的数学框架
- 4.2 UCCSD Ansatz
- 4.3 硬件高效Ansatz
- 4.4 ADAPT-VQE
- 4.5 对称性保持Ansatz
- 4.6 Ansatz的可训练性
- 4.7 特定问题的Ansatz设计
- 4.8 Ansatz复杂度的理论边界
- 4.9 小结

**核心公式**：
$$|\Psi_{UCCSD}\rangle = e^{\hat{T} - \hat{T}^\dagger}|\Phi_{HF}\rangle$$

---

### 第五章：激发态方法理论
**文件**：`05_excited_states_theory.md`

**内容**：
- 5.1 为什么需要激发态？
- 5.2 VQD（变分量子偏折）
- 5.3 SSVQE（子空间搜索VQE）
- 5.4 qEOM（量子运动方程）
- 5.5 折叠光谱方法
- 5.6 量子朗之万方法
- 5.7 振动和旋转激发
- 5.8 跃迁性质
- 5.9 收敛和精度分析
- 5.10 方法比较与选择
- 5.11 小结

**核心公式**：
$$L_k = \langle \hat{H} \rangle + \sum_{j<k} \beta_j |\langle \psi | \psi_j \rangle|^2$$

---

## 学习路线建议

```
第一章（二次量子化）
        │
        ↓
第二章（费米子映射）  ←── 必须先理解第一章
        │
        ↓
第三章（VQE理论）    ←── 核心算法
        │
        ├──→ 第四章（Ansatz设计）
        │
        └──→ 第五章（激发态）
```

## 配合代码学习

每章理论都有对应的代码实现：

| 理论章节 | 代码文件 |
|---------|---------|
| 第一章 | `core/fermion_operators.py` |
| 第二章 | `core/qubit_mapping.py` |
| 第三章 | `vqe/solver.py`, `vqe/measurement.py` |
| 第四章 | `ansatz/uccsd.py`, `ansatz/hardware_efficient.py`, `ansatz/adaptive.py` |
| 第五章 | （待扩展） |

## 推荐参考文献

### 教科书
1. **Szabo & Ostlund**: *Modern Quantum Chemistry* - 第一章的参考
2. **Nielsen & Chuang**: *Quantum Computation and Quantum Information* - 量子计算基础

### 综述论文
1. **McArdle et al. (2020)**: "Quantum computational chemistry" - Reviews of Modern Physics
2. **Cao et al. (2019)**: "Quantum Chemistry in the Age of Quantum Computing" - Chemical Reviews

### 关键原始论文
1. **Jordan & Wigner (1928)**: 费米子-自旋映射的原始论文
2. **Peruzzo et al. (2014)**: VQE的原始论文
3. **McClean et al. (2018)**: 贫瘠高原的理论分析
4. **Grimsley et al. (2019)**: ADAPT-VQE

## 符号约定

| 符号 | 含义 |
|------|------|
| $a_p^\dagger, a_p$ | 产生/湮灭算符 |
| $n_p$ | 数算符 |
| $\chi_p$ | 自旋轨道 |
| $h_{pq}$ | 单电子积分 |
| $g_{pqrs}$ | 双电子积分 |
| $X, Y, Z$ | Pauli矩阵 |
| $\theta$ | 变分参数 |
| $E_0$ | 基态能量 |

## 数学预备知识

1. **线性代数**：矩阵运算、本征值问题、张量积
2. **量子力学**：波函数、算符、测量
3. **量子计算**：量子比特、量子门、量子电路
4. **优化理论**：梯度下降、凸优化基础

---

*本文档持续更新中。欢迎反馈和建议。*
