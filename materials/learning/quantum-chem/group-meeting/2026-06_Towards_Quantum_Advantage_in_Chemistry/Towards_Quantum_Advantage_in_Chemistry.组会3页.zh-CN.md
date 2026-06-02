---
marp: true
theme: default
paginate: true
math: mathjax
title: 迈向化学中的量子优势 — 组会 3 页
description: Genin et al. arXiv:2512.13657v2 · iQCC / iQCC+PT
---

<!-- _class: lead -->

# 迈向化学中的量子优势

**Towards Quantum Advantage in Chemistry**  
Genin et al., arXiv:2512.13657v2 · OTI Lumionics + Samsung SAIT

经典 CPU 上运行 **Quantum Solver**（C++/OpenMPI）· 面向容错线路的 **iQCC + iQCC+PT**

---

## 1. 主要结论与经典方法对比

### 论文在回答什么

- 在 **商业相关尺度**（过渡金属磷光体、大活性空间）上，**量子原生算法** 能否在 **精度** 上优于常规 DFT/CC 流水线，并 **标定** 所需逻辑 qubit / 门深与 **经典仍可模拟的阈值**。
- **尚未** 在真实量子机上展示墙钟优势；标题 *Towards* =「迈向」。

### 核心结论（OLED 基准）

| 要点 | 内容 |
|------|------|
| 体系 | 14 个 Ir(III)/Pt(II) 磷光配合物；**$T_1\to S_0$** 能隙 vs **77 K PL** |
| 最优方法 | **iQCC+PT**：MAE **0.050 eV**，$R^2$ **0.941** |
| 规模标定 | **CAS(100,100) → 200 逻辑 qubit**；约 **$10^7$ CNOT**；**FCI 不可行**，经典 iQCC 仍可跑 |
| 阈值 | **≤ ~200 逻辑 qubit** 仍可用经典求解器完成 → 量子优势可能出现的 **尺度下界** |
| 对 Lee 论断 | **激发态 + 大 CAS + 量子原生 ansatz** 是对「弱相关基态难有量子优势」的 **限定性反例** |

### 相对实验：$T_1\to S_0$ 能隙（表 2，节选）

| 方法 | $R^2$ | MAE [eV] | 相对实验趋势 |
|------|:-----:|:--------:|--------------|
| RHF/ROHF | 0.74 | 0.38 | — |
| RO-CAM-B3LYP | 0.61 | 0.12 | — |
| RO-ωB97X | 0.85 | 0.22 | — |
| TD-CAM-B3LYP | 0.81 | 0.26 | — |
| CCSD | 0.81 | 0.22 | **红移**（能隙系统性偏小） |
| CR-CC(2,3) | 0.78 | **0.29** | **红移**，不宜作量子金标准 |
| iQCC（仅变分） | 0.88 | 0.12 | **蓝移**（能隙系统性偏大） |
| **iQCC+PT** | **0.94** | **0.05** | **略蓝移**，**全面最优** |

**化学要点**：iQCC 捕捉 **MLCT/LC 混合**；**PT 将 MAE 约减半**（0.12 → 0.05 eV）。建模省略 **SOC**；与 CC **同 GAMESS 积分**。

---

## 2. 实现配置：硬件、并行、线路与初始参数

### 计算平台与并行

| 项目 | 配置 |
|------|------|
| 软件 | 自研 **C++/OpenMPI** Quantum Solver（确定性 $\langle H\rangle$，无 shot） |
| 积分 | 修改版 **GAMESS** → JW → qubit $\hat H=\sum_k c_k \hat P_k$ |
| CPU | **2 × AMD EPYC 7702**（每台 **64 物理核**，共 **128 核**） |
| 并行 | **64 MPI 进程**（64→128 进程几乎无加速） |
| 200 qubit 墙钟 | Q1 CAS(100,100) 末步协议 **≈ 199.4 h**（表 SI.2-2） |
| 内存峰值 | **≈ 756 GB**（200 qubit；Hamiltonian **≈ 3.75×10⁹** Pauli 项） |

### 200 逻辑 qubit 线路规模（表 1，Q1）

| 量子比特 | 纠缠子数 | CNOT [×10⁶] | 变分 iQCC 能量 [$E_\mathrm{h}$] | CISD |
|:--------:|:--------:|:-----------:|:-------------------------------:|:----:|
| 200 | **1,500,000** | **10.2** | −1938.16627 | −1938.16010 |

*CNOT 计数假设 **全连接**、无 SWAP；真机需 SWAP 时门数会显著增加。*

### 几何 / 活性空间 / 参考态（初始条件）

| 步骤 | 参数 |
|------|------|
| 几何优化 | Gaussian 或 TeraChem；**B3LYP**；金属 **LANL2DZ-ECP**，配体 **6-31G\*\*** |
| CAS | ROHF 轨道；费米面下 $n_e$ 占据 + 上 $n_o$ 虚轨道（**$n_e=n_o$**） |
| Q1 最大 | **CAS(100,100) → 200 qubit**；原子数 $>87$ 的体系统一 CAS(100,100) |
| 参考态 $\|0\rangle$ | **HF/ROHF** 行列式；$S_0$ 用 RHF，$T_1$ 用 ROHF + **$S^2$ 惩罚** |
| 观测量 | $\Delta E_{T_1\to S_0}=E(S_0)-E(T_1)$；实验 $E_\mathrm{exp}=1240/\lambda$（nm→eV） |

### iQCC 分阶段超参数（OLED，§4.5）

| 阶段 | 穿衣步数 | 每步纠缠子 | Ansatz 阶数 | 停止条件 |
|:----:|:--------:|:----------:|:-----------:|----------|
| 1 | 8 | 28 | **6 阶** | 快速降能 |
| 2 | 6 | 46 | **4 阶** | |
| 3 | 循环 | 优化 **300**，仅穿衣 **46**/步 | **2 阶** | 全部 $\|\tau\|<0.012$ |
| 4 | 1（末步） | **30万–150万** | **1 阶** | Hamiltonian 项数触顶 |

**哈密顿量项数硬上限**：CAS(100,100) 为 **$3.5\times 10^9$**。

---

## 3. iQCC+PT 数学原理 · 与 UCCSD 区别 · 穿衣 / DIS / 纠缠子

### 共同框架（VQE 型）

$$
|\Psi(\tau)\rangle = \hat U(\tau)|0\rangle, \quad E = \langle 0|\hat U^\dagger \hat H \hat U|0\rangle
$$

**JW 映射**：$\hat H=\sum_k c_k \hat P_k$（Pauli 串，实系数）。

### QCC 变分 + 迭代穿衣（iQCC）

$$
\hat U(\tau)=\prod_k \exp\!\left(-\frac{\mathrm{i}}{2}\tau_k \hat T_k\right), \quad
g_\alpha = \mathrm{Im}\,\langle 0|\hat H \hat T_\alpha|0\rangle
$$

**DIS**：所有 $g_\alpha\neq 0$ 的 $\hat T_\alpha$；每步取 $|g_\alpha|$ 最大的 **$M$** 个变分。

**穿衣**：

$$
\hat H^{(n+1)} = \hat U^\dagger(\tau^{(n)}_\mathrm{opt})\,\hat H^{(n)}\,\hat U(\tau^{(n)}_\mathrm{opt})
$$

**纠缠子**：**1 个 $Y$（最低索引）+ 奇数个 $X$**，无 $Z$。

### iQCC+PT（EN2）

$$
\Delta E_\mathrm{EN2} = -\sum_k \frac{g_k^2}{D_k}, \quad D_k = E_0 - E_k, \quad E_\mathrm{EN2} = E_0 + \Delta E_\mathrm{EN2}
$$

变分只优化 **top-$M$**；PT 用 **完整 DIS** 补余项。

### 与 UCCSD 的主要区别

| 维度 | **iQCC / QCC** | **UCCSD** |
|------|----------------|-----------|
| 生成元 | **DIS**（Pauli 纠缠子） | 固定 $T_1,T_2$ |
| 扩展 | **穿衣** 更新 $\hat H^{(n)}$ | 固定池 / ADAPT 加长 $U$ |
| 后处理 | **EN2（iQCC+PT）** | CCSD(T) |
| 磷光 MAE | **0.05 eV** | CCSD 0.22 eV |
