# 专题问答：k-UpCCGSD 与 OO-VQE（原理、公式与实现）

**类型**：方法专题（非单篇 PDF 配套）  
**总索引**：[学习问答记录.md](./学习问答记录.md)（**条目 27** 与本文件同步索引）

## 使用说明

- 公式：行内 `$...$`；独立公式 `$$` 独占行，**块末 `$$` 后空一行**（见 `学习问答记录.md`）。

---

## 问答日志

### 条目 1

**文献：** J. Lee *et al.*, *J. Chem. Theory Comput.* **15**, 311 (2019)；arXiv:1810.02327（k-UpCCGSD）。W. Mizukami *et al.*, *Phys. Rev. Research* **2**, 033421 (2020)；arXiv:1910.11526（OO-UCC / OO-VQE）。

**标签：** `#k-UpCCGSD` `#UCC` `#OO-VQE` `#VQE` `#轨道优化`

**日期：** 2026-04-07

**问题：**

请详细解释 **k-UpCCGSD** 与 **OO-VQE** 的理论原理、主要数学公式，以及常见实现路径。

**回答：**

---

## A. k-UpCCGSD

### A.1 放在什么语境里？

**酉耦合簇（UCC）** 用幺正算符作用在参考态 $\lvert\Phi_0\rangle$（常为 Hartree–Fock）上构造试探波函数：

$$
\lvert\psi\rangle = \hat{U}(\boldsymbol{\theta})\lvert\Phi_0\rangle,\quad \hat{U}^\dagger\hat{U}=\mathbb{I}.
$$

典型写法是 **反对易的激发簇算符** $\hat{T}$（**非**幺正）进入 $\exp(\hat{T}-\hat{T}^\dagger)$，以保证波函数归一。若 $\hat{T}$ 只含 **相对于 $\lvert\Phi_0\rangle$ 的粒子–空穴单、双激发**，即得到常见的 **UCCSD**；若 $\hat{T}$ 在 **全自旋轨道指标** 上取更一般的单、双激发（不限于固定参考的占据–空轨道划分），则与 **广义** UCC（**UCCGSD**）相关。

**k-UpCCGSD**（k-unitary pair Coupled-Cluster **Generalized** Singles and Doubles）是 Lee 等提出的一种 **稀疏、可叠层加深** 的 UCC 族：**每一层**同时包含 **广义单激发** 与 **配对双激发（paired doubles，pCCD 型）**，共 **$k$ 层** 幺正因子相乘；在精度–线路深度之间折中，且 **线路深度标度** 优于完整 UCCGSD 与 UCCSD（见下文 A.4）。

### A.2 数学结构（费米子层）

**（1）广义单激发（每层一套独立参数）**

对自旋轨道指标 $p,q$，引入反对易的单体生成元（参数 $\theta^{(r)}_{pq}$，第 $r$ 层）：

$$
\hat{\kappa}^{(r)}_{\mathrm{S}} = \sum_{pq} \theta^{(r)}_{pq}\Bigl(\hat{a}^\dagger_p \hat{a}_q - \hat{a}^\dagger_q \hat{a}_p\Bigr).
$$

它生成 **轨道混合** 型的单粒子变换；**「广义」** 体现在 **不对 $(p,q)$ 施加「必须相对 HF 为粒子–空穴」的截断**（与 UCCSD 的激发池不同）。

**（2）配对双激发（pCCD 型）**

在空间轨道 $i,j$（或自旋轨道的一种配对约定）上，**同时** 提升 $\alpha$ 与 $\beta$ 电子的算符可写为（具体符号与指标顺序因文献/程序略异，以下是一种常见形式）：

$$
\hat{Q}_{ij} = \hat{a}^\dagger_{i\uparrow}\hat{a}^\dagger_{i\downarrow}\hat{a}_{j\downarrow}\hat{a}_{j\uparrow}.
$$

直观上，$\hat{Q}_{ij}$ 把 **空间轨道 $j$ 上的电子对** 迁到 **空间轨道 $i$**。配对双激发簇（第 $r$ 层）取这类算符的 **线性组合**（仅保留配对通道，而非全部 $\alpha\beta$ 双激发），参数记为 $\eta^{(r)}_{ij}$：

$$
\hat{T}^{(r)}_{\mathrm{pd}} = \sum_{(i,j)\in\mathcal{P}} \eta^{(r)}_{ij}\,\hat{Q}_{ij},
$$

其中 $\mathcal{P}$ 为论文/代码选定的 **稀疏配对集合**（k-UpCCGSD 的核心是 **用 pCCD 型双激发替代「全部广义双激发」**）。

**（3）单层幺正与 k 层 ansatz**

将单、双部分并入同一 **激发簇** $\hat{T}^{(r)}=\hat{T}^{(r)}_{\mathrm{S}}+\hat{T}^{(r)}_{\mathrm{pd}}$（文献中对「单、双是否分多个指数因子」有不同因式分解，**实现上以库定义为准**），定义：

$$
\hat{U}^{(r)}(\boldsymbol{\theta}^{(r)}) = \exp\Bigl(\hat{T}^{(r)} - \hat{T}^{(r)\dagger}\Bigr).
$$

**k-UpCCGSD** 波函数：

$$
\lvert\psi(\boldsymbol{\theta})\rangle = \hat{U}^{(k)}(\boldsymbol{\theta}^{(k)})\cdots \hat{U}^{(1)}(\boldsymbol{\theta}^{(1)})\,\lvert\Phi_0\rangle.
$$

**VQE** 中在量子设备上测量 $\langle\hat{H}\rangle$，经典优化器更新各层 $\boldsymbol{\theta}^{(r)}$。

### A.3 与 UCCSD / UCCGSD 的对比（概念）

| 对象 | 激发池特点 | 参数 / 线路（量级直觉） |
|------|------------|-------------------------|
| UCCSD | 相对 HF 的粒子–空穴 SD | 随空轨、占据数增长，常见讨论为 $\mathcal{O}((N-\eta)^2\eta)$ 量级项 |
| UCCGSD | **全** 广义 SD | 双激发项数 $\mathcal{O}(N^4)$，映射后线路更深 |
| k-UpCCGSD | 广义 **S** + **配对** D，重复 $k$ 次 | Lee 等给出线路深度 **$\mathcal{O}(kN)$** 量级的有利标度（$N$：自旋轨道数） |

**$k$ 的作用**：增大 $k$ 相当于 **加深同一算符族的可变参数容量**，在多个小分子基准上体现为 **系统可改进的精度–成本折中**（仍以具体体系验证为准）。

### A.4 实现要点（工程）

- **Qiskit Nature**：`second_q.circuit.library.UCC` 中 `excitations="gsd"`，`reps=k` 对应 **重复 GSD 堆叠**，与本专题数学叙述一致；再经 **Jordan–Wigner / Parity** 等映射为泡利串线路。本仓库 [`quantum_chem_bench`](../../../software/qc-x-chem/quantum_chem_bench/quantum_solvers/vqe_solver.py) 中 `vqe_kupccgsd` 即采用该构造（参见类 `VQEkUpCCGSDSolver`）。
- **PennyLane**：提供 [`qml.kUpCCGSD`](https://docs.pennylane.ai/en/stable/code/api/pennylane.kUpCCGSD.html) 模板，可直接与分子哈密顿量对接。
- **与 dft_qc_pipeline 的别名**：[`dft_qc_pipeline`](../../../software/qc-x-chem/dft_qc_pipeline/) 文档曾指出，部分路径下 `kupccgsd` 仅为 **重复 UCCSD 的近似**，与 Lee *et al.* 的 **GSD 池** 不完全等价；**论文对标** 时应以 **真实 `excitations='gsd'`** 的实现为准。

---

## B. OO-VQE（轨道优化 VQE）

### B.1 要解决什么问题？

普通 **UCC–VQE** 固定一组 **分子轨道**（常来自一次 HF/DFT），只优化 **簇振幅 / 线路参数** $\boldsymbol{\theta}$。若轨道 **不适合** 强关联或多参考特征，**在固定活性空间内** 往往需要 **更深电路或更大激发池** 才能补偿。

**OO-VQE**（常对应文献中的 **OO-UCC**：orbital-optimized UCC）把 **轨道参数** 一并纳入变分：**同时** 优化 $\boldsymbol{\theta}$ 与描述轨道旋转的参数 $\boldsymbol{\kappa}$，与经典 **MCSCF / CASSCF 中的轨道优化** 思想同构，被称为「量子版」轨道优化流程。

### B.2 数学表述

**（1）轨道酉变换**

用反对易算符 $\hat{\kappa}$ 生成 **单粒子酉** $\hat{R}(\boldsymbol{\kappa})=\exp(\hat{\kappa})$：

$$
\hat{\kappa} = \sum_{p>q} \kappa_{pq}\Bigl(\hat{a}^\dagger_p \hat{a}_q - \hat{a}^\dagger_q \hat{a}_p\Bigr).
$$

在二次量子化下，$\hat{\kappa}$ 对应 **分子轨道系数矩阵** 的反对称旋转（Lie 代数层面）；**积分** $h_{pq}(\boldsymbol{\kappa})$、$V_{pqrs}(\boldsymbol{\kappa})$ 随 $\boldsymbol{\kappa}$ **解析地变换**（与经典 SCF 中基组固定、MO 旋转等价）。

**（2）联合能量泛函**

以 UCC 型电子相关因子 $\hat{U}_{\mathrm{UCC}}(\boldsymbol{\theta})$ 为例，能量：

$$
E(\boldsymbol{\theta},\boldsymbol{\kappa}) = \langle\Phi_0\vert \hat{U}_{\mathrm{UCC}}^\dagger(\boldsymbol{\theta})\,\hat{H}(\boldsymbol{\kappa})\,\hat{U}_{\mathrm{UCC}}(\boldsymbol{\theta})\vert\Phi_0\rangle.
$$

等价视角包括：**先旋转轨道再写哈密顿量**，或 **对 $\hat{H}_0$ 做相似变换**；只要 **$\hat{H}$、$\lvert\Phi_0\rangle$、$\hat{U}_{\mathrm{UCC}}$ 在同一表象下一致** 即可。

**（3）优化循环（常见混合量子–经典）**

Mizukami *et al.* 采用 **嵌套或交替** 策略（实现细节因代码而异）：

1. **固定 $\boldsymbol{\kappa}$**：在量子侧（或模拟器）对 $\boldsymbol{\theta}$ 做 **标准 VQE**，降低 $E(\boldsymbol{\theta},\boldsymbol{\kappa})$。  
2. **固定 $\boldsymbol{\theta}$**：利用 **量子态给出的约化密度矩阵**（或能量对 $\boldsymbol{\kappa}$ 的解析梯度），在经典端做 **轨道梯度 /（近似）牛顿步** 更新 $\boldsymbol{\kappa}$。  
3. **更新积分** $\Rightarrow$ 更新 qubit 哈密顿量 $\Rightarrow$ 回到步骤 1，直至能量与轨道收敛。

**优点（文献强调）**：相对固定轨道的 UCC，在 **相同精度目标** 下可能用 **更小活性空间或更浅线路**；且全变分形式便于 **几何优化**（需 **对核坐标** 与 **$(\boldsymbol{\theta},\boldsymbol{\kappa})$** 的联合梯度，文献中讨论了解析一阶导）。

### B.3 扩展：态平均（SA-OO-VQE）

多个低能态 **近简并** 时，可对 **态平均能量** 做轨道优化，再在各态上做相关处理；**State-Averaged OO-VQE** 等有开源实现（例如文献地图中提到的 JOSS 2024 工作），与 **SA-CASSCF** 叙事平行。

### B.4 实现要点（工程）

- **理论参考**：Mizukami *et al.*, *Phys. Rev. Research* **2**, 033421 (2020)；arXiv:1910.11526。  
- **代码生态**：除自研混合循环外，可关注与 **PennyLane + 经典化学**、**Tequila**、**qiboml** 等结合的 OO 流程；具体 API 以各库文档为准。  
- **与 k-UpCCGSD 的组合**：原则上 **OO** 是 **外层轨道优化**，**k-UpCCGSD** 是 **内层 ansatz**；二者可组合为 **OO + k-UpCCGSD–VQE**，但实现与超参需单独调试。

---

## C. 一句话对照

- **k-UpCCGSD**：在 **广义单激发 + 配对双激发** 的 **稀疏 UCC 池** 上堆叠 **$k$ 层** 幺正，追求 **$\mathcal{O}(kN)$** 量级线路深度下的 **可加深、可改进** 精度。  
- **OO-VQE**：在 VQE 中 **联合变分轨道（$\boldsymbol{\kappa}$）与簇参数（$\boldsymbol{\theta}$）**，使哈密顿量与试探态 **在同一优化问题中对齐**，类比 **CASSCF 式轨道优化**。

---

*若需与课程表对齐，见同目录 [量子计算在计算化学中的方法与文献地图.md](./量子计算在计算化学中的方法与文献地图.md) 中 VQE 方法列表。*
