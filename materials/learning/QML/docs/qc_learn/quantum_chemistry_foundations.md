# 量子化学理论基础与近似方法

> **仓库路径**：`docs/qc_learn/quantum_chemistry_foundations.md`（原 QC-learn；对应第 1–2 周合并稿）。总览见 [README.md](README.md)；代码与推导补充见 [`quantum_chemistry/docs/`](../quantum_chemistry/docs/)。

## 学习目标

本文档系统介绍量子化学的数学基础和理论方法，从量子力学的基本数学语言到传统近似方法，为理解和使用机器学习方法求解薛定谔方程奠定理论基础。

## 目录

1. [量子力学数学基础](#1-量子力学数学基础)
2. [时间无关薛定谔方程的数学结构](#2-时间无关薛定谔方程的数学结构)
3. [变分原理](#3-变分原理)
4. [Hartree-Fock理论](#4-hartree-fock理论的数学推导)
5. [电子相关性的数学描述](#5-电子相关性的数学描述)
6. [后Hartree-Fock方法](#6-后hartree-fock方法)
7. [密度泛函理论](#7-密度泛函理论)
8. [基组展开的数学原理](#8-基组展开的数学原理)
9. [误差分析和数学性质](#9-误差分析和数学性质)
10. [理论思想总结](#10-理论思想总结)
11. [思考题](#11-思考题)
12. [总结](#12-总结)

---

本文档系统介绍量子化学的数学基础和理论方法，从量子力学的基本数学语言到传统近似方法。

## 1. 量子力学数学基础

### 1.1 希尔伯特空间（Hilbert Space）

#### 定义
希尔伯特空间 $\mathcal{H}$ 是一个完备的内积空间，满足：
- **内积性质**：对于任意 $|\psi\rangle, |\phi\rangle \in \mathcal{H}$，内积 $\langle\psi|\phi\rangle$ 满足：
  - 共轭对称性：$\langle\psi|\phi\rangle = \langle\phi|\psi\rangle^*$
  - 线性性：$\langle\psi|a\phi_1 + b\phi_2\rangle = a\langle\psi|\phi_1\rangle + b\langle\psi|\phi_2\rangle$
  - 正定性：$\langle\psi|\psi\rangle \geq 0$，且 $\langle\psi|\psi\rangle = 0$ 当且仅当 $|\psi\rangle = 0$

- **完备性**：所有柯西序列都收敛到空间内的元素

#### 柯西序列（Cauchy Sequence）
**定义**：在内积空间（或度量空间）中，序列 $\{|\psi_n\rangle\}_{n=1}^\infty$ 称为柯西序列，如果对于任意给定的 $\epsilon > 0$，存在正整数 $N$，使得当 $m, n > N$ 时，有：
$$\|\psi_m - \psi_n\| < \epsilon$$

其中 $\|\psi_m - \psi_n\| = \sqrt{\langle\psi_m - \psi_n|\psi_m - \psi_n\rangle}$ 是两个向量之间的距离。

**直观理解**：
- 柯西序列是"自我收敛"的序列：随着指标增大，序列中的元素彼此之间越来越接近
- 换句话说，当 $m$ 和 $n$ 都很大时，$|\psi_m\rangle$ 和 $|\psi_n\rangle$ 应该非常接近

**完备性的含义**：
- 如果一个内积空间是**完备的**（即希尔伯特空间），那么任何柯西序列都在该空间内有一个极限
- 这意味着不存在"收敛到空间外"的柯西序列，即空间的"洞"被填满了

**例子**：
- **有理数**：有理数序列 $\{1, 1.4, 1.41, 1.414, \ldots\}$（$\sqrt{2}$ 的近似）在有理数中不收敛，因为 $\sqrt{2}$ 不是有理数
- **实数**：同样的序列在实数中是柯西序列且收敛到 $\sqrt{2}$
- **希尔伯特空间**：所有柯西序列都收敛到空间内的元素，这使得我们可以安全地进行极限操作和展开

**物理意义**：
- 在量子力学中，完备性保证我们可以用本征函数展开任意波函数
- 当我们用基组展开波函数时，增加基函数数量得到更好的近似，这些近似序列是柯西序列，完备性保证它们收敛到真实的波函数（在基组完备的情况下）

#### 波函数空间
对于 $N$ 电子系统，波函数 $\psi(\mathbf{r}_1, \mathbf{r}_2, \ldots, \mathbf{r}_N)$ 属于 $L^2(\mathbb{R}^{3N})$ 空间，即平方可积函数空间：
$$\int |\psi(\mathbf{r}_1, \ldots, \mathbf{r}_N)|^2 d\mathbf{r}_1 \cdots d\mathbf{r}_N < \infty$$

内积定义为：
$$\langle\psi|\phi\rangle = \int \psi^*(\mathbf{r}_1, \ldots, \mathbf{r}_N) \phi(\mathbf{r}_1, \ldots, \mathbf{r}_N) d\mathbf{r}_1 \cdots d\mathbf{r}_N$$

#### 物理意义
- 波函数的模方 $|\psi|^2$ 表示概率密度
- 归一化条件：$\langle\psi|\psi\rangle = 1$
- 正交性：不同本征态之间正交，$\langle\psi_i|\psi_j\rangle = \delta_{ij}$

### 1.2 算符理论（Operator Theory）

#### 线性算符
算符 $\hat{A}$ 是线性的，如果：
$$\hat{A}(c_1|\psi_1\rangle + c_2|\psi_2\rangle) = c_1\hat{A}|\psi_1\rangle + c_2\hat{A}|\psi_2\rangle$$

#### 厄米算符（Hermitian Operator）
算符 $\hat{A}$ 是厄米的，如果 $\hat{A}^\dagger = \hat{A}$，其中 $\hat{A}^\dagger$ 是伴随算符，满足：
$$\langle\phi|\hat{A}\psi\rangle = \langle\hat{A}^\dagger\phi|\psi\rangle$$

**重要性质**：
1. 厄米算符的本征值是实数
2. 厄米算符的本征函数构成正交完备基
3. 可观测量的算符必须是厄米的

**证明**：设 $\hat{A}|\psi_n\rangle = a_n|\psi_n\rangle$，则：
$$\langle\psi_n|\hat{A}\psi_n\rangle = a_n\langle\psi_n|\psi_n\rangle = a_n$$
同时：
$$\langle\psi_n|\hat{A}\psi_n\rangle = \langle\hat{A}^\dagger\psi_n|\psi_n\rangle = \langle\hat{A}\psi_n|\psi_n\rangle = a_n^*\langle\psi_n|\psi_n\rangle = a_n^*$$
因此 $a_n = a_n^*$，即本征值为实数。

### 1.3 本征值问题（Eigenvalue Problem）

#### 时间无关薛定谔方程
$$\hat{H}|\psi_n\rangle = E_n|\psi_n\rangle$$

其中：
- $\hat{H}$ 是哈密顿算符（厄米算符）
- $E_n$ 是能量本征值（实数）
- $|\psi_n\rangle$ 是能量本征态

#### 谱定理
对于厄米算符 $\hat{H}$，存在正交归一的本征函数系 $\{|\psi_n\rangle\}$，使得任意波函数可以展开为：
$$|\psi\rangle = \sum_n c_n|\psi_n\rangle, \quad c_n = \langle\psi_n|\psi\rangle$$

#### 完备性关系
$$\sum_n |\psi_n\rangle\langle\psi_n| = \hat{I}$$

其中 $\hat{I}$ 是单位算符。

#### 将薛定谔方程投影到组态空间：详细推导

（注：以下内容详细介绍如何将连续空间的薛定谔方程投影到离散组态空间，这是CI方法的数学基础）

#### 一、基本思想

#### 1.1 问题的提出

**连续空间的问题**：
- 精确的薛定谔方程：$\hat{H}|\psi\rangle = E|\psi\rangle$ 在无限维的连续函数空间中
- 波函数 $|\psi(\mathbf{r}_1, \ldots, \mathbf{r}_N)\rangle$ 依赖于 $3N$ 个连续坐标
- 直接求解需要处理无限维积分，计算上不可行

**离散化的思想**：
- 将无限维的连续空间**投影**到有限维的离散空间
- 使用**基组展开**：用有限个基函数（如Slater行列式）的线性组合来近似波函数
- 将微分方程（薛定谔方程）转化为**矩阵本征值问题**

#### 1.2 投影方法的核心

**关键步骤**：
1. **选择基组**：$\{|\Phi_I\rangle\}_{I=1}^M$（例如，Slater行列式集合）
2. **波函数展开**：$|\psi\rangle = \sum_I c_I |\Phi_I\rangle$
3. **投影方程**：将薛定谔方程投影到每个基函数上
4. **矩阵方程**：得到矩阵本征值问题 $\mathbf{H}\mathbf{c} = E\mathbf{c}$

**物理意义**：
- 投影方法将"在连续空间中寻找波函数"转化为"在离散基组中寻找展开系数"
- 这是所有量子化学计算方法的基础

#### 二、数学推导

#### 2.1 波函数展开

**基组选择**：
选择一组正交归一的基函数（Slater行列式）：
$$\{|\Phi_I\rangle\}_{I=1}^M, \quad \langle\Phi_I|\Phi_J\rangle = \delta_{IJ}$$

其中：
- $|\Phi_0\rangle$ 是参考组态（通常是HF基态）
- $|\Phi_I\rangle$（$I \geq 1$）是激发组态（单激发、双激发等）

**波函数展开**：
将精确波函数 $|\psi\rangle$ 展开为基函数的线性组合：
$$|\psi\rangle = \sum_{I=0}^{M-1} c_I |\Phi_I\rangle$$

其中 $c_I$ 是展开系数（待求）。

**归一化条件**：
$$\langle\psi|\psi\rangle = \sum_{I,J} c_I^* c_J \langle\Phi_I|\Phi_J\rangle = \sum_I |c_I|^2 = 1$$

#### 2.2 薛定谔方程

**原始方程**：
$$\hat{H}|\psi\rangle = E|\psi\rangle$$

**代入展开**：
$$\hat{H}\sum_J c_J |\Phi_J\rangle = E\sum_J c_J |\Phi_J\rangle$$

**整理**：
$$\sum_J c_J \hat{H}|\Phi_J\rangle = E\sum_J c_J |\Phi_J\rangle$$

#### 2.3 投影操作

**投影到基函数 $|\Phi_I\rangle$**：
将方程两边同时左乘 $\langle\Phi_I|$（投影到第 $I$ 个基函数）：

$$\langle\Phi_I|\sum_J c_J \hat{H}|\Phi_J\rangle = \langle\Phi_I|E\sum_J c_J |\Phi_J\rangle$$

**展开左边**：
$$\sum_J c_J \langle\Phi_I|\hat{H}|\Phi_J\rangle = E\sum_J c_J \langle\Phi_I|\Phi_J\rangle$$

**利用正交性**：
由于 $\langle\Phi_I|\Phi_J\rangle = \delta_{IJ}$，右边简化为：
$$\sum_J c_J \langle\Phi_I|\hat{H}|\Phi_J\rangle = Ec_I$$

**矩阵形式**：
$$\sum_J H_{IJ} c_J = Ec_I$$

其中 $H_{IJ} = \langle\Phi_I|\hat{H}|\Phi_J\rangle$ 是哈密顿矩阵元。

#### 2.4 矩阵本征值问题

**对所有基函数投影**：
对每个 $I = 0, 1, \ldots, M-1$，我们得到一个方程：
$$\sum_J H_{IJ} c_J = Ec_I$$

**矩阵形式**：
将所有方程写成矩阵形式：
$$\begin{pmatrix}
H_{00} & H_{01} & H_{02} & \cdots \\
H_{10} & H_{11} & H_{12} & \cdots \\
H_{20} & H_{21} & H_{22} & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{pmatrix}
\begin{pmatrix}
c_0 \\
c_1 \\
c_2 \\
\vdots
\end{pmatrix}
= E
\begin{pmatrix}
c_0 \\
c_1 \\
c_2 \\
\vdots
\end{pmatrix}$$

**简写**：
$$\mathbf{H}\mathbf{c} = E\mathbf{c}$$

其中：
- $\mathbf{H}$ 是 $M \times M$ 的哈密顿矩阵，元素为 $H_{IJ} = \langle\Phi_I|\hat{H}|\Phi_J\rangle$
- $\mathbf{c} = (c_0, c_1, \ldots, c_{M-1})^T$ 是系数向量
- $E$ 是能量本征值

#### 三、矩阵元的计算

#### 3.1 哈密顿矩阵元

**定义**：
$$H_{IJ} = \langle\Phi_I|\hat{H}|\Phi_J\rangle$$

**展开哈密顿量**：
$$\hat{H} = \sum_i \hat{h}_i + \frac{1}{2}\sum_{i\neq j} \frac{1}{|\mathbf{r}_i - \mathbf{r}_j|}$$

其中：
- $\hat{h}_i = -\frac{1}{2}\nabla_i^2 - \sum_A \frac{Z_A}{|\mathbf{r}_i - \mathbf{R}_A|}$ 是单电子算符
- $\frac{1}{|\mathbf{r}_i - \mathbf{r}_j|}$ 是双电子算符

#### 3.2 Slater-Condon规则

**关键定理**：如果 $|\Phi_I\rangle$ 和 $|\Phi_J\rangle$ 是Slater行列式，那么：

1. **相同行列式**（$I = J$）：
   $$H_{II} = \langle\Phi_I|\hat{H}|\Phi_I\rangle = \sum_{i \in I} h_i + \frac{1}{2}\sum_{i,j \in I} (J_{ij} - K_{ij})$$
   其中 $h_i, J_{ij}, K_{ij}$ 是单电子和双电子积分。

2. **相差一个轨道**（单激发）：
   $$H_{IJ} = \langle\Phi_I|\hat{H}|\Phi_J\rangle = h_{ia} + \sum_{j \in I} (J_{ij} - K_{ij})$$
   其中 $i$ 是占据轨道，$a$ 是虚轨道。

3. **相差两个轨道**（双激发）：
   $$H_{IJ} = \langle\Phi_I|\hat{H}|\Phi_J\rangle = J_{ij} - K_{ij}$$
   其中 $i,j$ 是占据轨道，$a,b$ 是虚轨道。

4. **相差超过两个轨道**：
   $$H_{IJ} = 0$$
   这是**Slater-Condon规则**的核心：只有相差不超过两个轨道的行列式之间才有非零矩阵元。

##### 为什么相差超过两个轨道矩阵元为零？

**核心原因**：哈密顿量最多是双电子算符，没有三电子或更高阶的算符。

**哈密顿量的结构**：
$$\hat{H} = \underbrace{\sum_i \hat{h}_i}_{\text{单电子算符}} + \underbrace{\frac{1}{2}\sum_{i\neq j} \frac{1}{|\mathbf{r}_i - \mathbf{r}_j|}}_{\text{双电子算符}}$$

**单电子算符的贡献**：
- $\hat{h}_i$ 只作用于电子 $i$
- 积分时，其他电子的轨道必须正交归一配对
- 如果两个行列式相差超过1个轨道，必有轨道无法配对
- 由正交性 $\int \phi_a^* \phi_b = \delta_{ab}$，积分为零
- **结论**：单电子算符只能在相差0或1个轨道的行列式之间产生非零矩阵元

**双电子算符的贡献**：
- $\frac{1}{|\mathbf{r}_i - \mathbf{r}_j|}$ 同时作用于电子 $i$ 和 $j$
- 积分时，其他 $N-2$ 个电子的轨道必须配对
- 如果两个行列式相差超过2个轨道，必有轨道无法配对
- **结论**：双电子算符只能在相差0、1或2个轨道的行列式之间产生非零矩阵元

**具体例子**：

设两个4电子行列式相差3个轨道：
- $|\Phi_I\rangle = |\phi_1 \phi_2 \phi_3 \phi_4\rangle$
- $|\Phi_J\rangle = |\phi_1 \phi_5 \phi_6 \phi_7\rangle$

计算矩阵元时：
- 哈密顿量最多同时作用于2个电子
- 但有3对轨道不同：($\phi_2,\phi_5$), ($\phi_3,\phi_6$), ($\phi_4,\phi_7$)
- 即使算符作用于其中2对，第3对仍需要配对
- 由于 $\int \phi_3^* \phi_6 = 0$（正交），整个积分为零

**总结表**：

| 行列式相差轨道数 | 单电子算符 | 双电子算符 | 总矩阵元 |
|:---:|:---:|:---:|:---:|
| 0 | ✓ | ✓ | 非零 |
| 1 | ✓ | ✓ | 非零 |
| 2 | ✗ | ✓ | 非零 |
| ≥3 | ✗ | ✗ | **零** |

**物理直觉**：哈密顿量描述单电子运动和两电子相互作用，无法同时"改变"三个或更多电子的状态，因此相差超过两个轨道的行列式之间没有直接耦合。

#### 3.3 矩阵的稀疏性

**重要性质**：
- 哈密顿矩阵 $\mathbf{H}$ 是**稀疏的**（大部分元素为零）
- 只有少数矩阵元非零（根据Slater-Condon规则）
- 这大大减少了计算量

**例子**：
- 如果有 $M = 1000$ 个组态
- 完整矩阵有 $M^2 = 1,000,000$ 个元素
- 但根据Slater-Condon规则，只有约 $O(M)$ 个非零元素
- 稀疏性使得可以处理更大的系统

#### 四、求解矩阵本征值问题

#### 4.1 标准本征值问题

**矩阵方程**：
$$\mathbf{H}\mathbf{c} = E\mathbf{c}$$

**求解方法**：
- **对角化**：$\mathbf{H} = \mathbf{U}\mathbf{D}\mathbf{U}^\dagger$
  - $\mathbf{D}$ 是对角矩阵，对角元素是本征值 $E_0, E_1, \ldots, E_{M-1}$
  - $\mathbf{U}$ 是酉矩阵，列向量是对应的本征向量 $\mathbf{c}_0, \mathbf{c}_1, \ldots, \mathbf{c}_{M-1}$

- **基态能量**：$E_0 = \min\{E_i\}$（最低本征值）
- **基态波函数**：$|\psi_0\rangle = \sum_I c_{0,I} |\Phi_I\rangle$（对应 $E_0$ 的本征向量）

#### 4.2 变分原理的体现

**变分原理**：
- 投影方法自动满足变分原理
- 基态能量 $E_0$ 是真实基态能量的上界
- 增加基组大小（$M$ 增大），$E_0$ 单调下降，逼近精确值

**数学证明**：
设精确基态为 $|\psi_{exact}\rangle$，投影得到的基态为 $|\psi_0\rangle$，则：
$$E_0 = \langle\psi_0|\hat{H}|\psi_0\rangle \geq E_{exact}$$

等号成立当且仅当 $|\psi_0\rangle = |\psi_{exact}\rangle$（在基组完备时）。

#### 五、实际应用示例：CI方法

投影方法是组态相互作用（CI）方法的数学基础。CI方法将精确波函数展开为多个Slater行列式的线性组合：
$$|\Psi_{CI}\rangle = c_0|\Phi_0\rangle + \sum_{i,a} c_i^a|\Phi_i^a\rangle + \sum_{i<j,a<b} c_{ij}^{ab}|\Phi_{ij}^{ab}\rangle + \cdots$$

通过投影到组态空间，得到矩阵方程：$\mathbf{H}\mathbf{c} = E\mathbf{c}$

（注：关于CI方法的详细内容，包括截断级别、大小一致性等，参见第6.1节"组态相互作用"部分）

#### 六、投影方法总结

**投影方法的核心思想**：
1. **离散化**：将无限维连续空间投影到有限维离散空间
2. **基组展开**：用基函数的线性组合近似波函数
3. **矩阵方程**：将微分方程转化为矩阵本征值问题
4. **求解**：通过矩阵对角化得到能量和波函数

**数学框架**：
- **波函数展开**：$|\psi\rangle = \sum_I c_I |\Phi_I\rangle$
- **投影方程**：$\langle\Phi_I|\hat{H}|\psi\rangle = E\langle\Phi_I|\psi\rangle$
- **矩阵方程**：$\mathbf{H}\mathbf{c} = E\mathbf{c}$

**优势**：
- 将连续问题转化为离散问题
- 可以利用线性代数的成熟方法
- 矩阵的稀疏性减少计算量

**限制**：
- 基组截断误差
- 矩阵大小随基组指数增长（FCI）
- 需要高效的矩阵对角化算法

## 2. 时间无关薛定谔方程的数学结构

### 2.1 多体问题的数学表述

#### N电子系统的哈密顿量

**完整的分子哈密顿量**（包含核和电子）：
$$\hat{H}_{total} = \hat{T}_n + \hat{T}_e + \hat{V}_{nn} + \hat{V}_{ne} + \hat{V}_{ee}$$

其中：
- **核动能**：$\hat{T}_n = -\sum_A \frac{1}{2M_A} \nabla_A^2$（$M_A$ 是核质量）
- **电子动能**：$\hat{T}_e = -\frac{1}{2}\sum_{i=1}^N \nabla_i^2$
- **核-核相互作用**：$\hat{V}_{nn} = \frac{1}{2}\sum_{A\neq B} \frac{Z_A Z_B}{|\mathbf{R}_A - \mathbf{R}_B|}$
- **核-电子相互作用**：$\hat{V}_{ne} = -\sum_{i=1}^N \sum_{A=1}^{N_{nuc}} \frac{Z_A}{|\mathbf{r}_i - \mathbf{R}_A|}$
- **电子-电子相互作用**：$\hat{V}_{ee} = \frac{1}{2}\sum_{i\neq j}^N \frac{1}{|\mathbf{r}_i - \mathbf{r}_j|}$

**在Born-Oppenheimer近似下的电子哈密顿量**：
$$\hat{H}_e(\mathbf{R}) = \hat{T}_e + \hat{V}_{ne}(\mathbf{R}) + \hat{V}_{ee}$$

其中核坐标 $\{\mathbf{R}_A\}$ 被视为**固定参数**（不是量子变量）。

**重要说明：核-核相互作用项在哪里？**

**问题**：为什么电子哈密顿量中没有核-核相互作用项 $\hat{V}_{nn}$？

**答案**：核-核项并没有"消失"，而是：

1. **在电子问题中，核-核项是常数**：
   - 核-核相互作用：$\hat{V}_{nn} = \frac{1}{2}\sum_{A\neq B} \frac{Z_A Z_B}{|\mathbf{R}_A - \mathbf{R}_B|}$
   - 在Born-Oppenheimer近似下，核坐标 $\{\mathbf{R}_A\}$ 是**固定的参数**
   - 因此 $\hat{V}_{nn}$ 只是一个**常数**（对于给定的核构型），不作用于电子坐标

2. **电子哈密顿量 vs 总能量**：
   - **电子哈密顿量**（见上面定义）：只包含作用于电子坐标的算符，核-核项不作用于电子，所以不包含在内
   
   - **总能量**：$E_{total}(\mathbf{R}) = E_e(\mathbf{R}) + V_{nn}(\mathbf{R})$
     - $E_e(\mathbf{R})$ 是电子能量（电子哈密顿量的本征值）
     - $V_{nn}(\mathbf{R})$ 是核-核相互作用能（常数项）

3. **为什么分开处理？**
   - **电子问题**：求解 $\hat{H}_e(\mathbf{R}) \psi(\mathbf{R}; \mathbf{r}) = E_e(\mathbf{R}) \psi(\mathbf{R}; \mathbf{r})$
     - 这给出电子能量 $E_e(\mathbf{R})$（依赖于核构型）
   
   - **核-核项**：$V_{nn}(\mathbf{R}) = \frac{1}{2}\sum_{A\neq B} \frac{Z_A Z_B}{|\mathbf{R}_A - \mathbf{R}_B|}$
     - 这只是核坐标的函数，不涉及电子
     - 可以直接计算（不需要求解薛定谔方程）

4. **势能面**：
   - 总能量作为核坐标的函数：$E_{total}(\mathbf{R}) = E_e(\mathbf{R}) + V_{nn}(\mathbf{R})$
   - 这形成**势能面**，核在这个势能面上运动
   - 核-核项是势能面的重要组成部分

**总结**：
- 核-核项**没有消失**，只是不包含在电子哈密顿量中
- 它在总能量中作为常数项出现
- 这样分离使得问题更易处理：电子问题只涉及电子坐标，核-核项可以直接计算

#### 核-电子项可以"分离"吗？Born-Oppenheimer近似

（注：以下内容详细解释Born-Oppenheimer近似的数学原理，与上面关于核-核项的解释是相关的）

**关键问题**：波函数是否可以写成核坐标和电子坐标的乘积形式？
$$\Psi(\mathbf{R}_1, \ldots, \mathbf{R}_{N_{nuc}}, \mathbf{r}_1, \ldots, \mathbf{r}_N) \stackrel{?}{=} \chi(\mathbf{R}_1, \ldots, \mathbf{R}_{N_{nuc}}) \psi(\mathbf{r}_1, \ldots, \mathbf{r}_N)$$

**答案**：在精确量子力学中**不能**，但在Born-Oppenheimer近似下**可以**。

##### 精确量子力学：不能分离

**数学原因**：
- 完整的分子波函数 $\Psi(\mathbf{R}, \mathbf{r})$ 依赖于核坐标 $\mathbf{R}$ 和电子坐标 $\mathbf{r}$
- 核-电子相互作用项 $\hat{V}_{ne} = -\sum_{i,A} \frac{Z_A}{|\mathbf{r}_i - \mathbf{R}_A|}$ 同时依赖于 $\mathbf{R}_A$ 和 $\mathbf{r}_i$
- 类似于电子-电子项，这使得波函数**不能**写成乘积形式

**物理原因**：
- 核和电子是**耦合的**：核的运动影响电子，电子的运动也影响核
- 这种耦合使得核和电子的波函数不能分离

##### Born-Oppenheimer近似：可以分离

**基本思想**：
- 核的质量远大于电子（$M_A \gg m_e$，通常 $M_A/m_e \sim 10^3-10^5$）
- 核运动比电子慢得多（时间尺度分离）
- 电子可以"瞬时"适应核的位置

**数学表述**：
在Born-Oppenheimer近似下，分子波函数可以写成：
$$\Psi(\mathbf{R}, \mathbf{r}) = \chi(\mathbf{R}) \psi(\mathbf{R}; \mathbf{r})$$

其中：
- $\chi(\mathbf{R})$ 是核波函数
- $\psi(\mathbf{R}; \mathbf{r})$ 是电子波函数，**依赖于核坐标作为参数**

**关键点**：
- 电子波函数 $\psi(\mathbf{R}; \mathbf{r})$ 中的 $\mathbf{R}$ 是**参数**，不是变量
- 对于每个固定的核构型 $\mathbf{R}$，我们求解电子薛定谔方程：
  $$\hat{H}_e(\mathbf{R}) \psi(\mathbf{R}; \mathbf{r}) = E_e(\mathbf{R}) \psi(\mathbf{R}; \mathbf{r})$$
- 其中电子哈密顿量（见上面定义）依赖于 $\mathbf{R}$ 作为参数

**为什么核-电子项可以"分离"？**

在Born-Oppenheimer近似下：
1. **核坐标固定**：对于给定的核构型 $\mathbf{R}$，核-电子项 $\hat{V}_{ne}(\mathbf{R})$ 只是电子坐标的函数
2. **电子波函数求解**：对于固定的 $\mathbf{R}$，求解电子薛定谔方程得到 $\psi(\mathbf{R}; \mathbf{r})$
3. **势能面**：电子能量 $E_e(\mathbf{R})$ 作为核坐标的函数，加上核-核项 $V_{nn}(\mathbf{R})$ 得到总能量 $E_{total}(\mathbf{R}) = E_e(\mathbf{R}) + V_{nn}(\mathbf{R})$，形成势能面（见上面"势能面"部分的详细解释）
4. **核运动**：核在这个势能面上运动

**数学上**：
- 核-电子项：$\hat{V}_{ne} = -\sum_{i,A} \frac{Z_A}{|\mathbf{r}_i - \mathbf{R}_A|}$
- 当 $\mathbf{R}_A$ 固定时，这只是 $\mathbf{r}_i$ 的函数
- 因此可以包含在电子哈密顿量中，作为依赖于 $\mathbf{R}$ 的参数

**对比电子-电子项**：
- 电子-电子项：$\hat{V}_{ee} = \frac{1}{2}\sum_{i\neq j} \frac{1}{|\mathbf{r}_i - \mathbf{r}_j|}$
- 这**同时**依赖于多个电子坐标 $\mathbf{r}_i$ 和 $\mathbf{r}_j$
- 因此电子波函数不能写成单电子函数的乘积

**总结**：

| 项 | 精确量子力学 | Born-Oppenheimer近似 |
|---|---|---|
| 核-电子项 | 不能分离（核和电子耦合） | 可以分离（核坐标作为参数） |
| 电子-电子项 | 不能分离（电子相关） | 不能分离（电子相关） |

**实际应用**：
- 在量子化学计算中，几乎总是使用Born-Oppenheimer近似
- 对于每个核构型，求解电子结构问题
- 这大大简化了问题，将 $3(N+N_{nuc})$ 维问题分解为：
  - 电子问题：$3N$ 维（对于每个 $\mathbf{R}$）
  - 核问题：$3N_{nuc}$ 维（在势能面上）

#### 波函数空间
对于 $N$ 电子系统，波函数 $\psi(\mathbf{r}_1, \sigma_1, \ldots, \mathbf{r}_N, \sigma_N)$ 必须满足：

1. **反对称性**（费米子统计）：
   $$\psi(\ldots, \mathbf{r}_i, \sigma_i, \ldots, \mathbf{r}_j, \sigma_j, \ldots) = -\psi(\ldots, \mathbf{r}_j, \sigma_j, \ldots, \mathbf{r}_i, \sigma_i, \ldots)$$

2. **归一化**：
   $$\sum_{\sigma_1,\ldots,\sigma_N} \int |\psi(\mathbf{r}_1, \sigma_1, \ldots, \mathbf{r}_N, \sigma_N)|^2 d\mathbf{r}_1 \cdots d\mathbf{r}_N = 1$$

#### 维度灾难（Curse of Dimensionality）

维度灾难是指在处理高维问题时，计算复杂度、存储需求和采样点数随维数呈指数增长的现象。

##### 维度增长

- **单电子系统**：3维空间（3个空间坐标 $x, y, z$）
- **N电子系统**：$3N$ 维空间（每个电子3个坐标）
- **例子**：
  - H$_2$O（10电子）：30维
  - C$_6$H$_6$（42电子）：126维
  - 蛋白质分子（数千电子）：数千维

##### 基组展开的组合爆炸

在使用基组展开时，维度灾难表现为组合爆炸：

**重要概念澄清：基组、轨道和函数的关系**

**"轨道"就是函数！** 这是量子化学中的术语习惯：
- **轨道（orbital）**：单电子波函数，是空间坐标的函数 $\phi(\mathbf{r})$ 或 $\phi(\mathbf{x})$（包含自旋）
- **基组（basis set）**：一组基函数的集合 $\{\chi_\mu(\mathbf{r})\}_{\mu=1}^M$
- **轨道用基组展开**：任意轨道可以表示为基函数的线性组合：
  $$\phi_i(\mathbf{r}) = \sum_{\mu=1}^M c_{i\mu} \chi_\mu(\mathbf{r})$$

**为什么叫"轨道"而不直接叫"函数"？**
- 历史原因：早期量子力学中，电子的轨道概念来自原子结构（如s轨道、p轨道、d轨道）
- 物理含义：轨道描述了电子在空间的分布
- 约定俗成：量子化学领域一直沿用这个术语

**具体说明**：
1. **基组**：$\{\chi_1(\mathbf{r}), \chi_2(\mathbf{r}), \ldots, \chi_M(\mathbf{r})\}$ 是一组基函数
   - 例如：高斯型轨道（GTO）、Slater型轨道（STO）
   - 这些是**函数**，定义在空间坐标 $\mathbf{r}$ 上

2. **轨道**：$\phi_i(\mathbf{r})$ 也是**函数**，可以用基组展开：
   $$\phi_i(\mathbf{r}) = \sum_{\mu=1}^M c_{i\mu} \chi_\mu(\mathbf{r})$$

3. **占据轨道**：在 $N$ 电子系统中，有 $N$ 个电子，所以需要 $N$ 个轨道（每个电子一个轨道）
   - 这些是**占据轨道**（occupied orbitals）
   - 剩余的 $M-N$ 个轨道是**虚轨道**（virtual orbitals，未占据）

**为什么M个基函数能构造出M个轨道？**

这是线性代数的基本结论：

**线性无关性**：
- 假设基函数 $\{\chi_1, \chi_2, \ldots, \chi_M\}$ 是线性无关的
- 这意味着：$\sum_{\mu=1}^M a_\mu \chi_\mu = 0$ 当且仅当 $a_1 = a_2 = \cdots = a_M = 0$

**轨道构造**：
- 任意轨道可以表示为：$\phi_i = \sum_{\mu=1}^M c_{i\mu} \chi_\mu$
- 由于基函数是线性无关的，我们可以构造最多 $M$ 个线性无关的轨道
- 具体地，我们可以构造 $M$ 个正交归一的轨道 $\{\phi_1, \phi_2, \ldots, \phi_M\}$

**为什么恰好是M个？**
- 基组 $\{\chi_\mu\}$ 张成一个 $M$ 维的线性空间
- 在这个空间中，最多只能有 $M$ 个线性无关的向量（函数）
- 因此，最多可以构造 $M$ 个正交归一的轨道

**数学证明**（简要）：
- 如果我们尝试构造第 $(M+1)$ 个轨道 $\phi_{M+1} = \sum_{\mu=1}^M c_{M+1,\mu} \chi_\mu$
- 由于 $\{\chi_\mu\}$ 只有 $M$ 个，$\phi_{M+1}$ 必须能表示为前 $M$ 个轨道的线性组合
- 因此无法有 $M+1$ 个线性无关的轨道

**为什么需要构造M个轨道，而不是只构造N个占据轨道？**

**关键原因**：

1. **变分原理的要求**：
   - 在Hartree-Fock方法中，我们需要**优化轨道**
   - 通过变分原理，我们求解Fock方程：$\hat{F}\phi_i = \epsilon_i \phi_i$
   - Fock方程的解（本征函数）正好有 $M$ 个（对应 $M$ 个本征值 $\epsilon_1, \epsilon_2, \ldots, \epsilon_M$）
   - 我们无法只解出 $N$ 个，而必须解出所有 $M$ 个

2. **占据轨道的选择**：
   - 在解出所有 $M$ 个轨道后，按照能量排序：$\epsilon_1 \leq \epsilon_2 \leq \cdots \leq \epsilon_M$
   - 根据Aufbau原理，能量最低的 $N$ 个轨道被占据（每个轨道放2个电子，考虑自旋）
   - 这 $N$ 个是**占据轨道**（occupied orbitals）
   - 剩余的 $M-N$ 个是**虚轨道**（virtual orbitals）

3. **虚轨道的作用**：
   - **后HF方法需要**：CI、CC、MP等方法需要虚轨道来描述激发态
   - **激发过程**：电子从占据轨道激发到虚轨道
   - **相关能**：电子相关主要通过虚轨道的贡献来捕获
   - **变分优化**：即使在HF中，虚轨道也影响占据轨道的形状（通过优化过程）

**具体例子**：
- H$_2$O分子：10个电子
- 使用6-31G基组：约25个基函数（$M=25$）
- HF计算给出：
  - 5个占据轨道（$N=10$，但考虑自旋配对，空间轨道为5个）
  - 20个虚轨道（$M-N=20$）
- 如果我们只用5个轨道，无法：
  - 描述电子激发
  - 捕获电子相关
  - 进行后HF计算

**Slater行列式的构造**：
- 如果基组有 $M$ 个基函数 $\{\chi_\mu\}$，我们可以构造 $M$ 个轨道 $\{\phi_i\}$
- 要构造 $N$ 电子的Slater行列式，需要从 $M$ 个可能的轨道中选择 $N$ 个占据轨道
- Slater行列式的数量为：$\binom{M}{N} = \frac{M!}{N!(M-N)!}$
- 这包括了所有可能的占据-虚轨道分配方式

**总结**：
- **基组** = 一组基函数（如 $\{\chi_\mu\}$），有 $M$ 个
- **轨道** = 单电子波函数，最多可以构造 $M$ 个线性无关的正交轨道
- **占据轨道** = 能量最低的 $N$ 个轨道，被电子占据
- **虚轨道** = 剩余的 $M-N$ 个轨道，未占据但对描述电子相关至关重要

**数量级示例**：
- $M=20, N=10$：$\binom{20}{10} = 184,756$ 个行列式
- $M=50, N=20$：$\binom{50}{20} \approx 4.7 \times 10^{13}$ 个行列式
- $M=100, N=50$：$\binom{100}{50} \approx 1.0 \times 10^{29}$ 个行列式

**FCI（全组态相互作用）**：包含所有可能的激发，Slater行列式数量为：
$$\sum_{k=0}^{M-N} \binom{M}{N+k} = 2^M \quad \text{(对于轨道数等于基组大小的情况)}$$

##### 高维积分的困难

**网格积分的复杂度**：
- 在 $d$ 维空间中，如果每个维度用 $n$ 个格点
- 总格点数为：$n^d$
- **例子**：
  - 1维：$n = 10$ 个格点
  - 3维：$n^3 = 1,000$ 个格点
  - 6维（He原子）：$n^6 = 1,000,000$ 个格点
  - 30维（H$_2$O）：$n^{30} = 10^{30}$ 个格点（不可行！）

**蒙特卡洛方法的困难**：
- 虽然不依赖网格，但需要大量采样点
- 在高维空间中，大部分体积在边界附近（维度灾难的另一个表现）
- 有效采样区域变得非常小

##### 维度灾难的数学本质

**体积集中在边界**：

这是维度灾难的一个反直觉但重要的现象。让我们用数学来说明：

**单位超球的体积分布**：
- 在 $d$ 维空间中，半径为 $R$ 的球的体积为：$V_d(R) = \frac{\pi^{d/2}}{\Gamma(d/2+1)} R^d$
- 对于单位球（$R=1$），体积为：$V_d(1) = \frac{\pi^{d/2}}{\Gamma(d/2+1)}$
- 半径为 $r$ 的球的体积为：$V_d(r) = V_d(1) \cdot r^d$

**关键观察**：
考虑单位球内的体积分布。在半径 $r = 1-\epsilon$ 和 $r = 1$ 之间的球壳体积为：
$$\Delta V = V_d(1) - V_d(1-\epsilon) = V_d(1)[1 - (1-\epsilon)^d]$$

当 $d$ 很大时，$(1-\epsilon)^d \approx e^{-d\epsilon}$（当 $\epsilon$ 很小时）

**具体例子**：
- 对于 $d = 100$ 维单位球
- 考虑 $r > 0.99$ 的薄壳层（$\epsilon = 0.01$）
- 薄壳层的体积比例为：$1 - (0.99)^{100} \approx 1 - e^{-1} \approx 0.63$
- **63%的体积**在半径 $r > 0.99$ 的薄壳层中！

**为什么会出现这个现象？**
- 体积正比于 $r^d$，当 $d$ 很大时，这是一个非常陡峭的函数
- 当 $r$ 从 $0.9$ 增加到 $1.0$ 时，体积增加了 $(1/0.9)^d - 1$
- 对于 $d=100$，$(1/0.9)^{100} \approx 37,649$ 倍！
- 因此，即使是一个很薄的壳层，也包含了大部分体积

**直观理解（虽然很难直观想象高维）**：
想象一个多维的洋葱：
- 在低维（如3维），大部分体积在中心附近
- 但在高维（如100维），大部分"洋葱肉"都在最外层薄皮中
- 中心部分实际上体积很小

**对量子化学的意义**：
- 在 $3N$ 维的电子坐标空间中，波函数的大部分"权重"可能集中在边界区域
- 这使得均匀采样变得非常困难
- 需要智能的采样策略（如重要性采样）来有效采样高维空间

**距离失去意义**：
- 在高维空间中，所有点之间的距离都变得相似
- 最近邻和最远邻的距离比接近1
- 这使得基于距离的方法失效

**稀疏性**：
- $d$ 维单位立方体的体积为 $1^d = 1$
- 但如果用边长为 $0.1$ 的小立方体填充，需要 $10^d$ 个小立方体
- 当 $d=100$ 时，需要 $10^{100}$ 个小立方体（远超可观测宇宙的原子数）

##### 量子化学中的具体影响

**1. 精确方法不可扩展**：
- FCI方法：计算复杂度 $O(2^M)$，只能处理很小系统
- 即使中等大小的分子（如苯），FCI也是不可能的

**2. 近似方法的必要性**：
- Hartree-Fock：$O(M^4)$，多项式复杂度
- DFT：$O(M^3)$ 或 $O(M^4)$
- 后HF方法：CCSD为 $O(M^6)$，CCSDT为 $O(M^8)$

**3. 基组选择的重要性**：
- 必须平衡精度和计算成本
- 使用紧凑但准确的基组（如Dunning基组）
- 使用赝势减少核心电子

**4. 机器学习方法的机会**：
- 神经网络可能用更少的参数表示复杂函数
- 但仍然面临高维采样问题
- 需要开发特殊的技术（如重要性采样、变分方法）

##### 维度灾难的缓解策略

1. **维数约简**：
   - 使用自然轨道（自然轨道有更快的收敛性）
   - 主成分分析（PCA）降维

2. **稀疏表示**：
   - 利用波函数的稀疏性
   - 只计算重要的组态

3. **分层方法**：
   - 将问题分解为低维子问题
   - 使用多尺度方法

4. **近似方法**：
   - 接受近似以获得可扩展性
   - 使用物理直觉指导近似

5. **特殊方法**：
   - 变分蒙特卡洛（不依赖网格）
   - 机器学习方法（自适应表示）

##### 维度灾难在传统机器学习中的表现

**是的，传统机器学习在高维特征空间中也面临完全相同的问题！**

#### 1. 高维特征空间中的维度灾难

**问题**：当特征维度 $d$ 很大时（如 $d = 100, 1000, 10000$），会出现：

**1.1 体积集中在边界附近**

**数学表现**：
- 在 $d$ 维单位超立方体 $[0,1]^d$ 中
- 考虑中心区域：$[0.25, 0.75]^d$
- 中心区域的体积：$(0.5)^d$
- 当 $d=10$：中心体积 = $0.5^{10} \approx 0.001$（0.1%）
- 当 $d=100$：中心体积 = $0.5^{100} \approx 7.9 \times 10^{-31}$（几乎为零！）

**对机器学习的影响**：
- **数据稀疏**：大部分数据点都在边界附近，中心区域几乎没有数据
- **采样困难**：需要指数级的数据点才能覆盖空间
- **泛化困难**：模型难以学习中心区域的模式

**1.2 距离失去意义**

**数学表现**：
- 在 $d$ 维空间中，任意两点之间的距离：
  $$d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^d (x_i - y_i)^2}$$
- 当 $d$ 很大时，所有点之间的距离变得相似
- 最近邻和最远邻的距离比接近1

**对机器学习的影响**：
- **KNN失效**：最近邻算法无法区分"近"和"远"
- **聚类困难**：基于距离的聚类方法失效
- **相似性度量失效**：欧氏距离、余弦相似度等失去区分能力

**1.3 稀疏性**

**数学表现**：
- 在 $d$ 维单位立方体中，用边长为 $\epsilon$ 的小立方体填充
- 需要 $(\frac{1}{\epsilon})^d$ 个小立方体
- 当 $d=100, \epsilon=0.1$：需要 $10^{100}$ 个小立方体（不可行！）

**对机器学习的影响**：
- **数据需求指数增长**：需要指数级的数据才能覆盖特征空间
- **过拟合风险**：数据稀疏导致模型容易过拟合
- **特征选择困难**：难以判断哪些特征重要

#### 2. 对传统机器学习的具体影响

**2.1 数据需求**

**问题**：
- 低维（$d=2$）：100个数据点可能足够
- 高维（$d=100$）：可能需要 $10^{100}$ 个数据点（不可能！）

**实际影响**：
- **数据不足**：实际数据量远小于所需
- **样本外推困难**：模型难以泛化到未见过的区域
- **小样本学习困难**：高维小样本问题特别严重

**2.2 模型复杂度**

**问题**：
- 参数数量：通常随维度线性或多项式增长
- 但有效参数空间：随维度指数增长
- 模型容量：需要指数级容量才能表示复杂函数

**实际影响**：
- **过拟合**：模型容易记住训练数据
- **欠拟合**：简单模型无法捕获复杂模式
- **正则化困难**：需要更强的正则化

**2.3 计算复杂度**

**问题**：
- 许多算法的复杂度是 $O(d^n)$ 或 $O(2^d)$
- 存储需求：$O(d^n)$

**实际影响**：
- **训练时间**：指数增长
- **内存需求**：指数增长
- **实时预测**：变得不可行

**2.4 特征交互**

**问题**：
- 在 $d$ 维空间中，可能的特征交互有 $2^d$ 种
- 即使只考虑两两交互，也有 $\binom{d}{2} = O(d^2)$ 种

**实际影响**：
- **特征工程困难**：难以手动设计所有重要交互
- **自动特征学习**：需要强大的特征学习能力
- **可解释性**：难以理解高维特征交互

#### 3. 传统机器学习的解决方案

**3.1 降维（Dimensionality Reduction）**

**主成分分析（PCA）**：
- **思想**：找到数据的主要变化方向
- **方法**：将数据投影到低维子空间
- **优点**：保留主要信息，减少维度
- **缺点**：可能丢失重要信息

**线性判别分析（LDA）**：
- **思想**：找到最能区分类别的方向
- **方法**：最大化类间距离，最小化类内距离
- **优点**：有监督，保留判别信息

**t-SNE, UMAP**：
- **思想**：非线性降维，保持局部结构
- **方法**：流形学习
- **优点**：能发现非线性结构
- **缺点**：计算复杂度高

**3.2 特征选择（Feature Selection）**

**过滤方法**：
- **思想**：根据统计量选择特征
- **方法**：相关性、互信息、卡方检验等
- **优点**：快速，独立于模型
- **缺点**：可能忽略特征交互

**包装方法**：
- **思想**：使用模型性能选择特征
- **方法**：前向选择、后向消除、遗传算法等
- **优点**：考虑特征交互
- **缺点**：计算昂贵

**嵌入方法**：
- **思想**：在模型训练中自动选择
- **方法**：L1正则化（LASSO）、树模型的特征重要性
- **优点**：自动，考虑模型
- **缺点**：依赖于模型

**3.3 正则化（Regularization）**

**L1正则化（LASSO）**：
- **思想**：鼓励稀疏解（很多特征权重为0）
- **方法**：在损失函数中加入 $\lambda \sum_i |w_i|$
- **优点**：自动特征选择，防止过拟合
- **缺点**：可能过度稀疏

**L2正则化（Ridge）**：
- **思想**：限制参数大小
- **方法**：在损失函数中加入 $\lambda \sum_i w_i^2$
- **优点**：防止过拟合，数值稳定
- **缺点**：不进行特征选择

**弹性网络（Elastic Net）**：
- **思想**：结合L1和L2
- **方法**：$\lambda_1 \sum_i |w_i| + \lambda_2 \sum_i w_i^2$
- **优点**：兼顾两者优点

**3.4 核方法（Kernel Methods）**

**思想**：
- 将数据映射到高维特征空间
- 但在高维空间中只计算内积（通过核函数）
- 避免显式处理高维空间

**支持向量机（SVM）**：
- 使用核技巧，在隐式高维空间中分类
- 避免维度灾难（因为只计算内积）

**高斯过程（Gaussian Process）**：
- 使用核函数定义协方差
- 避免显式高维表示

**3.5 深度学习（Deep Learning）**

**自动特征学习**：
- 神经网络自动学习层次化特征
- 从低层特征到高层抽象

**表示学习**：
- 学习紧凑的表示
- 可能比原始特征维度更低但更有效

**正则化技术**：
- Dropout：随机丢弃神经元
- Batch Normalization：归一化
- 数据增强：增加数据多样性

**3.6 集成方法（Ensemble Methods）**

**随机森林**：
- 随机选择特征子集
- 每个树在低维子空间中学习
- 集成多个弱学习器

**梯度提升**：
- 逐步添加弱学习器
- 每个学习器关注之前模型的错误
- 有效利用特征

**3.7 流形学习（Manifold Learning）**

**思想**：
- 高维数据可能位于低维流形上
- 学习流形结构，在流形上操作

**方法**：
- Isomap：保持测地距离
- LLE：局部线性嵌入
- 自编码器：学习流形表示

#### 4. 量子化学中的机器学习：特殊考虑

**4.1 物理约束**

**对称性**：
- 利用旋转、平移、置换对称性
- 减少有效维度

**物理先验**：
- 使用物理知识指导特征设计
- 例如：原子环境描述符

**4.2 特殊方法**

**图神经网络**：
- 分子是图结构
- 在图上操作，避免高维坐标空间

**等变神经网络**：
- 保证旋转等变性
- 自动满足对称性

**4.3 混合方法**

**物理+机器学习**：
- 使用物理模型预处理
- 机器学习精化
- 结合两者优势

#### 5. 总结对比

| 方面 | 量子化学 | 传统机器学习 |
|---|---|---|
| **维度来源** | 电子坐标（$3N$维） | 特征维度（$d$维） |
| **体积集中** | 波函数在边界附近 | 数据在边界附近 |
| **距离失效** | 电子坐标距离 | 特征空间距离 |
| **稀疏性** | 波函数稀疏 | 数据稀疏 |
| **解决方案** | 变分方法、ML | 降维、正则化、深度学习 |

**共同点**：
- 都面临维度灾难
- 都需要智能的方法来处理高维
- 都受益于稀疏性和结构先验

**不同点**：
- 量子化学有物理约束（对称性、反对称性）
- 传统ML更灵活，但需要更多数据
- 量子化学可以结合物理知识

### 2.2 薛定谔方程的求解难点

#### 数学难点
1. **高维积分**：需要计算 $3N$ 维积分
2. **电子相关**：$\hat{V}_{ee}$ 项使得波函数不能写成单电子函数的乘积（详见下面的详细解释）
3. **反对称约束**：波函数必须满足费米子反对称性
4. **基组完备性**：需要无限大的基组才能精确表示

#### 为什么电子相互作用使得波函数不能写成乘积形式？

**波函数不能写成简单乘积有两个独立原因**：
1. **反对称性要求**（见4.1节）：费米子必须反对称 → 解决方案：Slater行列式
2. **电子相关**（见5.2节）：库仑相互作用使电子运动相关 → 需要多个Slater行列式

**简要总结**：
- 电子-电子相互作用项 $\frac{1}{|\mathbf{r}_i - \mathbf{r}_j|}$ 同时依赖于多个电子坐标
- Slater行列式解决了反对称性问题，但单个行列式仍忽略电子相关
- 精确波函数需要多个Slater行列式的线性组合（见第5、6节）

#### 物理难点
1. **电子相关能**：电子之间的瞬时相关
2. **交换相关**：费米子统计导致的交换能
3. **多体效应**：不能简单分解为单电子问题

## 3. 变分原理（Variational Principle）

### 3.1 变分原理的数学表述

#### 试探波函数与本征函数的关系

**重要概念**：试探波函数（trial wave function）通常**不是**精确的本征函数。

**定义**：
- **本征函数**：满足 $\hat{H}|\psi_n\rangle = E_n|\psi_n\rangle$ 的波函数，是精确解
- **试探波函数**：我们猜测或构造的近似波函数 $|\tilde{\psi}\rangle$，通常不满足本征方程

**关键区别**：
1. **本征函数是精确的**：
   - 满足 $\hat{H}|\psi_n\rangle = E_n|\psi_n\rangle$
   - 如果我们已经知道精确的本征函数，就不需要变分法了
   
2. **试探波函数是近似的**：
   - 通常不满足 $\hat{H}|\tilde{\psi}\rangle = E|\tilde{\psi}\rangle$（除非恰好是精确解）
   - 我们不知道精确解，所以用试探函数来近似

3. **特殊情况**：
   - 如果试探波函数**恰好是**精确的基态本征函数，那么：
     - $\hat{H}|\tilde{\psi}\rangle = E_0|\tilde{\psi}\rangle$
     - $E[\tilde{\psi}] = E_0$（达到精确值）
   - 但在实际中，这种情况几乎不可能发生

**实际应用**：
- 我们使用试探波函数（如Slater行列式、神经网络等）来**逼近**精确本征函数
- 通过优化参数，使试探函数尽可能接近精确解
- 变分原理告诉我们：即使试探函数不是精确解，我们也能得到能量上界

#### 定理
对于任意归一化的试探波函数 $|\tilde{\psi}\rangle$，其能量期望值满足：
$$E[\tilde{\psi}] = \frac{\langle\tilde{\psi}|\hat{H}|\tilde{\psi}\rangle}{\langle\tilde{\psi}|\tilde{\psi}\rangle} \geq E_0$$

其中 $E_0$ 是基态能量。

#### 证明
将 $|\tilde{\psi}\rangle$ 按能量本征态展开：
$$|\tilde{\psi}\rangle = \sum_n c_n|\psi_n\rangle$$

其中 $\{|\psi_n\rangle\}$ 是 $\hat{H}$ 的归一化本征态，$E_n$ 是对应本征值，且 $E_0 \leq E_1 \leq E_2 \leq \cdots$。

能量期望值为：
$$E[\tilde{\psi}] = \frac{\langle\tilde{\psi}|\hat{H}|\tilde{\psi}\rangle}{\langle\tilde{\psi}|\tilde{\psi}\rangle} = \frac{\sum_{n,m} c_n^* c_m \langle\psi_n|\hat{H}|\psi_m\rangle}{\sum_n |c_n|^2}$$

由于 $\langle\psi_n|\hat{H}|\psi_m\rangle = E_m \delta_{nm}$，得到：
$$E[\tilde{\psi}] = \frac{\sum_n |c_n|^2 E_n}{\sum_n |c_n|^2} \geq \frac{\sum_n |c_n|^2 E_0}{\sum_n |c_n|^2} = E_0$$

**关键结论**：
- 等号成立当且仅当 $|\tilde{\psi}\rangle$ 是基态本征函数 $|\psi_0\rangle$
- 这意味着：如果试探波函数**不是**精确的本征函数，那么 $E[\tilde{\psi}] > E_0$
- 只有当试探波函数**恰好是**精确的基态本征函数时，才能得到精确的基态能量

**物理意义**：
- 试探波函数可以表示为各个本征函数的叠加：$|\tilde{\psi}\rangle = \sum_n c_n|\psi_n\rangle$
- 如果 $c_0 = 1$ 且 $c_n = 0$（$n \neq 0$），则 $|\tilde{\psi}\rangle = |\psi_0\rangle$，这是精确的基态
- 在实际中，$|\tilde{\psi}\rangle$ 通常包含基态和激发态的混合，因此 $E[\tilde{\psi}] > E_0$

**例子**：
- Hartree-Fock波函数：是试探波函数，通常不是精确的本征函数
- 神经网络量子态：是试探波函数，通过优化可以逼近精确解
- 精确的基态：如果我们能构造出精确的基态，它既是本征函数，也可以作为试探函数（这时能量达到精确值）

### 3.2 变分原理的物理意义

1. **上界性质**：任何试探波函数给出的能量都是基态能量的上界
2. **优化方向**：通过优化波函数参数，可以不断降低能量，逼近基态
3. **误差估计**：能量误差与波函数误差的平方成正比

#### 试探波函数 vs 本征函数：总结

**核心答案**：试探波函数通常**不是**本征函数，但我们可以通过优化使其**逼近**本征函数。

**详细说明**：

1. **理想情况（我们不知道精确解）**：
   - 试探波函数 $|\tilde{\psi}\rangle$ 不是精确的本征函数
   - $\hat{H}|\tilde{\psi}\rangle \neq E|\tilde{\psi}\rangle$（一般情况）
   - $E[\tilde{\psi}] > E_0$（能量高于精确值）

2. **特殊情况（试探函数恰好是精确解）**：
   - 如果试探波函数恰好是精确的基态本征函数：$|\tilde{\psi}\rangle = |\psi_0\rangle$
   - 则 $\hat{H}|\tilde{\psi}\rangle = E_0|\tilde{\psi}\rangle$
   - $E[\tilde{\psi}] = E_0$（达到精确值）

3. **实际策略**：
   - 我们不知道精确的本征函数，所以使用试探函数
   - 通过变分优化，使试探函数尽可能接近精确本征函数
   - 能量会不断下降，逼近 $E_0$

**实际例子**：
- **Hartree-Fock方法**：
  - HF波函数是试探函数（单Slater行列式）
  - 通常不是精确的本征函数（因为忽略了电子相关）
  - HF能量 > 精确基态能量

- **CI方法**：
  - CI波函数是多个Slater行列式的线性组合（试探函数）
  - 随着包含更多行列式，越来越接近精确本征函数
  - FCI（包含所有可能行列式）在基组完备时给出精确本征函数

- **神经网络量子态**：
  - 神经网络参数化的波函数是试探函数
  - 通过优化神经网络参数，逼近精确本征函数
  - 理论上可以精确表示，但实际只能近似

### 3.3 变分法求解

#### 泛函变分
将波函数参数化：$|\psi(\boldsymbol{\theta})\rangle$，其中 $\boldsymbol{\theta}$ 是参数向量。

能量泛函：
$$E[\boldsymbol{\theta}] = \frac{\langle\psi(\boldsymbol{\theta})|\hat{H}|\psi(\boldsymbol{\theta})\rangle}{\langle\psi(\boldsymbol{\theta})|\psi(\boldsymbol{\theta})\rangle}$$

最优参数通过求解得到：
$$\frac{\partial E[\boldsymbol{\theta}]}{\partial \theta_i} = 0, \quad \forall i$$

#### 梯度计算
$$\frac{\partial E}{\partial \theta_i} = 2 \text{Re}\left[\frac{\langle\frac{\partial\psi}{\partial\theta_i}|\hat{H}|\psi\rangle}{\langle\psi|\psi\rangle} - E \frac{\langle\frac{\partial\psi}{\partial\theta_i}|\psi\rangle}{\langle\psi|\psi\rangle}\right]$$

## 4. Hartree-Fock理论的数学推导

### 4.1 单行列式近似与Slater行列式

（注：本节详细介绍Slater行列式的思想、数学原理和实际应用）

#### 一、思想来源：为什么需要Slater行列式？

#### 1.1 费米子的反对称性要求

**物理背景**：
- 电子是**费米子**（半整数自旋），必须遵守**泡利不相容原理**
- 两个电子不能处于完全相同的量子态
- 多电子波函数必须满足**反对称性**

**反对称性的要求**：
对于任意两个电子的交换，波函数必须改变符号：
$$\psi(\ldots, \mathbf{x}_i, \ldots, \mathbf{x}_j, \ldots) = -\psi(\ldots, \mathbf{x}_j, \ldots, \mathbf{x}_i, \ldots)$$

#### 1.2 简单乘积形式的失败（反对称性角度）

**注意**：波函数不能写成简单乘积有**两个独立的原因**：
1. **反对称性要求**（本节讨论）：费米子波函数必须反对称
2. **电子相关**（见5.2节）：库仑相互作用使电子运动相关

本节讨论第一个原因——反对称性。

**尝试1：简单乘积**
$$\psi(\mathbf{x}_1, \ldots, \mathbf{x}_N) = \phi_1(\mathbf{x}_1) \phi_2(\mathbf{x}_2) \cdots \phi_N(\mathbf{x}_N)$$

**问题**：不满足反对称性！
- 交换电子1和电子2：$\phi_1(\mathbf{x}_2) \phi_2(\mathbf{x}_1) \cdots \neq -\phi_1(\mathbf{x}_1) \phi_2(\mathbf{x}_2) \cdots$

**尝试2：对称化乘积**
$$\psi = \frac{1}{\sqrt{N!}} \sum_{\text{所有排列}} \phi_1(\mathbf{x}_{P(1)}) \phi_2(\mathbf{x}_{P(2)}) \cdots \phi_N(\mathbf{x}_{P(N)})$$

**问题**：这是**对称的**（适用于玻色子），但我们需要**反对称的**（费米子）！

#### 1.3 Slater的解决方案

**John C. Slater (1929)** 提出使用**行列式**来构造反对称波函数：
- 行列式天然具有反对称性
- 交换两行（对应交换两个电子）改变符号
- 两行相同（对应两个电子相同态）行列式为零（泡利原理）

#### 二、数学定义和构造

##### 2.1 Slater行列式的定义

对于 $N$ 电子系统，使用 $N$ 个单电子轨道 $\{\phi_1, \phi_2, \ldots, \phi_N\}$，Slater行列式定义为：

$$\psi_{SD}(\mathbf{x}_1, \ldots, \mathbf{x}_N) = \frac{1}{\sqrt{N!}} \det\begin{pmatrix}
\phi_1(\mathbf{x}_1) & \phi_1(\mathbf{x}_2) & \cdots & \phi_1(\mathbf{x}_N) \\
\phi_2(\mathbf{x}_1) & \phi_2(\mathbf{x}_2) & \cdots & \phi_2(\mathbf{x}_N) \\
\vdots & \vdots & \ddots & \vdots \\
\phi_N(\mathbf{x}_1) & \phi_N(\mathbf{x}_2) & \cdots & \phi_N(\mathbf{x}_N)
\end{pmatrix}$$

**简写形式**：
$$\psi_{SD}(\mathbf{x}_1, \ldots, \mathbf{x}_N) = \frac{1}{\sqrt{N!}} \det[\phi_i(\mathbf{x}_j)]$$

其中：
- $\mathbf{x}_i = (\mathbf{r}_i, \sigma_i)$ 是电子 $i$ 的坐标（空间坐标 $\mathbf{r}_i$ + 自旋 $\sigma_i$）
- $\phi_i(\mathbf{x})$ 是第 $i$ 个单电子轨道（自旋轨道）
- $\frac{1}{\sqrt{N!}}$ 是归一化因子

##### 2.2 行列式展开（两电子例子）

**两电子系统**（如He原子）：
$$\psi_{SD}(\mathbf{x}_1, \mathbf{x}_2) = \frac{1}{\sqrt{2}} \det\begin{pmatrix}
\phi_1(\mathbf{x}_1) & \phi_1(\mathbf{x}_2) \\
\phi_2(\mathbf{x}_1) & \phi_2(\mathbf{x}_2)
\end{pmatrix}$$

**展开行列式**：
$$\psi_{SD}(\mathbf{x}_1, \mathbf{x}_2) = \frac{1}{\sqrt{2}} \left[\phi_1(\mathbf{x}_1)\phi_2(\mathbf{x}_2) - \phi_1(\mathbf{x}_2)\phi_2(\mathbf{x}_1)\right]$$

**观察**：
- 第一项：$\phi_1(\mathbf{x}_1)\phi_2(\mathbf{x}_2)$（电子1在轨道1，电子2在轨道2）
- 第二项：$-\phi_1(\mathbf{x}_2)\phi_2(\mathbf{x}_1)$（电子1在轨道2，电子2在轨道1，带负号）
- 这自动包含了**交换项**，保证了反对称性

##### 2.3 三电子系统例子

$$\psi_{SD}(\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3) = \frac{1}{\sqrt{6}} \det\begin{pmatrix}
\phi_1(\mathbf{x}_1) & \phi_1(\mathbf{x}_2) & \phi_1(\mathbf{x}_3) \\
\phi_2(\mathbf{x}_1) & \phi_2(\mathbf{x}_2) & \phi_2(\mathbf{x}_3) \\
\phi_3(\mathbf{x}_1) & \phi_3(\mathbf{x}_2) & \phi_3(\mathbf{x}_3)
\end{pmatrix}$$

**展开**（包含 $3! = 6$ 项）：
$$\psi_{SD} = \frac{1}{\sqrt{6}} \left[
\begin{aligned}
&\phi_1(\mathbf{x}_1)\phi_2(\mathbf{x}_2)\phi_3(\mathbf{x}_3) \\
- &\phi_1(\mathbf{x}_1)\phi_2(\mathbf{x}_3)\phi_3(\mathbf{x}_2) \\
- &\phi_1(\mathbf{x}_2)\phi_2(\mathbf{x}_1)\phi_3(\mathbf{x}_3) \\
+ &\phi_1(\mathbf{x}_2)\phi_2(\mathbf{x}_3)\phi_3(\mathbf{x}_1) \\
+ &\phi_1(\mathbf{x}_3)\phi_2(\mathbf{x}_1)\phi_3(\mathbf{x}_2) \\
- &\phi_1(\mathbf{x}_3)\phi_2(\mathbf{x}_2)\phi_3(\mathbf{x}_1)
\end{aligned}
\right]$$

**符号规律**：
- 每一项对应一个电子排列
- 符号由排列的**奇偶性**决定（偶排列为正，奇排列为负）

#### 三、数学性质

##### 3.1 反对称性（核心性质）

**定理**：Slater行列式自动满足反对称性。

**证明**：
交换电子 $i$ 和 $j$，相当于交换行列式的第 $i$ 列和第 $j$ 列：
$$\psi_{SD}(\ldots, \mathbf{x}_i, \ldots, \mathbf{x}_j, \ldots) = \frac{1}{\sqrt{N!}} \det[\phi_k(\mathbf{x}_l)]$$

交换列后：
$$\psi_{SD}(\ldots, \mathbf{x}_j, \ldots, \mathbf{x}_i, \ldots) = \frac{1}{\sqrt{N!}} \det[\phi_k(\mathbf{x}_l')]$$

其中 $\mathbf{x}_l'$ 是交换后的坐标。

由于行列式交换两列改变符号：
$$\det[\phi_k(\mathbf{x}_l')] = -\det[\phi_k(\mathbf{x}_l)]$$

因此：
$$\psi_{SD}(\ldots, \mathbf{x}_j, \ldots, \mathbf{x}_i, \ldots) = -\psi_{SD}(\ldots, \mathbf{x}_i, \ldots, \mathbf{x}_j, \ldots)$$

**✓ 反对称性得证！**

##### 3.2 泡利不相容原理

**定理**：如果两个电子处于相同的轨道，Slater行列式为零。

**证明**：
如果 $\phi_i = \phi_j$（$i \neq j$），那么行列式有两行相同：
$$\det\begin{pmatrix}
\vdots & \vdots & \vdots \\
\phi_i(\mathbf{x}_1) & \phi_i(\mathbf{x}_2) & \cdots \\
\vdots & \vdots & \vdots \\
\phi_i(\mathbf{x}_1) & \phi_i(\mathbf{x}_2) & \cdots \\
\vdots & \vdots & \vdots
\end{pmatrix} = 0$$

因为行列式有两行相同，其值为零。

**物理意义**：
- 两个电子不能处于完全相同的量子态
- 这自动实现了**泡利不相容原理**

##### 3.3 归一化

**定理**：如果轨道是正交归一的，Slater行列式自动归一化。

**证明**：
$$\langle\psi_{SD}|\psi_{SD}\rangle = \int |\psi_{SD}(\mathbf{x}_1, \ldots, \mathbf{x}_N)|^2 d\mathbf{x}_1 \cdots d\mathbf{x}_N$$

对于正交归一的轨道：$\int \phi_i^*(\mathbf{x}) \phi_j(\mathbf{x}) d\mathbf{x} = \delta_{ij}$

可以证明（使用行列式的性质）：
$$\langle\psi_{SD}|\psi_{SD}\rangle = \frac{1}{N!} \sum_{P,Q} (-1)^{P+Q} \prod_{i=1}^N \int \phi_{P(i)}^*(\mathbf{x}_i) \phi_{Q(i)}(\mathbf{x}_i) d\mathbf{x}_i$$

由于轨道正交归一，只有当 $P = Q$ 时项才非零，且每个这样的项贡献为1。共有 $N!$ 个排列，因此：
$$\langle\psi_{SD}|\psi_{SD}\rangle = \frac{1}{N!} \cdot N! = 1$$

**✓ 归一化得证！**

##### 3.4 轨道正交归一条件

为了保证Slater行列式归一化，轨道必须满足：
$$\int \phi_i^*(\mathbf{x}) \phi_j(\mathbf{x}) d\mathbf{x} = \delta_{ij}$$

这保证了轨道之间是正交归一的。

#### 四、具体例子：He原子

##### 4.1 基态He原子

**电子数**：$N = 2$

**轨道选择**：
- $\phi_1(\mathbf{x}) = 1s(\mathbf{r}) \alpha(\sigma)$（1s轨道，自旋上）
- $\phi_2(\mathbf{x}) = 1s(\mathbf{r}) \beta(\sigma)$（1s轨道，自旋下）

**Slater行列式**：
$$\psi_{He}(\mathbf{x}_1, \mathbf{x}_2) = \frac{1}{\sqrt{2}} \det\begin{pmatrix}
1s(\mathbf{r}_1)\alpha(\sigma_1) & 1s(\mathbf{r}_2)\alpha(\sigma_2) \\
1s(\mathbf{r}_1)\beta(\sigma_1) & 1s(\mathbf{r}_2)\beta(\sigma_2)
\end{pmatrix}$$

**展开**：
$$\psi_{He} = \frac{1}{\sqrt{2}} \left[1s(\mathbf{r}_1)\alpha(\sigma_1) \cdot 1s(\mathbf{r}_2)\beta(\sigma_2) - 1s(\mathbf{r}_2)\alpha(\sigma_2) \cdot 1s(\mathbf{r}_1)\beta(\sigma_1)\right]$$

**物理意义**：
- 两个电子都在1s轨道上
- 但自旋相反（一个上，一个下）
- 这满足泡利原理：虽然空间部分相同，但自旋不同

##### 4.2 激发态He原子

**轨道选择**：
- $\phi_1(\mathbf{x}) = 1s(\mathbf{r}) \alpha(\sigma)$
- $\phi_2(\mathbf{x}) = 2s(\mathbf{r}) \alpha(\sigma)$（注意：两个都是自旋上！）

**Slater行列式**：
$$\psi_{He^*} = \frac{1}{\sqrt{2}} \left[1s(\mathbf{r}_1)\alpha(\sigma_1) \cdot 2s(\mathbf{r}_2)\alpha(\sigma_2) - 2s(\mathbf{r}_1)\alpha(\sigma_1) \cdot 1s(\mathbf{r}_2)\alpha(\sigma_2)\right]$$

**物理意义**：
- 一个电子在1s轨道，另一个在2s轨道
- 两个都是自旋上（这是允许的，因为空间部分不同）

#### 五、实际应用

##### 5.1 Hartree-Fock方法

**核心**：使用**单个Slater行列式**作为试探波函数：
$$\psi_{HF} = \frac{1}{\sqrt{N!}} \det[\phi_i(\mathbf{x}_j)]$$

**优化**：通过变分原理优化轨道 $\{\phi_i\}$，使得能量最小。

**优点**：
- 自动满足反对称性
- 自动满足泡利原理
- 计算相对简单

**缺点**：
- 只包含一个Slater行列式
- 忽略了电子相关（动态相关）

##### 5.2 组态相互作用（CI）

**核心**：使用**多个Slater行列式**的线性组合：
$$\psi_{CI} = \sum_I c_I \psi_{SD}^{(I)}$$

其中每个 $\psi_{SD}^{(I)}$ 是不同的Slater行列式（不同的轨道占据）。

**例子**：
- **参考组态**：$\psi_{SD}^{(0)} = \det[1s\alpha, 1s\beta]$（基态）
- **单激发**：$\psi_{SD}^{(1)} = \det[1s\alpha, 2s\alpha]$（一个电子激发）
- **双激发**：$\psi_{SD}^{(2)} = \det[2s\alpha, 2s\beta]$（两个电子都激发）

**优点**：
- 可以包含电子相关
- 可以描述激发态

**缺点**：
- 需要大量Slater行列式
- 计算复杂度高

##### 5.3 耦合簇方法（CC）

**核心**：使用**指数算符**作用在参考Slater行列式上：
$$\psi_{CC} = e^{\hat{T}} \psi_{SD}^{(0)}$$

其中 $\hat{T}$ 是激发算符，展开后包含多个Slater行列式。

**为什么用指数算符？**（简要）数学上：分离系统时 $e^{\hat{T}_A + \hat{T}_B} = e^{\hat{T}_A} e^{\hat{T}_B}$，自动满足大小一致性；指数展开用少量振幅（$\hat{T}$ 的系数）自动生成高激发（如 $\hat{T}_2^2$ 给出四激发），参数少而表达能力强。物理上：指数形式对应"连接关联"的乘积结构，符合多体理论中能量与波函数可分解的图像；$e^{\hat{T}}$ 可理解为对参考态的"完全相关化"。详细推导见第 6.2 节"耦合簇方法"及其中"3.6 指数算符的数学与物理意义"。

##### 5.4 全组态相互作用（FCI）

**核心**：包含**所有可能的Slater行列式**：
$$\psi_{FCI} = \sum_{\text{所有可能的占据}} c_I \psi_{SD}^{(I)}$$

在基组完备时，FCI给出精确解。

#### 六、Slater行列式的优缺点总结

##### 优点：
1. **自动满足反对称性**：行列式天然反对称
2. **自动满足泡利原理**：相同轨道使行列式为零
3. **数学简洁**：行列式有成熟的理论和计算方法
4. **物理直观**：每个轨道对应一个电子（在HF中）

##### 缺点：
1. **单行列式限制**：一个Slater行列式只能描述平均场，不能描述电子相关
2. **组合爆炸**：多个行列式时，数量呈指数增长
3. **基组依赖**：结果依赖于基组的选择

#### 七、重要说明

**轨道就是函数**：
- $\phi_i(\mathbf{x})$ 是单电子波函数（轨道），是坐标和自旋的函数
- 在量子化学中，"轨道"和"单电子波函数"是同义词
- 这些轨道可以用基组 $\{\chi_\mu(\mathbf{r})\}$ 展开：
  $$\phi_i(\mathbf{x}) = \sum_{\mu=1}^M c_{i\mu} \chi_\mu(\mathbf{r}) \alpha(\sigma) \quad \text{或} \quad \phi_i(\mathbf{x}) = \sum_{\mu=1}^M c_{i\mu} \chi_\mu(\mathbf{r}) \beta(\sigma)$$
  其中 $\alpha(\sigma)$ 和 $\beta(\sigma)$ 是自旋函数

### 4.2 Hartree-Fock能量泛函

#### 能量表达式
对于单Slater行列式，能量为：
$$E_{HF}[\{\phi_i\}] = \sum_{i=1}^N h_i + \frac{1}{2}\sum_{i,j=1}^N (J_{ij} - K_{ij})$$

其中：
- **单电子积分**：
  $$h_i = \int \phi_i^*(\mathbf{x}) \hat{h}(\mathbf{x}) \phi_i(\mathbf{x}) d\mathbf{x}$$
  
  其中 $\hat{h}(\mathbf{x}) = -\frac{1}{2}\nabla^2 - \sum_A \frac{Z_A}{|\mathbf{r} - \mathbf{R}_A|}$

- **库仑积分**：
  $$J_{ij} = \int \int \frac{|\phi_i(\mathbf{x}_1)|^2 |\phi_j(\mathbf{x}_2)|^2}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{x}_1 d\mathbf{x}_2$$

- **交换积分**：
  $$K_{ij} = \int \int \frac{\phi_i^*(\mathbf{x}_1) \phi_j(\mathbf{x}_1) \phi_j^*(\mathbf{x}_2) \phi_i(\mathbf{x}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{x}_1 d\mathbf{x}_2$$

#### 能量公式的详细推导

**一、能量期望值的定义**

对于单Slater行列式波函数 $|\Phi_0\rangle$，能量期望值为：
$$E_{HF} = \langle\Phi_0|\hat{H}|\Phi_0\rangle$$

其中电子哈密顿量为：
$$\hat{H} = \sum_{i=1}^N \hat{h}(\mathbf{x}_i) + \frac{1}{2}\sum_{i\neq j}^N \frac{1}{|\mathbf{r}_i - \mathbf{r}_j|}$$

**二、单电子项的期望值**

对于单电子算符，由于Slater行列式的反对称性，每个占据轨道贡献相同：
$$\langle\Phi_0|\sum_{i=1}^N \hat{h}(\mathbf{x}_i)|\Phi_0\rangle = \sum_{i=1}^N \int \phi_i^*(\mathbf{x}) \hat{h}(\mathbf{x}) \phi_i(\mathbf{x}) d\mathbf{x} = \sum_{i=1}^N h_i$$

**三、双电子项的期望值（两电子例子）**

对于两电子系统，Slater行列式为：
$$\Phi_0(\mathbf{x}_1, \mathbf{x}_2) = \frac{1}{\sqrt{2}}[\phi_1(\mathbf{x}_1)\phi_2(\mathbf{x}_2) - \phi_1(\mathbf{x}_2)\phi_2(\mathbf{x}_1)]$$

计算双电子积分：
$$\langle\Phi_0|\frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|}|\Phi_0\rangle = \frac{1}{2}\int \int [\phi_1^*(\mathbf{x}_1)\phi_2^*(\mathbf{x}_2) - \phi_1^*(\mathbf{x}_2)\phi_2^*(\mathbf{x}_1)] \frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|} [\phi_1(\mathbf{x}_1)\phi_2(\mathbf{x}_2) - \phi_1(\mathbf{x}_2)\phi_2(\mathbf{x}_1)] d\mathbf{x}_1 d\mathbf{x}_2$$

展开后得到4项：
1. **库仑项**：$\int \int \frac{|\phi_1(\mathbf{x}_1)|^2 |\phi_2(\mathbf{x}_2)|^2}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{x}_1 d\mathbf{x}_2 = J_{12}$
2. **交换项**：$-\int \int \frac{\phi_1^*(\mathbf{x}_1)\phi_2(\mathbf{x}_1)\phi_2^*(\mathbf{x}_2)\phi_1(\mathbf{x}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{x}_1 d\mathbf{x}_2 = -K_{12}$
3. **交换项**：$-\int \int \frac{\phi_1^*(\mathbf{x}_2)\phi_2(\mathbf{x}_2)\phi_2^*(\mathbf{x}_1)\phi_1(\mathbf{x}_1)}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{x}_1 d\mathbf{x}_2 = -K_{12}$
4. **库仑项**：$\int \int \frac{|\phi_1(\mathbf{x}_2)|^2 |\phi_2(\mathbf{x}_1)|^2}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{x}_1 d\mathbf{x}_2 = J_{12}$

合并：$\frac{1}{2}[J_{12} - K_{12} - K_{12} + J_{12}] = J_{12} - K_{12}$

**四、推广到N电子系统**

对于N电子系统，考虑所有电子对 $(i,j)$：
$$\langle\Phi_0|\frac{1}{2}\sum_{i\neq j}^N \frac{1}{|\mathbf{r}_i - \mathbf{r}_j|}|\Phi_0\rangle = \frac{1}{2}\sum_{i,j=1}^N (J_{ij} - K_{ij})$$

**注意**：当 $i = j$ 时，$J_{ii} = K_{ii}$，所以 $J_{ii} - K_{ii} = 0$，因此 $\sum_{i,j}$ 和 $\sum_{i\neq j}$ 等价。

**五、物理解释**

1. **单电子项** $\sum_i h_i$：每个电子的动能和核-电子相互作用
2. **库仑项** $\frac{1}{2}\sum_{i,j} J_{ij}$：电子之间的经典库仑排斥（使能量增加）
3. **交换项** $-\frac{1}{2}\sum_{i,j} K_{ij}$：费米子统计导致的交换能（使能量降低，稳定化）

**为什么有因子 $\frac{1}{2}$？**
- 每个电子对 $(i,j)$ 被计算了两次，但 $J_{ij} = J_{ji}$ 和 $K_{ij} = K_{ji}$，所以需要除以2。

**为什么交换项是负的？**
- 交换能是稳定化效应，由于泡利不相容原理，相同自旋的电子避免出现在同一位置，降低了库仑排斥。

### 4.3 Hartree-Fock方程的变分推导

（注：本节详细介绍Hartree-Fock方程的变分推导过程）

#### 一、变分问题的提出

##### 1.1 优化问题

**目标**：找到最优的轨道 $\{\phi_i\}$，使得Hartree-Fock能量最小：
$$\min_{\{\phi_i\}} E_{HF}[\{\phi_i\}] = \sum_{i=1}^N h_i + \frac{1}{2}\sum_{i,j=1}^N (J_{ij} - K_{ij})$$

**约束条件**：轨道必须正交归一
$$\int \phi_i^*(\mathbf{x}) \phi_j(\mathbf{x}) d\mathbf{x} = \delta_{ij}, \quad \forall i,j$$

##### 1.2 为什么需要约束？

**物理原因**：
- 轨道必须正交归一，这是量子力学的基本要求
- 保证波函数的归一化
- 保证不同轨道之间的独立性

**数学原因**：
- 如果没有约束，可以任意缩放轨道来降低能量（这是非物理的）
- 约束确保我们找到物理上有意义的解

#### 二、拉格朗日乘数法

##### 2.1 基本思想

**无约束优化**：$\min f(\mathbf{x})$
- 条件：$\nabla f = 0$

**有约束优化**：$\min f(\mathbf{x})$，约束 $g(\mathbf{x}) = 0$
- 不能直接令 $\nabla f = 0$（可能违反约束）
- **拉格朗日乘数法**：构造拉格朗日函数
  $$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) - \lambda g(\mathbf{x})$$
- 条件：$\nabla_{\mathbf{x}} \mathcal{L} = 0$ 和 $\nabla_{\lambda} \mathcal{L} = 0$

##### 2.2 应用到Hartree-Fock问题

**目标函数**：$E_{HF}[\{\phi_i\}]$

**约束函数**：$g_{ij}[\{\phi_i\}] = \int \phi_i^* \phi_j d\mathbf{x} - \delta_{ij} = 0$

**拉格朗日函数**：
$$\mathcal{L}[\{\phi_i\}] = E_{HF}[\{\phi_i\}] - \sum_{i,j} \lambda_{ij} \left(\int \phi_i^* \phi_j d\mathbf{x} - \delta_{ij}\right)$$

其中 $\lambda_{ij}$ 是拉格朗日乘数（待定常数）。

**为什么是 $\sum_{i,j}$？**
- 有 $N$ 个轨道，需要 $N \times N$ 个约束条件
- 每个约束对应一个拉格朗日乘数 $\lambda_{ij}$

#### 三、变分推导的详细步骤

##### 3.1 拉格朗日函数的展开

**完整形式**：
$$\mathcal{L}[\{\phi_i\}] = \sum_{i=1}^N h_i + \frac{1}{2}\sum_{i,j=1}^N (J_{ij} - K_{ij}) - \sum_{i,j} \lambda_{ij} \left(\int \phi_i^* \phi_j d\mathbf{x} - \delta_{ij}\right)$$

**展开各项**：
- 单电子项：$\sum_i h_i = \sum_i \int \phi_i^* \hat{h} \phi_i d\mathbf{x}$
- 库仑项：$\frac{1}{2}\sum_{i,j} J_{ij} = \frac{1}{2}\sum_{i,j} \int \int \frac{|\phi_i(\mathbf{x}_1)|^2 |\phi_j(\mathbf{x}_2)|^2}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{x}_1 d\mathbf{x}_2$
- 交换项：$-\frac{1}{2}\sum_{i,j} K_{ij} = -\frac{1}{2}\sum_{i,j} \int \int \frac{\phi_i^*(\mathbf{x}_1)\phi_j(\mathbf{x}_1)\phi_j^*(\mathbf{x}_2)\phi_i(\mathbf{x}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{x}_1 d\mathbf{x}_2$
- 约束项：$-\sum_{i,j} \lambda_{ij} \left(\int \phi_i^* \phi_j d\mathbf{x} - \delta_{ij}\right)$

##### 3.2 对 $\phi_k^*$ 的变分

**变分原理**：最优解满足
$$\frac{\delta \mathcal{L}}{\delta \phi_k^*} = 0, \quad \forall k$$

**关键技巧**：使用泛函导数（functional derivative）

**步骤1：单电子项的变分**

$$\frac{\delta}{\delta \phi_k^*} \sum_i \int \phi_i^* \hat{h} \phi_i d\mathbf{x} = \hat{h} \phi_k(\mathbf{x})$$

**推导**：
- 只有 $i = k$ 的项依赖于 $\phi_k^*$
- $\frac{\delta}{\delta \phi_k^*} \int \phi_k^* \hat{h} \phi_k d\mathbf{x} = \hat{h} \phi_k(\mathbf{x})$

**步骤2：库仑项的变分**

$$\frac{\delta}{\delta \phi_k^*} \frac{1}{2}\sum_{i,j} \int \int \frac{|\phi_i(\mathbf{x}_1)|^2 |\phi_j(\mathbf{x}_2)|^2}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{x}_1 d\mathbf{x}_2$$

**分析**：
- 当 $i = k$ 时：$\frac{\delta}{\delta \phi_k^*} |\phi_k(\mathbf{x}_1)|^2 = \phi_k(\mathbf{x}_1)$
- 贡献：$\sum_j \int \frac{|\phi_j(\mathbf{x}_2)|^2}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{x}_2 \phi_k(\mathbf{x}_1) = \sum_j \hat{J}_j(\mathbf{x}_1) \phi_k(\mathbf{x}_1)$

其中库仑算符定义为：
$$\hat{J}_j(\mathbf{x}_1) = \int \frac{|\phi_j(\mathbf{x}_2)|^2}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{x}_2$$

**结果**：
$$\frac{\delta}{\delta \phi_k^*} \frac{1}{2}\sum_{i,j} J_{ij} = \sum_j \hat{J}_j(\mathbf{x}) \phi_k(\mathbf{x})$$

**步骤3：交换项的变分**

$$\frac{\delta}{\delta \phi_k^*} \left(-\frac{1}{2}\sum_{i,j} \int \int \frac{\phi_i^*(\mathbf{x}_1)\phi_j(\mathbf{x}_1)\phi_j^*(\mathbf{x}_2)\phi_i(\mathbf{x}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{x}_1 d\mathbf{x}_2\right)$$

**分析**：
- 当 $i = k$ 时：$\frac{\delta}{\delta \phi_k^*} \phi_k^*(\mathbf{x}_1)\phi_k(\mathbf{x}_2) = \phi_k(\mathbf{x}_2)$
- 贡献：$-\sum_j \int \frac{\phi_j(\mathbf{x}_1)\phi_j^*(\mathbf{x}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{x}_2 \phi_k(\mathbf{x}_1) = -\sum_j \hat{K}_j(\mathbf{x}_1) \phi_k(\mathbf{x}_1)$

其中交换算符定义为：
$$\hat{K}_j(\mathbf{x}_1)\phi_k(\mathbf{x}_1) = \int \frac{\phi_j^*(\mathbf{x}_2)\phi_k(\mathbf{x}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{x}_2 \phi_j(\mathbf{x}_1)$$

**注意**：交换算符是**非局域**的（依赖于 $\phi_k$ 在 $\mathbf{x}_2$ 的值）。

**结果**：
$$\frac{\delta}{\delta \phi_k^*} \left(-\frac{1}{2}\sum_{i,j} K_{ij}\right) = -\sum_j \hat{K}_j(\mathbf{x}) \phi_k(\mathbf{x})$$

**步骤4：约束项的变分**

$$\frac{\delta}{\delta \phi_k^*} \left(-\sum_{i,j} \lambda_{ij} \int \phi_i^* \phi_j d\mathbf{x}\right) = -\sum_j \lambda_{kj} \phi_j(\mathbf{x})$$

**推导**：
- 当 $i = k$ 时，$\frac{\delta}{\delta \phi_k^*} \int \phi_k^* \phi_j d\mathbf{x} = \phi_j(\mathbf{x})$
- 贡献：$-\sum_j \lambda_{kj} \phi_j(\mathbf{x})$

##### 3.3 合并所有项

**变分方程**：
$$\frac{\delta \mathcal{L}}{\delta \phi_k^*} = \hat{h}(\mathbf{x}) \phi_k(\mathbf{x}) + \sum_j \hat{J}_j(\mathbf{x}) \phi_k(\mathbf{x}) - \sum_j \hat{K}_j(\mathbf{x}) \phi_k(\mathbf{x}) - \sum_j \lambda_{kj} \phi_j(\mathbf{x}) = 0$$

**定义Fock算符**：
$$\hat{F}_k(\mathbf{x}) = \hat{h}(\mathbf{x}) + \sum_j \left[\hat{J}_j(\mathbf{x}) - \hat{K}_j(\mathbf{x})\right]$$

**变分方程变为**：
$$\hat{F}_k(\mathbf{x}) \phi_k(\mathbf{x}) - \sum_j \lambda_{kj} \phi_j(\mathbf{x}) = 0$$

**注意**：$\hat{F}_k$ 依赖于所有占据轨道 $\{\phi_j\}$（因为 $\hat{J}_j$ 和 $\hat{K}_j$ 依赖于 $\phi_j$）。

#### 四、正则Hartree-Fock方程

##### 4.1 对角化拉格朗日乘数矩阵

**问题**：$\lambda_{ij}$ 矩阵不是对角的，方程耦合。

**解决方案**：通过酉变换对角化 $\lambda_{ij}$。

**关键观察**：
- 轨道可以任意酉变换而不改变Slater行列式（只改变表示）
- 我们可以选择使 $\lambda_{ij}$ 对角的表示

**结果**：在最优表示中，$\lambda_{ij} = \epsilon_i \delta_{ij}$（对角矩阵）

##### 4.2 正则Hartree-Fock方程

**对角化后**：
$$\hat{F}_k(\mathbf{x}) \phi_k(\mathbf{x}) - \epsilon_k \phi_k(\mathbf{x}) = 0$$

**进一步简化**：
- 对于所有占据轨道，Fock算符相同（因为都依赖于所有占据轨道）
- 可以写成统一形式：

$$\hat{F}(\mathbf{x}) \phi_i(\mathbf{x}) = \epsilon_i \phi_i(\mathbf{x}), \quad i = 1, 2, \ldots, N$$

其中：
$$\hat{F}(\mathbf{x}) = \hat{h}(\mathbf{x}) + \sum_{j=1}^N \left[\hat{J}_j(\mathbf{x}) - \hat{K}_j(\mathbf{x})\right]$$

**物理意义**：
- $\hat{F}$ 是**有效单电子哈密顿量**
- $\epsilon_i$ 是**轨道能量**（本征值）
- $\phi_i$ 是**轨道**（本征函数）

#### 五、Fock算符的详细形式

##### 5.1 库仑算符 $\hat{J}_j$

**定义**：
$$\hat{J}_j(\mathbf{x}_1) = \int \frac{|\phi_j(\mathbf{x}_2)|^2}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{x}_2$$

**物理意义**：
- 电子在轨道 $j$ 中产生的**平均库仑势**
- 这是**局域**的（只依赖于 $\mathbf{r}_1$）
- 类似于经典静电势

**作用**：
$$\hat{J}_j(\mathbf{x}_1) \phi_i(\mathbf{x}_1) = \left[\int \frac{|\phi_j(\mathbf{x}_2)|^2}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{x}_2\right] \phi_i(\mathbf{x}_1)$$

##### 5.2 交换算符 $\hat{K}_j$

**定义**：
$$\hat{K}_j(\mathbf{x}_1)\phi_i(\mathbf{x}_1) = \int \frac{\phi_j^*(\mathbf{x}_2)\phi_i(\mathbf{x}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{x}_2 \phi_j(\mathbf{x}_1)$$

**物理意义**：
- 由于费米子统计（反对称性）导致的**交换势**
- 这是**非局域**的（依赖于 $\phi_i$ 在 $\mathbf{x}_2$ 的值）
- 没有经典对应

**关键特性**：
- 交换算符是**积分算符**（不是乘法算符）
- 依赖于被作用的轨道 $\phi_i$

#### 六、自洽场（SCF）方法

##### 6.1 为什么需要迭代？

**问题**：Fock算符 $\hat{F}$ 依赖于轨道 $\{\phi_i\}$，而轨道是我们要找的！

**解决方案**：自洽迭代

##### 6.2 SCF迭代过程

1. **初始化**：猜测初始轨道 $\{\phi_i^{(0)}\}$

2. **构建Fock算符**：
   $$\hat{F}^{(n)}(\mathbf{x}) = \hat{h}(\mathbf{x}) + \sum_{j=1}^N \left[\hat{J}_j^{(n)}(\mathbf{x}) - \hat{K}_j^{(n)}(\mathbf{x})\right]$$
   其中 $\hat{J}_j^{(n)}$ 和 $\hat{K}_j^{(n)}$ 用当前轨道 $\{\phi_i^{(n)}\}$ 计算

3. **求解本征值问题**：
   $$\hat{F}^{(n)} \phi_i^{(n+1)} = \epsilon_i^{(n+1)} \phi_i^{(n+1)}$$
   得到新的轨道 $\{\phi_i^{(n+1)}\}$ 和轨道能量 $\{\epsilon_i^{(n+1)}\}$

4. **检查收敛**：
   - 轨道变化：$\max_i \|\phi_i^{(n+1)} - \phi_i^{(n)}\| < \epsilon$
   - 能量变化：$|E_{HF}^{(n+1)} - E_{HF}^{(n)}| < \epsilon$

5. **重复**：如果不收敛，回到步骤2

##### 6.3 收敛性

**为什么能收敛？**
- 每次迭代，能量单调下降（变分原理）
- 有下界（基态能量），所以必须收敛

**可能的问题**：
- 收敛到局部最优（不是全局最优）
- 振荡（需要阻尼）
- 不收敛（需要更好的初始猜测）

#### 七、总结

**变分推导的核心思想**：
1. **优化问题**：最小化能量，约束正交归一
2. **拉格朗日乘数法**：将有约束优化转化为无约束优化
3. **变分原理**：对轨道变分，得到最优条件
4. **对角化**：通过酉变换简化方程
5. **自洽迭代**：因为Fock算符依赖于轨道

**最终结果**：
$$\hat{F}(\mathbf{x}) \phi_i(\mathbf{x}) = \epsilon_i \phi_i(\mathbf{x})$$

这是Hartree-Fock方程，描述了最优轨道必须满足的条件。

**为什么Fock方程给出M个轨道？**

Fock方程 $\hat{F}\phi_i = \epsilon_i\phi_i$ 是一个本征值问题：
- Fock算符 $\hat{F}$ 在 $M$ 维基组空间中作用
- 因此有 $M$ 个本征值和对应的 $M$ 个本征函数（轨道）
- 本征值 $\epsilon_i$ 称为轨道能量（orbital energy）
- 按照能量排序：$\epsilon_1 \leq \epsilon_2 \leq \cdots \leq \epsilon_M$

**占据轨道的选择**：
- 能量最低的 $N$ 个轨道被占据（根据Aufbau原理）
- 这些是**占据轨道**（occupied orbitals）：$\phi_1, \phi_2, \ldots, \phi_N$
- 剩余的 $M-N$ 个是**虚轨道**（virtual orbitals）：$\phi_{N+1}, \phi_{N+2}, \ldots, \phi_M$
- 虚轨道能量通常为正值，表示电子激发到这些轨道需要能量

（关于占据轨道和虚轨道的详细解释，参见2.1节"为什么需要构造M个轨道"部分）

## 5. 电子相关性的数学描述

### 5.1 相关能定义

#### 精确相关能
$$E_{corr} = E_{exact} - E_{HF}$$

其中 $E_{exact}$ 是精确基态能量，$E_{HF}$ 是Hartree-Fock能量。

### 5.2 相关性的来源

#### 库仑相关（Coulomb Correlation）
电子之间的瞬时库仑排斥导致：
- **动态相关**：电子避免同时出现在同一空间区域
- **静态相关**：近简并态之间的相关

#### 为什么不能写成乘积形式：深入分析（电子相关角度）

**注意**：波函数不能写成简单乘积有**两个独立的原因**：
1. **反对称性要求**（见4.1节）：费米子波函数必须反对称，Slater行列式解决了这个问题
2. **电子相关**（本节讨论）：即使使用Slater行列式，单个行列式仍不能精确描述电子相关

本节讨论第二个原因——电子相关。

**简单乘积形式的失败**：

考虑最简单的两电子系统（如He原子）。如果波函数可以写成：
$$\psi(\mathbf{r}_1, \mathbf{r}_2) = \phi_1(\mathbf{r}_1) \phi_2(\mathbf{r}_2)$$

那么概率密度为：
$$|\psi(\mathbf{r}_1, \mathbf{r}_2)|^2 = |\phi_1(\mathbf{r}_1)|^2 |\phi_2(\mathbf{r}_2)|^2$$

这意味着：
- 电子1在 $\mathbf{r}_1$ 的概率：$P_1(\mathbf{r}_1) = |\phi_1(\mathbf{r}_1)|^2$
- 电子2在 $\mathbf{r}_2$ 的概率：$P_2(\mathbf{r}_2) = |\phi_2(\mathbf{r}_2)|^2$
- **联合概率**：$P(\mathbf{r}_1, \mathbf{r}_2) = P_1(\mathbf{r}_1) P_2(\mathbf{r}_2)$（独立！）

**但实际情况**：
- 由于库仑排斥 $\frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|}$，两个电子**避免**同时出现在同一位置
- 如果电子1在 $\mathbf{r}_1$，电子2在 $\mathbf{r}_1$ 附近的概率会**降低**
- 因此：$P(\mathbf{r}_1, \mathbf{r}_2) \neq P_1(\mathbf{r}_1) P_2(\mathbf{r}_2)$（相关！）

**数学证明（两电子系统）**：

如果 $\psi(\mathbf{r}_1, \mathbf{r}_2) = \phi_1(\mathbf{r}_1)\phi_2(\mathbf{r}_2)$ 是精确解，那么：
$$\hat{H}\psi = \left[\hat{h}_1 + \hat{h}_2 + \frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|}\right] \phi_1(\mathbf{r}_1)\phi_2(\mathbf{r}_2) = E \phi_1(\mathbf{r}_1)\phi_2(\mathbf{r}_2)$$

展开左边：
$$\hat{h}_1\phi_1 \cdot \phi_2 + \phi_1 \cdot \hat{h}_2\phi_2 + \frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|} \phi_1(\mathbf{r}_1)\phi_2(\mathbf{r}_2) = E \phi_1(\mathbf{r}_1)\phi_2(\mathbf{r}_2)$$

**矛盾**：
- 左边第三项 $\frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|} \phi_1(\mathbf{r}_1)\phi_2(\mathbf{r}_2)$ **不能**写成 $f_1(\mathbf{r}_1) f_2(\mathbf{r}_2)$ 的形式
- 因为 $\frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|}$ 同时依赖于两个坐标
- 但右边是乘积形式
- 因此等式**不可能**成立（除非 $\frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|}$ 项为零，但这是不可能的）

**结论**：精确波函数必须包含显式的电子-电子相关项。

**Hartree-Fock的近似**：

HF方法使用**单Slater行列式**（关于Slater行列式的详细解释，参见4.1节）：
$$\psi_{HF}(\mathbf{r}_1, \mathbf{r}_2) = \frac{1}{\sqrt{2}} \det\begin{pmatrix} \phi_1(\mathbf{r}_1) & \phi_1(\mathbf{r}_2) \\ \phi_2(\mathbf{r}_1) & \phi_2(\mathbf{r}_2) \end{pmatrix} = \frac{1}{\sqrt{2}}[\phi_1(\mathbf{r}_1)\phi_2(\mathbf{r}_2) - \phi_1(\mathbf{r}_2)\phi_2(\mathbf{r}_1)]$$

这**不是**简单乘积，而是**反对称化的乘积**（行列式形式）：
- 包含了交换项：$\phi_1(\mathbf{r}_2)\phi_2(\mathbf{r}_1)$
- 这捕获了**费米子统计**（交换相关）
- 但**仍然忽略了库仑相关**（动态相关）

**精确波函数需要更多项**：

#### 数学描述
精确波函数不能写成单Slater行列式，必须展开为多个行列式的线性组合：
$$\psi_{exact} = c_0 \psi_{HF} + \sum_{i,a} c_i^a \psi_i^a + \sum_{i<j,a<b} c_{ij}^{ab} \psi_{ij}^{ab} + \cdots$$

其中：
- $\psi_i^a$：单激发（一个电子从占据轨道 $i$ 激发到虚轨道 $a$）
- $\psi_{ij}^{ab}$：双激发（两个电子同时激发）
- 更高激发项...

**为什么需要这些项？**
- **单激发**：调整单个电子的分布以响应其他电子
- **双激发**：捕获两个电子同时移动的相关性
- **高激发**：捕获多个电子的联合相关

**物理图像**：
- 如果电子1移动到位置 $\mathbf{r}_1$，电子2会"感知"到并调整其分布
- 这种相关性需要多个Slater行列式来描述
- 单个Slater行列式（HF）只能描述平均场，不能描述瞬时相关

### 5.3 相关能的大小

#### 典型数值
- 小分子：相关能通常为总能量的 0.5-2%
- 但绝对值可能很大（几十到几百 kcal/mol）
- 化学键能主要由相关能贡献

#### 尺度分析
- **库仑能**：$O(N^2)$，但通过密度可以降低到 $O(N)$
- **交换能**：$O(N^2)$
- **相关能**：难以精确计算，需要多体方法

## 6. 后Hartree-Fock方法

### 6.1 组态相互作用（Configuration Interaction, CI）

#### 数学框架

##### 波函数展开
将精确波函数展开为多个Slater行列式的线性组合：
$$|\Psi_{CI}\rangle = \sum_I c_I |\Phi_I\rangle$$

其中 $|\Phi_I\rangle$ 是不同电子组态的Slater行列式，$c_I$ 是展开系数。

（注：关于将薛定谔方程投影到组态空间的详细推导，参见1.3节"将薛定谔方程投影到组态空间"部分）

##### 组态分类
- **参考组态**：$|\Phi_0\rangle$（通常是HF基态）
- **单激发**：$|\Phi_i^a\rangle$（一个电子从占据轨道 $i$ 激发到虚轨道 $a$）
- **双激发**：$|\Phi_{ij}^{ab}\rangle$
- **三激发、四激发**：等等

##### 截断级别
- **CIS**：只包含单激发（用于激发态）
- **CID**：只包含双激发
- **CISD**：单激发 + 双激发
- **CISDT**：单、双、三激发
- **FCI**：全组态相互作用（包含所有可能的激发）

#### 矩阵方程

##### 本征值问题
将薛定谔方程投影到组态空间：
$$\mathbf{H}\mathbf{c} = E\mathbf{c}$$

其中：
- $\mathbf{H}_{IJ} = \langle\Phi_I|\hat{H}|\Phi_J\rangle$ 是哈密顿矩阵
- $\mathbf{c} = (c_0, c_1, \ldots)^T$ 是系数向量
- $E$ 是能量本征值

（注：详细的投影推导参见1.3节）

##### 矩阵元计算
使用Slater-Condon规则计算矩阵元：
- **对角元**：$\langle\Phi_I|\hat{H}|\Phi_I\rangle$ 可以通过轨道能量和双电子积分计算
- **非对角元**：只有当两个行列式相差不超过两个轨道时，矩阵元才非零

（注：Slater-Condon规则的详细说明参见1.3节"矩阵元的计算"部分）

#### 大小一致性（Size Consistency）

##### 定义

**大小一致性**：若系统 $A + B$ 由两个**无相互作用**的片段 $A$、$B$ 组成（例如两分子相距无穷远），则方法应满足：
$$E(A+B) = E(A) + E(B)$$

且基态波函数应为两片段基态的张量积：$|\Psi_{A+B}\rangle = |\Psi_A\rangle \otimes |\Psi_B\rangle$。

##### 为什么 CISD 不满足大小一致性：完整数学推导

**1. 分离系统的精确波函数与哈密顿量**

设片段 $A$ 有 $N_A$ 个电子、$M_A$ 个自旋轨道，片段 $B$ 有 $N_B$ 个电子、$M_B$ 个自旋轨道；$A$ 与 $B$ 无相互作用时，总哈密顿量为：
$$\hat{H}_{A+B} = \hat{H}_A \otimes \hat{I}_B + \hat{I}_A \otimes \hat{H}_B$$

此时基态为**乘积态**：
$$|\Psi_{A+B}^{exact}\rangle = |\Psi_A^{exact}\rangle \otimes |\Psi_B^{exact}\rangle$$

总能量为：
$$E_{exact}(A+B) = E_{exact}(A) + E_{exact}(B)$$

**2. CISD 的组态空间**

对**单个**片段 $A$，CISD 波函数为：
$$|\Psi_A^{CISD}\rangle = c_0^A |\Phi_0^A\rangle + \sum_{i,a} c_i^{a,A} |\Phi_i^{a,A}\rangle + \sum_{i<j,a<b} c_{ij}^{ab,A} |\Phi_{ij}^{ab,A}\rangle$$

即参考组态 + 所有单激发 + 所有双激发（**无三激发、四激发**）。对 $B$ 同理。

**3. 复合系统 $A+B$ 的 CISD 空间**

复合系统的 CISD 空间定义为：参考 $|\Phi_0^{A+B}\rangle = |\Phi_0^A\rangle \otimes |\Phi_0^B\rangle$，加上**所有相对于该参考的单激发与双激发**。

关键点：在 $A+B$ 中，"双激发"是指**在整个大系统中只激发 2 个电子**（可来自 $A$、可来自 $B$、或各一个）。因此 $A+B$ 的 CISD 空间**包含**：
- 参考；
- 单激发（1 个电子在 $A$ 或 $B$ 中激发）；
- 双激发（2 个电子都在 $A$ 中激发，或都在 $B$ 中激发，或 1 个在 $A$、1 个在 $B$）。

**4. 两片段各自双激发的乘积不在 CISD 空间内**

若 $|\Psi_A^{CISD}\rangle$ 含有双激发项 $c_{ij}^{ab,A}|\Phi_{ij}^{ab,A}\rangle$，$|\Psi_B^{CISD}\rangle$ 含有双激发项 $c_{kl}^{cd,B}|\Phi_{kl}^{cd,B}\rangle$，则乘积态中会出现：
$$|\Phi_{ij}^{ab,A}\rangle \otimes |\Phi_{kl}^{cd,B}\rangle$$

这一项表示：在 $A$ 中有 2 个电子被激发，在 $B$ 中也有 2 个电子被激发，**总共 4 个电子被激发**，即相对于 $|\Phi_0^{A+B}\rangle$ 而言是一个**四激发**组态。

CISD 的定义是**只包含单激发和双激发**，不包含三激发、四激发。因此上述四激发组态**不在** $A+B$ 的 CISD 空间内。于是：
$$\text{CISD 空间}(A+B) \not\supset \operatorname{span}\bigl\{ |\Psi_A\rangle \otimes |\Psi_B\rangle : |\Psi_A\rangle, |\Psi_B\rangle \in \text{CISD 空间} \bigr\}$$

即"$A$ 的 CISD 波函数"与"$B$ 的 CISD 波函数"的**任意乘积**所张成的空间，并不完全落在 $A+B$ 的 CISD 空间里（缺四激发部分）。

**5. 能量不等式**

设 $|\Psi_{A+B}^{CISD}\rangle$ 是 $A+B$ 在 CISD 空间内得到的基态（能量最小）。由于 CISD 空间不包含"$A$ 双激发 × $B$ 双激发"这类四激发项，一般有：
$$|\Psi_{A+B}^{CISD}\rangle \neq |\Psi_A^{CISD}\rangle \otimes |\Psi_B^{CISD}\rangle$$

由变分原理，CISD 能量是精确能量的上界，故：
$$E_{CISD}(A+B) \geq E_{exact}(A+B) = E_{exact}(A) + E_{exact}(B)$$

另一方面，$|\Psi_A^{CISD}\rangle \otimes |\Psi_B^{CISD}\rangle$ 是 $A+B$ 的一个**试探态**，但它不在 CISD 空间内（因为含有四激发分量），所以**不能**作为 CISD 的优化解。在 CISD 约束下优化得到的 $E_{CISD}(A+B)$ 通常**严格大于** $E_{CISD}(A) + E_{CISD}(B)$，即：
$$E_{CISD}(A+B) > E_{CISD}(A) + E_{CISD}(B)$$

**数值示例**：两个远离的 He 原子，$E_{CISD}(A) = E_{CISD}(B) = E_{CISD}(\text{He})$。若大小一致，应有 $E_{CISD}(\text{He}_2) = 2 E_{CISD}(\text{He})$。实际计算中 $E_{CISD}(\text{He}_2)$ 会**高于** $2 E_{CISD}(\text{He})$，多出的部分即大小不一致误差。

**6. 小结（CI）**

| 内容 | 结论 |
|------|------|
| 精确 $A+B$ 波函数 | $|\Psi_A\rangle \otimes |\Psi_B\rangle$（乘积态） |
| CISD 空间 | 仅单、双激发，无四激发 |
| 乘积态中的四激发 | $A$ 中双激发 × $B$ 中双激发 = 四激发，不在 CISD 内 |
| 能量 | $E_{CISD}(A+B) > E_{CISD}(A) + E_{CISD}(B)$ |

##### 解决方案
- **CISDTQ**：显式加入四激发，使空间包含"双×双"型组态，可恢复大小一致性（代价为计算量大幅增加）。
- **CC方法**：通过指数算符自动生成高激发，乘积形式 $e^{\hat{T}_A}e^{\hat{T}_B}$ 自然满足大小一致性（见下节）。



### 6.2 耦合簇方法（Coupled Cluster, CC）

（注：本节详细介绍耦合簇方法的思想、数学推导与求解流程。）

#### 一、基本思想与动机

##### 1.1 核心问题

如何用**单个**参考 Slater 行列式 $|\Phi_0\rangle$（通常为 HF 基态）构造包含电子相关的波函数，并使其在增大系统时具有正确的**大小一致性**（见 6.1 节）？

##### 1.2 CI 的局限为何难以接受

CI 波函数为线性组合：
$$|\Psi_{CI}\rangle = c_0|\Phi_0\rangle + \sum_{i,a} c_i^a|\Phi_i^a\rangle + \sum_{i<j,a<b} c_{ij}^{ab}|\Phi_{ij}^{ab}\rangle + \cdots$$

- **截断到 CISD** 时：不包含四激发，导致 $E_{CISD}(A+B) > E_{CISD}(A) + E_{CISD}(B)$，即大小不一致。
- **若包含到 FCI**：组态数 $\binom{M}{N}$ 随体系指数增长，计算不可行。
- **本质**：线性组合中各组态系数 $c_I$ 彼此独立；截断即强制部分 $c_I=0$，无法自然得到“分离时能量相加”的乘积结构。

##### 1.3 CC 的出发点：用算符代替系数

**思路**：不直接优化各组态系数，而是引入**激发算符** $\hat{T}$，令波函数为：
$$|\Psi_{CC}\rangle = e^{\hat{T}} |\Phi_0\rangle, \quad \hat{T} = \hat{T}_1 + \hat{T}_2 + \hat{T}_3 + \cdots$$

- $\hat{T}_1$：单激发算符（振幅 $t_i^a$）
- $\hat{T}_2$：双激发算符（振幅 $t_{ij}^{ab}$）
- 更高 $\hat{T}_k$ 类推。

**为何用指数**：指数展开 $e^{\hat{T}} = \hat{I} + \hat{T} + \frac{1}{2}\hat{T}^2 + \cdots$ 会**自动**生成高激发（如 $\hat{T}_2^2$ 产生四激发），且对分离系统有 $e^{\hat{T}_A+\hat{T}_B}=e^{\hat{T}_A}e^{\hat{T}_B}$，从而**自动满足大小一致性**（详见本节“大小一致性证明”）。

##### 1.4 从薛定谔方程到 CC 方程

将 $|\Psi_{CC}\rangle = e^{\hat{T}}|\Phi_0\rangle$ 代入薛定谔方程 $\hat{H}|\Psi\rangle = E|\Psi\rangle$：
$$\hat{H} e^{\hat{T}} |\Phi_0\rangle = E\, e^{\hat{T}} |\Phi_0\rangle$$

左乘 $e^{-\hat{T}}$（因 $e^{-\hat{T}}e^{\hat{T}}=\hat{I}$）得**相似变换形式**：
$$e^{-\hat{T}} \hat{H} e^{\hat{T}} |\Phi_0\rangle = E |\Phi_0\rangle$$

定义**相似变换哈密顿量**：
$$\bar{H} = e^{-\hat{T}} \hat{H} e^{\hat{T}}$$

则上式变为 $\bar{H}|\Phi_0\rangle = E|\Phi_0\rangle$，即**参考态 $|\Phi_0\rangle$ 是 $\bar{H}$ 的本征态，本征值为 $E$**。后续的能量公式与振幅方程都由此出发，对 $\bar{H}$ 在参考态与激发态上的投影得到（见第四节）。

#### 二、激发算符的详细定义

##### 2.1 二次量子化记号与约定

**产生/湮灭算符**：
- $a_p^\dagger$：在自旋轨道 $p$ 上产生一个电子
- $a_p$：在自旋轨道 $p$ 上湮灭一个电子

**反对易关系**：
$$\{a_p, a_q^\dagger\} = \delta_{pq}, \quad \{a_p, a_q\} = 0, \quad \{a_p^\dagger, a_q^\dagger\} = 0$$

**占据与虚轨道**：在 HF 参考态 $|\Phi_0\rangle$ 下，被占据的自旋轨道指标记为 $i,j,k,\ldots$（占据轨道），未被占据的记为 $a,b,c,\ldots$（虚轨道）。总自旋轨道数记为 $M$，电子数 $N$。

##### 2.2 单激发算符 $\hat{T}_1$

**定义**：
$$\hat{T}_1 = \sum_{i \in occ} \sum_{a \in virt} t_i^a \, a_a^\dagger a_i$$

- $i$：占据轨道；$a$：虚轨道。
- $a_a^\dagger a_i$：从 $i$ 湮灭一个电子，在 $a$ 产生一个电子，即“单激发”。
- $t_i^a$：单激发振幅（待求实或复数参数）。

**作用**：$\hat{T}_1 |\Phi_0\rangle$ 是**所有单激发 Slater 行列式**的线性组合，系数为 $t_i^a$。

**为何没有 $1/2$ 等因子**：$(i,a)$ 与 $(a,i)$ 在求和中是不同指标对，每个单激发只出现一次，故无需对称化因子。

##### 2.3 双激发算符 $\hat{T}_2$

**定义**：
$$\hat{T}_2 = \frac{1}{4} \sum_{ij} \sum_{ab} t_{ij}^{ab} \, a_a^\dagger a_b^\dagger a_j a_i$$

**$\frac{1}{4}$ 因子的来源**：在无约束求和 $\sum_{i,j,a,b}$ 中，$(i,j)$ 与 $(j,i)$、$(a,b)$ 与 $(b,a)$ 各算一次，但费米子算符满足 $a_i^\dagger a_j^\dagger = -a_j^\dagger a_i^\dagger$，故同一物理双激发会以不同顺序出现 4 次。约定振幅对称化：$t_{ij}^{ab} = -t_{ji}^{ab} = -t_{ij}^{ba} = t_{ji}^{ba}$，并限制求和为 $i<j,\,a<b$ 时可写为：
$$\hat{T}_2 = \sum_{i<j} \sum_{a<b} t_{ij}^{ab} \, (a_a^\dagger a_b^\dagger a_j a_i - a_a^\dagger a_b^\dagger a_i a_j - \cdots)$$
等价地，在**无约束**求和下用系数 $\frac{1}{4}$ 避免同一组态被重复计数 4 次，从而与“独立振幅个数”一致。

**作用**：$\hat{T}_2 |\Phi_0\rangle$ 是**所有双激发组态** $|\Phi_{ij}^{ab}\rangle$ 的线性组合，系数由 $t_{ij}^{ab}$ 及反对称化给出；双激发是电子相关的主要来源。

#### 2.4 更高激发算符

**三激发算符**：
$$\hat{T}_3 = \frac{1}{36}\sum_{i,j,k,a,b,c} t_{ijk}^{abc} \hat{a}_a^\dagger \hat{a}_b^\dagger \hat{a}_c^\dagger \hat{a}_k \hat{a}_j \hat{a}_i$$

**四激发算符**：
$$\hat{T}_4 = \frac{1}{576}\sum_{i,j,k,l,a,b,c,d} t_{ijkl}^{abcd} \hat{a}_a^\dagger \hat{a}_b^\dagger \hat{a}_c^\dagger \hat{a}_d^\dagger \hat{a}_l \hat{a}_k \hat{a}_j \hat{a}_i$$

#### 三、指数算符的展开和作用

##### 3.1 指数算符的定义与在组态空间中的形式

**CC 波函数**：
$$|\Psi_{CC}\rangle = e^{\hat{T}} |\Phi_0\rangle, \qquad \hat{T} = \hat{T}_1 + \hat{T}_2 + \hat{T}_3 + \cdots$$

形式上也可写成**组态的线性组合**（与 CI 类比）：
$$|\Psi_{CC}\rangle = |\Phi_0\rangle + \sum_{i,a} c_i^a |\Phi_i^a\rangle + \sum_{i<j,a<b} c_{ij}^{ab} |\Phi_{ij}^{ab}\rangle + \sum_{i<j<k,a<b<c} c_{ijk}^{abc} |\Phi_{ijk}^{abc}\rangle + \cdots$$

但这里的系数 $c_i^a,\,c_{ij}^{ab},\,c_{ijk}^{abc},\ldots$ **不是独立参数**：它们由 $\hat{T}$ 的振幅 $t_i^a,\,t_{ij}^{ab},\ldots$ 通过指数展开**代数地**给出（例如 $c_{ij}^{ab}$ 中含 $t_{ij}^{ab}$ 以及 $t_i^a t_j^b$ 等）。因此 CC 用**少量振幅**（$t$）生成**大量组态系数**（$c$），并自动满足大小一致性；而 CI 中每个 $c_I$ 都是独立参数。

##### 3.2 指数算符的级数展开

**泰勒展开**：
$$e^{\hat{T}} = \hat{I} + \hat{T} + \frac{1}{2!}\hat{T}^2 + \frac{1}{3!}\hat{T}^3 + \frac{1}{4!}\hat{T}^4 + \cdots$$

其中 $\hat{I}$ 是单位算符。

#### 3.3 指数算符作用在参考态上

**展开结果**：
$$|\Psi_{CC}\rangle = e^{\hat{T}} |\Phi_0\rangle = |\Phi_0\rangle + \hat{T}|\Phi_0\rangle + \frac{1}{2!}\hat{T}^2|\Phi_0\rangle + \frac{1}{3!}\hat{T}^3|\Phi_0\rangle + \cdots$$

**逐项分析**：

**第0项**：$|\Phi_0\rangle$
- 参考组态（HF基态）

**第1项**：$\hat{T}|\Phi_0\rangle = (\hat{T}_1 + \hat{T}_2 + \cdots)|\Phi_0\rangle$
- 单激发：$\hat{T}_1|\Phi_0\rangle = \sum_{i,a} t_i^a |\Phi_i^a\rangle$
- 双激发：$\hat{T}_2|\Phi_0\rangle = \sum_{i<j,a<b} t_{ij}^{ab} |\Phi_{ij}^{ab}\rangle$
- 等等...

**第2项**：$\frac{1}{2!}\hat{T}^2|\Phi_0\rangle = \frac{1}{2}(\hat{T}_1 + \hat{T}_2)^2|\Phi_0\rangle$
- 展开：$\frac{1}{2}(\hat{T}_1^2 + 2\hat{T}_1\hat{T}_2 + \hat{T}_2^2)|\Phi_0\rangle$
- **关键**：$\hat{T}_1^2$ 可以生成双激发（两个单激发）
- **关键**：$\hat{T}_2^2$ 可以生成四激发（两个双激发）

**第3项及更高项**：
- 包含更高阶的激发
- 例如：$\hat{T}_1^3$ 可以生成三激发

#### 3.4 具体例子：CCSD（只包含单激发和双激发）

**CCSD波函数**：
$$|\Psi_{CCSD}\rangle = e^{\hat{T}_1 + \hat{T}_2} |\Phi_0\rangle$$

**展开**：
$$|\Psi_{CCSD}\rangle = |\Phi_0\rangle + \hat{T}_1|\Phi_0\rangle + \hat{T}_2|\Phi_0\rangle + \frac{1}{2}\hat{T}_1^2|\Phi_0\rangle + \hat{T}_1\hat{T}_2|\Phi_0\rangle + \frac{1}{2}\hat{T}_2^2|\Phi_0\rangle + \cdots$$

**各项的Slater行列式内容**：

1. **$|\Phi_0\rangle$**：参考组态（0激发）

2. **$\hat{T}_1|\Phi_0\rangle$**：所有单激发组态
   - $|\Phi_i^a\rangle$：电子从轨道 $i$ 激发到 $a$

3. **$\hat{T}_2|\Phi_0\rangle$**：所有双激发组态
   - $|\Phi_{ij}^{ab}\rangle$：电子从轨道 $i,j$ 激发到 $a,b$

4. **$\frac{1}{2}\hat{T}_1^2|\Phi_0\rangle$**：通过两个单激发生成的双激发
   - 例如：$\hat{T}_1$ 将电子1从 $i$ 激发到 $a$，再 $\hat{T}_1$ 将电子2从 $j$ 激发到 $b$
   - 结果：双激发 $|\Phi_{ij}^{ab}\rangle$

5. **$\hat{T}_1\hat{T}_2|\Phi_0\rangle$**：单激发和双激发的组合
   - 可以生成三激发

6. **$\frac{1}{2}\hat{T}_2^2|\Phi_0\rangle$**：两个双激发的组合
   - 可以生成四激发

**关键观察**：
- 即使只包含 $\hat{T}_1$ 和 $\hat{T}_2$，指数展开会**自动生成**更高激发
- 这是CC方法比CI方法更强大的原因之一

#### 3.5 为什么指数形式能表示更精确的波函数？

**1. 自动包含高激发**：
- CI方法：需要显式包含所有需要的激发
- CC方法：指数展开自动生成高激发
- 例如：CCSD自动包含四激发（通过 $\hat{T}_2^2$）

**2. 大小一致性**：
- CI方法：CISD不是大小一致的
- CC方法：指数形式自动保证大小一致性

**3. 更紧凑的表示**：
- 用较少的参数（振幅）可以表示更多的组态
- 通过指数展开，参数数量是多项式的，但组态数量是指数的

**4. 物理意义**：
- 激发算符 $\hat{T}$ 可以理解为"相关算符"
- 指数形式 $e^{\hat{T}}$ 表示"完全相关化"
- 类似于统计物理中的配分函数

#### 3.6 指数算符的数学与物理意义

下面从数学和物理两方面说明**为什么用指数算符** $e^{\hat{T}}$ 而不是线性组合。

##### 数学上为什么用指数

**（1）分离系统时变成乘积：大小一致性**

对两个互不作用的子系统 $A$ 和 $B$，各自的激发算符只作用在各自电子上，因此**对易**：$[\hat{T}_A, \hat{T}_B] = 0$。于是：
$$e^{\hat{T}_{A+B}} = e^{\hat{T}_A + \hat{T}_B} = e^{\hat{T}_A} e^{\hat{T}_B}$$

即总波函数是两子系统的波函数形式"乘在一起"（各自 $e^{\hat{T}}$ 作用各自的参考态再张量积），能量自然满足 $E_{CC}(A+B) = E_{CC}(A) + E_{CC}(B)$，即**大小一致性**。

若用 CI 的线性形式 $|\Psi_{CI}\rangle = c_0|\Phi_0\rangle + \sum c_i^a|\Phi_i^a\rangle + \cdots$，对 $A+B$ 必须在"大空间"里展开；若只做到 CISD，不会自动拆成两边的 CISD 之和，必须显式加入四激发等才能修正。指数形式则一次保证：只要两边各自用各自的 $\hat{T}$，大系统自然呈乘积。

**（2）用少量参数生成高激发**

指数展开 $e^{\hat{T}} = \hat{I} + \hat{T} + \frac{1}{2!}\hat{T}^2 + \cdots$ 作用在 $|\Phi_0\rangle$ 上时，$\hat{T}|\Phi_0\rangle$ 给出单、双激发等；$\frac{1}{2}\hat{T}^2|\Phi_0\rangle$ 中 $\hat{T}_2^2$ 自动给出**四激发**；更高次项自动给出六激发、八激发等。因此只优化 $t_i^a$、$t_{ij}^{ab}$ 等有限个振幅，波函数中却已包含由它们"乘积"出来的高激发。即：**参数是多项式数量**（$\hat{T}$ 的振幅），通过指数生成**指数多的组态**，且高激发的系数由低激发振幅代数地决定，不必再独立拟合。这样比 CISD 多包含了高激发，又比 FCI 省参数。

##### 物理上为什么用指数

**（1）"连接关联"的乘积结构（linked cluster）**

多体理论中，有物理意义的是**连接（linked）**的关联；若系统可拆成不相互作用的两块 $A$ 和 $B$，总能量应为 $E_A + E_B$，波函数应为两边波函数的张量积。若波函数写成 $|\Psi\rangle = e^{\hat{T}}|\Phi_0\rangle$，且 $\hat{T}$ 只包含**连接型**激发（如 $\hat{T}_2$ 对应"一对电子一起激发"的连通图），则能量和波函数仅由这些连接振幅决定；**非连接**部分（如 $A$ 中一个双激发与 $B$ 中一个双激发独立）会自然以乘积 $e^{\hat{T}_A}e^{\hat{T}_B}$ 出现，不破坏 $E(A+B)=E(A)+E(B)$。因此指数形式对应：用"连接在一起"的关联（$\hat{T}$）作为基本量，再通过指数自动生成所有"多块关联的乘积"，既满足大小一致性，又符合"相关可分解"的物理图像。

**（2）"完全相关化"的直观**

$e^{\hat{T}}$ 可理解为：用算符 $\hat{T}$ 对参考态 $|\Phi_0\rangle$ 做**完全相关化**。$\hat{T}$ 代表单次、双次激发等的叠加，$e^{\hat{T}}$ 则是"把这些激发以所有可能方式叠加"。统计物理中配分函数或波函数也常写成指数形式，对应所有涨落/激发的累积。因此指数 = 把 $\hat{T}$ 所代表的关联以一致的方式全部作用上去。

##### 与 CI 的对比小结

| 方面 | CI：$c_0|\Phi_0\rangle + \sum c_I|\Phi_I\rangle$ | CC：$e^{\hat{T}}|\Phi_0\rangle$ |
|------|--------------------------------------------------|----------------------------------|
| 形式 | 线性组合 | 指数算符作用参考态 |
| 高激发 | 需显式加 CISDT、CISDTQ 等 | $\hat{T}^2,\hat{T}^3$ 自动生成 |
| 大小一致性 | CISD 不满足，需显式补四激发等 | $e^{\hat{T}_A+\hat{T}_B}=e^{\hat{T}_A}e^{\hat{T}_B}$ 自动满足 |
| 参数 | 每个组态一个系数 | 只优化 $\hat{T}$ 的振幅，更紧凑 |
| 物理 | 各组态权重独立 | 关联以"连接振幅 + 指数"组织 |

#### 四、投影方程和求解

##### 4.1 能量方程从何而来

由 $e^{-\hat{T}}\hat{H}e^{\hat{T}}|\Phi_0\rangle = E|\Phi_0\rangle$，左乘 $\langle\Phi_0|$ 并利用 $\langle\Phi_0|\Phi_0\rangle=1$，得：
$$E_{CC} = \langle\Phi_0| e^{-\hat{T}}\hat{H}e^{\hat{T}} |\Phi_0\rangle$$

因此 **CC 能量** 就是相似变换哈密顿量 $\bar{H} = e^{-\hat{T}}\hat{H}e^{\hat{T}}$ 在参考态上的期望值。

**为何用相似变换**：
- $e^{-\hat{T}}e^{\hat{T}}=\hat{I}$，故 $\bar{H}$ 与 $\hat{H}$ **本征值相同**（相似变换不改变谱）。
- 将 $\hat{H}$ 换成 $\bar{H}$ 后，本征态从 $e^{\hat{T}}|\Phi_0\rangle$ 变为 $|\Phi_0\rangle$，在参考态上的期望值即能量，便于与振幅方程在同一套“参考态 + 激发态”基下写出。

**Baker-Campbell-Hausdorff（BCH）展开**：
$$e^{-\hat{T}}\hat{H}e^{\hat{T}} = \hat{H} + [\hat{H}, \hat{T}] + \frac{1}{2!}[[\hat{H}, \hat{T}], \hat{T}] + \frac{1}{3!}[[[\hat{H}, \hat{T}], \hat{T}], \hat{T}] + \cdots$$

- $\hat{H}$ 是单、双电子算符之和；$\hat{T}$ 是单、双激发算符之和。
- 对易子 $[\hat{H}, \hat{T}]$ 仍由单、双电子与单、双激发组合，产生有限多种算符类型。
- 逐次对易后，**CCSD** 下（仅 $\hat{T}_1,\hat{T}_2$）该级数在有限步后**必然截断**：例如 $[[[\hat{H},\hat{T}_2],\hat{T}_2],\hat{T}_2],\hat{T}_2]$ 中会出现超过双电子的算符，在 Fock 空间矩阵元中为零。因此实际计算中 BCH 只需算到有限项（CCSD 常到 $\hat{T}_2^4$ 量级）。

##### 4.2 振幅方程从何而来

$\bar{H}|\Phi_0\rangle = E|\Phi_0\rangle$ 等价于：**$\bar{H}|\Phi_0\rangle$ 在任意与 $|\Phi_0\rangle$ 正交的态上分量为零**（即 $\bar{H}|\Phi_0\rangle$ 与 $|\Phi_0\rangle$ 平行）。故对任意激发组态 $|\Phi_I\rangle \neq |\Phi_0\rangle$，应有：
$$\langle\Phi_I| e^{-\hat{T}}\hat{H}e^{\hat{T}} |\Phi_0\rangle = 0$$

这就是 **CC 振幅方程**（投影方程）：激发通道 $I$ 上的“残差”为零。

**CCSD 时的具体形式**：
- **单激发**：$\langle\Phi_i^a| \bar{H} |\Phi_0\rangle = 0$，对所有 $(i,a)$。
- **双激发**：$\langle\Phi_{ij}^{ab}| \bar{H} |\Phi_0\rangle = 0$，对所有 $i<j,\,a<b$。

未知量为振幅 $t_i^a$、$t_{ij}^{ab}$；上述方程是**关于振幅的非线性方程组**（因 $\bar{H}$ 中含 $e^{-\hat{T}}\hat{H}e^{\hat{T}}$，展开后是 $t$ 的多项式）。

**物理含义**：要求相似变换后的哈密顿量 $\bar{H}$ 在参考态 $|\Phi_0\rangle$ 上的作用**没有单、双激发分量**，即 $|\Phi_0\rangle$ 是 $\bar{H}$ 在“参考+单+双”子空间内的本征态；通过调节 $t$ 使残差为零，即得到自洽的 CC 振幅。

##### 4.3 求解流程（迭代）

振幅方程无解析解，需**迭代**：

1. **初猜**：通常取 MP2 给出的双激发振幅 $t_{ij}^{ab}$，单激发 $t_i^a$ 取零（或从 HF 轨道的小修正得到）。
2. **第 $n$ 步**：
   - 用当前振幅 $t^{(n)}$ 构造 $\bar{H}$（或其在需要的 Slater 基上的矩阵元）。
   - 计算**残差**：
     - 单激发残差 $r_i^a = \langle\Phi_i^a|\bar{H}|\Phi_0\rangle$；
     - 双激发残差 $r_{ij}^{ab} = \langle\Phi_{ij}^{ab}|\bar{H}|\Phi_0\rangle$。
   - 通过“残差 → 振幅更新”的规则（源于线性化振幅方程或 Newton 步）得到 $t^{(n+1)}$。
3. **收敛判据**：$\max_I |r_I| < \epsilon$ 且/或 $|E^{(n+1)}-E^{(n)}| < \epsilon$。

**注意**：CC 能量 $E_{CC}$ 由 $\langle\Phi_0|\bar{H}|\Phi_0\rangle$ 给出，**不是**变分极小化得到的；因此 CC 能量可略低于真实基态能量（非变分）。振幅方程保证的是 $|\Phi_0\rangle$ 为 $\bar{H}$ 在所用激发空间内的本征态。

#### 五、截断级别

##### 截断级别
- **CCD**：$\hat{T} = \hat{T}_2$（只包含双激发）
  - 忽略单激发
  - 适用于闭壳层系统

- **CCSD**：$\hat{T} = \hat{T}_1 + \hat{T}_2$（单激发 + 双激发）
  - 最常用的CC方法
  - 自动包含四激发（通过 $\hat{T}_2^2$）

- **CCSDT**：包含三激发
  - 更高精度
  - 计算复杂度 $O(N^3 M^5)$

- **CCSDTQ**：包含四激发
  - 接近FCI精度
  - 计算复杂度 $O(N^4 M^6)$

#### 六、与CI方法的对比

| 方面 | CI方法 | CC方法 |
|------|--------|--------|
| **波函数形式** | 线性组合 | 指数形式 |
| **大小一致性** | ✗（CISD不是） | ✓（自动满足） |
| **高激发** | 需要显式包含 | 自动生成 |
| **参数效率** | 较低 | 较高 |
| **计算复杂度** | 类似 | 类似 |

#### 七、实际应用与截断选择

**CCSD**：
- 只含 $\hat{T}_1 + \hat{T}_2$，自动包含 $\hat{T}_2^2$ 等带来的四激发成分；计算量 $O(N^2 M^4)$。
- 适用于小到中等分子、单参考性较好的体系；能量误差常 < 1 kcal/mol，键长、振动频率也较可靠。

**CCSD(T)**（“金标准”）：
- 在 CCSD 基础上，用 **Møller-Plesset 型微扰** 加入**三激发 (T)** 的贡献，不显式求解 $\hat{T}_3$ 方程。
- 能量精度通常优于 CCSD，计算量 $O(N^7)$（随体系增大比 CCSD 贵很多）。
- 适用于需要高精度单点能、结合能、反应能垒等的单参考体系。

**更高截断**：
- **CCSDT**：显式包含 $\hat{T}_3$，$O(N^3 M^5)$，仅在必要时使用。
- **CCSDTQ**：含 $\hat{T}_4$，接近 FCI，仅用于极小体系或基准比较。

**使用限制**：
- 计算成本随电子数和基组增大迅速上升，通常限于中等大小、单参考占主导的分子。
- 强相关体系（多参考）需多参考 CC 或其它方法；CC 单参考方法在此时可能收敛差或精度不足。

#### 大小一致性证明（完整数学推导）

##### 1. 分离系统的设定

设系统 $A+B$ 由无相互作用片段 $A$、$B$ 组成：
$$\hat{H}_{A+B} = \hat{H}_A \otimes \hat{I}_B + \hat{I}_A \otimes \hat{H}_B$$

轨道与电子也按片段划分：$A$ 的轨道指标集合为 $\mathcal{O}_A$，电子在 $A$ 上；$B$ 的为 $\mathcal{O}_B$。参考态取为乘积态：
$$|\Phi_0^{A+B}\rangle = |\Phi_0^A\rangle \otimes |\Phi_0^B\rangle$$

##### 2. 激发算符的对易性 $[\hat{T}_A, \hat{T}_B] = 0$

$\hat{T}_A$ 只含 $A$ 的轨道的产生/湮灭算符（$i,a \in \mathcal{O}_A$），$\hat{T}_B$ 只含 $B$ 的轨道的产生/湮灭算符（$k,b \in \mathcal{O}_B$）。由于 $A$ 与 $B$ 的轨道互不相同，任意 $a_p^\dagger a_q$（$p,q \in \mathcal{O}_A$）与 $a_r^\dagger a_s$（$r,s \in \mathcal{O}_B$）**对易**（费米子算符作用在不同轨道上时对易）。因此 $\hat{T}_A$ 与 $\hat{T}_B$ 的每一项都对易，有：
$$\boxed{[\hat{T}_A, \hat{T}_B] = 0}$$

##### 3. 指数分解：$e^{\hat{T}_A + \hat{T}_B} = e^{\hat{T}_A} e^{\hat{T}_B}$

一般地，若两个算符 $\hat{X}$、$\hat{Y}$ 满足 $[\hat{X}, \hat{Y}] = 0$，则：
$$e^{\hat{X} + \hat{Y}} = e^{\hat{X}} e^{\hat{Y}}$$

**证明**：由 Baker-Campbell-Hausdorff 公式，$\ln(e^{\hat{X}}e^{\hat{Y}}) = \hat{X} + \hat{Y} + \frac{1}{2}[\hat{X},\hat{Y}] + \cdots$。当 $[\hat{X},\hat{Y}]=0$ 时，所有对易子项为零，故 $\ln(e^{\hat{X}}e^{\hat{Y}}) = \hat{X}+\hat{Y}$，即 $e^{\hat{X}}e^{\hat{Y}} = e^{\hat{X}+\hat{Y}}$。取 $\hat{X}=\hat{T}_A$，$\hat{Y}=\hat{T}_B$，即得：
$$e^{\hat{T}_{A+B}} = e^{\hat{T}_A + \hat{T}_B} = e^{\hat{T}_A} e^{\hat{T}_B}$$

（这里 $\hat{T}_{A+B}$ 表示复合系统的激发算符，在分离情形下等于 $\hat{T}_A + \hat{T}_B$，且只含 $A$ 的激发与 $B$ 的激发的直和。）

##### 4. CC 波函数的乘积形式

复合系统的 CC 波函数（截断到单双激发时，$\hat{T}_{A+B} = \hat{T}_A + \hat{T}_B$）为：
$$|\Psi_{CC}^{A+B}\rangle = e^{\hat{T}_A + \hat{T}_B} |\Phi_0^A\rangle \otimes |\Phi_0^B\rangle = e^{\hat{T}_A} e^{\hat{T}_B} |\Phi_0^A\rangle \otimes |\Phi_0^B\rangle$$

由于 $\hat{T}_B$ 只作用在 $B$ 的轨道上，$e^{\hat{T}_B}$ 对 $|\Phi_0^A\rangle$ 无影响（等价于恒等）；同理 $e^{\hat{T}_A}$ 对 $|\Phi_0^B\rangle$ 无影响。因此：
$$e^{\hat{T}_A} e^{\hat{T}_B} \bigl( |\Phi_0^A\rangle \otimes |\Phi_0^B\rangle \bigr) = \bigl( e^{\hat{T}_A} |\Phi_0^A\rangle \bigr) \otimes \bigl( e^{\hat{T}_B} |\Phi_0^B\rangle \bigr) = |\Psi_{CC}^A\rangle \otimes |\Psi_{CC}^B\rangle$$

即复合系统 CC 波函数为两片段 CC 波函数的**张量积**。

##### 5. 能量的可加性

能量期望值为：
$$E_{CC}(A+B) = \langle\Psi_{CC}^{A+B}|\hat{H}_{A+B}|\Psi_{CC}^{A+B}\rangle = \langle\Psi_{CC}^{A+B}| \bigl( \hat{H}_A \otimes \hat{I}_B + \hat{I}_A \otimes \hat{H}_B \bigr) |\Psi_{CC}^{A+B}\rangle$$

将 $|\Psi_{CC}^{A+B}\rangle = |\Psi_{CC}^A\rangle \otimes |\Psi_{CC}^B\rangle$ 代入，并利用 $\hat{H}_A$ 只作用在 $A$、$\hat{H}_B$ 只作用在 $B$，以及 $\langle\Psi_{CC}^A|\Psi_{CC}^A\rangle = \langle\Psi_{CC}^B|\Psi_{CC}^B\rangle = 1$（归一化），得：
$$E_{CC}(A+B) = \langle\Psi_{CC}^A|\hat{H}_A|\Psi_{CC}^A\rangle \cdot \langle\Psi_{CC}^B|\Psi_{CC}^B\rangle + \langle\Psi_{CC}^A|\Psi_{CC}^A\rangle \cdot \langle\Psi_{CC}^B|\hat{H}_B|\Psi_{CC}^B\rangle = E_{CC}(A) + E_{CC}(B)$$

因此：
$$\boxed{E_{CC}(A+B) = E_{CC}(A) + E_{CC}(B)}$$

CC 方法自动满足大小一致性。

##### 6. 与 CI 的对比（为何指数形式能自动包含“四激发”）

在 CC 中，$e^{\hat{T}_A} e^{\hat{T}_B}$ 展开后会出现例如 $\hat{T}_{2,A} \hat{T}_{2,B}$ 的项，作用在 $|\Phi_0^A\rangle \otimes |\Phi_0^B\rangle$ 上即产生“$A$ 中双激发 × $B$ 中双激发”的**四激发**型组态。这些项是由**指数乘积**自然生成的，不需要在 $\hat{T}$ 中显式加入四激发算符 $\hat{T}_4$。相反，CI 的波函数是线性组合 $c_0|\Phi_0\rangle + \sum c_I|\Phi_I\rangle$，若截断到 CISD，则四激发系数被强制为零，乘积态 $|\Psi_A^{CISD}\rangle \otimes |\Psi_B^{CISD}\rangle$ 无法被 CISD 空间表示，导致大小不一致。

#### 本节小结（6.2 耦合簇方法）

- **波函数**：$|\Psi_{CC}\rangle = e^{\hat{T}}|\Phi_0\rangle$，$\hat{T} = \hat{T}_1 + \hat{T}_2 + \cdots$；用振幅 $t_i^a,\,t_{ij}^{ab}$ 等参数化。
- **方程来源**：将 $e^{-\hat{T}}\hat{H}e^{\hat{T}}|\Phi_0\rangle = E|\Phi_0\rangle$ 对参考态投影得能量 $E = \langle\Phi_0|\bar{H}|\Phi_0\rangle$，对单/双激发态投影得振幅方程 $\langle\Phi_I|\bar{H}|\Phi_0\rangle=0$。
- **求解**：振幅方程非线性，用迭代（初猜常取 MP2）+ 残差收敛；能量由 $\langle\Phi_0|\bar{H}|\Phi_0\rangle$ 计算。
- **性质**：指数形式自动含高激发、大小一致；CC 能量非变分（可略低于真值）；与微扰结合得 CCSD(T) 等“金标准”方法。

#### 计算复杂度
- **CCSD**：$O(N^2 M^4)$，其中 $N$ 是占据轨道数，$M$ 是基函数数
- **CCSDT**：$O(N^3 M^5)$
- **CCSDTQ**：$O(N^4 M^6)$

### 6.3 Møller-Plesset微扰理论（MP）

#### 数学框架

##### 哈密顿量分解
$$\hat{H} = \hat{H}_0 + \lambda \hat{V}$$

其中：
- **零级哈密顿量**：$\hat{H}_0 = \sum_i \hat{F}_i$（Fock算符之和）
- **微扰**：$\hat{V} = \hat{H} - \hat{H}_0$

##### 微扰展开
能量和波函数按 $\lambda$ 展开：
$$E = E^{(0)} + \lambda E^{(1)} + \lambda^2 E^{(2)} + \lambda^3 E^{(3)} + \cdots$$
$$|\Psi\rangle = |\Psi^{(0)}\rangle + \lambda |\Psi^{(1)}\rangle + \lambda^2 |\Psi^{(2)}\rangle + \cdots$$

#### 各级修正

##### 零级（MP0）
$$E^{(0)} = \sum_i \epsilon_i$$
这是轨道能量之和，但重复计算了电子相互作用。

##### 一级（MP1）
$$E^{(1)} = \langle\Phi_0|\hat{V}|\Phi_0\rangle = E_{HF} - E^{(0)}$$

因此：
$$E_{MP1} = E_{HF}$$

##### 二级（MP2）
$$E^{(2)} = -\sum_{i<j,a<b} \frac{|\langle\Phi_0|\hat{V}|\Phi_{ij}^{ab}\rangle|^2}{\epsilon_a + \epsilon_b - \epsilon_i - \epsilon_j}$$

这是最重要的相关能修正。

##### 三级（MP3）和四级（MP4）
包含更复杂的项，计算复杂度急剧增加。

#### 收敛性

##### 问题
MP级数可能不收敛，特别是对于强相关系统。

##### 原因
- 微扰参数 $\lambda$ 可能不够小
- 零级波函数可能不是好的起点

##### 解决方案
- 使用多参考方法
- 使用CC方法（非微扰）



## 7. 密度泛函理论

### 7.0 DFT的核心思想：为什么用密度代替波函数？

#### 问题的提出

**波函数方法的困难**：
- $N$ 电子波函数：$\psi(\mathbf{r}_1, \mathbf{r}_2, \ldots, \mathbf{r}_N)$ 是 $3N$ 维函数
- 存储和计算复杂度随 $N$ 指数增长
- 例如：100个电子需要处理300维空间（不可行！）

**DFT的革命性思想**：
- 用**电子密度** $\rho(\mathbf{r})$ 代替波函数
- 密度只是3维函数：$\rho(\mathbf{r}) = \rho(x, y, z)$
- 复杂度从 $3N$ 维降到3维！

#### 电子密度的定义

$$\rho(\mathbf{r}) = N \int |\psi(\mathbf{r}, \mathbf{r}_2, \ldots, \mathbf{r}_N)|^2 d\mathbf{r}_2 \cdots d\mathbf{r}_N$$

**物理意义**：
- $\rho(\mathbf{r})$ 是在位置 $\mathbf{r}$ 找到**任意一个**电子的概率密度
- 归一化：$\int \rho(\mathbf{r}) d\mathbf{r} = N$（总电子数）

#### 关键问题

**能否只用密度就完全描述系统？**

直觉上似乎不行：
- 波函数包含所有量子信息
- 密度似乎丢失了很多信息（相位、相关性等）

**Hohenberg和Kohn的回答（1964年）**：可以！至少对于基态。

### 7.1 Hohenberg-Kohn定理：DFT的理论基础

#### 第一定理（存在性定理）

**定理**：外势 $v_{ext}(\mathbf{r})$ 由基态电子密度 $\rho_0(\mathbf{r})$ 唯一确定（除了一个常数）。

**通俗解释**：
- 给定一个基态密度 $\rho_0(\mathbf{r})$
- 就能唯一确定外势 $v_{ext}(\mathbf{r})$（即原子核的位置和电荷）
- 进而确定哈密顿量 $\hat{H}$
- 从而确定所有性质（能量、激发态等）

**意义**：密度包含了系统的全部信息！

##### 证明（反证法）

假设存在两个不同的外势 $v_{ext}^{(1)}$ 和 $v_{ext}^{(2)}$，它们给出相同的基态密度 $\rho_0$，但不同的基态波函数 $|\Psi_0^{(1)}\rangle$ 和 $|\Psi_0^{(2)}\rangle$。

对应的哈密顿量为：
$$\hat{H}^{(i)} = \hat{T} + \hat{V}_{ee} + \int v_{ext}^{(i)}(\mathbf{r}) \hat{\rho}(\mathbf{r}) d\mathbf{r}$$

基态能量为：
$$E_0^{(i)} = \langle\Psi_0^{(i)}|\hat{H}^{(i)}|\Psi_0^{(i)}\rangle$$

使用变分原理：
$$E_0^{(1)} < \langle\Psi_0^{(2)}|\hat{H}^{(1)}|\Psi_0^{(2)}\rangle = E_0^{(2)} + \int [v_{ext}^{(1)}(\mathbf{r}) - v_{ext}^{(2)}(\mathbf{r})] \rho_0(\mathbf{r}) d\mathbf{r}$$

同样：
$$E_0^{(2)} < E_0^{(1)} + \int [v_{ext}^{(2)}(\mathbf{r}) - v_{ext}^{(1)}(\mathbf{r})] \rho_0(\mathbf{r}) d\mathbf{r}$$

相加得到矛盾：$E_0^{(1)} + E_0^{(2)} < E_0^{(1)} + E_0^{(2)}$

因此，外势由密度唯一确定。

**链条关系**：
$$\rho_0(\mathbf{r}) \xrightarrow{\text{唯一确定}} v_{ext}(\mathbf{r}) \xrightarrow{\text{确定}} \hat{H} \xrightarrow{\text{确定}} \psi_0, E_0, \text{所有性质}$$

#### 第二定理（变分原理）

**定理**：对于任意试探密度 $\tilde{\rho}(\mathbf{r})$，如果它是 $v$-可表示的（即存在某个外势的基态密度），则：
$$E[\tilde{\rho}] \geq E_0$$

**通俗解释**：
- 能量是密度的泛函：$E = E[\rho]$
- 真实基态密度 $\rho_0$ 使能量最小
- 任何其他密度给出的能量都更高
- 这是DFT版本的变分原理

**能量泛函**：
$$E[\rho] = F[\rho] + \int v_{ext}(\mathbf{r}) \rho(\mathbf{r}) d\mathbf{r}$$

**普适泛函**（与系统无关，只依赖于密度）：
$$F[\rho] = T[\rho] + V_{ee}[\rho]$$

其中：
- $T[\rho]$：动能泛函（密度的泛函）
- $V_{ee}[\rho]$：电子-电子相互作用能泛函

**问题**：$F[\rho]$ 的精确形式未知！这是DFT的核心困难。

### 7.2 Kohn-Sham方法：DFT的实用方案

#### 直接DFT的困难

**问题**：如何从密度计算动能 $T[\rho]$？

**Thomas-Fermi模型**（1927年）尝试：
$$T_{TF}[\rho] = C_F \int \rho^{5/3}(\mathbf{r}) d\mathbf{r}$$

**失败原因**：
- 无法描述化学键（所有分子都不稳定！）
- 动能的局域近似太粗糙

#### Kohn-Sham的天才想法（1965年）

**核心思路**：引入一个**虚构的无相互作用系统**，它具有与真实系统**相同的密度**。

**为什么这样做？**
- 无相互作用系统的动能可以精确计算（通过轨道）
- 把"不知道如何计算"的部分集中到一个小项中

#### Kohn-Sham系统的构造

**虚构系统**：$N$ 个**无相互作用**的电子，在有效势 $v_{KS}(\mathbf{r})$ 中运动。

**哈密顿量**：
$$\hat{H}_{KS} = \sum_{i=1}^N \left[-\frac{1}{2}\nabla_i^2 + v_{KS}(\mathbf{r}_i)\right]$$

**关键要求**：选择 $v_{KS}(\mathbf{r})$ 使得无相互作用系统的密度等于真实系统的密度：
$$\rho_{KS}(\mathbf{r}) = \rho_{真实}(\mathbf{r})$$

**为什么无相互作用系统更容易处理？**
- 无相互作用 → 波函数可以写成单Slater行列式
- 动能可以精确计算：$T_s = \sum_i \langle\phi_i|-\frac{1}{2}\nabla^2|\phi_i\rangle$

#### 能量泛函的巧妙分解

**Kohn-Sham的分解策略**：把能量分成"可以精确计算的部分"和"需要近似的部分"。

$$E[\rho] = \underbrace{T_s[\rho]}_{\text{无相互作用动能}} + \underbrace{E_H[\rho]}_{\text{经典库仑能}} + \underbrace{E_{xc}[\rho]}_{\text{交换相关能}} + \underbrace{\int v_{ext}(\mathbf{r}) \rho(\mathbf{r}) d\mathbf{r}}_{\text{外势能}}$$

**各项详解**：

1. **无相互作用动能** $T_s[\rho]$：
   $$T_s[\rho] = \sum_{i=1}^N \langle\phi_i|-\frac{1}{2}\nabla^2|\phi_i\rangle = -\frac{1}{2}\sum_{i=1}^N \int \phi_i^*(\mathbf{r}) \nabla^2 \phi_i(\mathbf{r}) d\mathbf{r}$$
   - 这是**无相互作用**系统的动能（不是真实动能！）
   - 可以通过Kohn-Sham轨道精确计算

2. **Hartree能**（经典库仑能）$E_H[\rho]$：
   $$E_H[\rho] = \frac{1}{2}\int\int \frac{\rho(\mathbf{r})\rho(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|} d\mathbf{r} d\mathbf{r}'$$
   - 把电子当作经典电荷分布
   - 是电子-电子相互作用的**经典近似**
   - 可以从密度精确计算

3. **外势能**：
   $$E_{ext}[\rho] = \int v_{ext}(\mathbf{r}) \rho(\mathbf{r}) d\mathbf{r}$$
   - $v_{ext}(\mathbf{r}) = -\sum_A \frac{Z_A}{|\mathbf{r} - \mathbf{R}_A|}$（核-电子相互作用）
   - 可以从密度精确计算

4. **交换相关能** $E_{xc}[\rho]$（**关键！**）：
   $$E_{xc}[\rho] = \underbrace{(T[\rho] - T_s[\rho])}_{\text{动能修正}} + \underbrace{(V_{ee}[\rho] - E_H[\rho])}_{\text{非经典电子相互作用}}$$
   
   包含：
   - **动能修正**：真实动能与无相互作用动能之差
   - **交换能**：费米子统计导致的能量降低
   - **相关能**：电子相关导致的能量降低
   
   **这是DFT中唯一需要近似的部分！**

#### Kohn-Sham方程的推导

**变分原理**：最小化能量泛函 $E[\rho]$，约束密度归一化。

**结果**：Kohn-Sham方程（单电子方程）
$$\left[-\frac{1}{2}\nabla^2 + v_{eff}(\mathbf{r})\right] \phi_i(\mathbf{r}) = \epsilon_i \phi_i(\mathbf{r})$$

**有效势**：
$$v_{eff}(\mathbf{r}) = v_{ext}(\mathbf{r}) + v_H(\mathbf{r}) + v_{xc}(\mathbf{r})$$

各部分：
- **外势**：$v_{ext}(\mathbf{r}) = -\sum_A \frac{Z_A}{|\mathbf{r} - \mathbf{R}_A|}$（核-电子吸引）
- **Hartree势**：$v_H(\mathbf{r}) = \int \frac{\rho(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|} d\mathbf{r}'$（电子-电子经典排斥）
- **交换相关势**：$v_{xc}(\mathbf{r}) = \frac{\delta E_{xc}[\rho]}{\delta \rho(\mathbf{r})}$（交换相关泛函的泛函导数）

#### 密度的自洽计算

**密度由Kohn-Sham轨道给出**：
$$\rho(\mathbf{r}) = \sum_{i=1}^N |\phi_i(\mathbf{r})|^2$$

**自洽场（SCF）循环**：

```
1. 猜测初始密度 ρ(0)
2. 计算有效势 v_eff = v_ext + v_H[ρ] + v_xc[ρ]
3. 求解Kohn-Sham方程得到轨道 {φ_i}
4. 计算新密度 ρ_new = Σ|φ_i|²
5. 检查收敛：|ρ_new - ρ_old| < ε？
   - 是：结束
   - 否：用ρ_new更新，回到步骤2
```

**与Hartree-Fock的对比**：

| 方面 | Hartree-Fock | Kohn-Sham DFT |
|-----|-------------|---------------|
| 基本变量 | 轨道 $\{\phi_i\}$ | 密度 $\rho(\mathbf{r})$ |
| 交换 | 精确（非局域） | 近似（通常局域） |
| 相关 | 忽略 | 包含（近似） |
| 计算复杂度 | $O(N^4)$ | $O(N^3)$-$O(N^4)$ |

### 7.3 交换相关泛函：DFT的核心近似

#### 为什么交换相关泛函如此重要？

**回顾**：DFT的精确性完全取决于 $E_{xc}[\rho]$ 的近似质量。

**精确定义**：
$$E_{xc}[\rho] = \underbrace{(T[\rho] - T_s[\rho])}_{\text{动能相关修正}} + \underbrace{(V_{ee}[\rho] - E_H[\rho])}_{\text{非经典电子相互作用}}$$

**分解为交换和相关**：
$$E_{xc}[\rho] = E_x[\rho] + E_c[\rho]$$

- **交换能** $E_x[\rho]$：来自费米子统计（泡利原理），使同自旋电子避免彼此
- **相关能** $E_c[\rho]$：来自电子的瞬时相关运动

#### 泛函近似的层次结构（Jacob's Ladder）

John Perdew提出的"雅各布天梯"比喻：从"凡间"（简单近似）到"天堂"（化学精度）。

##### 第一阶梯：LDA（局域密度近似）

**思想**：假设每个点的交换相关能只取决于该点的密度。

$$E_{xc}^{LDA}[\rho] = \int \epsilon_{xc}(\rho(\mathbf{r})) \rho(\mathbf{r}) d\mathbf{r}$$

**$\epsilon_{xc}(\rho)$ 的来源**：
- 来自**均匀电子气**（jellium模型）的精确计算
- 交换部分：$\epsilon_x(\rho) = -\frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3} \rho^{1/3}$
- 相关部分：通过量子蒙特卡洛计算（Ceperley-Alder），用解析公式拟合

**优点**：
- 简单，计算快
- 对固体（接近均匀）效果不错

**缺点**：
- 对分子（不均匀）误差较大
- 高估结合能
- 低估键长

##### 第二阶梯：GGA（广义梯度近似）

**思想**：考虑密度的变化率（梯度），因为真实系统密度不均匀。

$$E_{xc}^{GGA}[\rho] = \int f(\rho(\mathbf{r}), |\nabla\rho(\mathbf{r})|) d\mathbf{r}$$

**常见GGA泛函**：
- **PBE**（Perdew-Burke-Ernzerhof）：基于物理约束，无经验参数
- **BLYP**（Becke-Lee-Yang-Parr）：含经验参数，对分子效果好
- **PW91**：PBE的前身

**改进**：
- 更好的分子几何
- 更准确的结合能
- 更好的反应势垒

##### 第三阶梯：meta-GGA

**思想**：进一步包含动能密度 $\tau(\mathbf{r}) = \frac{1}{2}\sum_i |\nabla\phi_i|^2$

$$E_{xc}^{meta-GGA}[\rho] = \int f(\rho, \nabla\rho, \tau) d\mathbf{r}$$

**常见泛函**：TPSS, SCAN

##### 第四阶梯：杂化泛函（Hybrid Functionals）

**思想**：混合精确的Hartree-Fock交换和DFT交换。

$$E_{xc}^{hybrid} = a E_x^{HF} + (1-a) E_x^{DFT} + E_c^{DFT}$$

**为什么要混合？**
- HF交换是精确的，但没有相关
- DFT交换是近似的，但与DFT相关配合好
- 混合可以取长补短

**常见杂化泛函**：
- **B3LYP**：$E_{xc} = 0.2 E_x^{HF} + 0.8 E_x^{Slater} + 0.72 \Delta E_x^{B88} + 0.81 E_c^{LYP} + 0.19 E_c^{VWN}$
  - 量子化学中最常用的泛函
  - 对有机分子效果很好
- **PBE0**：25% HF交换 + 75% PBE交换 + PBE相关
- **HSE**：用于固体，屏蔽的HF交换

##### 第五阶梯：双杂化泛函（Double Hybrid）

**思想**：进一步加入MP2相关能。

$$E_{xc}^{DH} = a_x E_x^{HF} + (1-a_x) E_x^{DFT} + b E_c^{MP2} + (1-b) E_c^{DFT}$$

**常见泛函**：B2PLYP, XYG3

**优点**：精度接近CCSD
**缺点**：计算成本高（$O(N^5)$）

#### 如何选择泛函？

| 应用 | 推荐泛函 |
|-----|---------|
| 有机分子几何和频率 | B3LYP, PBE0 |
| 反应能和热化学 | B3LYP, M06-2X |
| 过渡金属 | PBE0, TPSSh |
| 固体和表面 | PBE, HSE |
| 弱相互作用 | ωB97X-D, B3LYP-D3 |
| 高精度 | 双杂化泛函 |

### 7.4 DFT的优势与局限

#### 优点

1. **计算效率**：
   - 复杂度：$O(N^3)$（纯泛函）到 $O(N^4)$（杂化泛函）
   - 比CCSD的 $O(N^6)$ 快得多
   - 可以处理数百甚至数千原子的系统

2. **包含电子相关**：
   - 与HF不同，DFT通过 $E_{xc}$ 包含相关效应
   - 对很多系统给出比HF更准确的结果

3. **实用性**：
   - 对分子几何、振动频率、反应能等性质效果好
   - 是目前应用最广泛的量子化学方法

#### 局限性

1. **交换相关泛函未知**：
   - 精确的 $E_{xc}[\rho]$ 不知道
   - 所有泛函都是近似
   - 不同泛函对不同问题效果不同（需要经验选择）

2. **无系统改进途径**：
   - 波函数方法：HF → MP2 → CCSD → CCSDT → ... → 精确
   - DFT：没有系统的改进路径
   - 无法保证更复杂的泛函一定更好

3. **自相互作用误差**：
   - Hartree能包含电子与自己的相互作用
   - HF交换正好抵消这个误差
   - 近似的DFT交换不能完全抵消
   - 导致：电荷转移态误差、解离曲线错误等

4. **强相关系统**：
   - 过渡金属、镧系/锕系化合物
   - 多参考态系统
   - 标准DFT可能严重失效

5. **激发态**：
   - 基态DFT只能计算基态
   - 激发态需要时间依赖DFT（TD-DFT）
   - TD-DFT对某些激发态不准确

6. **弱相互作用**：
   - 标准泛函难以描述范德华力
   - 需要加色散修正（如DFT-D3）

#### DFT与波函数方法的对比

| 方面 | DFT | 波函数方法 |
|-----|-----|-----------|
| 基本变量 | 密度 $\rho(\mathbf{r})$ | 波函数 $\psi$ |
| 理论基础 | Hohenberg-Kohn定理 | 薛定谔方程 |
| 近似位置 | 交换相关泛函 | 波函数截断 |
| 系统改进 | 无 | 有（CI/CC层次） |
| 计算成本 | 低 | 高 |
| 适用系统 | 大系统 | 小到中等系统 |
| 相关能 | 包含（近似） | 可以精确（如FCI） |



## 8. 基组展开的数学原理

### 8.1 基组完备性

#### 数学表述
基函数集合 $\{\chi_\mu(\mathbf{r})\}$ 是完备的，如果任意函数 $f(\mathbf{r}) \in L^2$ 可以表示为：
$$f(\mathbf{r}) = \lim_{M \to \infty} \sum_{\mu=1}^M c_\mu \chi_\mu(\mathbf{r})$$

#### 收敛性
随着基组增大，能量单调下降（变分原理）：
$$E(M+1) \leq E(M)$$

#### 完备基组极限（CBS）
$$E_{CBS} = \lim_{M \to \infty} E(M)$$

### 8.2 基组类型

#### 原子轨道基组
- **STO**：Slater型轨道，$\chi(\mathbf{r}) = r^{n-1} e^{-\zeta r} Y_{lm}(\theta,\phi)$
- **GTO**：高斯型轨道，$\chi(\mathbf{r}) = x^l y^m z^n e^{-\alpha r^2}$

#### 基组大小
- **最小基组**：每个原子一个轨道
- **双zeta（DZ）**：每个原子轨道用两个函数
- **三zeta（TZ）**：三个函数
- **四zeta（QZ）**：四个函数

#### 极化函数
添加角动量更高的函数，例如在碳原子上添加 $d$ 函数。

#### 弥散函数
添加指数很小的函数，描述远离原子核的电子。

#### "紧凑但精确"的基组：Dunning基组为例

**问题**：什么是"紧凑但精确"的基组？

**核心概念**：
- **紧凑（Compact）**：基函数数量少，计算效率高
- **精确（Accurate）**：能够达到高精度结果
- **看似矛盾**：通常更多基函数 = 更高精度，但计算成本也更高

**Dunning基组的设计哲学**：

##### 1. 紧凑性（Compactness）

**定义**：用尽可能少的基函数达到目标精度。

**实现方式**：
- **优化的指数**：仔细选择高斯函数的指数 $\alpha$，使其在关键区域有好的覆盖
- **收缩基组（Contracted Basis Sets）**：
  - 将多个原始高斯函数（primitive Gaussians）线性组合成收缩函数（contracted functions）
  - 例如：$(6s, 3p)$ 表示6个原始s函数收缩成3个s函数
  - 减少基函数数量，但保持灵活性

**例子**：cc-pVDZ（Dunning基组）
- **cc** = correlation-consistent（相关一致）
- **p** = polarized（极化）
- **VDZ** = Valence Double Zeta（价层双zeta）
- 对于C原子：$(9s, 4p, 1d) \rightarrow [3s, 2p, 1d]$（9个原始s函数收缩成3个s函数）

##### 2. 精确性（Accuracy）

**定义**：能够准确描述电子结构，特别是：
- 价层电子（化学键）
- 电子相关效应
- 激发态

**实现方式**：
- **相关一致设计**：
  - 基组设计时考虑电子相关
  - 不同角动量函数（s, p, d, f...）的指数系统化选择
  - 使得相关能计算更准确

- **系统化改进**：
  - **cc-pVDZ**：双zeta + 极化（价层2个函数，1个d函数）
  - **cc-pVTZ**：三zeta + 极化（价层3个函数，1个d函数，1个f函数）
  - **cc-pVQZ**：四zeta + 极化（价层4个函数，更多极化函数）
  - 系统化地增加基组大小，可以外推到CBS极限

##### 3. 为什么"紧凑但精确"是可能的？

**关键洞察**：

**1. 不是所有基函数都同等重要**：
- 价层电子（参与化学键）需要更精确的描述
- 核心电子（靠近原子核）可以用较少函数描述
- 虚轨道（未占据）的精度要求较低

**2. 优化的基函数选择**：
- 传统基组：可能包含冗余或低效的基函数
- Dunning基组：每个基函数都经过优化，贡献最大化

**3. 收缩策略**：
- 原始函数：很多个（如9个s函数）
- 收缩函数：较少个（如3个s函数）
- 但收缩函数是原始函数的线性组合，保留了大部分信息

**数学表述**：

**原始基组**：$\{\chi_\mu^{primitive}\}_{\mu=1}^{M_{prim}}$
- 例如：9个原始s函数

**收缩基组**：$\{\chi_\nu^{contracted}\}_{\nu=1}^{M_{cont}}$
- 例如：3个收缩s函数
- 其中：$\chi_\nu^{contracted} = \sum_{\mu} c_{\nu\mu} \chi_\mu^{primitive}$

**关键**：$M_{cont} < M_{prim}$，但收缩基组几乎保留了原始基组的表达能力。

##### 4. Dunning基组系列

**cc-pVDZ（Correlation-Consistent Polarized Valence Double Zeta）**：
- **大小**：中等（对于C：14个基函数）
- **精度**：中等（误差通常 < 5 kcal/mol）
- **用途**：快速计算，初步研究

**cc-pVTZ（Triple Zeta）**：
- **大小**：较大（对于C：30个基函数）
- **精度**：高（误差通常 < 1 kcal/mol）
- **用途**：标准高精度计算

**cc-pVQZ（Quadruple Zeta）**：
- **大小**：很大（对于C：55个基函数）
- **精度**：很高（误差通常 < 0.1 kcal/mol）
- **用途**：极高精度，接近CBS

**aug-cc-pVXZ（Augmented）**：
- **aug** = 添加弥散函数
- **用途**：描述负离子、激发态、弱相互作用

##### 5. 紧凑 vs 精确的权衡

**传统基组（如STO-3G）**：
- **紧凑**：✓ 基函数很少
- **精确**：✗ 精度低（误差可能 > 50 kcal/mol）

**大基组（如cc-pV6Z）**：
- **紧凑**：✗ 基函数很多（计算昂贵）
- **精确**：✓ 精度很高（接近CBS）

**Dunning基组（如cc-pVTZ）**：
- **紧凑**：✓ 相对紧凑（比大基组小）
- **精确**：✓ 高精度（比小基组精确得多）
- **平衡**：在紧凑性和精确性之间找到最佳平衡

##### 6. 实际性能对比

**例子**：H$_2$O分子的能量计算

| 基组 | 基函数数 | 能量误差 (kcal/mol) | 计算时间 |
|------|---------|---------------------|----------|
| STO-3G | 7 | ~50 | 很快 |
| 6-31G | 13 | ~10 | 快 |
| cc-pVDZ | 24 | ~3 | 中等 |
| cc-pVTZ | 58 | ~0.5 | 慢 |
| cc-pVQZ | 115 | ~0.1 | 很慢 |

**观察**：
- cc-pVTZ用58个基函数达到0.5 kcal/mol精度
- 如果用传统方法，可能需要更多基函数才能达到相同精度
- 这就是"紧凑但精确"的含义

##### 7. 为什么Dunning基组"紧凑但精确"？

**设计原则**：

1. **相关一致**：
   - 基组设计时考虑电子相关
   - 不同角动量函数的指数系统化选择
   - 使得相关能计算更准确

2. **优化的指数**：
   - 通过大量测试优化
   - 每个基函数都贡献最大化
   - 避免冗余

3. **系统化**：
   - 可以系统化地增加基组大小
   - 可以外推到CBS极限
   - 可预测的精度改进

4. **物理直觉**：
   - 价层电子需要更精确描述
   - 核心电子可以用较少函数
   - 极化函数捕获电子相关

##### 8. 总结

**"紧凑但精确"的含义**：
- **紧凑**：用相对较少的基函数
- **精确**：达到高精度结果
- **关键**：通过优化设计，在紧凑性和精确性之间找到最佳平衡

**Dunning基组的优势**：
- 系统化设计
- 相关一致
- 优化的指数
- 可外推到CBS

**实际应用**：
- cc-pVTZ是标准的高精度基组
- 在精度和计算成本之间取得良好平衡
- 广泛用于量子化学计算

### 8.3 基组误差

#### 基组截断误差
$$E_{CBS} - E(M) = \sum_{k=M+1}^\infty a_k$$

通常随 $M$ 指数衰减。

#### 基组叠加误差（BSSE）
在分子计算中，由于基组不完备，原子能量被高估，导致结合能被高估。

##### 修正方法（Counterpoise修正）
$$E_{corrected} = E_{AB} - E_A^{AB} - E_B^{AB}$$

其中 $E_A^{AB}$ 是原子 $A$ 在 $AB$ 的完整基组中的能量。



### 8.4 赝势方法的数学基础

#### 全电子 vs 赝势

#### 全电子计算
显式处理所有电子，包括核心电子。

#### 赝势方法
用有效势替代核心电子，只处理价电子。

#### 赝势构造

#### 要求
1. **价轨道**：在价电子区域，赝轨道与全电子轨道相同
2. **能量**：赝轨道能量与全电子轨道能量相同
3. **归一化**：赝轨道归一化

#### 数学表述
在价电子区域 $r > r_c$（截断半径）：
$$\phi_{ps}(r) = \phi_{AE}(r), \quad r > r_c$$

在核心区域 $r < r_c$：
$$\phi_{ps}(r) = \text{smooth function}$$

#### 投影增强波（PAW）

#### 思想
将全电子波函数表示为赝波函数和核心修正的叠加：
$$|\psi_{AE}\rangle = |\tilde{\psi}\rangle + \sum_i (|\phi_i\rangle - |\tilde{\phi}_i\rangle) \langle\tilde{p}_i|\tilde{\psi}\rangle$$

其中：
- $|\tilde{\psi}\rangle$ 是赝波函数
- $|\phi_i\rangle$ 是全电子原子轨道
- $|\tilde{\phi}_i\rangle$ 是赝原子轨道
- $|\tilde{p}_i\rangle$ 是投影算符



## 9. 误差分析和数学性质

### 9.1 误差来源

#### 方法误差
- **HF**：忽略电子相关
- **CISD**：忽略高激发
- **DFT**：交换相关泛函近似

#### 基组误差
- 基组不完备
- 基组叠加误差

#### 数值误差
- 积分精度
- 矩阵对角化精度
- SCF收敛精度

### 9.2 误差估计

#### 基组外推
使用多个基组大小，外推到CBS极限：
$$E(M) = E_{CBS} + A e^{-\alpha M}$$

#### 方法层次
比较不同级别的方法，估计方法误差。

### 9.3 计算复杂度

#### 方法比较
| 方法 | 复杂度 | 可扩展性 |
|------|--------|----------|
| HF | $O(N^4)$ | 好 |
| MP2 | $O(N^5)$ | 中等 |
| CCSD | $O(N^6)$ | 中等 |
| CCSDT | $O(N^8)$ | 差 |
| DFT | $O(N^3)$-$O(N^4)$ | 好 |
| FCI | 指数 | 不可扩展 |



## 10. 理论思想总结

### 10.1 近似方法的层次结构

1. **单参考方法**：HF → MP2 → CCSD → CCSDT
2. **多参考方法**：CASSCF → MRCI → MRCC
3. **密度泛函方法**：LDA → GGA → 杂化 → 双杂化

### 10.2 精度与效率的权衡

- **高精度**：CCSDT, FCI（计算昂贵）
- **中等精度**：CCSD, MP2（实用）
- **快速方法**：DFT, HF（大系统）

### 10.3 可扩展性挑战

传统方法的计算复杂度随系统大小快速增长，限制了可处理系统的规模。



## 11. 思考题

1. 为什么CI方法不是大小一致的？CC方法如何解决这个问题？
2. MP微扰级数为什么不总是收敛？
3. Hohenberg-Kohn定理的物理意义是什么？
4. Kohn-Sham方法如何将多体问题简化为单电子问题？
5. 基组误差如何系统性地减小？
6. 传统方法的可扩展性瓶颈在哪里？



## 12. 总结

第3周将学习如何使用经典机器学习方法（神经网络、变分蒙特卡洛等）近似求解薛定谔方程，探索新的计算范式。

