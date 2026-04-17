# 第五章：激发态方法理论

## 5.1 为什么需要激发态？

### 5.1.1 物理意义

激发态对于理解：
- **光谱**：吸收/发射光谱来自态间跃迁
- **光化学**：光激发后的化学反应
- **能量传递**：分子间能量转移机制
- **材料性质**：带隙、光学性质

### 5.1.2 数学定义

对于哈密顿量 $\hat{H}$，本征态按能量排序：
$$\hat{H}|E_n\rangle = E_n|E_n\rangle, \quad E_0 \leq E_1 \leq E_2 \leq \cdots$$

- $|E_0\rangle$：基态
- $|E_n\rangle$（$n > 0$）：第n激发态

**激发能**：$\Delta E_n = E_n - E_0$

---

## 5.2 VQD（变分量子偏折）

### 5.2.1 核心思想

**问题**：变分原理只保证找到基态，如何找激发态？

**解决**：添加惩罚项，使优化过程避开已找到的态。

### 5.2.2 算法框架

**第一步**：用标准VQE找基态
$$E_0 = \min_{\boldsymbol{\theta}} \langle \psi(\boldsymbol{\theta}) | \hat{H} | \psi(\boldsymbol{\theta}) \rangle$$
得到基态 $|\psi_0\rangle$。

**第二步**：找第一激发态，代价函数加惩罚项
$$L_1(\boldsymbol{\theta}) = \langle \psi(\boldsymbol{\theta}) | \hat{H} | \psi(\boldsymbol{\theta}) \rangle + \beta_0 |\langle \psi(\boldsymbol{\theta}) | \psi_0 \rangle|^2$$

**一般形式**：找第k激发态
$$L_k(\boldsymbol{\theta}) = \langle \psi(\boldsymbol{\theta}) | \hat{H} | \psi(\boldsymbol{\theta}) \rangle + \sum_{j=0}^{k-1} \beta_j |\langle \psi(\boldsymbol{\theta}) | \psi_j \rangle|^2$$

### 5.2.3 惩罚项的作用

当 $|\psi(\boldsymbol{\theta})\rangle$ 与已知态 $|\psi_j\rangle$ 有重叠时，惩罚项增加。

**选择 $\beta_j$**：
- 需要 $\beta_j > E_k - E_j$ 保证激发态是最小值
- 实际中通常取 $\beta_j \gg |E_k - E_j|$

### 5.2.4 重叠的测量

计算 $|\langle \psi | \phi \rangle|^2$ 需要**SWAP测试**：

```
|0⟩ ─── H ─── • ─── H ─── 测量
              │
|ψ⟩ ─────── SWAP ─────────
              │
|φ⟩ ───────────────────────

P(0) = (1 + |⟨ψ|φ⟩|²)/2
```

或使用**计算基展开**：
$$|\langle \psi | \phi \rangle|^2 = \left|\sum_x \psi^*(x) \phi(x)\right|^2$$

### 5.2.5 VQD的数学保证

**定理**：如果惩罚系数足够大且优化全局收敛，VQD找到的第k个态是第k激发态。

**证明要点**：
惩罚项使代价函数在低能态附近有极大值（鞍点），最小值必然在正交补空间中。

---

## 5.3 SSVQE（子空间搜索VQE）

### 5.3.1 核心思想

**同时**优化多个正交态，而不是逐个求解。

### 5.3.2 算法框架

定义多个参数化态：
$$|\psi_k(\boldsymbol{\theta})\rangle = U_k(\boldsymbol{\theta})|0\rangle$$

**代价函数**（加权能量和）：
$$L(\boldsymbol{\theta}) = \sum_k w_k \langle \psi_k | \hat{H} | \psi_k \rangle$$

其中 $w_0 > w_1 > w_2 > \cdots > 0$。

**正交约束**：
$$\langle \psi_j | \psi_k \rangle = \delta_{jk}$$

### 5.3.3 权重的选择

**定理**：如果 $w_k - w_{k+1} > E_K - E_0$（$K$ 是最高目标态），则SSVQE的全局最小值对应前K+1个本征态。

**实际选择**：
- $w_k = K - k$（线性递减）
- $w_k = e^{-\alpha k}$（指数递减）

### 5.3.4 构建正交态

**方法1**：不同初态
$$|\psi_k\rangle = U(\boldsymbol{\theta})|k\rangle$$
用不同计算基态作为初态。

**方法2**：正交化层
在电路中加入确保正交的结构。

**方法3**：Gram-Schmidt
优化后对态进行正交化。

### 5.3.5 SSVQE vs VQD

| 特性 | VQD | SSVQE |
|-----|-----|-------|
| 求解方式 | 逐个 | 同时 |
| 代价函数 | 加惩罚 | 加权和 |
| 正交性 | 惩罚软约束 | 硬约束 |
| 测量复杂度 | 需要SWAP测试 | 只需能量 |
| 误差累积 | 可能 | 较少 |

---

## 5.4 qEOM（量子运动方程）

### 5.4.1 经典EOM-CCSD

在耦合簇框架中，激发态通过运动方程求解：

**右本征值问题**：
$$[\bar{H}, R_k] = \omega_k R_k$$

其中 $\bar{H} = e^{-\hat{T}} \hat{H} e^{\hat{T}}$ 是相似变换的哈密顿量，$R_k$ 是激发算符。

### 5.4.2 量子化版本

**思想**：在VQE得到的近似基态上，线性响应求激发态。

**步骤**：
1. VQE得到基态 $|\psi_0\rangle$
2. 构建激发流形 $\{R_k |\psi_0\rangle\}$
3. 在该子空间中对角化 $\hat{H}$

### 5.4.3 激发算符的选择

**单激发**：
$$R_i^a = a_a^\dagger a_i$$

**双激发**：
$$R_{ij}^{ab} = a_a^\dagger a_b^\dagger a_j a_i$$

### 5.4.4 qEOM矩阵元

需要计算：
$$H_{kl} = \langle \psi_0 | R_k^\dagger \hat{H} R_l | \psi_0 \rangle$$
$$S_{kl} = \langle \psi_0 | R_k^\dagger R_l | \psi_0 \rangle$$

**广义本征值问题**：
$$\mathbf{H} \mathbf{c} = E \mathbf{S} \mathbf{c}$$

### 5.4.5 量子电路测量

$H_{kl}$ 和 $S_{kl}$ 可以通过量子电路测量：

```
|0⟩ ─── H ─── • ─────────── • ─── H ─── 测量
              │             │
|ψ₀⟩ ──── R_k† ───── H ─── R_l ────────
```

这是受控酉门的期望值测量。

### 5.4.6 优势与局限

**优势**：
- 一次VQE可得多个激发态
- 避免重新优化
- 保持尺寸一致性

**局限**：
- 依赖基态质量
- 激发算符选择有限
- 测量复杂度高

---

## 5.5 折叠光谱方法

### 5.5.1 核心思想

构造修改后的哈密顿量，使目标激发态变成"基态"。

### 5.5.2 能量折叠

定义**折叠哈密顿量**：
$$\hat{H}_\mu = (\hat{H} - \mu)^2$$

其本征值为 $(E_n - \mu)^2$。

当 $\mu \approx E_k$ 时，$|E_k\rangle$ 变成 $\hat{H}_\mu$ 的基态。

### 5.5.3 算法

1. **估计**：粗略估计目标激发能 $\mu$
2. **构造**：$\hat{H}_\mu = (\hat{H} - \mu)^2$
3. **VQE**：对 $\hat{H}_\mu$ 运行VQE
4. **提取**：最小化 $\langle \hat{H}_\mu \rangle$ 给出接近 $E_k$ 的态

### 5.5.4 挑战

**问题**：$\hat{H}^2$ 包含 $O(M^2)$ 个Pauli项（$M$ 是原始项数）

**缓解**：
- 只保留重要的交叉项
- 使用随机化方法

---

## 5.6 量子朗之万方法

### 5.6.1 思想

使用虚时演化结合随机采样来探索激发态。

### 5.6.2 虚时Schrödinger方程

$$\frac{\partial}{\partial \tau} |\psi(\tau)\rangle = -(\hat{H} - E_0) |\psi(\tau)\rangle$$

解为：
$$|\psi(\tau)\rangle = e^{-(\hat{H}-E_0)\tau} |\psi(0)\rangle$$

长时间演化 $\tau \to \infty$ 后，投影到基态。

### 5.6.3 激发态的访问

从不同初态出发，或在虚时演化中加入随机"扰动"，可以访问不同能量态。

---

## 5.7 振动和旋转激发

### 5.7.1 Born-Oppenheimer近似后

电子能量定义势能面 $V(\mathbf{R})$，核运动在该势能面上。

**振动**：核在平衡位置附近振动
**旋转**：分子整体旋转

### 5.7.2 简谐近似

在平衡位置 $\mathbf{R}_0$ 展开：
$$V(\mathbf{R}) \approx V(\mathbf{R}_0) + \frac{1}{2} \sum_{ij} K_{ij} (R_i - R_{0,i})(R_j - R_{0,j})$$

**简谐振动能级**：
$$E_v = \hbar \omega (v + \frac{1}{2})$$

### 5.7.3 VQE在振动问题中的应用

对于非谐振动，可以：
1. 将振动哈密顿量映射到量子比特
2. 使用VQE求解振动态

---

## 5.8 跃迁性质

### 5.8.1 跃迁偶极矩

电子跃迁的偶极矩：
$$\boldsymbol{\mu}_{0k} = \langle E_0 | \hat{\boldsymbol{\mu}} | E_k \rangle$$

其中 $\hat{\boldsymbol{\mu}} = -e \sum_i \mathbf{r}_i$ 是电偶极算符。

### 5.8.2 振子强度

**振子强度**表征跃迁概率：
$$f_{0k} = \frac{2m_e \omega_{0k}}{3\hbar} |\boldsymbol{\mu}_{0k}|^2$$

### 5.8.3 量子计算测量

需要测量 $\langle \psi_0 | \hat{\mu} | \psi_k \rangle$

**方法**：
1. 制备 $|\psi_0\rangle$ 和 $|\psi_k\rangle$
2. 使用Hadamard测试获取实部和虚部

$$\text{Re}\langle \psi_0 | O | \psi_k \rangle = \frac{1}{2}(\langle + | O | + \rangle - \langle - | O | - \rangle)$$

其中 $|+\rangle = (|\psi_0\rangle + |\psi_k\rangle)/\sqrt{2}$

---

## 5.9 收敛和精度分析

### 5.9.1 误差来源（激发态）

除了基态VQE的误差外，还有：
1. **正交性误差**：态之间不完全正交
2. **累积误差**：逐个求解时误差累积
3. **子空间误差**：激发算符选择不完备

### 5.9.2 状态平均误差

对于SSVQE，平均态能量误差：
$$\bar{\epsilon} = \frac{1}{K+1} \sum_{k=0}^K |E_k^{VQE} - E_k^{exact}|$$

### 5.9.3 激发能精度

激发能误差往往比绝对能量误差小（误差抵消）：
$$\epsilon_{\Delta E} = |\Delta E^{VQE} - \Delta E^{exact}| < \epsilon_0 + \epsilon_k$$

---

## 5.10 方法比较与选择

### 5.10.1 综合比较

| 方法 | 测量复杂度 | 实现复杂度 | 精度 | 适用场景 |
|------|-----------|-----------|------|---------|
| VQD | 高（SWAP测试） | 中 | 好 | 少量激发态 |
| SSVQE | 中 | 高 | 中 | 多个态同时 |
| qEOM | 高 | 高 | 好 | 多激发态 |
| 折叠光谱 | 很高 | 低 | 依赖 | 特定能量态 |

### 5.10.2 选择指南

- **只需1-2个激发态**：VQD
- **需要多个激发态且能量接近**：SSVQE
- **需要全谱且基态精度高**：qEOM
- **目标能量已知**：折叠光谱

### 5.10.3 实践建议

1. **从VQD开始**：最直接
2. **惩罚系数**：从大值开始，必要时调整
3. **正交性检查**：计算态间重叠
4. **与经典比较**：用小系统验证

---

## 5.11 小结

### 5.11.1 核心公式

**VQD代价函数**：
$$L_k = \langle \hat{H} \rangle + \sum_{j<k} \beta_j |\langle \psi | \psi_j \rangle|^2$$

**SSVQE代价函数**：
$$L = \sum_k w_k \langle \psi_k | \hat{H} | \psi_k \rangle$$

**qEOM矩阵元**：
$$H_{kl} = \langle \psi_0 | R_k^\dagger H R_l | \psi_0 \rangle$$

### 5.11.2 关键洞察

1. 变分原理可以扩展到激发态（通过约束或加权）
2. 正交性是关键约束
3. 测量开销通常比基态VQE更大
4. 误差分析更复杂，但激发能可能有误差抵消

### 5.11.3 前沿方向

- **量子相位估计辅助**：用QPE初始化VQE
- **神经网络辅助**：学习态到态的映射
- **动态激发态**：非绝热动力学
- **共振态**：处理连续谱中的准束缚态
