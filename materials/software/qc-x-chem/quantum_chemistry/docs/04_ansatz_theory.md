# 第四章：Ansatz设计理论

## 4.1 Ansatz的数学框架

### 4.1.1 一般定义

**Ansatz**（德语，意为"假设"）是参数化的量子态：

$$|\psi(\boldsymbol{\theta})\rangle = U(\boldsymbol{\theta})|0\rangle^{\otimes n}$$

其中 $U(\boldsymbol{\theta})$ 是参数化酉算符。

### 4.1.2 表达能力的度量

**可达态空间**：
$$\mathcal{S}_U = \{|\psi(\boldsymbol{\theta})\rangle : \boldsymbol{\theta} \in \mathbb{R}^p\}$$

**完备性**：如果 $\mathcal{S}_U$ 包含整个Hilbert空间（或目标子空间），称为完备。

**过参数化**：参数数 $p$ 远大于独立参数需求。

### 4.1.3 Ansatz复杂度层次

| 层次 | 特点 | 例子 |
|-----|------|------|
| 精确 | 可表示任意态 | 全连接深电路 |
| 化学精确 | 可达化学精度 | UCCSD |
| 近似 | 捕获主要相关 | 单层UCC |
| 平均场 | 仅描述平均场态 | HF ansatz |

---

## 4.2 UCCSD Ansatz

### 4.2.1 耦合簇理论回顾

传统耦合簇（CC）波函数：
$$|\Psi_{CC}\rangle = e^{\hat{T}}|\Phi_0\rangle$$

其中：
- $|\Phi_0\rangle$：参考态（通常是HF态）
- $\hat{T} = \hat{T}_1 + \hat{T}_2 + \cdots$：激发算符

**单激发算符**：
$$\hat{T}_1 = \sum_{i \in occ} \sum_{a \in vir} t_i^a \, a_a^\dagger a_i$$

**双激发算符**：
$$\hat{T}_2 = \frac{1}{4} \sum_{i,j \in occ} \sum_{a,b \in vir} t_{ij}^{ab} \, a_a^\dagger a_b^\dagger a_j a_i$$

### 4.2.2 幺正化

**问题**：$e^{\hat{T}}$ 不是幺正的（$\hat{T}$ 不是反厄米的）

**解决**：使用幺正耦合簇（UCC）
$$|\Psi_{UCC}\rangle = e^{\hat{T} - \hat{T}^\dagger}|\Phi_0\rangle$$

因为 $(\hat{T} - \hat{T}^\dagger)^\dagger = \hat{T}^\dagger - \hat{T} = -(\hat{T} - \hat{T}^\dagger)$，所以 $e^{\hat{T} - \hat{T}^\dagger}$ 是幺正的。

### 4.2.3 UCCSD的显式形式

定义**幺正激发算符**：

$$\hat{\tau}_i^a = a_a^\dagger a_i - a_i^\dagger a_a$$

$$\hat{\tau}_{ij}^{ab} = a_a^\dagger a_b^\dagger a_j a_i - a_i^\dagger a_j^\dagger a_b a_a$$

UCCSD波函数：
$$|\Psi_{UCCSD}\rangle = \exp\left(\sum_{ia} \theta_i^a \hat{\tau}_i^a + \sum_{ijab} \theta_{ij}^{ab} \hat{\tau}_{ij}^{ab}\right)|\Phi_{HF}\rangle$$

### 4.2.4 参数数量

设有 $n_{occ}$ 个占据轨道，$n_{vir}$ 个虚轨道：

- **单激发**：$n_{occ} \times n_{vir}$ 个参数
- **双激发**：$\binom{n_{occ}}{2} \times \binom{n_{vir}}{2}$ 个参数

**总参数数**：$O(n_{occ}^2 \times n_{vir}^2)$

### 4.2.5 Trotterization

指数 $e^{A+B}$ 一般不等于 $e^A e^B$（除非 $[A,B]=0$）。

**一阶Trotter分解**：
$$e^{\sum_k \theta_k \hat{\tau}_k} \approx \prod_k e^{\theta_k \hat{\tau}_k}$$

误差为 $O(\theta^2)$。

**高阶Trotter**：
$$e^{A+B} = (e^{A/2n} e^{B/n} e^{A/2n})^n + O(1/n^2)$$

### 4.2.6 量子电路实现

单激发 $e^{\theta(a_a^\dagger a_i - a_i^\dagger a_a)}$ 的电路：

JW变换后：
$$a_a^\dagger a_i - a_i^\dagger a_a \to \frac{i}{2}(X_a Y_i - Y_a X_i) \prod_{i<k<a} Z_k$$

**电路**（简化，$i<a$ 相邻）：
```
|i⟩ ─── RY(θ/2) ─── •  ─── RY(-θ/2) ───
                     │
|a⟩ ────────────── ⊕ ─── RY(θ/2) ─────
```

完整的单激发门在PennyLane中是 `qml.SingleExcitation`。

### 4.2.7 电路深度分析

对于UCCSD：
- 每个单激发门：$O(n)$ 深度（由于Z串）
- 每个双激发门：$O(n)$ 深度
- 总门数：$O(n^4)$（双激发主导）

**瓶颈**：UCCSD电路在大系统上非常深。

---

## 4.3 硬件高效Ansatz

### 4.3.1 设计哲学

**核心思想**：不追求物理意义，而是：
1. 使用硬件原生门
2. 尊重硬件拓扑
3. 最小化电路深度

### 4.3.2 层状结构

$$U(\boldsymbol{\theta}) = \prod_{l=1}^L \left[ \prod_i R_Y(\theta_{l,i}^Y) R_Z(\theta_{l,i}^Z) \cdot \text{Entangle}(l) \right]$$

**旋转层**：
$$\prod_i R_Y(\theta_i^Y) R_Z(\theta_i^Z)$$

**纠缠层**：
- Linear: CNOT$(0,1)$, CNOT$(1,2)$, ..., CNOT$(n-2,n-1)$
- Circular: 加上 CNOT$(n-1,0)$
- Full: 所有 $\binom{n}{2}$ 对

### 4.3.3 参数数量

$$p = n \times n_{rot} \times L$$

其中：
- $n$：量子比特数
- $n_{rot}$：每量子比特旋转门数（通常2-3）
- $L$：层数

### 4.3.4 表达能力分析

**定理**（通用性）：足够深的交替旋转-纠缠电路可以近似任意幺正算符。

**所需深度**：$L = O(4^n / n)$ 层足以实现任意n量子比特幺正。

**实际考量**：
- 目标不是任意幺正，而是基态
- 可能需要的深度远小于理论上界
- 但也可能陷入贫瘠高原

### 4.3.5 与UCCSD比较

| 特性 | UCCSD | 硬件高效 |
|-----|-------|---------|
| 物理意义 | 强 | 弱 |
| 电路深度 | $O(n^4)$ | $O(nL)$ |
| 参数数量 | $O(n_{occ}^2 n_{vir}^2)$ | $O(nL)$ |
| 初始化 | MP2/HF | 随机 |
| 贫瘠高原 | 较少 | 常见 |
| 硬件适配 | 差 | 好 |

---

## 4.4 ADAPT-VQE

### 4.4.1 核心思想

**问题**：如何选择最优ansatz？

**ADAPT方法**：从简单开始，迭代添加最重要的算符。

### 4.4.2 算法流程

1. **初始化**：$|\psi_0\rangle = |\Phi_{HF}\rangle$，算符池 $\mathcal{P} = \{\hat{\tau}_k\}$

2. **选择**：计算每个池中算符的梯度
   $$g_k = \left|\frac{\partial E}{\partial \theta_k}\right|_{\theta_k=0} = |\langle \psi | [H, \hat{\tau}_k] | \psi \rangle|$$
   
3. **添加**：选择 $k^* = \arg\max_k |g_k|$，将 $e^{\theta_{new} \hat{\tau}_{k^*}}$ 加入ansatz

4. **优化**：优化所有参数

5. **迭代**：重复2-4直到 $\max_k |g_k| < \epsilon$

### 4.4.3 梯度计算

关键公式：
$$\frac{\partial E}{\partial \theta_k}\Big|_{\theta_k=0} = \langle \psi | [H, \hat{\tau}_k - \hat{\tau}_k^\dagger] | \psi \rangle = 2i \, \text{Im}\langle \psi | H \hat{\tau}_k | \psi \rangle$$

对于反厄米算符 $A = \hat{\tau}_k - \hat{\tau}_k^\dagger$：
$$\frac{d}{d\theta} \langle \psi | e^{-\theta A} H e^{\theta A} | \psi \rangle \Big|_{\theta=0} = \langle \psi | [H, A] | \psi \rangle$$

### 4.4.4 算符池选择

**Fermionic Pool**（原始ADAPT）：
$$\mathcal{P} = \{\hat{\tau}_i^a, \hat{\tau}_{ij}^{ab}\}$$
所有单激发和双激发。

**Qubit Pool**（qubit-ADAPT）：
$$\mathcal{P} = \{P_\alpha : P_\alpha \in \{I, X, Y, Z\}^{\otimes n}\}$$
直接在Pauli级别选择。

**Sparse Pool**：
只包含最可能重要的算符。

### 4.4.5 ADAPT的优势

1. **紧凑电路**：只包含必要算符
2. **系统改进**：添加更多算符单调降低能量
3. **避免贫瘠高原**：逐步构建，每步梯度显著
4. **可解释性**：每个算符有明确物理意义

### 4.4.6 ADAPT的代价

- 每次迭代需要计算所有池算符的梯度
- 池大小 $|\mathcal{P}| = O(n^4)$（UCCSD池）
- 总测量次数可能很大

---

## 4.5 对称性保持Ansatz

### 4.5.1 利用对称性的好处

如果 $[\hat{H}, \hat{S}] = 0$（$\hat{S}$ 是对称算符）：
1. 基态在 $\hat{S}$ 的本征空间中
2. 只需在该子空间中搜索
3. 减少参数空间维度

### 4.5.2 粒子数守恒

**对称性**：$[\hat{H}, \hat{N}] = 0$，其中 $\hat{N} = \sum_p a_p^\dagger a_p$

**保持方法**：
- UCCSD自然保持（激发算符保粒子数）
- 硬件高效需要特殊设计（如只用 $XX+YY$ 型纠缠）

**粒子数保持电路**：
```
RY-RZ rotations that preserve Hamming weight
Entangling: exp(-iθ(XX+YY)) type gates
```

### 4.5.3 自旋对称性

**对称性**：$[\hat{H}, \hat{S}^2] = 0$，$[\hat{H}, \hat{S}_z] = 0$

**$\hat{S}_z$ 守恒**：分别处理 $\alpha$ 和 $\beta$ 自旋

**$\hat{S}^2$ 守恒**：更复杂，需要自旋适配激发

### 4.5.4 点群对称性

分子有点群对称性（如 $C_{2v}$，$D_{2h}$ 等）

**利用**：
1. 轨道按对称性分类
2. 只允许同对称性轨道间的激发
3. 大大减少参数数量

---

## 4.6 Ansatz的可训练性

### 4.6.1 损失景观分析

**好的ansatz**：损失函数有明确的梯度方向，无过多局部极小

**坏的ansatz**：贫瘠高原、大量局部极小

### 4.6.2 贫瘠高原的数学

**定理**（McClean et al., 2018）：对于形成2-设计的参数化电路：

$$\mathbb{E}_{\boldsymbol{\theta}}[\partial_k E] = 0$$

$$\text{Var}_{\boldsymbol{\theta}}[\partial_k E] \leq O(2^{-n})$$

**含义**：梯度期望为零且方差指数小。

### 4.6.3 避免贫瘠高原

**策略**：

1. **局部代价函数**：
   $$C_{local} = \sum_i c_i \langle O_i \rangle$$
   其中 $O_i$ 只作用于少数量子比特。

2. **浅层电路**：
   深度 $L < O(\log n)$ 可以避免。

3. **好的初始化**：
   - 从恒等电路开始（$\theta \approx 0$）
   - 从HF态开始（化学问题）
   - 分层预训练

4. **相关性限制的ansatz**：
   限制每个量子比特只与少数量子比特纠缠。

### 4.6.4 Ansatz的Lipschitz连续性

**定义**：如果存在常数 $L$ 使得
$$|E(\boldsymbol{\theta}) - E(\boldsymbol{\theta}')| \leq L \|\boldsymbol{\theta} - \boldsymbol{\theta}'\|$$

则损失函数是Lipschitz连续的。

**意义**：Lipschitz常数小意味着函数变化缓慢，优化更容易。

---

## 4.7 特定问题的Ansatz设计

### 4.7.1 分子基态

**推荐**：
1. 小分子（< 20 量子比特）：UCCSD
2. 中分子（20-50 量子比特）：ADAPT-VQE或简化UCC
3. 大分子（> 50 量子比特）：需要活性空间约化

### 4.7.2 激发态

**方法**：
1. VQD（变分量子偏折）
2. SSVQE（子空间搜索）
3. qEOM（量子运动方程）

**Ansatz修改**：可能需要更多参数来表示正交态。

### 4.7.3 强关联系统

**特点**：HF是坏的参考态

**策略**：
1. 多参考UCCSD
2. 更深的ansatz
3. ADAPT-VQE自适应构建

### 4.7.4 周期性系统

**挑战**：无限系统 → k空间采样

**Ansatz设计**：
1. 每个k点独立ansatz
2. 或用实空间局域ansatz + 周期边界

---

## 4.8 Ansatz复杂度的理论边界

### 4.8.1 下界

**定理**：达到能量精度 $\epsilon$ 需要的ansatz复杂度（参数数/电路深度）有下界。

对于一般哈密顿量，可能需要 $O(e^n)$ 复杂度。

### 4.8.2 上界（特定问题）

对于**局域哈密顿量**基态，存在 $O(\text{poly}(n))$ 复杂度的ansatz可以达到常数精度。

### 4.8.3 量子优势的条件

VQE可能有量子优势当：
1. 经典方法（如CCSD）失败
2. 但存在相对浅的量子电路可以表示基态
3. 这种情况在强关联系统中可能存在

---

## 4.9 小结

### 4.9.1 Ansatz选择指南

```
             问题类型
                │
       ┌────────┴────────┐
       ↓                  ↓
   弱关联             强关联
       │                  │
   UCCSD/HW           ADAPT-VQE
       │                  │
  硬件限制?          硬件限制?
   ↓    ↓             ↓    ↓
  是    否           是    否
   │    │             │    │
  HW  UCCSD       qubit-ADAPT  ADAPT
```

### 4.9.2 关键公式

**UCCSD波函数**：
$$|\Psi_{UCCSD}\rangle = e^{\hat{T} - \hat{T}^\dagger}|\Phi_{HF}\rangle$$

**激发算符**：
$$\hat{\tau}_i^a = a_a^\dagger a_i - a_i^\dagger a_a$$

**ADAPT梯度**：
$$g_k = \langle \psi | [H, \hat{\tau}_k - \hat{\tau}_k^\dagger] | \psi \rangle$$

### 4.9.3 实践建议

1. **从简单开始**：先尝试少层硬件高效
2. **利用物理**：化学问题优先UCCSD
3. **监控梯度**：梯度太小则换ansatz
4. **利用对称性**：大幅减少参数
5. **迭代改进**：ADAPT-VQE自动优化结构
