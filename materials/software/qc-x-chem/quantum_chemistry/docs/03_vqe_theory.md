# 第三章：变分量子本征求解器（VQE）理论

## 📚 本章概览

**什么是VQE？**

**变分量子本征求解器**（Variational Quantum Eigensolver, VQE）是一种**混合量子-经典算法**，用于在量子计算机上求解分子的基态能量。

**核心思想**：
1. 用量子计算机制备参数化的量子态
2. 测量这个态的能量
3. 用经典计算机优化参数，使能量最小
4. 重复直到找到基态能量

**为什么需要VQE？**

**问题**：我们想计算分子的基态能量（最低能量）

**经典方法的问题**：
- 对于大分子，经典计算非常困难（指数复杂度）
- 强关联系统（如过渡金属配合物）经典方法精度不够

**量子计算的优势**：
- 量子态可以表示指数级的信息
- 可以模拟量子系统（分子本身就是量子系统）

**VQE的特点**：
- **混合算法**：结合量子计算和经典优化
- **适合NISQ时代**：不需要容错量子计算
- **灵活**：可以处理各种分子系统

**学习路径**：
1. 理解变分原理（为什么最小化能量能找到基态）
2. 理解VQE算法框架（如何混合量子-经典）
3. 理解能量测量（如何在量子计算机上测量能量）
4. 理解参数优化（如何找到最优参数）
5. 理解实际挑战（贫瘠高原、误差等）

**本章结构**：
- **3.1 变分原理**：理论基础，为什么最小化能量能找到基态
- **3.2 VQE算法框架**：整体流程，混合量子-经典结构
- **3.3 能量测量**：如何在量子计算机上测量能量
- **3.4 参数化量子电路**：如何构造试探波函数
- **3.5 梯度计算**：如何计算梯度来优化参数
- **3.6 优化器**：经典优化算法
- **3.7 贫瘠高原**：深层电路的挑战
- **3.8 误差分析**：各种误差来源
- **3.9 与其他方法比较**：VQE的优势和局限
- **3.10 小结**：核心概念总结

---

**什么是VQE？**

**变分量子本征求解器**（Variational Quantum Eigensolver, VQE）是一种**混合量子-经典算法**，用于在量子计算机上求解分子的基态能量。

**核心思想**：
1. 用量子计算机制备参数化的量子态
2. 测量这个态的能量
3. 用经典计算机优化参数，使能量最小
4. 重复直到找到基态能量

**为什么需要VQE？**

**问题**：我们想计算分子的基态能量（最低能量）

**经典方法的问题**：
- 对于大分子，经典计算非常困难（指数复杂度）
- 强关联系统（如过渡金属配合物）经典方法精度不够

**量子计算的优势**：
- 量子态可以表示指数级的信息
- 可以模拟量子系统（分子本身就是量子系统）

**VQE的特点**：
- **混合算法**：结合量子计算和经典优化
- **适合NISQ时代**：不需要容错量子计算
- **灵活**：可以处理各种分子系统

**学习路径**：
1. 理解变分原理（为什么最小化能量能找到基态）
2. 理解VQE算法框架（如何混合量子-经典）
3. 理解能量测量（如何在量子计算机上测量能量）
4. 理解参数优化（如何找到最优参数）
5. 理解实际挑战（贫瘠高原、误差等）

---

## 3.1 变分原理

### 3.1.0 为什么需要变分原理？

**核心问题**：如何找到分子的基态能量？

**直接方法**：求解Schrödinger方程
$$\hat{H}|\psi\rangle = E|\psi\rangle$$

**问题**：
- 对于大分子，精确求解几乎不可能
- 需要找到所有本征态和本征值

**变分方法**：不直接求解，而是**最小化能量期望值**

**关键洞察**：如果我们能找到一个态，它的能量期望值最小，那么这个态就是基态！

### 3.1.1 Rayleigh-Ritz变分原理

**定理**：对于任意归一化态 $|\psi\rangle$，有：

$$\boxed{E_0 \leq \langle \psi | \hat{H} | \psi \rangle}$$

其中 $E_0$ 是哈密顿量 $\hat{H}$ 的基态能量。

**等号成立条件**：当且仅当 $|\psi\rangle$ 是基态 $|E_0\rangle$。

#### 直观理解

**物理意义**：
- 左边：真实的基态能量 $E_0$（最低可能的能量）
- 右边：任意猜测的态 $|\psi\rangle$ 的能量期望值

**定理说**：无论你猜什么态，它的能量期望值**永远不会低于**真实的基态能量！

**类比**：
- 想象你在找一座山的最低点（基态能量）
- 无论你站在哪里（任意态），你所在位置的高度（能量期望值）**永远不会低于**真正的最低点
- 只有当你站在最低点时（基态），高度才等于最低点

**数学表达**：
```
真实基态能量 ≤ 任意猜测态的能量期望值
     E₀      ≤    ⟨ψ|H|ψ⟩
```

### 3.1.2 证明

**步骤1：将任意态展开为本征态**

设 $\hat{H}$ 的本征态为 $\{|E_n\rangle\}$，本征值为 $\{E_n\}$，且 $E_0 \leq E_1 \leq E_2 \leq \cdots$

任意归一化态 $|\psi\rangle$ 可以展开为：
$$|\psi\rangle = \sum_n c_n |E_n\rangle$$

其中展开系数满足归一化条件：
$$\sum_n |c_n|^2 = 1$$

**物理意义**：
- $|c_n|^2$ 是态 $|\psi\rangle$ 中"包含"本征态 $|E_n\rangle$ 的概率
- 归一化条件：所有概率之和为1

**步骤2：计算能量期望值**

$$\langle \psi | \hat{H} | \psi \rangle = \left\langle \sum_m c_m^* \langle E_m | \right| \hat{H} \left| \sum_n c_n |E_n\rangle \right\rangle$$

展开：
$$= \sum_{m,n} c_m^* c_n \langle E_m | \hat{H} | E_n \rangle$$

由于 $|E_n\rangle$ 是 $\hat{H}$ 的本征态：
$$\hat{H}|E_n\rangle = E_n|E_n\rangle$$

所以：
$$\langle E_m | \hat{H} | E_n \rangle = E_n \langle E_m | E_n \rangle = E_n \delta_{mn}$$

（因为本征态正交：$\langle E_m | E_n \rangle = \delta_{mn}$）

因此：
$$\langle \psi | \hat{H} | \psi \rangle = \sum_{m,n} c_m^* c_n E_n \delta_{mn} = \sum_n c_n^* c_n E_n = \sum_n |c_n|^2 E_n$$

**步骤3：证明不等式**

由于 $E_0 \leq E_1 \leq E_2 \leq \cdots$（基态能量最低），我们有：
$$E_n \geq E_0 \quad \text{对所有} \quad n$$

所以：
$$\langle \psi | \hat{H} | \psi \rangle = \sum_n |c_n|^2 E_n \geq \sum_n |c_n|^2 E_0 = E_0 \sum_n |c_n|^2 = E_0$$

**等号成立**：当且仅当 $|\psi\rangle$ 只包含基态，即 $|\psi\rangle = |E_0\rangle$（可能差一个相位因子 $e^{i\phi}$）

**证明完成** ✓

### 3.1.3 变分方法的应用

**思想**：构造参数化的试探波函数 $|\psi(\boldsymbol{\theta})\rangle$，最小化：

$$E(\boldsymbol{\theta}) = \langle \psi(\boldsymbol{\theta}) | \hat{H} | \psi(\boldsymbol{\theta}) \rangle$$

最优参数 $\boldsymbol{\theta}^*$ 给出基态能量的上界。

#### 详细解释

**参数化试探波函数**：
- $\boldsymbol{\theta} = (\theta_1, \theta_2, \ldots, \theta_p)$ 是**可调参数**
- 通过改变参数，我们可以改变量子态 $|\psi(\boldsymbol{\theta})\rangle$
- 这就像"调整旋钮"来改变系统的状态

**优化过程**：
1. 从某个初始参数 $\boldsymbol{\theta}_0$ 开始
2. 计算能量 $E(\boldsymbol{\theta}_0)$
3. 调整参数，使能量降低
4. 重复直到能量不再降低

**最优参数**：
$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} E(\boldsymbol{\theta})$$

**结果**：
$$E(\boldsymbol{\theta}^*) \geq E_0$$

如果我们的参数化足够好，$E(\boldsymbol{\theta}^*)$ 会非常接近 $E_0$。

**例子**：H₂分子

- 基态能量：$E_0 = -1.137$ Ha（Hartree，能量单位）
- 如果我们用参数化态，可能得到：$E(\boldsymbol{\theta}^*) = -1.136$ Ha
- 误差：$0.001$ Ha（非常接近！）

**关键优势**：
- 不需要精确求解Schrödinger方程
- 只需要找到能量最小的参数
- 适合在量子计算机上实现

### 3.1.1 Rayleigh-Ritz变分原理

**定理**：对于任意归一化态 $|\psi\rangle$，有：

$$\boxed{E_0 \leq \langle \psi | \hat{H} | \psi \rangle}$$

其中 $E_0$ 是哈密顿量 $\hat{H}$ 的基态能量。

**等号成立条件**：当且仅当 $|\psi\rangle$ 是基态 $|E_0\rangle$。

### 3.1.2 证明

设 $\hat{H}$ 的本征态为 $\{|E_n\rangle\}$，本征值为 $\{E_n\}$，且 $E_0 \leq E_1 \leq E_2 \leq \cdots$

任意态可展开：
$$|\psi\rangle = \sum_n c_n |E_n\rangle, \quad \sum_n |c_n|^2 = 1$$

则：
$$\langle \psi | \hat{H} | \psi \rangle = \sum_n |c_n|^2 E_n \geq E_0 \sum_n |c_n|^2 = E_0 \quad \square$$

### 3.1.3 变分方法的应用

**思想**：构造参数化的试探波函数 $|\psi(\boldsymbol{\theta})\rangle$，最小化：

$$E(\boldsymbol{\theta}) = \langle \psi(\boldsymbol{\theta}) | \hat{H} | \psi(\boldsymbol{\theta}) \rangle$$

最优参数 $\boldsymbol{\theta}^*$ 给出基态能量的上界。

---

## 3.2 VQE算法框架

### 3.2.0 为什么是"混合"算法？

**关键问题**：为什么需要经典计算机和量子计算机一起工作？

**答案**：
- **量子计算机**：擅长制备和测量量子态（指数级信息）
- **经典计算机**：擅长优化和数值计算（快速、精确）

**分工**：
- 量子计算机：制备参数化态，测量能量
- 经典计算机：优化参数，决定下一步怎么做

**优势**：
- 充分利用两种计算机的优势
- 不需要容错量子计算（适合NISQ时代）

### 3.2.1 混合量子-经典结构

```
     ┌─────────────────────────────────────────┐
     │           经典计算机                      │
     │  ┌─────────────────────────────────┐   │
     │  │  优化器: min E(θ)                │   │
     │  │  θ_{k+1} = θ_k - η∇E(θ_k)      │   │
     │  │                                  │   │
     │  │  任务：                            │   │
     │  │  - 接收能量值 E(θ)                │   │
     │  │  - 计算梯度或更新参数              │   │
     │  │  - 发送新参数 θ_{k+1}            │   │
     │  └───────────────┬─────────────────┘   │
     │                  │ θ_k                   │
     │                  ↓                       │
     │           发送参数到量子计算机            │
     └──────────────────┼──────────────────────┘
                        ↓
     ┌──────────────────┼──────────────────────┐
     │           量子计算机                      │
     │                  ↓                       │
     │           接收参数 θ_k                   │
     │                  ↓                       │
     │  |0⟩ ─── U(θ_k) ───┬─── 测量 ───→ E(θ_k)│
     │                  │                       │
     │    参数化电路     哈密顿量期望值         │
     │                  ↓                       │
     │           发送能量值到经典计算机         │
     └──────────────────┴──────────────────────┘
```

#### 详细解释每个部分

**经典计算机的任务**：

1. **优化器**：
   - 接收量子计算机测量的能量值 $E(\boldsymbol{\theta}_k)$
   - 根据优化算法（如梯度下降）计算新参数 $\boldsymbol{\theta}_{k+1}$
   - 发送新参数到量子计算机

2. **优化算法**：
   - 梯度下降：$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta \nabla E(\boldsymbol{\theta}_k)$
   - 其中 $\eta$ 是学习率（步长）
   - $\nabla E$ 是能量对参数的梯度

**量子计算机的任务**：

1. **态制备**：
   - 接收参数 $\boldsymbol{\theta}_k$
   - 应用参数化量子电路 $U(\boldsymbol{\theta}_k)$ 到初始态 $|0\rangle$
   - 得到参数化态：$|\psi(\boldsymbol{\theta}_k)\rangle = U(\boldsymbol{\theta}_k)|0\rangle$

2. **能量测量**：
   - 测量哈密顿量 $\hat{H}$ 的期望值
   - $E(\boldsymbol{\theta}_k) = \langle \psi(\boldsymbol{\theta}_k) | \hat{H} | \psi(\boldsymbol{\theta}_k) \rangle$
   - 发送能量值到经典计算机

**循环过程**：
```
经典计算机 → 发送参数 → 量子计算机
量子计算机 → 测量能量 → 经典计算机
经典计算机 → 更新参数 → 量子计算机
...（重复直到收敛）
```

### 3.2.2 VQE流程（详细步骤）

#### 步骤1：初始化

**选择初始参数** $\boldsymbol{\theta}_0$：

**常见策略**：
- **随机初始化**：参数从均匀分布或正态分布中随机选择
- **从HF态开始**：使用Hartree-Fock方法的解作为初始参数（更智能）
- **零初始化**：所有参数设为0（可能不是最好的选择）

**例子**：对于H₂分子，如果有3个参数：
$$\boldsymbol{\theta}_0 = (0.1, -0.2, 0.05)$$

#### 步骤2：态制备

**在量子计算机上制备参数化态**：

$$|\psi(\boldsymbol{\theta}_k)\rangle = U(\boldsymbol{\theta}_k)|0\rangle^{\otimes n}$$

**详细过程**：
1. 初始化：所有量子比特处于 $|0\rangle$ 态
   $$|0\rangle^{\otimes n} = |00\cdots 0\rangle$$
   
2. 应用参数化电路 $U(\boldsymbol{\theta}_k)$：
   - 这包含旋转门（$R_X, R_Y, R_Z$）和纠缠门（CNOT）
   - 参数 $\boldsymbol{\theta}_k$ 控制旋转角度
   
3. 得到参数化态 $|\psi(\boldsymbol{\theta}_k)\rangle$

**例子**：简单的2量子比特电路
```
q0: |0⟩ ─── R_Y(θ₁) ────●─── R_Y(θ₂) ──── |ψ(θ)⟩
                        │
q1: |0⟩ ────────────────X─────────────────
```

#### 步骤3：能量测量

**测量哈密顿量的期望值**：

$$E(\boldsymbol{\theta}_k) = \langle \psi(\boldsymbol{\theta}_k) | \hat{H} | \psi(\boldsymbol{\theta}_k) \rangle$$

**详细过程**（见3.3节）：
1. 将哈密顿量分解为Pauli串：$\hat{H} = \sum_i c_i P_i$
2. 测量每个Pauli串的期望值：$\langle P_i \rangle$
3. 加权求和：$E = \sum_i c_i \langle P_i \rangle$

**例子**：H₂分子
- 哈密顿量有15个Pauli项
- 需要测量15次（或分组后更少）
- 得到能量值，例如：$E(\boldsymbol{\theta}_k) = -1.12$ Ha

#### 步骤4：参数更新

**经典优化器更新参数**：

**梯度下降**：
$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta \nabla E(\boldsymbol{\theta}_k)$$

**详细过程**：
1. 计算梯度 $\nabla E(\boldsymbol{\theta}_k)$（见3.5节）
2. 选择学习率 $\eta$（如 $\eta = 0.01$）
3. 更新参数：$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta \nabla E$

**例子**：
- 当前参数：$\boldsymbol{\theta}_k = (0.1, -0.2, 0.05)$
- 能量：$E = -1.12$ Ha
- 梯度：$\nabla E = (0.5, -0.3, 0.2)$
- 学习率：$\eta = 0.01$
- 新参数：$\boldsymbol{\theta}_{k+1} = (0.1 - 0.01×0.5, -0.2 - 0.01×(-0.3), 0.05 - 0.01×0.2) = (0.095, -0.197, 0.048)$

#### 步骤5：迭代

**重复步骤2-4**，直到满足收敛条件。

**迭代过程**：
```
k=0: θ₀ → E(θ₀) = -1.10 Ha
k=1: θ₁ → E(θ₁) = -1.12 Ha  (能量降低！)
k=2: θ₂ → E(θ₂) = -1.135 Ha (能量继续降低)
k=3: θ₃ → E(θ₃) = -1.136 Ha (能量几乎不变)
k=4: θ₄ → E(θ₄) = -1.136 Ha (收敛！)
```

### 3.2.3 收敛条件

**什么时候停止迭代？**

**常用收敛判据**：

1. **能量变化**：
   $$|E_{k+1} - E_k| < \epsilon$$
   - 如果能量变化很小（如 $< 10^{-6}$ Ha），认为收敛
   - **优点**：直观，容易理解
   - **缺点**：可能陷入局部最小值

2. **梯度范数**：
   $$\|\nabla E(\boldsymbol{\theta}_k)\| < \epsilon$$
   - 如果梯度很小，说明接近最小值
   - **优点**：理论上更严格
   - **缺点**：需要计算梯度（额外开销）

3. **最大迭代次数**：
   - 如果达到最大迭代次数（如1000次），停止
   - **优点**：防止无限循环
   - **缺点**：可能未收敛

**实际例子**：

对于H₂分子：
- 初始能量：$E_0 = -1.10$ Ha
- 迭代50次后：$E_{50} = -1.136$ Ha
- 能量变化：$|E_{50} - E_{49}| = 0.0001$ Ha $< 10^{-4}$ Ha
- **判断**：收敛！停止迭代

**典型收敛曲线**：
```
能量 (Ha)
  ↑
-1.0│
    │  ●
-1.1│    ●
    │      ●
-1.2│        ●
    │          ●●●●●●●
-1.3│                    ●●●●●●●●●
    └────────────────────────────────→ 迭代次数
     0  10  20  30  40  50  60
```

**观察**：
- 开始时能量快速下降
- 后来能量变化很小（接近收敛）
- 最终达到基态能量附近

---

## 3.3 能量测量

### 3.3.0 核心问题：如何在量子计算机上测量能量？

**问题**：我们想测量 $E = \langle \psi | \hat{H} | \psi \rangle$

**挑战**：
- 量子计算机不能直接测量算符 $\hat{H}$
- 只能测量**Pauli算符**（$X, Y, Z$）的期望值

**解决方案**：
1. 将哈密顿量 $\hat{H}$ 分解为Pauli串之和
2. 测量每个Pauli串的期望值
3. 加权求和得到总能量

### 3.3.1 哈密顿量分解

**量子比特哈密顿量写为Pauli串之和**：
$$\hat{H} = \sum_i c_i P_i$$

其中：
- $P_i$ 是**Pauli串**（如 $Z_0 Z_1 X_2$，多个Pauli算符的张量积）
- $c_i$ 是**实系数**（来自JW变换，见第二章）

**例子**：H₂分子的哈密顿量（简化）

$$\hat{H} = -0.8126 I + 0.1712 Z_0 + 0.1712 Z_1 - 0.2228 Z_2 - 0.2228 Z_3$$
$$+ 0.1686 Z_0 Z_1 + 0.1206 Z_0 Z_2 + \cdots$$
$$+ 0.0453 X_0 X_1 Y_2 Y_3 + \cdots$$

**观察**：
- 有**15个Pauli项**
- 每项是Pauli串（如 $Z_0$, $Z_0 Z_1$, $X_0 X_1 Y_2 Y_3$）
- 每项有系数（如 $0.1712$, $-0.2228$）

**为什么可以这样分解？**

**数学原理**：
- 所有 $n$ 量子比特的算符都可以写成Pauli串的线性组合
- Pauli串构成**完备基**（就像向量可以用基向量展开）
- 这类似于Fourier展开：任何函数都可以写成正弦函数的和

### 3.3.2 期望值计算

**由线性性**：
$$\langle \hat{H} \rangle = \left\langle \sum_i c_i P_i \right\rangle = \sum_i c_i \langle P_i \rangle$$

**详细推导**：

$$\langle \psi | \hat{H} | \psi \rangle = \left\langle \psi \left| \sum_i c_i P_i \right| \psi \right\rangle$$

由于期望值是线性的：
$$= \sum_i c_i \langle \psi | P_i | \psi \rangle = \sum_i c_i \langle P_i \rangle$$

**物理意义**：
- 总能量 = 所有Pauli项的能量贡献之和
- 每项贡献 = 系数 × 该项的期望值

**例子**：H₂分子（简化，只有3项）

$$\hat{H} = c_1 Z_0 + c_2 Z_1 + c_3 Z_0 Z_1$$

测量：
- $\langle Z_0 \rangle = 0.8$（量子比特0的Z期望值）
- $\langle Z_1 \rangle = 0.9$（量子比特1的Z期望值）
- $\langle Z_0 Z_1 \rangle = 0.7$（两个量子比特的关联）

计算能量：
$$E = c_1 \times 0.8 + c_2 \times 0.9 + c_3 \times 0.7$$

**关键点**：每个 $\langle P_i \rangle$ 需要**单独测量**！

**为什么不能同时测量所有项？**

因为不同的Pauli串可能**不对易**，不能同时精确测量。

例如：$Z_0$ 和 $X_0$ 不对易，不能同时测量它们的精确值。

**解决方案**：分别测量每项（或分组测量，见3.3.5节）

### 3.3.3 单个Pauli串的测量

#### 核心问题：如何测量 $\langle P \rangle$？

**问题**：量子计算机只能直接测量**Z基**（计算基），即测量 $|0\rangle$ 或 $|1\rangle$。

**解决方案**：通过**基变换**，将其他Pauli算符变换到Z基测量。

#### 详细步骤

**步骤1：基变换**

**目标**：将非Z的Pauli变换到Z基

**规则**：
- **$X \to Z$**：应用 **$H$**（Hadamard门）
  - $H|0\rangle = |+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$
  - $H|1\rangle = |-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$
  - $H$ 将X基变换到Z基

- **$Y \to Z$**：应用 **$HS^\dagger$**
  - $S^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}$（相位门）
  - $HS^\dagger$ 将Y基变换到Z基

- **$Z \to Z$**：不需要变换（已经在Z基）

**为什么这样变换？**

**X到Z的变换**：
- $X$ 的本征态是 $|+\rangle$ 和 $|-\rangle$
- $Z$ 的本征态是 $|0\rangle$ 和 $|1\rangle$
- $H$ 门将 $|+\rangle \leftrightarrow |0\rangle$，$|-\rangle \leftrightarrow |1\rangle$
- 所以测量 $H|\psi\rangle$ 在Z基，等价于测量 $|\psi\rangle$ 在X基

**Y到Z的变换**：
- $Y$ 的本征态是 $|\pm i\rangle = \frac{1}{\sqrt{2}}(|0\rangle \pm i|1\rangle)$
- $S^\dagger$ 旋转相位，$H$ 变换到Z基
- $HS^\dagger$ 将Y基变换到Z基

**步骤2：测量**

在计算基（Z基）测量所有相关量子比特，得到测量结果 $m_j \in \{0, 1\}$。

**步骤3：计算期望值**

$$\langle P \rangle = \mathbb{E}[(-1)^{\oplus_j m_j}]$$

其中 $\oplus_j m_j$ 表示所有测量结果的**异或**（XOR）。

**为什么是异或？**

**原因**：Pauli串 $P = P_0 \otimes P_1 \otimes \cdots$ 的期望值是所有量子比特测量结果的**乘积**。

对于Z基测量：
- 测量结果 $m_j = 0$ 对应 $+1$
- 测量结果 $m_j = 1$ 对应 $-1$

所以：
$$\langle P \rangle = \mathbb{E}[\prod_j (-1)^{m_j}] = \mathbb{E}[(-1)^{\sum_j m_j}]$$

对于**Z算符**，这是正确的。但对于**X和Y算符**（经过基变换后），需要根据变换调整符号。

**实际上**，对于一般的Pauli串，公式是：
$$\langle P \rangle = \mathbb{E}[(-1)^{\oplus_j m_j}]$$

其中异或 $\oplus$ 考虑了基变换的影响。

#### 完整例子：测量 $\langle X_0 Y_1 Z_2 \rangle$

**目标**：测量 $X_0 Y_1 Z_2$ 的期望值

**步骤1：基变换**

```
q0: ─── H ────────────── (X → Z)
q1: ─── S† ─── H ─────── (Y → Z)
q2: ────────────────── (Z，不需要变换)
```

**详细电路**：
```
q0: |ψ⟩ ─── H ────────────── |·⟩ ─→ m0
q1: |ψ⟩ ─── S† ─── H ─────── |·⟩ ─→ m1  
q2: |ψ⟩ ────────────────── |·⟩ ─→ m2
```

**步骤2：测量**

在Z基测量所有3个量子比特：
- 量子比特0：$m_0 \in \{0, 1\}$
- 量子比特1：$m_1 \in \{0, 1\}$
- 量子比特2：$m_2 \in \{0, 1\}$

**步骤3：计算期望值**

$$\langle X_0 Y_1 Z_2 \rangle = \mathbb{E}[(-1)^{m_0 \oplus m_1 \oplus m_2}]$$

**具体计算**：

假设我们进行了1000次测量，得到：
- 500次：$(m_0, m_1, m_2) = (0, 0, 0)$ → $(-1)^0 = +1$
- 200次：$(m_0, m_1, m_2) = (0, 0, 1)$ → $(-1)^1 = -1$
- 150次：$(m_0, m_1, m_2) = (0, 1, 0)$ → $(-1)^1 = -1$
- 150次：$(m_0, m_1, m_2) = (1, 0, 0)$ → $(-1)^1 = -1$
- 其他组合：...

**期望值**：
$$\langle X_0 Y_1 Z_2 \rangle = \frac{500 \times (+1) + 200 \times (-1) + 150 \times (-1) + 150 \times (-1) + \cdots}{1000}$$

**简化例子**：如果所有测量结果都是 $(0, 0, 0)$：
$$\langle X_0 Y_1 Z_2 \rangle = \frac{1000 \times (+1)}{1000} = +1$$

#### 更简单的例子：测量 $\langle Z_0 \rangle$

**目标**：测量 $Z_0$ 的期望值

**步骤1：基变换**
- $Z_0$ 已经在Z基，**不需要变换**

**步骤2：测量**
```
q0: |ψ⟩ ────────────────── |·⟩ ─→ m0
```

**步骤3：计算**
$$\langle Z_0 \rangle = \mathbb{E}[(-1)^{m_0}]$$

**具体**：
- 如果 $m_0 = 0$（测量到 $|0\rangle$）：贡献 $+1$
- 如果 $m_0 = 1$（测量到 $|1\rangle$）：贡献 $-1$

**例子**：1000次测量
- 800次测量到 $|0\rangle$（$m_0 = 0$）：$+1$
- 200次测量到 $|1\rangle$（$m_0 = 1$）：$-1$

$$\langle Z_0 \rangle = \frac{800 \times (+1) + 200 \times (-1)}{1000} = \frac{600}{1000} = 0.6$$

**物理意义**：
- $\langle Z_0 \rangle = 0.6$ 意味着量子比特0更倾向于处于 $|0\rangle$ 态
- 如果 $\langle Z_0 \rangle = 1$，则完全在 $|0\rangle$ 态
- 如果 $\langle Z_0 \rangle = -1$，则完全在 $|1\rangle$ 态

### 3.3.4 测量统计误差

#### 为什么需要多次测量？

**问题**：单次测量只能得到 $\pm 1$，不是期望值！

**例子**：测量 $\langle Z_0 \rangle$
- 单次测量：可能得到 $+1$（测量到 $|0\rangle$）或 $-1$（测量到 $|1\rangle$）
- 但期望值可能是 $0.6$（不是 $\pm 1$！）

**解决方案**：**多次测量**，取平均

**原理**：
- 每次测量是**随机**的
- 多次测量的**平均值**接近期望值
- 测量次数越多，估计越准确

**例子**：测量 $\langle Z_0 \rangle = 0.6$

| 测量次数 | 测量结果 | 平均值 | 误差 |
|---------|---------|--------|------|
| 10次 | +1, -1, +1, +1, -1, +1, +1, +1, -1, +1 | 0.6 | 0.0 |
| 100次 | ... | 0.58 | 0.02 |
| 1000次 | ... | 0.601 | 0.001 |
| 10000次 | ... | 0.6001 | 0.0001 |

**观察**：测量次数越多，平均值越接近真实值！

#### 统计误差分析

**单次测量的方差**：

对于Pauli算符 $P$（本征值为 $\pm 1$）：
- 测量结果：$+1$ 或 $-1$
- 概率：$P(+1) = \frac{1 + \langle P \rangle}{2}$，$P(-1) = \frac{1 - \langle P \rangle}{2}$

**方差**：
$$\text{Var}[P] = \mathbb{E}[P^2] - (\mathbb{E}[P])^2 = 1 - \langle P \rangle^2$$

**$N_s$ 次测量的方差**：

由于测量是独立的，平均值的方差是：
$$\text{Var}[\langle P \rangle] = \frac{\text{Var}[P]}{N_s} = \frac{1 - \langle P \rangle^2}{N_s}$$

**标准误差**（标准差）：
$$\sigma_P = \sqrt{\text{Var}[\langle P \rangle]} = \frac{\sqrt{1 - \langle P \rangle^2}}{\sqrt{N_s}}$$

**关键观察**：
- 误差与 $\sqrt{N_s}$ 成反比
- 要减少一半误差，需要**4倍**测量次数
- 要减少10倍误差，需要**100倍**测量次数

**例子**：$\langle Z_0 \rangle = 0.6$

$$\sigma_{Z_0} = \frac{\sqrt{1 - 0.6^2}}{\sqrt{N_s}} = \frac{\sqrt{0.64}}{\sqrt{N_s}} = \frac{0.8}{\sqrt{N_s}}$$

| 测量次数 | 标准误差 |
|---------|---------|
| 100 | 0.08 |
| 1000 | 0.025 |
| 10000 | 0.008 |

#### 总能量误差

**总能量**：
$$E = \sum_i c_i \langle P_i \rangle$$

**总能量误差**：

由于各项独立测量，总方差的方差是各项方差之和：
$$\text{Var}[\langle H \rangle] = \text{Var}\left[\sum_i c_i \langle P_i \rangle\right] = \sum_i c_i^2 \text{Var}[\langle P_i \rangle]$$

代入每项的方差：
$$= \sum_i c_i^2 \frac{1 - \langle P_i \rangle^2}{N_{s,i}}$$

**标准误差**：
$$\sigma_E = \sqrt{\sum_i c_i^2 \frac{1 - \langle P_i \rangle^2}{N_{s,i}}}$$

**例子**：H₂分子（简化，3项）

假设：
- $c_1 = 0.1$，$\langle P_1 \rangle = 0.8$，$N_{s,1} = 1000$
- $c_2 = 0.2$，$\langle P_2 \rangle = 0.6$，$N_{s,2} = 1000$
- $c_3 = 0.15$，$\langle P_3 \rangle = 0.7$，$N_{s,3} = 1000$

计算：
$$\sigma_E = \sqrt{0.1^2 \times \frac{1-0.8^2}{1000} + 0.2^2 \times \frac{1-0.6^2}{1000} + 0.15^2 \times \frac{1-0.7^2}{1000}}$$
$$= \sqrt{0.01 \times 0.00036 + 0.04 \times 0.00064 + 0.0225 \times 0.00051}$$
$$= \sqrt{0.0000036 + 0.0000256 + 0.0000115} = \sqrt{0.0000407} \approx 0.0064 \text{ Ha}$$

**结论**：总能量误差约为 $0.0064$ Ha（约4 mHa）

### 3.3.5 测量优化：Pauli分组

#### 问题：如何减少测量次数？

**挑战**：H₂分子有15个Pauli项，如果分别测量，需要15次（或更多，因为每次需要多次测量取平均）

**关键洞察**：**对易的Pauli串可以同时测量**！

**原理**：
- 如果 $[P_1, P_2] = 0$（对易），则 $P_1$ 和 $P_2$ 有共同本征态
- 可以在**同一次测量**中同时得到 $\langle P_1 \rangle$ 和 $\langle P_2 \rangle$
- 这大大减少了测量次数！

#### 分组策略

**策略1：QWC (Qubit-wise Commuting)**

**定义**：两个Pauli串 $P_1$ 和 $P_2$ 是QWC的，如果：
- 在每个量子比特上，$P_1$ 和 $P_2$ 的Pauli算符**相同**或**其一为I**

**判断方法**：
- 比较每个量子比特位置上的Pauli算符
- 如果都是 $X, X$ 或 $Y, Y$ 或 $Z, Z$ 或 $X, I$ 等 → QWC
- 如果是 $X, Y$ 或 $X, Z$ → 不是QWC

**例子1**：$\{Z_0, Z_0 Z_1, Z_1\}$

- $Z_0$：量子比特0上是Z，量子比特1上是I
- $Z_0 Z_1$：量子比特0上是Z，量子比特1上是Z
- $Z_1$：量子比特0上是I，量子比特1上是Z

**检查**：
- 量子比特0：$Z, Z, I$ → 都是Z或I ✓
- 量子比特1：$I, Z, Z$ → 都是Z或I ✓

**结论**：它们是QWC的，可以**同时测量**！

**测量电路**：
```
q0: |ψ⟩ ────────────────── |·⟩ ─→ m0
q1: |ψ⟩ ────────────────── |·⟩ ─→ m1
```

**计算**：
- $\langle Z_0 \rangle = \mathbb{E}[(-1)^{m_0}]$
- $\langle Z_0 Z_1 \rangle = \mathbb{E}[(-1)^{m_0 \oplus m_1}]$
- $\langle Z_1 \rangle = \mathbb{E}[(-1)^{m_1}]$

**一次测量得到三个期望值！** ✓

**例子2**：$\{X_0, X_0 X_1, X_1\}$

- 都是X算符，需要Hadamard门变换
- 但变换后可以同时测量

**测量电路**：
```
q0: |ψ⟩ ─── H ──────────── |·⟩ ─→ m0
q1: |ψ⟩ ─── H ──────────── |·⟩ ─→ m1
```

**计算**：
- $\langle X_0 \rangle = \mathbb{E}[(-1)^{m_0}]$
- $\langle X_0 X_1 \rangle = \mathbb{E}[(-1)^{m_0 \oplus m_1}]$
- $\langle X_1 \rangle = \mathbb{E}[(-1)^{m_1}]$

**策略2：General Commuting（一般对易）**

**定义**：两个Pauli串 $P_1$ 和 $P_2$ 对易，如果 $[P_1, P_2] = 0$

**判断方法**：
- 计算对易子：$[P_1, P_2] = P_1 P_2 - P_2 P_1$
- 如果等于0，则对易

**优点**：
- 可以找到**更多**可以同时测量的Pauli串
- 进一步减少测量次数

**缺点**：
- 判断更复杂（需要计算对易子）
- 可能需要更复杂的基变换

**例子**：$\{X_0 Y_1, Y_0 X_1\}$

**检查对易**：
$$[X_0 Y_1, Y_0 X_1] = X_0 Y_1 Y_0 X_1 - Y_0 X_1 X_0 Y_1$$

由于不同量子比特上的算符对易：
$$= X_0 Y_0 Y_1 X_1 - Y_0 X_0 X_1 Y_1$$

由于 $X_0 Y_0 = -Y_0 X_0$（同一量子比特上反对易）：
$$= -Y_0 X_0 Y_1 X_1 - Y_0 X_0 X_1 Y_1 = -Y_0 X_0 (Y_1 X_1 + X_1 Y_1) = 0$$

**结论**：它们对易，可以同时测量！

#### 分组效果

**例子**：H₂分子（15个Pauli项）

**不分组**：
- 需要15次测量（每次测量一个Pauli项）
- 总测量次数：$15 \times N_s$（$N_s$ 是每次的测量次数）

**QWC分组后**：
- 可能分成5-6组
- 总测量次数：$6 \times N_s$（减少约60%！）

**General Commuting分组后**：
- 可能分成3-4组
- 总测量次数：$4 \times N_s$（减少约73%！）

**实际效果**：
- 如果 $N_s = 1000$（每次测量1000次取平均）
- 不分组：$15 \times 1000 = 15,000$ 次测量
- QWC分组：$6 \times 1000 = 6,000$ 次测量（节省60%）
- General分组：$4 \times 1000 = 4,000$ 次测量（节省73%）

**结论**：Pauli分组可以**显著减少**测量次数，提高VQE效率！

---

## 3.4 参数化量子电路（Ansatz）

### 3.4.0 什么是Ansatz？

**Ansatz**（德语，意思是"尝试"或"假设"）是**参数化的量子电路**，用于制备试探波函数。

**核心思想**：
- 我们不知道基态的确切形式
- 但我们可以用**参数化的电路**来"尝试"各种可能的态
- 通过优化参数，找到最接近基态的态

**类比**：
- 就像用多项式拟合数据：$f(x) = a_0 + a_1 x + a_2 x^2 + \cdots$
- 参数 $a_0, a_1, a_2, \ldots$ 可以调整
- Ansatz的参数 $\boldsymbol{\theta}$ 也可以调整

### 3.4.1 一般形式

$$|\psi(\boldsymbol{\theta})\rangle = U(\boldsymbol{\theta})|0\rangle^{\otimes n}$$

其中：
- $|0\rangle^{\otimes n} = |00\cdots 0\rangle$：初始态（所有量子比特在 $|0\rangle$）
- $U(\boldsymbol{\theta})$：参数化的酉算符（量子电路）
- $\boldsymbol{\theta} = (\theta_1, \theta_2, \ldots, \theta_p)$：可调参数

**物理意义**：
- 从简单的初始态 $|00\cdots 0\rangle$ 开始
- 通过参数化电路 $U(\boldsymbol{\theta})$ 变换
- 得到复杂的参数化态 $|\psi(\boldsymbol{\theta})\rangle$

**例子**：2量子比特系统

$$|\psi(\theta_1, \theta_2)\rangle = U(\theta_1, \theta_2)|00\rangle$$

通过改变 $\theta_1$ 和 $\theta_2$，我们可以得到不同的态。

### 3.4.2 层状结构

**典型ansatz由多层组成**：
$$U(\boldsymbol{\theta}) = \prod_{l=1}^L U_l(\boldsymbol{\theta}_l) = U_L(\boldsymbol{\theta}_L) \cdots U_2(\boldsymbol{\theta}_2) U_1(\boldsymbol{\theta}_1)$$

**结构**：
```
|0⟩ ─── U₁(θ₁) ─── U₂(θ₂) ─── ... ─── U_L(θ_L) ─── |ψ(θ)⟩
```

**每层包含**：
1. **旋转门**：单量子比特参数化门（$R_X, R_Y, R_Z$）
   - 作用：旋转量子比特的态
   - 参数：旋转角度 $\theta$

2. **纠缠门**：两量子比特门（CNOT, CZ）
   - 作用：创建量子比特之间的纠缠
   - 参数：无（固定门）

**例子**：简单的2层ansatz（2量子比特）

```
q0: |0⟩ ─── R_Y(θ₁) ────●─── R_Y(θ₃) ──── |ψ(θ)⟩
                        │
q1: |0⟩ ─── R_Y(θ₂) ────X─── R_Y(θ₄) ────
```

**解释**：
- 第1层：$R_Y(\theta_1)$ 和 $R_Y(\theta_2)$ 旋转两个量子比特
- CNOT门创建纠缠
- 第2层：$R_Y(\theta_3)$ 和 $R_Y(\theta_4)$ 进一步旋转

**参数**：$\boldsymbol{\theta} = (\theta_1, \theta_2, \theta_3, \theta_4)$

### 3.4.3 旋转门详解

**旋转门**是参数化的单量子比特门，用于旋转量子比特的态。

#### $R_X(\theta)$：绕X轴旋转

$$R_X(\theta) = e^{-i\frac{\theta}{2}X} = \cos\frac{\theta}{2} I - i\sin\frac{\theta}{2} X$$

**矩阵形式**：
$$R_X(\theta) = \begin{pmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

**作用**：
- $R_X(\pi)|0\rangle = |1\rangle$（翻转）
- $R_X(\pi/2)|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle - i|1\rangle)$（叠加态）

**物理意义**：在Bloch球上绕X轴旋转角度 $\theta$

#### $R_Y(\theta)$：绕Y轴旋转

$$R_Y(\theta) = e^{-i\frac{\theta}{2}Y} = \cos\frac{\theta}{2} I - i\sin\frac{\theta}{2} Y$$

**矩阵形式**：
$$R_Y(\theta) = \begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

**作用**：
- $R_Y(\pi)|0\rangle = |1\rangle$（翻转）
- $R_Y(\pi/2)|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) = |+\rangle$（叠加态）

**物理意义**：在Bloch球上绕Y轴旋转角度 $\theta$

#### $R_Z(\theta)$：绕Z轴旋转

$$R_Z(\theta) = e^{-i\frac{\theta}{2}Z} = \cos\frac{\theta}{2} I - i\sin\frac{\theta}{2} Z$$

**矩阵形式**：
$$R_Z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

**作用**：
- $R_Z(\theta)|0\rangle = e^{-i\theta/2}|0\rangle$（只改变相位）
- $R_Z(\theta)|1\rangle = e^{i\theta/2}|1\rangle$（只改变相位）

**物理意义**：在Bloch球上绕Z轴旋转角度 $\theta$（只改变相位，不改变概率）

#### 为什么用 $\theta/2$？

**原因**：为了与物理旋转对应

- 物理上，旋转角度 $\theta$ 对应 $R(\theta) = e^{-i\theta G/2}$
- 这确保 $R(2\pi) = -I$（旋转 $2\pi$ 得到负号，这是量子力学的特性）

**验证**：
$$R_X(2\pi) = e^{-i\pi X} = \cos\pi I - i\sin\pi X = -I \quad \checkmark$$

### 3.4.4 Ansatz的表达能力

**定义**：Ansatz的表达能力是其能表示的态空间大小。

#### 表达能力的重要性

**问题**：如果ansatz的表达能力不足，可能无法表示基态！

**例子**：
- **表达能力不足**：只能表示简单的态（如 $|00\rangle, |01\rangle$）
- 如果基态是复杂的纠缠态，可能无法达到

**表达能力过强**：
- 可以表示几乎所有态
- 但可能导致**贫瘠高原**（见3.7节）
- 优化变得困难

#### 过表达 vs 欠表达

**过表达(Overparameterization)**：参数数多于需要

**例子**：2量子比特系统
- 需要：3个参数（一般2量子比特态需要3个实参数）
- 实际：10个参数（ansatz有10个参数）

**影响**：
- **优点**：可能更容易优化（更多自由度）
- **缺点**：可能导致贫瘠高原，容易过拟合

**欠表达(Underparameterization)**：参数不足

**例子**：复杂分子系统
- 需要：100个参数才能表示基态
- 实际：10个参数（ansatz只有10个参数）

**影响**：
- **优点**：梯度景观可能更好（不会陷入贫瘠高原）
- **缺点**：**无法达到基态**（表达能力不足）

#### 如何选择Ansatz？

**策略**：
1. **物理启发**：使用基于物理原理的ansatz（如UCCSD，见第四章）
2. **平衡**：在表达能力和训练难度之间平衡
3. **从简单开始**：先尝试简单的ansatz，如果不够再增加复杂度

**常见Ansatz类型**：
- **硬件高效**：适合特定硬件，但可能表达能力不足
- **UCCSD**：基于量子化学，表达能力好，但可能复杂
- **QAOA**：适合组合优化问题

**总结**：
- Ansatz是VQE的"试探函数"
- 需要足够的表达能力，但不能太强
- 选择需要平衡表达能力和训练难度

---

## 3.5 梯度计算

### 3.5.0 为什么需要梯度？

**问题**：如何找到使能量最小的参数？

**方法1：随机搜索**
- 随机尝试不同的参数
- 效率低，不实用

**方法2：梯度下降**
- 计算能量对参数的梯度
- 沿着梯度**下降**的方向更新参数
- 更高效！

**梯度下降**：
$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta \nabla E(\boldsymbol{\theta}_k)$$

其中 $\nabla E = \left(\frac{\partial E}{\partial \theta_1}, \frac{\partial E}{\partial \theta_2}, \ldots\right)$ 是梯度。

**关键问题**：如何在量子计算机上计算梯度？

**挑战**：
- 不能直接计算导数（量子计算机不是微分器）
- 但可以通过**测量**来估计梯度

**解决方案**：**参数位移规则**（Parameter Shift Rule）

### 3.5.1 参数位移规则（Parameter Shift Rule）

**定理**：对于形如 $U(\theta) = e^{-i\frac{\theta}{2}G}$ 的门（$G$ 的本征值为 $\pm 1$），有：

$$\boxed{\frac{\partial}{\partial \theta} \langle \psi(\theta) | H | \psi(\theta) \rangle = \frac{1}{2}\left[ E(\theta + \frac{\pi}{2}) - E(\theta - \frac{\pi}{2}) \right]}$$

#### 直观理解

**核心思想**：通过**测量两个不同参数的能量**来计算梯度！

**步骤**：
1. 测量 $E(\theta + \pi/2)$（参数增加 $\pi/2$）
2. 测量 $E(\theta - \pi/2)$（参数减少 $\pi/2$）
3. 计算差值：$\frac{E(\theta + \pi/2) - E(\theta - \pi/2)}{2}$

**类比**：数值微分
- 经典数值微分：$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$
- 参数位移规则：$\frac{\partial E}{\partial \theta} = \frac{E(\theta + \pi/2) - E(\theta - \pi/2)}{2}$

**关键区别**：
- 经典：$h$ 很小（如 $10^{-6}$）
- 量子：位移是 $\pi/2$（固定值，不是小量！）

**为什么是 $\pi/2$？**

**原因**：对于旋转门 $R(\theta) = e^{-i\theta G/2}$，当 $G$ 的本征值为 $\pm 1$ 时，$\pi/2$ 的位移有特殊性质。

**例子**：$R_Y(\theta) = e^{-i\theta Y/2}$

- $Y$ 的本征值是 $\pm 1$
- $R_Y(\theta + \pi/2) = R_Y(\theta) R_Y(\pi/2)$
- 这个位移正好对应"正交"的旋转

#### 具体例子

**目标**：计算 $\frac{\partial E}{\partial \theta_1}$（能量对参数 $\theta_1$ 的梯度）

**步骤1**：测量 $E(\theta_1 + \pi/2, \theta_2, \ldots)$
- 将 $\theta_1$ 增加 $\pi/2$，其他参数不变
- 在量子计算机上测量能量

**步骤2**：测量 $E(\theta_1 - \pi/2, \theta_2, \ldots)$
- 将 $\theta_1$ 减少 $\pi/2$，其他参数不变
- 在量子计算机上测量能量

**步骤3**：计算梯度
$$\frac{\partial E}{\partial \theta_1} = \frac{1}{2}[E(\theta_1 + \pi/2, \theta_2, \ldots) - E(\theta_1 - \pi/2, \theta_2, \ldots)]$$

**例子**：假设
- $E(\theta_1 + \pi/2) = -1.12$ Ha
- $E(\theta_1 - \pi/2) = -1.10$ Ha

则：
$$\frac{\partial E}{\partial \theta_1} = \frac{1}{2}[-1.12 - (-1.10)] = \frac{1}{2}[-0.02] = -0.01$$

**物理意义**：
- 梯度为**负**，意味着增加 $\theta_1$ 会**降低**能量
- 所以应该**增加** $\theta_1$（沿着负梯度方向）

### 3.5.2 参数位移规则的证明

#### 证明思路

**目标**：证明 $\frac{\partial E}{\partial \theta} = \frac{1}{2}[E(\theta + \pi/2) - E(\theta - \pi/2)]$

**策略**：
1. 写出能量期望值的表达式
2. 计算对参数的导数
3. 利用 $G$ 的性质（本征值 $\pm 1$）简化
4. 得到参数位移规则

#### 详细证明

**步骤1：设置**

设 $|\psi(\theta)\rangle = U(\theta)V|0\rangle$，其中：
- $V$ 是 $\theta$ **之前**的所有门（不依赖于 $\theta$）
- $U(\theta) = e^{-i\frac{\theta}{2}G}$ 是依赖于 $\theta$ 的门
- $G$ 是Pauli算符（本征值为 $\pm 1$）

**能量期望值**：
$$E(\theta) = \langle \psi(\theta) | H | \psi(\theta) \rangle = \langle 0 | V^\dagger U(\theta)^\dagger H U(\theta) V | 0 \rangle$$

**步骤2：计算 $U(\theta)$ 的导数**

由于 $U(\theta) = e^{-i\frac{\theta}{2}G}$，利用指数函数的导数：

$$\frac{d U(\theta)}{d\theta} = \frac{d}{d\theta} e^{-i\frac{\theta}{2}G} = -\frac{i}{2}G e^{-i\frac{\theta}{2}G} = -\frac{i}{2}G U(\theta)$$

类似地：
$$\frac{d U(\theta)^\dagger}{d\theta} = \frac{i}{2}G U(\theta)^\dagger$$

**步骤3：计算能量对参数的导数**

$$\frac{dE}{d\theta} = \frac{d}{d\theta} \langle 0 | V^\dagger U(\theta)^\dagger H U(\theta) V | 0 \rangle$$

使用乘积法则：
$$= \langle 0 | V^\dagger \frac{dU(\theta)^\dagger}{d\theta} H U(\theta) V | 0 \rangle + \langle 0 | V^\dagger U(\theta)^\dagger H \frac{dU(\theta)}{d\theta} V | 0 \rangle$$

代入导数：
$$= \langle 0 | V^\dagger \frac{i}{2}G U(\theta)^\dagger H U(\theta) V | 0 \rangle + \langle 0 | V^\dagger U(\theta)^\dagger H \left(-\frac{i}{2}G\right) U(\theta) V | 0 \rangle$$

整理：
$$= \frac{i}{2}\langle 0 | V^\dagger U(\theta)^\dagger G H U(\theta) V | 0 \rangle - \frac{i}{2}\langle 0 | V^\dagger U(\theta)^\dagger H G U(\theta) V | 0 \rangle$$

定义 $H_{eff} = U(\theta)^\dagger H U(\theta)$（有效哈密顿量）：
$$= \frac{i}{2}\langle \psi(\theta) | G H_{eff} | \psi(\theta) \rangle - \frac{i}{2}\langle \psi(\theta) | H_{eff} G | \psi(\theta) \rangle$$

$$= \frac{i}{2}\langle \psi(\theta) | [G, H_{eff}] | \psi(\theta) \rangle$$

其中 $[G, H_{eff}] = G H_{eff} - H_{eff} G$ 是对易子。

**步骤4：利用 $G$ 的性质**

**关键性质**：$G$ 是Pauli算符，满足 $G^2 = I$（因为本征值为 $\pm 1$）

**技巧**：利用 $e^{i\frac{\pi}{2}G}$ 的性质

由于 $G^2 = I$，我们有：
$$e^{i\frac{\pi}{2}G} = \cos\frac{\pi}{2} I + i\sin\frac{\pi}{2} G = iG$$

所以：
$$G = -i e^{i\frac{\pi}{2}G}$$

**步骤5：推导参数位移规则**

**关键观察**：
$$U(\theta + \pi/2) = e^{-i\frac{\theta + \pi/2}{2}G} = e^{-i\frac{\theta}{2}G} e^{-i\frac{\pi}{4}G} = U(\theta) e^{-i\frac{\pi}{4}G}$$

类似地：
$$U(\theta - \pi/2) = U(\theta) e^{i\frac{\pi}{4}G}$$

**计算** $E(\theta + \pi/2)$：
$$E(\theta + \pi/2) = \langle 0 | V^\dagger U(\theta + \pi/2)^\dagger H U(\theta + \pi/2) V | 0 \rangle$$
$$= \langle 0 | V^\dagger e^{i\frac{\pi}{4}G} U(\theta)^\dagger H U(\theta) e^{-i\frac{\pi}{4}G} V | 0 \rangle$$

**计算** $E(\theta - \pi/2)$：
$$E(\theta - \pi/2) = \langle 0 | V^\dagger e^{-i\frac{\pi}{4}G} U(\theta)^\dagger H U(\theta) e^{i\frac{\pi}{4}G} V | 0 \rangle$$

**差值**：
$$E(\theta + \pi/2) - E(\theta - \pi/2) = \langle 0 | V^\dagger [e^{i\frac{\pi}{4}G} - e^{-i\frac{\pi}{4}G}] U(\theta)^\dagger H U(\theta) e^{-i\frac{\pi}{4}G} V | 0 \rangle$$
$$+ \langle 0 | V^\dagger e^{-i\frac{\pi}{4}G} U(\theta)^\dagger H U(\theta) [e^{-i\frac{\pi}{4}G} - e^{i\frac{\pi}{4}G}] V | 0 \rangle$$

**利用** $e^{i\frac{\pi}{4}G} - e^{-i\frac{\pi}{4}G} = 2i\sin\frac{\pi}{4} G = i\sqrt{2} G$：

经过详细计算（这里省略中间步骤），可以证明：
$$\frac{1}{2}[E(\theta + \pi/2) - E(\theta - \pi/2)] = \frac{i}{2}\langle \psi(\theta) | [G, H_{eff}] | \psi(\theta) \rangle = \frac{dE}{d\theta}$$

**证明完成** ✓

#### 为什么这个规则有用？

**优势**：
1. **不需要计算导数**：只需要测量两个能量值
2. **精确**：不是近似，是**精确**的梯度
3. **适合量子计算机**：只需要运行量子电路，不需要经典微分

**代价**：
- 每个参数需要**2次**能量测量
- 如果有 $p$ 个参数，需要 $2p$ 次测量
- 但这是值得的，因为梯度信息很有用

### 3.5.3 梯度计算代价

#### 计算复杂度分析

**对于 $p$ 个参数**：

**参数位移规则**：
- 每个参数需要**2次**能量评估（$\theta_i \pm \pi/2$）
- 总共需要 **$2p$ 次**电路执行

**例子**：
- 如果 $p = 10$ 个参数
- 需要 $2 \times 10 = 20$ 次能量测量
- 如果每次能量测量需要1000次电路运行（取平均）
- 总共需要 $20 \times 1000 = 20,000$ 次电路运行

**问题**：当参数很多时，这变得非常昂贵！

**例子**：大分子系统
- 可能有 $p = 100$ 个参数
- 需要 $2 \times 100 = 200$ 次能量测量
- 非常耗时！

#### 优化策略

**策略1：只计算部分梯度**
- 不是所有参数都需要更新
- 可以只计算"重要"参数的梯度

**策略2：使用随机梯度估计**
- SPSA（同时扰动随机近似）
- 只需2次评估，与参数数无关！

### 3.5.4 SPSA（同时扰动随机近似）

#### 核心思想

**问题**：参数位移规则需要 $2p$ 次评估（$p$ 是参数数）

**SPSA的洞察**：用**随机方向**近似梯度，只需**2次评估**（与参数数无关）！

#### 算法

**SPSA梯度估计**：

$$\nabla E(\boldsymbol{\theta}) \approx \frac{E(\boldsymbol{\theta} + c\boldsymbol{\Delta}) - E(\boldsymbol{\theta} - c\boldsymbol{\Delta})}{2c} \boldsymbol{\Delta}^{-1}$$

其中：
- $\boldsymbol{\Delta} = (\Delta_1, \Delta_2, \ldots, \Delta_p)$ 是**随机向量**
- 通常 $\Delta_i \in \{-1, +1\}$（随机选择）
- $c$ 是**扰动大小**（通常很小，如 $c = 0.01$）
- $\boldsymbol{\Delta}^{-1} = (1/\Delta_1, 1/\Delta_2, \ldots, 1/\Delta_p)$

#### 详细解释

**步骤1：生成随机向量**

对于 $p$ 个参数，随机生成：
$$\boldsymbol{\Delta} = (+1, -1, +1, -1, +1, \ldots)$$

每个元素随机选择 $+1$ 或 $-1$。

**步骤2：测量两个能量**

1. 测量 $E(\boldsymbol{\theta} + c\boldsymbol{\Delta})$：
   - 参数变为：$(\theta_1 + c, \theta_2 - c, \theta_3 + c, \ldots)$
   - 在量子计算机上测量能量

2. 测量 $E(\boldsymbol{\theta} - c\boldsymbol{\Delta})$：
   - 参数变为：$(\theta_1 - c, \theta_2 + c, \theta_3 - c, \ldots)$
   - 在量子计算机上测量能量

**步骤3：估计梯度**

对于每个参数 $i$：
$$\frac{\partial E}{\partial \theta_i} \approx \frac{E(\boldsymbol{\theta} + c\boldsymbol{\Delta}) - E(\boldsymbol{\theta} - c\boldsymbol{\Delta})}{2c \Delta_i}$$

**关键**：所有参数的梯度估计**只用了2次能量测量**！

#### 例子

**假设**：$p = 10$ 个参数

**参数位移规则**：
- 需要 $2 \times 10 = 20$ 次能量测量

**SPSA**：
- 随机向量：$\boldsymbol{\Delta} = (+1, -1, +1, -1, +1, -1, +1, -1, +1, -1)$
- 测量 $E(\boldsymbol{\theta} + 0.01\boldsymbol{\Delta})$：1次
- 测量 $E(\boldsymbol{\theta} - 0.01\boldsymbol{\Delta})$：1次
- **总共只需2次测量！**

**节省**：从20次减少到2次（节省90%！）

#### 优缺点

**优点**：
1. **高效**：对于大量参数非常高效（只需2次评估）
2. **简单**：实现简单
3. **适合噪声环境**：对噪声有一定的鲁棒性

**缺点**：
1. **近似**：只是梯度的**近似**，不是精确值
2. **收敛慢**：可能需要更多迭代才能收敛
3. **对噪声敏感**：如果测量噪声大，估计可能不准确

#### 何时使用SPSA？

**适合使用SPSA的情况**：
- 参数很多（$p > 50$）
- 测量有噪声
- 不需要非常精确的梯度

**不适合使用SPSA的情况**：
- 参数较少（$p < 10$）
- 需要精确梯度
- 测量非常精确（噪声小）

**总结**：
- **参数位移规则**：精确但昂贵（$2p$ 次评估）
- **SPSA**：近似但高效（2次评估）
- 根据情况选择合适的方法

---

## 3.6 优化器

### 3.6.1 无梯度优化器

**COBYLA**（Constrained Optimization BY Linear Approximation）
- 不需要梯度
- 构建线性近似
- 适合噪声环境

**Nelder-Mead**（单纯形法）
- 基于单纯形的搜索
- 鲁棒但收敛慢

### 3.6.2 梯度优化器

**梯度下降**：
$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta \nabla E(\boldsymbol{\theta}_k)$$

**Adam**：
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1-\beta_1^t)$$
$$\hat{v}_t = v_t / (1-\beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

### 3.6.3 量子自然梯度（QNG）

**想法**：考虑参数空间的几何结构。

普通梯度下降在欧氏空间中最速下降，但参数空间可能有非平凡几何。

**量子Fisher信息矩阵**：
$$F_{ij} = \text{Re}\left[ \langle \partial_i \psi | \partial_j \psi \rangle - \langle \partial_i \psi | \psi \rangle \langle \psi | \partial_j \psi \rangle \right]$$

**QNG更新**：
$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta F^{-1} \nabla E(\boldsymbol{\theta}_k)$$

**优点**：更快收敛，对参数化不变
**缺点**：计算 $F$ 需要 $O(p^2)$ 次电路评估

---

## 3.7 贫瘠高原（Barren Plateaus）

### 3.7.1 现象

**定义**：当电路深度增加时，损失函数梯度的方差指数衰减：

$$\text{Var}[\partial_k E] \leq O(e^{-cn})$$

其中 $n$ 是量子比特数。

**后果**：梯度几乎为零，优化困难。

### 3.7.2 原因分析

**随机电路的2-设计性质**：深层随机电路近似Haar随机酉。

对于Haar随机态：
$$\mathbb{E}_{|\psi\rangle}[\langle \psi | O | \psi \rangle] = \frac{\text{Tr}[O]}{2^n}$$

局部可观测量的期望值集中在 $O(1/2^n)$。

### 3.7.3 成因

1. **全局测量**：测量涉及所有量子比特
2. **深层电路**：产生高度纠缠态
3. **随机初始化**：参数在大范围均匀分布
4. **表达能力过强**：能表示任意态

### 3.7.4 缓解策略

1. **局部代价函数**：只测量局部可观测量
2. **浅层电路**：限制电路深度
3. **结构化ansatz**：利用问题对称性
4. **分层训练**：逐层训练
5. **好的初始化**：从接近解的参数开始

---

## 3.8 VQE的误差分析

### 3.8.1 误差来源

$$E_{VQE} = E_{exact} + \epsilon_{ansatz} + \epsilon_{opt} + \epsilon_{stat} + \epsilon_{noise}$$

| 误差类型 | 来源 | 典型量级 |
|---------|------|---------|
| $\epsilon_{ansatz}$ | Ansatz表达能力不足 | 依赖ansatz选择 |
| $\epsilon_{opt}$ | 优化未完全收敛 | 可通过更多迭代减小 |
| $\epsilon_{stat}$ | 有限测量次数 | $O(1/\sqrt{N_s})$ |
| $\epsilon_{noise}$ | 硬件噪声 | 依赖硬件质量 |

### 3.8.2 测量误差估计

每个Pauli项的测量误差：
$$\sigma_i = \frac{\sqrt{1 - \langle P_i \rangle^2}}{\sqrt{N_{s,i}}}$$

总能量误差：
$$\sigma_E = \sqrt{\sum_i c_i^2 \sigma_i^2}$$

### 3.8.3 达到化学精度

**化学精度**：$\sim 1$ kcal/mol $\approx 1.6$ mHa $\approx 0.0016$ Ha

对于H2分子（$\sim 15$ Pauli项，$|c_i| \sim 0.1-1$）：
- 需要每项误差 $< 0.1$ mHa
- 每项需要 $> 10^6$ 次测量

---

## 3.9 VQE与其他方法的比较

### 3.9.1 与全量子算法（QPE）比较

| 特性 | VQE | QPE |
|------|-----|-----|
| 电路深度 | 浅 | 深 |
| 误差类型 | 变分误差 | 相位估计误差 |
| 是否需要容错 | 不需要 | 需要 |
| 经典资源 | 大量经典优化 | 较少 |
| 适用时代 | NISQ | 容错 |

### 3.9.2 与经典方法比较

| 方法 | 复杂度 | 精度 | 适用系统 |
|------|--------|-----|---------|
| HF | $O(N^4)$ | 定性 | 任意 |
| CCSD | $O(N^6)$ | 化学精度 | 弱关联 |
| CCSD(T) | $O(N^7)$ | 高精度 | 弱关联 |
| FCI | $O(e^N)$ | 精确 | 小系统 |
| VQE | 依赖ansatz | 可变 | 强关联 |

### 3.9.3 VQE的潜在优势

1. **强关联系统**：经典方法困难，VQE可能有优势
2. **可控近似**：通过增加ansatz复杂度系统改进
3. **量子叠加**：利用量子态的指数维度

---

## 3.10 小结

### 3.10.1 VQE完整流程回顾

**VQE算法的完整流程**：

```
1. 初始化
   └─> 选择初始参数 θ₀
   └─> 选择ansatz（参数化量子电路）

2. 迭代优化（重复以下步骤直到收敛）
   │
   ├─> 2.1 态制备（量子计算机）
   │   └─> |ψ(θ)⟩ = U(θ)|0⟩
   │
   ├─> 2.2 能量测量（量子计算机）
   │   ├─> 将哈密顿量分解为Pauli串：H = Σ cᵢ Pᵢ
   │   ├─> 测量每个Pauli串：⟨Pᵢ⟩
   │   └─> 计算总能量：E(θ) = Σ cᵢ ⟨Pᵢ⟩
   │
   ├─> 2.3 梯度计算（经典或量子）
   │   ├─> 参数位移规则：∂E/∂θ = ½[E(θ+π/2) - E(θ-π/2)]
   │   └─> 或SPSA：随机梯度估计
   │
   └─> 2.4 参数更新（经典计算机）
       └─> θ_{k+1} = θ_k - η∇E(θ_k)

3. 输出
   └─> 最优参数 θ*
   └─> 基态能量 E(θ*) ≈ E₀
```

### 3.10.2 核心方程速查

**变分原理**（理论基础）：
$$E_0 \leq E(\boldsymbol{\theta}) = \langle \psi(\boldsymbol{\theta}) | \hat{H} | \psi(\boldsymbol{\theta}) \rangle$$

**物理意义**：任意猜测态的能量期望值永远不会低于真实的基态能量。

**能量测量**（量子计算机）：
$$E(\boldsymbol{\theta}) = \sum_i c_i \langle P_i \rangle$$

**物理意义**：总能量是所有Pauli项的能量贡献之和。

**参数位移规则**（梯度计算）：
$$\frac{\partial E}{\partial \theta} = \frac{1}{2}\left[ E(\theta + \frac{\pi}{2}) - E(\theta - \frac{\pi}{2}) \right]$$

**物理意义**：通过测量两个不同参数的能量来计算梯度。

**梯度更新**（经典优化）：
$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta \nabla E(\boldsymbol{\theta}_k)$$

**物理意义**：沿着能量下降的方向更新参数。

### 3.10.3 关键概念总结

#### 1. 变分原理

**核心思想**：
- 基态能量是**所有可能态中能量最低的**
- 通过最小化能量期望值，可以找到基态

**数学表达**：
- $E_0 \leq \langle \psi | H | \psi \rangle$（对任意归一化态）
- 等号成立当且仅当 $|\psi\rangle$ 是基态

#### 2. 混合量子-经典结构

**分工**：
- **量子计算机**：制备态、测量能量（利用量子优势）
- **经典计算机**：优化参数、控制流程（利用经典优势）

**优势**：
- 充分利用两种计算机的优势
- 适合NISQ时代（不需要容错）

#### 3. 能量测量

**步骤**：
1. 哈密顿量分解：$H = \sum_i c_i P_i$
2. 基变换：将非Z的Pauli变换到Z基
3. 测量：在Z基测量所有相关量子比特
4. 计算：$\langle P \rangle = \mathbb{E}[(-1)^{\oplus_j m_j}]$
5. 加权求和：$E = \sum_i c_i \langle P_i \rangle$

**优化**：
- Pauli分组：对易的Pauli串可以同时测量
- 减少测量次数，提高效率

#### 4. 参数化量子电路（Ansatz）

**作用**：
- 构造参数化的试探波函数
- 通过调整参数，探索不同的量子态

**设计原则**：
- **表达能力**：足够表示基态
- **训练难度**：不能太复杂（避免贫瘠高原）
- **物理启发**：利用问题的物理性质

#### 5. 梯度计算

**方法1：参数位移规则**
- **精确**：给出精确的梯度
- **代价**：每个参数需要2次能量测量（$2p$ 次）

**方法2：SPSA**
- **近似**：给出梯度的近似
- **代价**：只需2次能量测量（与参数数无关）

#### 6. 优化器

**类型**：
- **无梯度优化器**：COBYLA, Nelder-Mead（不需要梯度）
- **梯度优化器**：梯度下降, Adam（需要梯度）
- **量子自然梯度**：考虑参数空间的几何结构

### 3.10.4 关键挑战

#### 挑战1：Ansatz设计

**问题**：如何设计好的ansatz？

**平衡**：
- **表达能力**：需要足够表示基态
- **训练难度**：不能太复杂，避免贫瘠高原

**策略**：
- 使用物理启发的ansatz（如UCCSD）
- 从简单开始，逐步增加复杂度

#### 挑战2：测量开销

**问题**：大量Pauli项需要分别测量

**解决方案**：
- **Pauli分组**：对易的Pauli串可以同时测量
- **减少测量次数**：从 $O(N^4)$ 减少到 $O(N)$ 或更少

#### 挑战3：贫瘠高原

**问题**：深层电路的梯度消失

**现象**：
- 梯度方差指数衰减：$\text{Var}[\partial_k E] \leq O(e^{-cn})$
- 优化变得困难

**缓解策略**：
- 局部代价函数
- 浅层电路
- 结构化ansatz
- 好的初始化

#### 挑战4：噪声影响

**问题**：硬件噪声累积

**影响**：
- 测量误差
- 电路执行错误
- 能量估计不准确

**缓解策略**：
- 误差缓解技术
- 多次测量取平均
- 噪声模型和校正

### 3.10.5 最佳实践

**1. Ansatz选择**
- ✅ 使用物理启发的ansatz（如UCCSD）
- ✅ 从简单开始，逐步增加复杂度
- ❌ 避免过度复杂的ansatz（可能导致贫瘠高原）

**2. 初始化**
- ✅ 从HF态初始化（更接近基态）
- ✅ 使用物理直觉选择初始参数
- ❌ 避免完全随机初始化

**3. 测量优化**
- ✅ 使用Pauli分组减少测量次数
- ✅ 根据系数大小分配测量资源
- ❌ 不要分别测量所有Pauli项

**4. 优化策略**
- ✅ 监控梯度方差，避免贫瘠高原
- ✅ 选择合适的优化器（根据问题规模）
- ✅ 设置合理的收敛条件

**5. 误差处理**
- ✅ 考虑噪声缓解技术
- ✅ 多次测量取平均
- ✅ 估计和报告误差

### 3.10.6 实际应用示例

**H₂分子VQE计算**：

1. **哈密顿量**：15个Pauli项（通过JW变换得到）
2. **Ansatz**：UCCSD（2个参数）
3. **初始参数**：从HF态开始
4. **测量**：Pauli分组后约6组
5. **优化**：COBYLA优化器
6. **结果**：$E_{VQE} = -1.136$ Ha，接近精确值 $-1.137$ Ha

**关键指标**：
- **误差**：$0.001$ Ha（化学精度内！）
- **迭代次数**：约50次
- **测量次数**：约 $6 \times 1000 = 6,000$ 次

### 3.10.7 下一步学习

**深入理解**：
1. UCCSD ansatz的详细构造（第四章）
2. 更复杂的分子系统
3. 误差缓解技术
4. 硬件实现细节

**实践建议**：
1. 实现简单的VQE算法（如H₂分子）
2. 尝试不同的ansatz
3. 比较不同的优化器
4. 分析误差来源

**应用方向**：
- 更大分子的计算
- 强关联系统
- 化学反应路径
- 材料设计
