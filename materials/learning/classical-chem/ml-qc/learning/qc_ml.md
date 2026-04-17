# 第3周：经典机器学习在量子化学中的应用

## 学习目标

本周学习如何使用经典机器学习方法近似求解薛定谔方程，重点关注神经网络表示波函数、变分蒙特卡洛方法、深度学习求解电子结构问题等，理解其数学原理和理论思想。

## 1. 神经网络量子态（Neural Network Quantum States, NNQS）

### 1.1 波函数的神经网络表示

#### 基本思想
用神经网络参数化波函数，利用神经网络的表示能力来捕获电子相关。

#### 数学表述
波函数表示为：
$$\psi(\mathbf{x}; \boldsymbol{\theta}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\theta})$$

其中 $\mathcal{N}$ 是神经网络，$\boldsymbol{\theta}$ 是网络参数，$\mathbf{x} = (\mathbf{r}_1, \sigma_1, \ldots, \mathbf{r}_N, \sigma_N)$ 是电子坐标。

#### 反对称性处理

##### Slater-Jastrow形式
$$\psi(\mathbf{x}; \boldsymbol{\theta}) = \det[\phi_i(\mathbf{r}_j)] \times J(\mathbf{r}_1, \ldots, \mathbf{r}_N; \boldsymbol{\theta})$$

其中：
- **Slater行列式**：保证反对称性
- **Jastrow因子**：神经网络 $J$ 捕获电子相关

##### 反对称神经网络
直接构造反对称的神经网络，例如使用反对称层（Antisymmetric Layer）。

### 1.2 神经网络架构

#### 全连接神经网络
$$\psi(\mathbf{x}) = f_L \circ f_{L-1} \circ \cdots \circ f_1(\mathbf{x})$$

其中每层为：
$$f_l(\mathbf{h}_l) = \sigma(W_l \mathbf{h}_l + \mathbf{b}_l)$$

- $W_l$ 是权重矩阵
- $\mathbf{b}_l$ 是偏置向量
- $\sigma$ 是激活函数（如 tanh, ReLU）

#### 卷积神经网络（CNN）
对于具有平移对称性的系统，可以使用CNN。

#### 循环神经网络（RNN）
对于序列结构，可以使用RNN。

#### 图神经网络（GNN）
对于分子系统，可以使用GNN，节点表示原子，边表示化学键。

### 1.3 表示理论

#### 通用逼近定理
对于任意连续函数，存在一个足够大的神经网络可以任意精度逼近。

#### 量子态表示能力
- **单隐藏层网络**：可以表示任意量子态（理论上）
- **实际限制**：参数数量、训练难度

#### 与基组展开的对比
- **基组展开**：$\psi = \sum_i c_i \phi_i$，需要大量基函数
- **神经网络**：$\psi = \mathcal{N}(\mathbf{x})$，参数可能更少但表示能力更强

### 1.4 变分优化

#### 能量泛函
$$E[\boldsymbol{\theta}] = \frac{\langle\psi(\boldsymbol{\theta})|\hat{H}|\psi(\boldsymbol{\theta})\rangle}{\langle\psi(\boldsymbol{\theta})|\psi(\boldsymbol{\theta})\rangle}$$

#### 梯度计算
$$\frac{\partial E}{\partial \theta_i} = 2 \text{Re}\left[\frac{\langle\frac{\partial\psi}{\partial\theta_i}|\hat{H}|\psi\rangle}{\langle\psi|\psi\rangle} - E \frac{\langle\frac{\partial\psi}{\partial\theta_i}|\psi\rangle}{\langle\psi|\psi\rangle}\right]$$

#### 随机梯度下降
使用蒙特卡洛采样估计期望值，然后使用梯度下降优化参数。

## 2. 变分蒙特卡洛方法（Variational Monte Carlo, VMC）

### 2.1 蒙特卡洛积分

#### 期望值计算
哈密顿量的期望值为：
$$E = \frac{\int \psi^*(\mathbf{x}) \hat{H} \psi(\mathbf{x}) d\mathbf{x}}{\int |\psi(\mathbf{x})|^2 d\mathbf{x}}$$

#### 重要性采样
引入概率分布 $p(\mathbf{x}) = |\psi(\mathbf{x})|^2 / \int |\psi(\mathbf{x}')|^2 d\mathbf{x}'$，则：
$$E = \int \frac{\psi^*(\mathbf{x}) \hat{H} \psi(\mathbf{x})}{|\psi(\mathbf{x})|^2} p(\mathbf{x}) d\mathbf{x} = \mathbb{E}_p\left[\frac{\hat{H}\psi(\mathbf{x})}{\psi(\mathbf{x})}\right]$$

其中 $\hat{H}\psi(\mathbf{x})/\psi(\mathbf{x})$ 是局域能量。

#### 蒙特卡洛估计
$$E \approx \frac{1}{M} \sum_{m=1}^M \frac{\hat{H}\psi(\mathbf{x}_m)}{\psi(\mathbf{x}_m)}$$

其中 $\{\mathbf{x}_m\}$ 是从 $p(\mathbf{x})$ 采样的配置。

### 2.2 Metropolis-Hastings采样

#### 算法
1. 从当前配置 $\mathbf{x}$ 提议新配置 $\mathbf{x}'$
2. 计算接受概率：
   $$A(\mathbf{x}'|\mathbf{x}) = \min\left(1, \frac{|\psi(\mathbf{x}')|^2}{|\psi(\mathbf{x})|^2} \frac{T(\mathbf{x}|\mathbf{x}')}{T(\mathbf{x}'|\mathbf{x})}\right)$$
3. 以概率 $A$ 接受新配置

#### 提议分布
通常使用高斯随机游走：
$$\mathbf{x}' = \mathbf{x} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2)$$

### 2.3 VMC与神经网络结合

#### 算法流程
1. **初始化**：随机初始化神经网络参数 $\boldsymbol{\theta}$
2. **采样**：使用Metropolis-Hastings从 $|\psi(\boldsymbol{\theta})|^2$ 采样
3. **估计能量**：计算能量期望值
4. **计算梯度**：使用采样估计能量梯度
5. **更新参数**：使用梯度下降更新 $\boldsymbol{\theta}$
6. **重复**：回到步骤2，直到收敛

#### 梯度估计
$$\frac{\partial E}{\partial \theta_i} \approx \frac{2}{M} \sum_{m=1}^M \text{Re}\left[\left(\frac{\hat{H}\psi(\mathbf{x}_m)}{\psi(\mathbf{x}_m)} - E\right) \frac{\partial \ln\psi(\mathbf{x}_m)}{\partial \theta_i}\right]$$

这是无偏估计。

### 2.4 方差减小技术

#### 问题
蒙特卡洛估计的方差可能很大，导致训练不稳定。

#### 控制变量
使用控制变量减少方差：
$$\mathbb{E}[X] = \mathbb{E}[X - c(Y - \mathbb{E}[Y])]$$

其中 $Y$ 是易于计算期望的变量，$c$ 是协方差系数。

#### 重采样
使用重采样技术提高采样效率。

## 3. 深度学习求解电子结构问题

### 3.1 神经网络基组（Neural Network Basis Sets）

#### 思想
用神经网络生成基函数，而不是使用固定的原子轨道基组。

#### 数学表述
轨道表示为：
$$\phi_i(\mathbf{r}) = \sum_{\mu} c_{i\mu} \chi_\mu(\mathbf{r}; \boldsymbol{\theta})$$

其中 $\chi_\mu$ 是神经网络生成的基函数。

#### 优势
- 自适应：基函数可以根据系统优化
- 紧凑：可能用更少的基函数达到相同精度

### 3.2 深度学习势能面

#### 问题
传统方法计算势能面需要大量量子化学计算。

#### 解决方案
用神经网络拟合势能面：
$$E(\mathbf{R}) = \mathcal{N}(\mathbf{R}; \boldsymbol{\theta})$$

其中 $\mathbf{R}$ 是原子坐标。

#### 训练数据
从高精度量子化学计算（如CCSD(T)）生成训练数据。

#### 应用
- 分子动力学模拟
- 反应路径搜索
- 光谱计算

### 3.3 端到端学习

#### 思想
直接从原子坐标和原子类型预测分子性质，不显式求解薛定谔方程。

#### 架构
$$\text{分子结构} \to \text{神经网络} \to \text{分子性质}$$

#### 挑战
- 需要大量训练数据
- 可解释性差
- 外推能力有限

## 4. 机器学习势能面构建

### 4.1 原子中心描述符

#### 问题
神经网络需要固定大小的输入，但分子大小可变。

#### 解决方案
使用原子中心描述符，每个原子用局部环境描述。

#### 描述符类型
- **Coulomb矩阵**：$M_{ij} = \begin{cases} Z_i^2/2 & i=j \\ Z_i Z_j / |\mathbf{R}_i - \mathbf{R}_j| & i\neq j \end{cases}$
- **原子环境描述符**：描述每个原子周围的化学环境
- **图描述符**：使用图神经网络

### 4.2 神经网络势能面

#### Behler-Parrinello神经网络
$$E = \sum_i E_i$$

其中每个原子的能量 $E_i$ 是神经网络函数：
$$E_i = \mathcal{N}(\mathbf{G}_i; \boldsymbol{\theta})$$

$\mathbf{G}_i$ 是原子 $i$ 的对称函数描述符。

#### 对称函数
保证旋转和平移不变性：
$$G_i = \sum_j f(|\mathbf{R}_i - \mathbf{R}_j|, Z_j)$$

### 4.3 主动学习

#### 思想
智能选择需要计算的配置，而不是随机采样。

#### 算法
1. 用少量数据训练初始模型
2. 用模型预测不确定性大的区域
3. 在这些区域进行量子化学计算
4. 更新训练数据，重新训练
5. 重复直到收敛

## 5. 数学原理：表示理论

### 5.1 神经网络的表示能力

#### 通用逼近定理
对于任意连续函数 $f: \mathbb{R}^n \to \mathbb{R}$ 和任意 $\epsilon > 0$，存在单隐藏层神经网络 $\mathcal{N}$，使得：
$$\|f - \mathcal{N}\|_\infty < \epsilon$$

#### 深度网络的优势
- **参数效率**：深度网络可能用更少的参数表示复杂函数
- **层次特征**：自动学习层次化特征

### 5.2 波函数表示的复杂度

#### 传统基组
需要 $O(e^N)$ 个基函数（FCI）。

#### 神经网络
理论上可以用 $O(\text{poly}(N))$ 个参数表示（但实际可能更复杂）。

#### 实际限制
- 训练难度
- 局部最优
- 过拟合

### 5.3 优化理论

#### 非凸优化
能量泛函关于神经网络参数是非凸的，存在多个局部最优。

#### 优化方法
- **随机梯度下降（SGD）**
- **Adam优化器**
- **自然梯度**：考虑参数空间的几何结构

#### 自然梯度
$$\tilde{\nabla}_\theta E = F^{-1} \nabla_\theta E$$

其中 $F$ 是Fisher信息矩阵：
$$F_{ij} = \mathbb{E}_p\left[\frac{\partial \ln\psi}{\partial \theta_i} \frac{\partial \ln\psi}{\partial \theta_j}\right]$$

## 6. 优化理论

### 6.1 梯度下降

#### 基本更新
$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \eta \nabla_\theta E$$

其中 $\eta$ 是学习率。

#### 学习率调度
- **固定学习率**
- **自适应学习率**：Adam, RMSprop
- **学习率衰减**

### 6.2 随机优化

#### 小批量梯度
使用小批量样本估计梯度：
$$\nabla_\theta E \approx \frac{1}{B} \sum_{b=1}^B \nabla_\theta E(\mathbf{x}_b)$$

#### 优势
- 计算效率高
- 可能跳出局部最优

### 6.3 二阶方法

#### 牛顿法
$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - H^{-1} \nabla_\theta E$$

其中 $H$ 是Hessian矩阵。

#### 准牛顿法
使用近似Hessian，如BFGS、L-BFGS。

## 7. 泛函分析视角

### 7.1 函数空间

#### 波函数空间
波函数属于 $L^2(\mathbb{R}^{3N})$，这是无限维函数空间。

#### 神经网络空间
神经网络参数化了一个有限维子空间，但通过调整参数可以探索整个空间。

### 7.2 逼近理论

#### 最佳逼近
寻找在给定参数数量下，对目标函数的最佳逼近。

#### 神经网络 vs 基组
- **基组**：固定基函数，优化系数
- **神经网络**：同时优化基函数和系数

### 7.3 收敛性

#### 能量收敛
随着网络增大，能量单调下降（变分原理）。

#### 波函数收敛
需要证明波函数也收敛到精确解。

## 8. 实际应用和挑战

### 8.1 成功案例

#### 小分子系统
神经网络量子态在小分子（如H$_2$O, CH$_4$）上取得了接近FCI的精度。

#### 周期性系统
在周期性系统（如固体）上也取得了成功。

### 8.2 挑战

#### 可扩展性
- 大系统的采样困难
- 网络参数数量增长
- 训练时间增长

#### 数值稳定性
- 梯度爆炸/消失
- 采样效率
- 能量方差

#### 初始化
好的初始化对训练成功至关重要。

### 8.3 改进方向

#### 架构改进
- 更好的反对称性处理
- 更高效的网络结构

#### 训练改进
- 更好的优化算法
- 自适应采样

#### 理论理解
- 理解神经网络的表示能力
- 理解优化动力学

## 9. 理论思想总结

### 9.1 新范式

#### 传统方法
使用解析的基函数和明确的物理模型。

#### 机器学习方法
使用数据驱动的表示，自动学习复杂模式。

### 9.2 优势

1. **表示能力**：神经网络可能更紧凑地表示复杂波函数
2. **可扩展性**：可能突破传统方法的限制
3. **灵活性**：可以适应不同系统

### 9.3 局限性

1. **训练难度**：非凸优化，可能陷入局部最优
2. **可解释性**：难以理解网络学到了什么
3. **理论保证**：缺乏严格的理论保证

## 10. 思考题

1. 神经网络如何保证波函数的反对称性？
2. VMC方法中的采样效率如何影响训练？
3. 神经网络的表示能力与传统基组相比如何？
4. 如何理解神经网络量子态的优化动力学？
5. 机器学习方法在哪些方面可能超越传统方法？
6. 如何评估神经网络量子态的准确性？

## 11. 下周预告

第4周将学习量子机器学习方法，包括变分量子本征求解器（VQE）、量子神经网络等，探索量子-经典混合算法在量子化学中的应用。

