# 量子机器学习完整指南

## 目录

### 第一部分：基本理论
- [1.1 为什么 |0⟩ = [1, 0]ᵀ](#11-为什么-0--beginbmatrix-1--0-endbmatrix)
- [1.2 三种测量返回类型详解](#第一部分基本理论)
- [1.3 PauliZ 算符详解](#第一部分基本理论)
- [1.4 量子特征提取基础](#第一部分基本理论)
- [1.5 特征 vs 标签](#第一部分基本理论)
- [1.6 量子态维度与特征提取](#第一部分基本理论)
- [1.7 量子特征提取作为升维方法](#第一部分基本理论)
- [1.8 数据重上传](#第一部分基本理论)
- [1.9 量子机器学习的非线性表达](#19-量子机器学习的非线性表达)
- [1.10 应用场景与最佳实践](#110-应用场景与最佳实践)
- [1.11 批量处理](#111-批量处理)
- [1.12 测量策略](#112-测量策略)

### 第二部分：量子数据编码
- [2.1 学习目标](#21-学习目标)
- [2.2 核心问题：为什么需要编码](#22-核心问题为什么需要编码)
- [2.3 Angle Embedding（角度编码）](#23-angle-embedding角度编码-最常用)
- [2.4 Amplitude Embedding（振幅编码）](#24-amplitude-embedding振幅编码-表达密度最高)
- [2.5 Basis Embedding（基态编码）](#25-basis-embedding基态编码-只适合离散数据)
- [2.6 三种编码对比总结](#26-三种编码对比总结)
- [2.7 选择指南](#27-选择指南什么时候用哪种编码)
- [2.8 常见问题](#28-常见问题)

### 第三部分：量子核方法
- [3.1 概述](#31-概述)
- [3.2 数据预处理：标准化与缩放到 [0, π]](#32-数据预处理标准化与缩放到-0-π)
- [3.3 量子核方法应用对比](#33-量子核方法应用对比)
- [3.4 量子核应用的其他可能形式组合](#34-量子核应用的其他可能形式组合)
- [3.5 参考](#35-参考)

### 第四部分：量子变分层和量子神经网络
- [4.1 概述](#41-概述)
- [4.2 输出缩放问题](#42-输出缩放问题传统ml和量子ml的共同挑战)
- [4.3 两种核心架构方式概述](#43-两种核心架构方式概述)
- [4.4 架构详细说明](#44-架构详细说明)
- [4.5 架构对比与选择建议](#45-架构对比与选择建议)
- [4.6 参考](#46-参考)

### 第五部分：代码实现详解
- [5.1 数据编码模块实现说明](#51-数据编码模块实现说明)
- [5.2 量子神经网络模块实现说明](#52-量子神经网络模块实现说明)
- [5.3 MNIST分类模块说明](#53-mnist分类模块说明)
- [5.4 量子核方法模块说明](#54-量子核方法模块说明)
- [5.5 其他功能模块说明](#55-其他功能模块说明)

---

## 第一部分：基本理论

### 1.1 

为什么单量子比特的基态 $|0\rangle$ 是列向量 $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$？

---

#### 详细解释

### 1. 量子态是向量空间中的向量

单量子比特的量子态存在于**二维复向量空间** $\mathbb{C}^2$ 中。这个向量空间需要两个基向量来构成一个**标准正交基**。

### 2. 计算基态的约定

在量子计算中，我们选择两个**计算基态**作为标准基：

- **$|0\rangle$**：第一个基向量
- **$|1\rangle$**：第二个基向量

### 4. 类比笛卡尔坐标系和经典比特

#### 4.1 类比笛卡尔坐标系

在**二维笛卡尔坐标系**中：
- **x轴**和**y轴**是2维空间的基向量
- **x轴单位向量**：$\hat{e}_x = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$（在x方向，系数为1；在y方向，系数为0）
- **y轴单位向量**：$\hat{e}_y = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$（在x方向，系数为0；在y方向，系数为1）

任意二维向量可以表示为：
$$\vec{v} = v_x \hat{e}_x + v_y \hat{e}_y = v_x \begin{bmatrix} 1 \\ 0 \end{bmatrix} + v_y \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} v_x \\ v_y \end{bmatrix}$$

**量子基态的表示完全类似**：
- **$|0\rangle$** 和 **$|1\rangle$** 是2维复向量空间的基向量
- **$|0\rangle$**：$\begin{bmatrix} 1 \\ 0 \end{bmatrix}$（在 $|0\rangle$ 方向，系数为1；在 $|1\rangle$ 方向，系数为0）
- **$|1\rangle$**：$\begin{bmatrix} 0 \\ 1 \end{bmatrix}$（在 $|0\rangle$ 方向，系数为0；在 $|1\rangle$ 方向，系数为1）

任意量子态可以表示为：
$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle = \alpha \begin{bmatrix} 1 \\ 0 \end{bmatrix} + \beta \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} \alpha \\ \beta \end{bmatrix}$$

其中 $\alpha, \beta \in \mathbb{C}$ 是复数系数。

#### 4.2 类比经典比特

这类似于经典计算中的表示：

- **经典比特**：0 和 1 是两个离散状态
- **量子比特**：$|0\rangle$ 和 $|1\rangle$ 是两个基向量，但可以处于叠加态

### 5. 归一化条件

任意量子态 $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle = \begin{bmatrix} \alpha \\ \beta \end{bmatrix}$ 必须满足**归一化条件**：

$$|\alpha|^2 + |\beta|^2 = 1$$

这确保了测量概率的总和为 1。
### 7. 正交归一性

两个基态满足**正交归一性**：

- **正交性**：$\langle 0|1\rangle = 0$
  $$\langle 0|1\rangle = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = 1 \cdot 0 + 0 \cdot 1 = 0$$

- **归一性**：$\langle 0|0\rangle = 1$，$\langle 1|1\rangle = 1$
  $$\langle 0|0\rangle = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = 1 \cdot 1 + 0 \cdot 0 = 1$$

### 8. 物理意义

- **$|0\rangle$** 表示量子比特处于"0 态"
- **$|1\rangle$** 表示量子比特处于"1 态"
- 测量 $|0\rangle$ 时，得到 0 的概率为 $|\langle 0|0\rangle|^2 = 1$（100%）
- 测量 $|0\rangle$ 时，得到 1 的概率为 $|\langle 1|0\rangle|^2 = 0$（0%）

---

#### 总结

**$|0\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ 是一个约定**：

1. 在标准基 $\{|0\rangle, |1\rangle\}$ 下，$|0\rangle$ 对应第一个基向量
2. $|1\rangle$ 对应第二个基向量
3. 这是量子计算中的**标准表示方法**，类似于：
   - **笛卡尔坐标系**中，x轴单位向量是 $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$，y轴单位向量是 $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$
   - **经典计算**中比特的表示（但量子比特可以处于叠加态）

这个约定使得：
- 量子态可以用向量表示（就像笛卡尔坐标系中的向量）
- 量子门可以用矩阵表示（就像坐标变换矩阵）
- 量子操作可以用矩阵乘法计算（就像线性变换）
- 整个量子计算框架具有统一的数学基础

---

#### 扩展：

在量子力学中，我们使用 **Dirac 记号**：
- **Ket** $|\psi\rangle$：列向量（右矢）
- **Bra** $\langle\psi|$：行向量（左矢，是 ket 的共轭转置）

这种表示方法的好处：
- **内积**：$\langle\phi|\psi\rangle$ 表示行向量乘以列向量，得到标量
- **外积**：$|\psi\rangle\langle\phi|$ 表示列向量乘以行向量，得到矩阵
- **一致性**：与量子力学的数学框架完全一致

---

## 第四部分：量子神经网络架构详解

### 4.1 概述

本部分详细解释量子神经网络的不同架构设计，回答"为什么要先用经典神经网络处理特征"以及不同架构方式的区别和适用场景。

---

### 4.2 输出缩放问题：传统ML和量子ML的共同挑战

在回归任务中，无论是传统机器学习还是量子机器学习，都可能面临输出范围受限的问题。

#### 4.2.1 问题场景

**传统ML场景：** 使用 `Tanh` 激活函数的神经网络，输出范围被限制在 **[-1, 1]**，但目标值可能是 **50、100、200** 等大数值。

**量子ML场景：** 量子层输出范围通常是 **[-1, 1]**（Z期望值），但回归任务的目标值范围可能不同。

**共同问题：** 输出范围与目标值范围不匹配，导致训练困难。

#### 4.2.2 解决方案：缩放目标值（推荐）

**原理：** 将目标值缩放到模型的输出范围，训练后反缩放预测值。

```python
from sklearn.preprocessing import MinMaxScaler

# 训练时：将目标值缩放到 [-1, 1]
y_scaler = MinMaxScaler(feature_range=(-1, 1))
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

# 预测时：反缩放
predictions_scaled = model(X_test)  # 范围 [-1, 1]
predictions = y_scaler.inverse_transform(predictions_scaled.numpy())  # 原始范围
```

**优势：**
- ✅ 训练稳定：目标值和输出都在相同范围
- ✅ 梯度合理：损失值在合理范围内
- ✅ 实现简单：只需要在训练前缩放，预测时反缩放
- ✅ 通用性强：适用于传统ML和量子ML

**详细说明：** 传统ML的缩放方法（方法1：缩放目标值、方法2：输出层无激活、方法3：添加缩放层）同样适用于量子ML。推荐使用方法1（缩放目标值）。

---

### 4.3 两种核心架构方式概述

在量子机器学习中，主要有两种核心架构方式：

**方式1：神经网络特征处理 → 量子编码和变分网络 → 传统分类器/回归器 或 量子直接做回归**

```
输入数据 → [经典NN特征处理] → [量子编码+变分网络] → [传统分类器/回归器] 或 [量子直接回归]
```

**特点：**
- 经典神经网络先做特征处理和变换
- 然后将处理后的特征给量子进行编码和变分计算
- 最后可以用传统分类器/回归器，或者量子直接做回归
- **训练方式**：端到端联合训练

**方式2：量子编码和纠缠 → 测量输出量子特征 → 传统机器学习方法做回归/分类**

```
输入数据 → [量子编码+纠缠等] → [测量输出量子特征] → [传统ML方法（回归/分类）]
```

**特点：**
- 直接用量子编码和纠缠等操作处理原始数据
- 测量输出量子处理后的特征（通常是多个量子比特的期望值）
- 将量子特征输入到传统机器学习方法（如SVM、Ridge回归、随机森林等）进行最终预测
- **训练方式**：两阶段训练（可改进为端到端训练）

---

### 4.4 架构详细说明

#### 4.4.1 方式1：神经网络特征处理 → 量子层 → 传统分类器/回归器 或 量子直接回归

**变体A：神经网络特征处理 → 量子层 → 传统分类器/回归器**

```
输入数据 → [经典NN特征处理] → [量子编码+变分网络] → [传统分类器/回归器] → 输出
```

**核心实现：**
```python
class QuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits, n_layers, n_classes=2, task='classification'):
        self.feature_processor = nn.Sequential(...)  # 经典特征处理层
        self.quantum_layer = QuantumLayer(n_qubits, n_layers)  # 量子层
        self.output_layer = nn.Linear(1, n_classes if task=='classification' else 1)
    
    def forward(self, x):
        processed_features = self.feature_processor(x)  # 步骤1：经典特征处理
        quantum_output = self.quantum_layer(processed_features)  # 步骤2：量子计算
        return self.output_layer(quantum_output)  # 步骤3：传统分类器/回归器
```

**为什么需要经典预处理层？**

经典神经网络预处理层的作用包括：
1. **特征变换**：将原始特征映射到适合量子编码的空间（如 [0, π]）
2. **非线性变换**：量子编码（如 AngleEmbedding）本身是线性的，需要非线性变换来增强表达能力
3. **特征组合**：学习特征之间的交互，生成更有用的组合特征

**工作流程：**
1. **经典神经网络特征处理**：特征变换、规范化、特征组合
2. **量子编码和变分网络**：AngleEmbedding编码 + 变分旋转门 + 纠缠门 → 量子测量值（标量）
3. **传统分类器/回归器**：线性变换 → 最终预测结果

**特点：** 端到端学习，表达能力强，适合复杂任务。详细对比和选择建议参见4.5节。

---

**变体B：神经网络特征处理 → 量子层 → 量子直接做回归**

```
输入数据 → [经典NN特征处理] → [量子编码+变分网络] → [量子直接回归] → 输出
```

**核心实现：**
```python
class QuantumDirectRegression(nn.Module):
    def __init__(self, n_qubits, n_layers):
        self.feature_processor = nn.Sequential(...)  # 经典特征处理层
        self.quantum_layer = QuantumLayer(n_qubits, n_layers)  # 量子层（直接输出回归值）
    
    def forward(self, x):
        processed_features = self.feature_processor(x)  # 步骤1：经典特征处理
        return self.quantum_layer(processed_features)  # 步骤2：量子直接回归（需要缩放）
```

**工作流程：**
1. **经典神经网络特征处理**：同变体A
2. **量子编码和变分网络**：同变体A
3. **量子直接回归**：直接使用量子输出（范围 [-1, 1]）作为回归值，需要缩放（参见4.2节）

**特点：** 参数最少，纯量子优势，适合简单回归任务。详细对比和选择建议参见4.5节。

---

#### 4.4.2 方式2：量子编码和纠缠 → 测量输出量子特征 → 传统机器学习方法

**架构流程：**
```
输入数据 → [量子编码+纠缠等] → [测量输出量子特征] → [传统ML方法] → 输出
```

**工作流程：**
1. **量子编码和纠缠**：AngleEmbedding编码 + 变分旋转门 + CNOT纠缠 → 量子态
2. **测量输出量子特征**：测量所有量子比特的Z期望值 → 量子特征向量 `[⟨Z₀⟩, ⟨Z₁⟩, ..., ⟨Z_{n-1}⟩]`（n_qubits 维）
3. **传统机器学习方法**：使用传统ML方法（Ridge回归、SVR、随机森林等）→ 最终预测结果

**特点：** 充分利用量子优势，可以使用成熟的传统ML方法，灵活性高。两阶段训练，对输入数据要求高（需要预处理到 [0, π]）。详细对比和选择建议参见4.5节。

**注意：** 输入数据需要预先预处理到 [0, π] 范围，传统ML层输出也需要缩放（参见4.2节）。

---

#### 4.4.3 方式2改进：端到端训练（量子特征提取 + 传统ML联合训练）

**问题：** 方式2的两阶段训练是否可以改进为端到端训练？

**答案：可以！** 通过在量子特征处理时加入可训练参数，并将量子层和传统ML模型组合成一个PyTorch模型，可以实现端到端训练。

**核心思想：**
- 量子层包含可训练参数（变分旋转门的角度）
- 传统ML模型也包含可训练参数（如Ridge回归的权重）
- 将两者组合成一个PyTorch模型，可以联合训练

**优势：**
- ✅ 端到端学习，可以针对最终任务优化量子特征
- ✅ 统一训练，训练流程更简单
- ✅ 梯度可以从传统ML模型反向传播到量子层

**挑战：**
- ⚠️ 参数耦合：量子层参数和传统ML参数存在耦合，可能不容易训练
- ⚠️ 训练难度：联合训练可能比分开训练更困难
- ⚠️ 梯度问题：量子层的梯度可能很小（Barren Plateaus问题）

**解决方案：**
1. **使用不同的学习率**（推荐）：量子层使用较小的学习率（如0.001），ML层使用较大的学习率（如0.01）
2. **分阶段训练**：先训练量子层，再联合训练
3. **使用学习率调度器**：根据损失调整学习率
4. **梯度裁剪**：防止梯度爆炸
5. **批量归一化**：在传统ML层中添加BatchNorm

**端到端训练 vs 两阶段训练对比：**

| 特性 | 两阶段训练 | 端到端训练 |
|------|----------|----------|
| **训练方式** | 先训练量子层，再训练ML层 | 联合训练 |
| **参数耦合** | 无耦合 | 存在耦合 |
| **训练难度** | 较简单 | 较困难 |
| **效果** | 中等 | 可能更好（针对任务优化） |
| **灵活性** | 高（可以切换ML方法） | 中（ML方法固定） |
| **梯度传播** | 无 | 有（可以反向传播） |

**选择建议：**
- **需要最大化性能**：使用端到端训练
- **需要灵活性**：使用两阶段训练
- **最佳实践**：先两阶段训练获得初始模型，再端到端微调

**注意：** 端到端训练中，传统ML层输出也需要缩放（参见4.2节）。

---

---

### 4.5 架构对比与选择建议

#### 4.5.1 详细对比表

| 特性 | 方式1-A：NN→量子→传统分类器/回归器 | 方式1-B：NN→量子直接回归 | 方式2：量子特征提取→传统ML（两阶段） | 方式2改进：量子特征提取→传统ML（端到端） |
|------|--------------------------------------|----------------------------------|-------------------------|------------------------------|
| **架构** | 经典NN → 量子层 → 传统分类器/回归器 | 经典NN → 量子层（直接回归） | 量子层 → 传统ML方法 | 量子层（可训练）→ 传统ML层（可训练） |
| **训练方式** | 端到端联合训练 | 端到端联合训练 | 两阶段训练 | 端到端联合训练 |
| **参数数量** | 多（经典NN + 量子 + 传统ML） | 中（经典NN + 量子） | 中（量子 + 传统ML） | 中（量子 + 传统ML） |
| **表达能力** | 最强 | 中 | 中 | 中-高（针对任务优化） |
| **量子优势** | 中等 | 高 | 最高 | 最高 |
| **传统ML优势** | 高 | 低 | 高 | 高 |
| **灵活性** | 高 | 中 | 高（可切换ML方法） | 中（ML方法固定） |
| **训练难度** | 中 | 低 | 低 | 高（参数耦合） |
| **适用任务** | 复杂任务 | 简单回归任务 | 中等复杂度任务 | 中等复杂度任务（需要最大化性能） |
| **数据要求** | 低（可处理原始数据） | 低（可处理原始数据） | 高（需要预处理） | 高（需要预处理） |
| **计算成本** | 高 | 中 | 中 | 中-高 |
| **可解释性** | 中 | 低 | 高（量子特征可分析） | 中（端到端训练） |

#### 4.5.2 选择建议

**方式1-A（NN→量子→传统分类器/回归器）推荐场景：**
- ✅ 复杂任务，需要强大的特征学习能力
- ✅ 原始数据未预处理或预处理不充分
- ✅ 需要端到端学习
- ✅ 有足够的计算资源和训练数据

**方式1-B（NN→量子直接回归）推荐场景：**
- ✅ 简单回归任务
- ✅ 目标值范围在 [-1, 1] 或可以缩放（参见4.2节）
- ✅ 希望最小化参数
- ✅ 纯量子计算场景

**方式2（量子特征提取→传统ML，两阶段）推荐场景：**
- ✅ 中等复杂度任务
- ✅ 希望利用量子优势进行特征提取
- ✅ 数据已经预处理到合适范围
- ✅ 需要量子特征用于其他任务
- ✅ 希望使用成熟的传统ML方法（如SVM、随机森林等）
- ✅ 需要灵活切换不同的ML方法

**方式2改进（量子特征提取→传统ML，端到端）推荐场景：**
- ✅ 中等复杂度任务，希望最大化模型性能
- ✅ 数据已经预处理到合适范围
- ✅ 有足够的计算资源和时间
- ✅ 可以接受更复杂的训练过程（参数耦合问题）
- ✅ 不需要频繁切换ML方法

---

### 4.6 参考

- [PennyLane 文档 - Quantum Neural Networks](https://pennylane.ai/qml/demos/tutorial_variational_classifier.html)
- [Variational Quantum Classifiers](https://pennylane.ai/qml/demos/tutorial_variational_classifier.html)
- [Hybrid Quantum-Classical Neural Networks](https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html)

---

## 第五部分：量子核方法详解

### 3.1 概述

本部分详细解释量子核方法的原理、实现和应用，包括数据预处理、核函数计算、以及与经典机器学习方法的组合使用。

---

### 3.2 数据预处理：标准化与缩放到 [0, π]

**核心要点：** 标准化（可选，推荐）和缩放到 [0, π]（必须）是两个不同的步骤。`AngleEmbedding` 将特征值映射为旋转角度，必须在 [0, π] 范围内。

**预处理流程：**
```
原始数据 → PCA降维（可选） → 标准化（可选，推荐） → 缩放到[0,π]（必须） → 量子核函数
```

**实现代码：**

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def scale_to_0_pi(x):
    """将数据缩放到 [0, π] 范围"""
    x = np.asarray(x, dtype=float)
    mn = x.min()
    mx = x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn) * np.pi

# 完整预处理流程
pca = PCA(n_components=n_qubits)  # 步骤1：PCA降维（可选）
X_reduced = pca.fit_transform(X_original)

scaler = StandardScaler()  # 步骤2：标准化（可选，推荐）
    X_scaled = scaler.fit_transform(X_reduced)

X_quantum = scale_to_0_pi(X_scaled)  # 步骤3：缩放到[0,π]（必须！）
```

**常见问题：**
- **Q: 已标准化还需要缩放到[0,π]吗？** A: 必须！标准化后范围是[-3,3]，AngleEmbedding需要[0,π]
- **Q: 可以直接缩放到[0,π]不做标准化吗？** A: 可以但不推荐，标准化有助于训练稳定性
- **Q: 顺序重要吗？** A: 重要！必须先标准化再缩放到[0,π]

---

### 3.3 量子核方法应用对比

**核心区别：** 第三部分（MNIST分类）和第五部分（通用回归）使用相同的量子核函数，但任务类型、ML算法、数据集和计算方式不同。

| 特性 | 第三部分：MNIST分类 | 第五部分：通用回归 |
|------|------------------|-----------------|
| **任务类型** | 分类（Classification） | 回归（Regression） |
| **ML算法** | SVM | KernelRidge |
| **数据集** | MNIST专用 | 通用 |
| **计算方式** | 并行计算 | 串行计算 |
| **核函数** | 相同 | 相同 |

**代码对比：**

```python
# 第三部分：分类（SVM）
    kernel_fn = make_quantum_kernel_circuit(n_qubits)
K_train = compute_kernel_matrix_parallel(X_train, X_train, kernel_fn, n_workers=4)
svm = SVC(kernel='precomputed')
    svm.fit(K_train, y_train)
    y_pred = svm.predict(K_test)  # 输出：0 或 1

# 第五部分：回归（KernelRidge）
kernel_fn = make_quantum_kernel_ridge(n_qubits)
K_train = compute_kernel_matrix(X_train, X_train, kernel_fn)
    krr = KernelRidge(kernel='precomputed', alpha=1.0)
    krr.fit(K_train, y_train)
    y_pred = krr.predict(K_test)  # 输出：连续值
```

**选择建议：**
- **MNIST分类** → 第三部分（并行计算，完整流程）
- **通用回归** → 第五部分（串行计算，灵活使用）
- **大规模数据** → 使用第三部分的并行计算函数
- **混合使用** → 第三部分的并行计算 + 第五部分的核函数

---

### 3.4 量子核应用的其他可能形式组合

**核心思路：** 量子核可以与任何支持 `kernel='precomputed'` 的算法直接组合；对于不支持的算法，使用 `KernelPCA` 从核矩阵提取特征。

#### 3.4.1 已实现的组合

```python
# 1. 量子核 + SVM（分类）- 第三部分
svm = SVC(kernel='precomputed')
svm.fit(K_train, y_train)

# 2. 量子核 + KernelRidge（回归）- 第五部分
krr = KernelRidge(kernel='precomputed', alpha=1.0)
krr.fit(K_train, y_train)

# 注意：VQC属于量子神经网络架构，参见第四部分
```

#### 3.4.2 其他可能的组合形式

**模式A：直接使用precomputed kernel（简单）**

```python
# 通用流程
kernel_fn = make_quantum_kernel_ridge(n_qubits=4)
K_train = compute_kernel_matrix(X_train, X_train, kernel_fn, n_workers=4)
K_test = compute_kernel_matrix(X_test, X_train, kernel_fn, n_workers=4)

# 多类分类
svm_multi = SVC(kernel='precomputed', multi_class='ovr')  # 或 'ovo'
svm_multi.fit(K_train, y_train)

# 异常检测
ocsvm = OneClassSVM(kernel='precomputed', nu=0.1)
ocsvm.fit(K_train)

# 降维
kpca = KernelPCA(kernel='precomputed', n_components=2)
X_reduced = kpca.fit_transform(K_train)
```

**模式B：使用KernelPCA提取特征（中等）**

```python
# 通用流程：核矩阵 → KernelPCA → 特征 → ML算法
kernel_fn = make_quantum_kernel_ridge(n_qubits=4)
K_train = compute_kernel_matrix(X_train, X_train, kernel_fn, n_workers=4)
K_test = compute_kernel_matrix(X_test, X_train, kernel_fn, n_workers=4)

kpca = KernelPCA(kernel='precomputed', n_components=10)
X_train_features = kpca.fit_transform(K_train)
X_test_features = kpca.transform(K_test)

# 逻辑回归
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_features, y_train)

# 随机森林
rf = RandomForestClassifier(n_estimators=100)  # 或 Regressor
rf.fit(X_train_features, y_train)

# 神经网络
mlp = MLPClassifier(hidden_layer_sizes=(100, 50))  # 或 Regressor
mlp.fit(X_train_features, y_train)

# 聚类
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X_train_features)
```

**模式C：自定义核类包装（困难）**

```python
# 高斯过程：需要自定义核类
class QuantumKernelWrapper(Kernel):
    def __init__(self, quantum_kernel_fn):
        self.quantum_kernel_fn = quantum_kernel_fn
    
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        K = np.zeros((len(X), len(Y)))
        for i in range(len(X)):
            for j in range(len(Y)):
                K[i, j] = self.quantum_kernel_fn(X[i], Y[j])
        return K if not eval_gradient else (K, np.zeros((K.shape[0], K.shape[1], 0)))
    
    def diag(self, X):
        return np.array([self.quantum_kernel_fn(x, x) for x in X])

quantum_kernel = QuantumKernelWrapper(make_quantum_kernel_ridge(n_qubits=4))
gpr = GaussianProcessRegressor(kernel=quantum_kernel, alpha=1e-6)
gpr.fit(X_train, y_train)
y_pred, y_std = gpr.predict(X_test, return_std=True)
```

#### 3.4.3 组合方式总结表

| 组合方式 | 任务类型 | 实现模式 | 是否需要特征提取 |
|---------|---------|---------|----------------|
| 量子核 + SVM | 分类 | 模式A | 否 |
| 量子核 + KernelRidge | 回归 | 模式A | 否 |
| 量子核 + 多类分类 | 分类 | 模式A | 否 |
| 量子核 + 异常检测 | 异常检测 | 模式A | 否 |
| 量子核 + 降维 | 降维 | 模式A | 否 |
| 量子核 + 逻辑回归 | 分类 | 模式B | 是（KernelPCA） |
| 量子核 + 随机森林 | 分类/回归 | 模式B | 是（KernelPCA） |
| 量子核 + 神经网络 | 分类/回归 | 模式B | 是（KernelPCA） |
| 量子核 + 聚类 | 聚类 | 模式B | 是（KernelPCA） |
| 量子核 + 高斯过程 | 回归 | 模式C | 否（需自定义核） |

#### 3.4.4 选择建议

**根据任务类型：**
- **二分类**：量子核 + SVM（第三部分）或 VQC分类（第四部分）
- **多类分类**：量子核 + SVM（multi_class='ovr'）
- **回归**：量子核 + KernelRidge（第五部分）或 VQC回归（第四部分）
- **异常检测**：量子核 + OneClassSVM
- **聚类**：量子核 + KernelPCA + KMeans
- **降维/可视化**：量子核 + KernelPCA

**性能对比：**
- **大规模数据**：VQC方法（内存效率高，O(n×epochs)）
- **小规模数据**：量子核方法（表达能力强，O(n²)或O(n³)）
- **需要不确定性估计**：量子核 + 高斯过程
- **需要可解释性**：量子核 + 随机森林

---

### 3.5 参考

- [PennyLane 文档 - Quantum Kernels](https://pennylane.ai/qml/demos/tutorial_quantum_kernels.html)
- [sklearn - Kernel Methods](https://scikit-learn.org/stable/modules/kernel_ridge.html)
- [sklearn - Precomputed Kernels](https://scikit-learn.org/stable/modules/svm.html#using-precomputed-kernels)

---

---

### 1.10 批量处理

**核心方法：** 在 `@qml.qnode` 中添加 `batch_params=True` 启用批量处理。

**代码对比：**

```python
# 循环版本（QuantumLayer）
@qml.qnode(dev, interface="torch")  # 没有 batch_params
        def quantum_circuit(inputs, weights):
    # ...
            return qml.expval(qml.PauliZ(0))
    
    def forward(self, x):
        outputs = []
        for i in range(x.shape[0]):
        outputs.append(self.quantum_circuit(x[i], self.weights))
    return torch.stack(outputs)  # O(batch_size) 次调用

# 批量版本（QuantumLayerBatch）
@qml.qnode(dev, interface="torch", batch_params=True)  # 关键！
        def quantum_circuit(inputs, weights):
    # ...
            return qml.expval(qml.PauliZ(0))
    
    def forward(self, x):
    return self.quantum_circuit(x, self.weights)  # O(1) 次调用，x形状: (batch_size, n_qubits)
```

**关键规则：**
- `batch_params=True` 只对第一个参数（`inputs`）启用批量处理
- `weights` 必须是单个值，所有样本共享
- 支持的设备：`default.qubit`, `lightning.gpu`, `lightning.qubit`

**选择建议：**
- 大批量数据（>32）：使用批量版本
- 小批量数据（≤32）：可以使用循环版本
- GPU设备：推荐批量版本

---

### 1.9 量子机器学习的非线性表达

**核心问题：** 经典机器学习通过激活函数（如 ReLU、Sigmoid）实现非线性。量子机器学习如何实现非线性表达？

**答案：** 量子机器学习通过**量子特征映射**、**纠缠**和**变分量子电路**实现非线性表达。

#### 1.9.1 五种非线性表达方式

| 方式 | 非线性来源 | 特点 | 推荐度 |
|------|-----------|------|--------|
| **1. 量子特征映射** | 三角函数（cos/sin） | 基础非线性 | ⭐⭐⭐ |
| **2. 量子纠缠** | 量子比特间的非线性关联 | 关键机制 | ⭐⭐⭐⭐⭐ |
| **3. 变分量子电路** | 可训练的非线性变换 | 灵活强大 | ⭐⭐⭐⭐⭐ |
| **4. 数据重上传** | 多次编码+变分层组合 | 增强非线性 | ⭐⭐⭐⭐ |
| **5. 量子核方法** | 隐式非线性映射 | 理论保证 | ⭐⭐⭐⭐ |

#### 1.9.2 方式1：量子特征映射的非线性

**角度编码的非线性：**
$$|\phi(\bm{x})\rangle = \bigotimes_{i=1}^{n} R_Y(x_i) |0\rangle^{\otimes n}$$

**非线性来源：**
- 旋转门 $R_Y(\theta)$ 包含 $\cos(\theta/2)$ 和 $\sin(\theta/2)$，这些**三角函数是非线性的**
- 输入 $x_i$ 通过 $\cos(x_i/2)$ 和 $\sin(x_i/2)$ 变换

**例子：**
```python
@qml.qnode(dev)
def angle_encoding(x):
    qml.AngleEmbedding(x, wires=range(2), rotation="Y")
    return qml.expval(qml.PauliZ(0))

# 输入线性序列，输出非线性（cos函数）
x = np.linspace(0, 2*np.pi, 100)
y = [angle_encoding([xi, 0]) for xi in x]  # y = cos(x)（非线性）
```

#### 1.9.3 方式2：量子纠缠的非线性关联 ⭐ **关键机制**

**核心思想：** 纠缠门（CNOT、CZ）创建量子比特之间的**非线性关联**，这是经典计算无法实现的。

**数学表示：**
- **无纠缠**：$|\psi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle$（可分离，线性组合）
- **有纠缠**：$|\psi\rangle \neq |\psi_1\rangle \otimes |\psi_2\rangle$（不可分离，非线性关联）

**例子：**
```python
@qml.qnode(dev)
def entanglement_nonlinear(x1, x2):
    # 角度编码
    qml.RY(x1, wires=0)
    qml.RY(x2, wires=1)
    
    # 纠缠（创建非线性关联）
    qml.CNOT(wires=[0, 1])
    
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

# 输出：x1 和 x2 的非线性关联（无法用线性组合表示）
```

**关键理解：**
- 纠缠允许模型学习**特征之间的复杂非线性交互**
- 这些交互是**非经典的**，无法用简单的线性组合表示

#### 1.9.4 方式3：变分量子电路的可训练非线性 ⭐ **最灵活**

**数学表示：**
$$|\psi(\bm{x}, \bm{\theta})\rangle = U_L(\bm{\theta}_L) \cdots U_2(\bm{\theta}_2) E(\bm{x}) U_1(\bm{\theta}_1) |0\rangle^{\otimes n}$$

**非线性来源：**
1. **编码层**：角度编码提供初始非线性
2. **变分层**：可训练的旋转门和纠缠门进一步非线性变换
3. **多层组合**：多层量子门的组合产生**复合非线性**

**优势：**
- ✅ **可训练**：参数可以通过梯度下降优化
- ✅ **灵活**：可以学习任务特定的非线性变换
- ✅ **强大**：多层变分电路可以表示复杂的非线性函数

#### 1.9.5 方式4：数据重上传增强非线性

**数学表示：**
$$|\psi(\bm{x}, \bm{\theta})\rangle = U_L(\bm{\theta}_L) E(\bm{x}) \cdots U_2(\bm{\theta}_2) E(\bm{x}) U_1(\bm{\theta}_1) E(\bm{x}) |0\rangle^{\otimes n}$$

**非线性增强：**
- **多次编码**：数据被多次编码，每次编码都经过非线性变换
- **变分层插入**：在编码层之间插入可训练的变分层
- **复合非线性**：多层非线性变换的组合产生**更强的非线性**

#### 1.9.6 方式5：量子核方法的隐式非线性

**量子核函数：**
$$K(\bm{x}_i, \bm{x}_j) = |\langle \phi(\bm{x}_i) | \phi(\bm{x}_j) \rangle|^2$$

**非线性来源：**
1. **量子特征映射**：$|\phi(\bm{x})\rangle$ 本身是非线性的
2. **内积的模平方**：$|\cdot|^2$ 是非线性操作
3. **高维特征空间**：量子态存在于 $2^n$ 维希尔伯特空间

#### 1.9.7 非线性表达能力的来源总结

**五种来源：**

1. **三角函数**（角度编码）：$\cos$ 和 $\sin$ 提供基础非线性
2. **量子叠加**：多个基态的叠加提供非线性表达能力
3. **量子纠缠**：创建量子比特之间的非线性关联
4. **高维特征空间**：$2^n$ 维空间提供强大的表达能力
5. **多层组合**：多层量子门的组合产生复合非线性

**核心优势：**
- ✅ **指数级特征空间**：$n$ 个量子比特对应 $2^n$ 维空间
- ✅ **非经典关联**：纠缠提供经典计算无法实现的非线性
- ✅ **可训练非线性**：变分电路可以学习最优的非线性变换

**实际建议：**
- 使用**角度编码 + 纠缠 + 变分电路**的组合
- 采用**先旋转再纠缠**的顺序
- 根据任务复杂度选择合适的层数和重上传次数

> **详细说明**：关于量子机器学习非线性表达的完整解释，包括数学推导、代码示例和实际应用建议，请参见 [`../notes/qml_training_landscape_compendium.md`](../notes/qml_training_landscape_compendium.md)（Part 3）。

---

### 1.10 批量处理

**核心方法：** 在 `@qml.qnode` 中添加 `batch_params=True` 启用批量处理。

**代码对比：**

```python
# 循环版本（QuantumLayer）
@qml.qnode(dev, interface="torch")  # 没有 batch_params
def quantum_circuit(inputs, weights):
    # ...
    return qml.expval(qml.PauliZ(0))
    
def forward(self, x):
    outputs = []
    for i in range(x.shape[0]):
        outputs.append(self.quantum_circuit(x[i], self.weights))
    return torch.stack(outputs)  # O(batch_size) 次调用

# 批量版本（QuantumLayerBatch）
@qml.qnode(dev, interface="torch", batch_params=True)  # 关键！
def quantum_circuit(inputs, weights):
    # ...
    return qml.expval(qml.PauliZ(0))
    
def forward(self, x):
    return self.quantum_circuit(x, self.weights)  # O(1) 次调用，x形状: (batch_size, n_qubits)
```

**关键规则：**
- `batch_params=True` 只对第一个参数（`inputs`）启用批量处理
- `weights` 必须是单个值，所有样本共享
- 支持的设备：`default.qubit`, `lightning.gpu`, `lightning.qubit`

**选择建议：**
- 大批量数据（>32）：使用批量版本
- 小批量数据（≤32）：可以使用循环版本
- GPU设备：推荐批量版本

---

### 1.11 测量策略

**三种策略：**

| 策略 | 代码 | 适用场景 |
|------|------|---------|
| **只测量第一个** | `qml.expval(qml.PauliZ(0))` | 有纠缠 + 单输出任务 |
| **测量所有取平均** | `sum([qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]) / n_qubits` | 无纠缠 + 单输出任务 |
| **测量所有返回向量** | `torch.tensor([qml.expval(qml.PauliZ(i)) for i in range(n_qubits)])` | 多输出任务 |

**选择原则：**
- **有纠缠**：只测量第一个量子比特通常足够（信息已通过纠缠传递）
- **无纠缠**：必须测量所有量子比特，否则会丢失信息
- **需要鲁棒性**：测量所有量子比特取平均可以减少噪声影响



---

---

## 第八部分：量子数据编码学习指南

### 2.1 学习目标
理解三种主要编码方式（Angle/Amplitude/Basis）的：
- **表达能力**：能表示什么样的数据
- **输入约束**：对输入数据有什么要求
- **计算代价**：需要多少量子比特、多少门操作
- **适用场景**：什么时候用哪种编码

---

### 2.2 核心问题：为什么需要编码？

**经典数据**（比如你的吸附能特征 `[1.5, -0.2, 0.8]`）是**实数向量**，但**量子态**是**复数向量**（且必须归一化：所有振幅的平方和=1）。

**编码（Encoding/Embedding）**就是把经典数据"装进"量子态的过程。

---

### 2.3 Angle Embedding（角度编码）⭐ **最常用**

### 概念直觉
- **思路**：把每个特征值映射成一个**旋转角度**，用旋转门（RY/RX/RZ）作用在量子比特上
- **数学表达**：对于特征 $x_i$，在 qubit $i$ 上作用 $R_Y(x_i)$ 或 $R_X(x_i)$

### 代码分析（从你的 `AngleEncoding.py` 看起）

```python
# 你的 AngleEncoding.py 核心部分
@qml.qnode(dev, interface="torch")
def circuit(x):
    qml.AngleEmbedding(features=x, wires=range(n_qubits), rotation="X")
    return qml.state()
```

**关键点**：
- `features=x`：输入是长度为 `n_qubits` 的向量
- `rotation="X"`：用 X 轴旋转（也可以选 "Y" 或 "Z"）
- **输入维度 = qubit 数**：这是最直观的对应关系

### 输入约束
- ✅ **输入维度**：必须等于 qubit 数（比如 2 个特征 → 2 个 qubits）
- ✅ **输入范围**：通常需要缩放到 $[0, \pi]$ 或 $[-\pi, \pi]$（因为旋转角是周期性的）
- ✅ **数据类型**：连续实数

### 表达能力
- **状态空间维度**：$2^n$（n 个 qubits）
- **能表示的信息**：每个特征独立编码到一个 qubit，**没有纠缠**（除非后续加 CNOT）
- **优势**：直观、稳定、适合你的 **2-5 维特征**

### 实际例子（用你的数据）
假设你有 3 个特征：`[1.5, -0.2, 0.8]`
- 需要 **3 个 qubits**
- 缩放后（比如到 $[0, \pi]$）：`[π, 0.1π, 0.6π]`
- 每个 qubit 独立旋转，得到 3-qubit 量子态（维度 $2^3 = 8$）

---

### 2.4 Amplitude Embedding（振幅编码）🔬 **表达密度最高**

### 概念直觉
- **思路**：直接把特征向量的**归一化版本**当作量子态的**振幅**
- **数学表达**：如果输入是 $[x_1, x_2, ..., x_d]$，归一化后得到 $|\psi\rangle = \sum_{i=0}^{2^n-1} \frac{x_i}{\|x\|} |i\rangle$

### 代码分析（从你的 `AmplitudeEncoding.py` 看起）

```python
# 你的 AmplitudeEncoding.py 核心部分
@qml.qnode(dev, interface="torch")
def amplitude_circuit(f):
    qml.AmplitudeEmbedding(features=f, wires=range(3), normalize=True)
    return qml.state()
```

**关键点**：
- `normalize=True`：**必须归一化**（因为量子态要求 $\sum |\psi_i|^2 = 1$）
- **输入维度必须是 $2^n$**：3 个 qubits → 需要 8 维输入；4 个 qubits → 需要 16 维输入

### 输入约束
- ⚠️ **输入维度**：必须是 $2^n$（否则需要 padding 到最近的 $2^n$）
- ✅ **归一化**：必须归一化（`normalize=True` 会自动处理）
- ✅ **数据类型**：连续实数（可以是负数，归一化后变成复数振幅）

### 表达能力
- **状态空间维度**：$2^n$（n 个 qubits）
- **能表示的信息**：**所有 $2^n$ 个基态的叠加**，信息密度最高
- **优势**：理论上能编码最多信息（一个 $2^n$ 维向量）
- **劣势**：对输入维度要求严格；归一化可能丢失原始尺度信息

### 实际例子（用你的数据）
假设你有 3 个特征：`[1.5, -0.2, 0.8]`
- 需要 padding 到 $2^2 = 4$ 维：`[1.5, -0.2, 0.8, 0.0]`
- 归一化后：`[0.85, -0.11, 0.45, 0.0]`（近似值）
- 需要 **2 个 qubits**（因为 $2^2 = 4$）
- 量子态是这 4 个振幅的叠加

**注意**：你的 3 维特征被 padding 后，实际只用了 2 个 qubits（比 Angle 编码用的 qubits 还少！）

---

### 2.5 Basis Embedding（基态编码）🔢 **只适合离散数据**

### 概念直觉
- **思路**：把输入当作**二进制位**，直接制备到对应的**基态**（比如 `[1,0,1]` → $|101\rangle$）
- **数学表达**：输入 $[b_0, b_1, ..., b_{n-1}]$ → 量子态 $|b_0 b_1 ... b_{n-1}\rangle$（**确定态，不是叠加**）

### 代码分析（从你的 `Basis-Encoding.py` 看起）

```python
# 你的 Basis-Encoding.py 核心部分
@qml.qnode(dev, interface="torch")
def circuit(x):
    qml.BasisEmbedding(x, wires=range(3))
    return qml.state()
```

**关键点**：
- 输入必须是 **0 或 1**（二进制位）
- 输出是**确定态**（不是叠加）：比如 `[1,1,0]` → $|110\rangle$（概率=1，其他基态概率=0）

### 输入约束
- ⚠️ **输入维度**：等于 qubit 数
- ⚠️ **输入值**：必须是 **0 或 1**（整数/二进制）
- ⚠️ **连续特征不能用**：必须先离散化（比如阈值化：>0.5 → 1，否则 → 0）

### 表达能力
- **状态空间维度**：$2^n$
- **能表示的信息**：只能表示 $2^n$ 个**确定基态**之一（不是叠加）
- **优势**：最简单、最直接
- **劣势**：**信息损失最大**（连续特征被硬编码成 0/1）

### 实际例子（用你的数据）
假设你有 3 个特征：`[1.5, -0.2, 0.8]`
- 先阈值化：`[1, 0, 1]`（>0 → 1，否则 → 0）
- 需要 **3 个 qubits**
- 输出是**确定态** $|101\rangle$（概率=1，其他基态概率=0）

**注意**：这种编码**不适合你的连续吸附能特征**，更适合离散分类标签或二进制特征。

---

### 2.6 三种编码对比总结

| 编码方式 | 输入维度要求 | 输入值要求 | Qubit 数 | 信息密度 | 适用场景 |
|---------|------------|-----------|---------|---------|---------|
| **Angle** | = qubit 数 | 连续实数 | n | 中等 | ⭐ **你的 2-5 维特征** |
| **Amplitude** | = $2^n$ | 连续实数（需归一化） | $\lceil \log_2(d) \rceil$ | 最高 | 高维向量（图像/传感器） |
| **Basis** | = qubit 数 | 0/1（二进制） | n | 最低 | 离散/分类标签 |

---

#### 2.6.1 实验：运行你的对比脚本

### 任务 1：理解输出

运行：
```bash
conda activate qml_torch
python /Users/shl/nvidia/QML/week1_encoding_compare.py
```

**观察重点**：
1. **`<Z_i>` 期望值**：这是最常用的"量子特征"，范围 $[-1, 1]$
2. **Top 概率基态**：看看哪种编码产生的量子态更"集中"或更"分散"

### 任务 2：用你自己的特征测试

修改 `week1_encoding_compare.py` 中的输入，换成你的真实吸附能特征（2-5 维），观察：
- Angle 编码的 `<Z_i>` 是否更"区分不同样本"
- Amplitude 编码在 padding 后是否保留了足够信息

---

### 2.7 选择指南：什么时候用哪种编码？

### 场景 1：你的吸附能预测（N≈200，特征 2-5 维）
**推荐：Angle Embedding**
- ✅ 输入维度刚好匹配（2-5 个特征 → 2-5 个 qubits）
- ✅ 不需要 padding，信息不丢失
- ✅ 稳定、可解释

### 场景 2：高维特征（比如图像，784 维）
**推荐：Amplitude Embedding（但需要先 PCA 降维）**
- 先 PCA 到 $2^n$ 维（比如 8 维 → 3 qubits）
- 然后用 Amplitude 编码

### 场景 3：离散分类标签
**推荐：Basis Embedding**
- 比如类别标签 `[0, 1, 2]` → 二进制 `[00, 01, 10]` → Basis 编码

---

#### 2.7.1 本周学习检查清单

- [ ] 能解释为什么需要编码（经典数据 → 量子态）
- [ ] 能说出三种编码的输入约束
- [ ] 能运行 `week1_encoding_compare.py` 并看懂输出
- [ ] 能判断你的 2-5 维特征应该用哪种编码
- [ ] 理解 `<Z_i>` 期望值是什么，为什么常用它作为"量子特征"

---

#### 2.7.2 延伸阅读

- **PennyLane 文档**：[AngleEmbedding](https://docs.pennylane.ai/en/stable/code/api/pennylane.AngleEmbedding.html), [AmplitudeEmbedding](https://docs.pennylane.ai/en/stable/code/api/pennylane.AmplitudeEmbedding.html)
- **qml-tutorial**：第 2 章"量子数据编码"
- **你的代码**：`week1_encoding_compare.py` 里有详细的实现和可视化

---

### 2.8 常见问题

**Q: 为什么 Angle 编码最常用？**
A: 因为它最直观（1 个特征 → 1 个 qubit），不需要 padding，适合中小维度特征。

**Q: Amplitude 编码的"信息密度最高"是什么意思？**
A: 它能把 $2^n$ 维向量直接编码到 $n$ 个 qubits 的量子态里，理论上能表示最多信息。

**Q: Basis 编码为什么不适合连续特征？**
A: 因为它只能表示确定基态（0/1），连续特征必须先离散化，会丢失信息。

---

**下一步**：Week 2 我们会学"测量"（如何从量子态提取特征），以及为什么"局部测量 vs 全局测量"会影响可训练性（贫瘠高原问题）。

---

## 第九部分：代码实现详解

> **注意**：关于数据编码的详细学习指南，请参见第二部分。本节主要提供代码实现的补充说明。

### 5.1 数据编码模块实现说明

#### 5.1.1 编码结果数据类（EncodingResult）

```python
@dataclass(frozen=True)
class EncodingResult:
    """
    编码结果数据类
    
    属性：
        name: 编码方法名称
        n_qubits: 使用的量子比特数
        probs: 测量概率分布（2^n_qubits 维）
        z_exps: Z方向期望值列表（n_qubits 维）
    """
    name: str
    n_qubits: int
    probs: np.ndarray
    z_exps: np.ndarray
```

**说明：**
- 这是一个不可变的数据类（frozen=True），用于封装编码结果
- `probs` 包含完整的概率分布信息（维度 2^n）
- `z_exps` 包含每个量子比特的 Z 期望值（维度 n），这是最常用的量子特征

#### 5.1.2 数据预处理函数

**scale_to_0_pi(x)**: 将数据缩放到 [0, π] 范围（用于角度编码）

**pad_to_pow2(x)**: 将向量填充到长度为 2^n（用于幅度编码）

**详细说明：**
- `np.asarray(x, dtype=float).ravel()`: 将输入转换为1维浮点数组
  - `np.asarray()`: 转换为numpy数组
  - `.ravel()`: 展平多维数组为一维（返回视图，不复制数据）
  - 为什么需要 `ravel()`：幅度编码需要1维向量，无论输入是什么形状
- `math.ceil(math.log2(d))`: 计算需要的量子比特数
  - `ceil()`: 向上取整（向正无穷方向）
  - `floor()`: 向下取整（向负无穷方向）
  - `round()`: 四舍五入
  - 为什么使用 `ceil()`：log2(d) 可能不是整数，但量子比特数必须是整数

> **详细说明**：关于三种编码方式的原理、优缺点、输入约束等，请参见第二部分。

### 5.2 量子神经网络模块实现说明

#### 5.2.1 QuantumLayer（量子层）

**功能：**
- 将经典数据编码到量子态，通过变分量子电路处理，然后测量得到输出

**架构：**
1. 数据编码层：AngleEmbedding（将特征映射为旋转角度）
2. 变分层：多层旋转门 + 纠缠门
3. 测量层：测量 Z 方向期望值

**注意：**
- 当前实现使用循环逐个处理样本
- 如需批量处理，可以使用 `QuantumLayerBatch`（使用 `batch_params=True`）

**重要：旋转门和纠缠门的顺序**

在变分层中，**旋转门和纠缠门的顺序会影响最终的量子态**，从而影响模型的表达能力。

**两种顺序对比：**

| 顺序 | 代码结构 | 表达能力 | 推荐度 |
|------|---------|---------|--------|
| **先旋转再纠缠**（推荐） | 先 `RY/RZ`，再 `CNOT` | 强（可创建任意纠缠态） | ✅ 推荐 |
| **先纠缠再旋转** | 先 `CNOT`，再 `RY/RZ` | 弱（受限） | ❌ 不推荐 |

**为什么顺序重要？**

量子门操作**不满足交换律**：$U_1 U_2 \neq U_2 U_1$。因此，先旋转再纠缠和先纠缠再旋转会产生**不同的量子态**。

**物理意义：**
- **先旋转再纠缠**：先准备每个量子比特的局部状态，再通过CNOT建立关联。可以创建任意纠缠态（包括最大纠缠态），表达能力更强。
- **先纠缠再旋转**：如果初始态是基态（如 $|00\rangle$），CNOT对基态无影响，等价于只旋转，无法创建某些纠缠态。

**实际应用：**
- ✅ **推荐**：先旋转再纠缠（这是 `StronglyEntanglingLayers` 的标准做法）
- ❌ **不推荐**：先纠缠再旋转（除非有特殊需求）

**代码示例：**
```python
# 推荐方式：先旋转再纠缠
for layer in range(n_layers):
    # 步骤1：旋转
    for qubit in range(n_qubits):
        qml.RY(weights[layer, qubit, 0], wires=qubit)
        qml.RZ(weights[layer, qubit, 1], wires=qubit)
    
    # 步骤2：纠缠
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
```

> **详细说明**：关于旋转门和纠缠门顺序的完整解释，包括数学推导、物理意义和实际应用建议，请参见 [`../notes/qml_training_landscape_compendium.md`](../notes/qml_training_landscape_compendium.md)（Part 4）。

#### 5.2.2 QuantumLayerBatch（批量处理版本）

**与 QuantumLayer 的区别：**
- 支持批量处理：可以一次性处理整个批次的数据，无需循环
- 使用 `batch_params=True` 启用 PennyLane 的批量处理功能
- 性能可能更好（特别是使用 GPU 时）

**关键点：**
- `batch_params=True` 允许 `inputs` 参数是批量数据
- 只有 `inputs` 参数会被批量处理，`weights` 参数必须是单个值（所有样本共享）

#### 5.2.3 QuantumNeuralNetwork（完整的量子神经网络）

**架构：**
1. 经典预处理层：对输入特征进行非线性变换
2. 量子层：执行量子计算（编码 + 变分层 + 测量）
3. 经典分类层：将量子输出映射到类别

**设计问题解答：为什么要先用经典神经网络处理特征？**

1. **为什么需要经典预处理层？**
   - 目的：特征变换、非线性变换、维度适配、数据规范化
   - 优势：提高表达能力、减少量子层负担、训练稳定性、灵活性

2. **直接给量子变分层不行吗？**
   - 可以，但有限制：简单任务可以，复杂任务可能效果不佳

3. **量子做特征处理后给神经网络拟合行不行？**
   - 可以！这是另一种架构（量子特征提取 + 经典分类器）

**三种架构对比：**
- **方式A：经典预处理 → 量子层 → 经典分类**（当前实现）
  - 优势：表达能力最强，可以处理复杂数据
  - 劣势：参数多，训练时间长
- **方式B：量子特征提取 → 经典分类器**
  - 优势：利用量子特征提取能力，经典分类器简单高效
  - 劣势：量子层需要承担更多特征学习任务
- **方式C：直接量子分类**
  - 优势：最简洁，参数最少
  - 劣势：表达能力可能有限

#### 5.2.4 QuantumClassifier（简单的量子分类器）

**特点：**
- 更简单的版本，直接使用 VQC 进行分类
- 使用 `StronglyEntanglingLayers` 作为可学习层
- 适合二分类任务

### 5.3 MNIST分类模块说明

**两种方法：**
1. **量子核SVM**：使用量子核函数 + 经典SVM（分类任务）
   - 实现函数：`train_quantum_kernel_svm()`
   - 使用：量子核 + SVC(kernel='precomputed')
   
2. **VQC分类**：使用变分量子电路直接分类（端到端量子分类器）
   - 实现函数：`train_vqc_classifier()`
   - 使用：QuantumClassifier（变分量子电路）
   - 特点：不依赖经典ML方法，纯量子端到端学习

**数据预处理：**
- 从 OpenML 加载 MNIST 数据
- 过滤指定数字（例如 3 和 6）
- PCA 降维到 n_qubits 维
- 标准化和归一化
- **必须缩放到 [0, π] 范围**（AngleEmbedding 的要求）

### 5.4 量子核方法模块说明

**主要应用：**
1. **量子核岭回归**：用于回归（量子核 + KernelRidge）
   - 实现函数：`make_quantum_kernel(n_qubits, return_float=True)` + KernelRidge
   - 使用：量子核 + KernelRidge(kernel='precomputed')
   
2. **VQC回归**：使用变分量子电路直接回归（端到端量子回归器）
   - 实现函数：`train_vqc_regressor()`
   - 使用：QuantumClassifier（变分量子电路）用于回归
   - 特点：不依赖经典ML方法，纯量子端到端学习

**与第三部分的区别：**
- 第三部分：专门针对MNIST数据集的分类任务
- 第四部分：通用的量子核方法模块，支持分类和回归

**主要区别：**
1. 任务类型：第三部分 = 分类（SVM + VQC），第四部分 = 回归（KernelRidge + VQC）
2. 数据集：第三部分 = MNIST专用，第四部分 = 通用（可用于任何数据集）
3. 计算方式：都支持并行计算
4. 应用场景：第三部分 = 图像分类（MNIST），第四部分 = 通用回归任务

> **注意**：关于量子核应用的其他可能形式组合，请参见第三部分3.4节。

### 5.5 其他功能模块说明

#### 5.5.1 量子测量模式详解：Local、Global 和 Local Vector

在量子神经网络中，有三种主要的测量模式用于分析 Barren Plateaus 和量子态特性：
1. **Local 测量**：测量单个量子比特
2. **Global 测量**：测量所有量子比特的全局关联
3. **Local Vector 测量**：测量所有量子比特并取平均

##### 1. Local 测量（局部测量）

**定义：**
只测量单个量子比特的 Z 方向期望值。

**代码实现：**
```python
mode = "local"
obs = qml.PauliZ(0)  # 只测量第0个量子比特
return qml.expval(obs)  # 返回 ⟨Z₀⟩
```

**物理意义：**
- **测量单个量子比特在 Z 方向的投影**
- 反映该量子比特处于 |0⟩ 和 |1⟩ 的概率差
- **范围**：[-1, 1]

**数学表示：**
$$\langle Z_0 \rangle = \langle \psi | Z_0 | \psi \rangle = P(|0\rangle_0) - P(|1\rangle_0)$$

**特点：**
- ✅ 计算成本低
- ✅ 梯度稳定
- ✅ 适合大多数任务
- ❌ 对纠缠不敏感

##### 2. Global 测量（全局测量）

**定义：**
测量所有量子比特的 Z 方向张量积的期望值。

**代码实现：**
```python
mode = "global"
obs = qml.PauliZ(0)  # 从 Z₀ 开始
for i in range(1, n_qubits):
    obs = obs @ qml.PauliZ(i)  # 张量积：Z₀ ⊗ Z₁ ⊗ ... ⊗ Z_{n-1}
return qml.expval(obs)  # 返回 ⟨Z₀ ⊗ Z₁ ⊗ ... ⊗ Z_{n-1}⟩
```

**物理意义：**
- **测量所有量子比特的全局关联**
- 反映所有量子比特的联合状态
- **范围**：[-1, 1]
- **对纠缠态特别敏感**

**数学表示：**
$$\langle Z_0 \otimes Z_1 \otimes \cdots \otimes Z_{n-1} \rangle = \langle \psi | Z_0 \otimes Z_1 \otimes \cdots \otimes Z_{n-1} | \psi \rangle$$

**关键理解：**
- 这是**所有量子比特的联合测量**，不是单个测量
- 反映所有量子比特之间的**全局关联**
- 对纠缠态特别敏感

**张量积的含义：**
对于 3 量子比特系统，$Z_0 \otimes Z_1 \otimes Z_2$ 是一个 8×8 的矩阵，对角线元素为：
- |000⟩: +1
- |001⟩: -1
- |010⟩: -1
- |011⟩: +1
- |100⟩: -1
- |101⟩: +1
- |110⟩: +1
- |111⟩: -1

**Global 测量与不同量子态：**
- |000⟩ 态：$\langle Z_0 \otimes Z_1 \otimes Z_2 \rangle = +1$（所有量子比特都是 |0⟩）
- |111⟩ 态：$\langle Z_0 \otimes Z_1 \otimes Z_2 \rangle = +1$（所有量子比特都是 |1⟩）
- |001⟩ 态：$\langle Z_0 \otimes Z_1 \otimes Z_2 \rangle = -1$（奇数个 |1⟩）
- Bell 态 |Φ⁺⟩ = $(|00\rangle + |11\rangle)/\sqrt{2}$：$\langle Z_0 \otimes Z_1 \rangle = +1$（两个量子比特状态相同）
- Bell 态 |Ψ⁺⟩ = $(|01\rangle + |10\rangle)/\sqrt{2}$：$\langle Z_0 \otimes Z_1 \rangle = -1$（两个量子比特状态相反）

**Global 测量与纠缠的关系：**
- GHZ 态：$|GHZ\rangle = (|000\rangle + |111\rangle)/\sqrt{2}$，$\langle Z_0 \otimes Z_1 \otimes Z_2 \rangle = +1$（所有量子比特完全关联）
- W 态：$|W\rangle = (|001\rangle + |010\rangle + |100\rangle)/\sqrt{3}$，$\langle Z_0 \otimes Z_1 \otimes Z_2 \rangle = -1/3$（部分关联）

**特点：**
- ✅ 能捕获全局关联
- ✅ 对纠缠敏感
- ✅ 适合检测量子关联
- ❌ 可能遇到 Barren Plateaus
- ❌ 计算成本较高

##### 3. Local Vector 测量（局部向量测量）

**定义：**
测量所有量子比特的 Z 期望值，然后取平均。

**代码实现：**
```python
mode = "local_vector"
obs = [qml.PauliZ(i) for i in range(n_qubits)]  # 所有量子比特的 Z
return [qml.expval(o) for o in obs]  # 返回 [⟨Z₀⟩, ⟨Z₁⟩, ..., ⟨Z_{n-1}⟩]
# 然后取平均
```

**物理意义：**
- **测量每个量子比特的局部 Z 期望值**
- 取平均得到整体统计
- **范围**：每个值 [-1, 1]，平均值也在 [-1, 1]

**数学表示：**
$$\text{平均} = \frac{1}{n} \sum_{i=0}^{n-1} \langle Z_i \rangle = \frac{1}{n} \sum_{i=0}^{n-1} \langle \psi | Z_i | \psi \rangle$$

**特点：**
- ✅ 结合了 Local 和 Global 的优点
- ✅ 计算成本中等
- ✅ 梯度相对稳定
- ✅ 能捕获所有量子比特的信息
- ⚠️ 对纠缠的敏感性中等

##### 三种测量的对比

| 测量模式 | 可观测量 | 输出 | 物理意义 | 对纠缠的敏感性 | 计算成本 | 梯度稳定性 |
|---------|---------|------|---------|--------------|---------|-----------|
| **Local** | $\langle Z_0 \rangle$ | 标量 | 单个量子比特的状态 | 低 | 低 | 高 |
| **Global** | $\langle Z_0 \otimes Z_1 \otimes ... \otimes Z_{n-1} \rangle$ | 标量 | 所有量子比特的全局关联 | **高** | 高 | 低（可能 Barren Plateaus） |
| **Local Vector** | $[\langle Z_0 \rangle, \langle Z_1 \rangle, ..., \langle Z_{n-1} \rangle]$ | 向量（取平均） | 所有量子比特的平均状态 | 中等 | 中等 | 中等 |

##### 实际应用场景

**1. Barren Plateaus 分析**

Global 测量更容易出现 Barren Plateaus，因为 Global 测量对深层电路的梯度更敏感。

- **Local 测量**：梯度可能较大，容易训练
- **Global 测量**：梯度可能很小（Barren Plateaus），难以训练
- **Local Vector 测量**：梯度中等，训练难度中等

**2. 纠缠检测**

Global 测量可以检测纠缠：
- 如果 $\langle Z_0 \otimes Z_1 \otimes ... \otimes Z_{n-1} \rangle$ 接近 ±1，说明强关联
- 如果接近 0，说明弱关联或没有纠缠
- 注意：Global 测量不能完全区分经典关联和量子纠缠，需要结合其他方法（如纠缠熵）来区分

**3. 任务选择建议**

- **简单任务/大多数情况**：使用 **Local 测量**
  - 计算成本低
  - 梯度稳定
  - 适合大多数任务
  - 例如：回归、二分类

- **需要检测全局关联**：使用 **Global 测量**（注意 Barren Plateaus）
  - 能捕获全局关联
  - 对纠缠敏感
  - 但可能遇到 Barren Plateaus
  - 例如：需要检测全局模式的任务

- **平衡方案**：使用 **Local Vector 测量**
  - 结合了 Local 和 Global 的优点
  - 计算成本中等
  - 梯度相对稳定
  - 例如：需要捕获所有量子比特信息但不想遇到 Barren Plateaus 的任务

##### 详细示例

**示例1：3量子比特系统的测量**

```python
import pennylane as qml
import numpy as np

dev = qml.device("default.qubit", wires=3)

# Local 测量
@qml.qnode(dev)
def local_measurement(state):
    qml.StatePrep(state, wires=range(3))
    return qml.expval(qml.PauliZ(0))  # 只测量第一个量子比特

# Global 测量
@qml.qnode(dev)
def global_measurement(state):
    qml.StatePrep(state, wires=range(3))
    obs = qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2)
    return qml.expval(obs)  # 测量所有量子比特的关联

# Local Vector 测量
@qml.qnode(dev)
def local_vector_measurement(state):
    qml.StatePrep(state, wires=range(3))
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

# 测试不同的量子态
# |000⟩ 态
state_000 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
print(f"Local: {local_measurement(state_000)}")        # +1
print(f"Global: {global_measurement(state_000)}")      # +1
print(f"Local Vector: {local_vector_measurement(state_000)}")  # [+1, +1, +1]

# |111⟩ 态
state_111 = np.array([0, 0, 0, 0, 0, 0, 0, 1])
print(f"Local: {local_measurement(state_111)}")        # -1
print(f"Global: {global_measurement(state_111)}")      # +1
print(f"Local Vector: {local_vector_measurement(state_111)}")  # [-1, -1, -1]

# GHZ 态：(|000⟩ + |111⟩)/√2
state_ghz = np.array([1/np.sqrt(2), 0, 0, 0, 0, 0, 0, 1/np.sqrt(2)])
print(f"Local: {local_measurement(state_ghz)}")        # 0
print(f"Global: {global_measurement(state_ghz)}")      # +1
print(f"Local Vector: {local_vector_measurement(state_ghz)}")  # [0, 0, 0]
```

**示例2：Barren Plateaus 分析**

```python
def analyze_barren_plateaus(n_qubits, n_layers, mode="local"):
    """
    分析不同测量模式下的梯度大小
    """
    from pennylane import numpy as pnp
    
    dev = qml.device("default.qubit", wires=n_qubits)
    
    # 创建可观测量
    if mode == "local":
        obs = qml.PauliZ(0)
    elif mode == "global":
        obs = qml.PauliZ(0)
        for i in range(1, n_qubits):
            obs = obs @ qml.PauliZ(i)
    elif mode == "local_vector":
        obs = [qml.PauliZ(i) for i in range(n_qubits)]
    
    @qml.qnode(dev, interface="autograd")
    def circuit(weights):
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        if mode == "local_vector":
            return [qml.expval(o) for o in obs]
        return qml.expval(obs)
    
    # 随机初始化权重
    weights = pnp.random.normal(0, 1, size=(n_layers, n_qubits, 3), requires_grad=True)
    
    # 计算梯度
    def scalar_out(w):
        out = circuit(w)
        if mode == "local_vector":
            out = pnp.stack(out).mean()
        return out
    
    grad = qml.grad(scalar_out)(weights)
    return float(pnp.mean(pnp.abs(grad)))

# 比较不同模式
n_qubits = 10
n_layers = 5

grad_local = analyze_barren_plateaus(n_qubits, n_layers, mode="local")
grad_global = analyze_barren_plateaus(n_qubits, n_layers, mode="global")
grad_local_vector = analyze_barren_plateaus(n_qubits, n_layers, mode="local_vector")

print(f"Local 测量梯度: {grad_local}")
print(f"Global 测量梯度: {grad_global}")
print(f"Local Vector 测量梯度: {grad_local_vector}")

# 通常结果：
# Local 测量梯度: 较大（例如 0.1）
# Global 测量梯度: 很小（例如 0.0001，Barren Plateaus）
# Local Vector 测量梯度: 中等（例如 0.01）
```

##### 总结

**核心要点：**

1. **Local 测量**：
   - 测量单个量子比特
   - 计算简单，梯度稳定
   - 对纠缠不敏感
   - 适合大多数任务

2. **Global 测量**：
   - 测量所有量子比特的全局关联
   - 能捕获全局关联，对纠缠敏感
   - 但可能遇到 Barren Plateaus
   - 适合需要检测全局模式的任务

3. **Local Vector 测量**：
   - 测量所有量子比特并取平均
   - 平衡了计算成本和表达能力
   - 梯度相对稳定
   - 适合需要捕获所有量子比特信息但不想遇到 Barren Plateaus 的任务

**选择建议：**
- **简单任务/大多数情况**：使用 **Local 测量**
- **需要检测全局关联**：使用 **Global 测量**（注意 Barren Plateaus）
- **平衡方案**：使用 **Local Vector 测量**

**Global 测量的核心：**
Global 测量的核心是测量所有量子比特的全局关联，对纠缠态特别敏感，是检测量子关联的重要工具。但需要注意，Global 测量在深层电路中容易出现 Barren Plateaus 问题，导致梯度消失，训练困难。

#### 5.5.2 Block Encoding 演示

**功能：**
- Block Encoding 是量子线性代数中的关键技术，用于将经典矩阵编码到量子电路中

**原理：**
- 使用 LCU（Linear Combination of Unitaries）方法：
  1. 将矩阵分解为 Pauli 字符串的线性组合
  2. 使用 PREP + SELECT + PREP† 模式实现

---
