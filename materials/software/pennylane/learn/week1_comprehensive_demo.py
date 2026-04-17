"""
Week 1 - 量子机器学习基础综合演示

本脚本整合了所有 Week 1 的核心功能：
1. 三种测量返回类型演示 (state, probs, expval)
   - state(): 返回完整量子态矢（复数振幅）
   - probs(): 返回概率分布（测量各基态的概率）
   - expval(): 返回期望值（测量某个可观测量的期望值）
2. PauliZ 算符详解
   - PauliZ 算符用于测量量子比特在 Z 方向的期望值
   - 期望值范围在 [-1, 1] 之间
3. 量子特征提取 + 经典机器学习
   - 使用量子电路将经典数据编码到量子态
   - 通过测量提取量子特征
   - 使用经典机器学习模型（如 Ridge 回归）进行预测
4. 单特征数据的量子特征提取和升维
   - 将 1 维特征扩展到多维量子特征
   - 展示多种升维方法
5. 数据重上传的三种实现方式
   - 不同的旋转角度
   - 不同的编码方式
   - 变分层

所有输出图片保存到 week0-fig/ 文件夹

运行：
  python3 week1_comprehensive_demo.py
"""

from __future__ import annotations

import os
import numpy as np
import pennylane as qml
import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端（适用于无图形界面的服务器环境）
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# 创建输出文件夹（如果不存在则创建）
OUTPUT_DIR = "week0-fig"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# 第一部分：三种测量返回类型演示
# ============================================================================

def demo_measurement_types(n_qubits: int = 3):
    """
    演示三种测量返回类型：state, probs, expval
    """
    print("=" * 70)
    print("第一部分：三种测量返回类型演示")
    print("=" * 70)
    
    dev = qml.device("default.qubit", wires=n_qubits)
    
    # 生成测试数据（随机角度值，范围在 [0, π]）
    # 这些值将作为量子比特的旋转角度
    x = np.random.rand(n_qubits) * np.pi
    
    # 1. state() - 返回完整的量子态矢
    # 返回所有基态的复数振幅，维度为 2^n_qubits
    @qml.qnode(dev)
    def state_circuit(x):
        # 使用角度编码将经典数据编码到量子态
        # rotation="Y" 表示绕 Y 轴旋转
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        # 返回完整的量子态矢（复数振幅）
        return qml.state()
    
    state = state_circuit(x)
    print(f"\n1. qml.state() - 完整量子态矢:")
    print(f"   维度: {len(state)} (2^{n_qubits} = {2**n_qubits})")
    print(f"   类型: {type(state[0])} (复数，包含实部和虚部)")
    print(f"   前3个振幅: {state[:3]}")
    
    # 2. probs() - 返回概率分布
    # 返回测量各基态的概率，维度为 2^n_qubits，所有概率之和为 1
    @qml.qnode(dev)
    def probs_circuit(x):
        # 角度编码
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        # 返回所有量子比特的测量概率分布
        # 结果是一个长度为 2^n_qubits 的数组，每个元素是对应基态被测量到的概率
        return qml.probs(wires=range(n_qubits))
    
    probs = probs_circuit(x)
    print(f"\n2. qml.probs() - 概率分布:")
    print(f"   维度: {len(probs)} (2^{n_qubits} = {2**n_qubits})")
    print(f"   类型: {type(probs[0])} (实数，非负)")
    print(f"   概率和: {probs.sum():.6f} (应该 = 1，满足归一化条件)")
    print(f"   前3个概率: {probs[:3]}")
    
    # 3. expval() - 返回期望值
    # 返回某个可观测量的期望值，这里是 PauliZ 算符
    # PauliZ 测量量子比特在 Z 方向的期望值，范围在 [-1, 1]
    @qml.qnode(dev)
    def z_circuit(x, idx):
        # 角度编码
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        # 返回第 idx 个量子比特的 PauliZ 期望值
        # ⟨Z⟩ = ⟨ψ|Z|ψ⟩，表示测量该量子比特得到 |0⟩ 和 |1⟩ 的概率差
        return qml.expval(qml.PauliZ(idx))
    
    # 计算所有量子比特的 Z 期望值
    z_exps = [z_circuit(x, i) for i in range(n_qubits)]
    print(f"\n3. qml.expval(qml.PauliZ(i)) - Z 期望值:")
    print(f"   维度: {len(z_exps)} (n_qubits = {n_qubits})")
    print(f"   类型: {type(z_exps[0])} (实数)")
    print(f"   值: {z_exps}")
    print(f"   范围: [{min(z_exps):.4f}, {max(z_exps):.4f}] (理论上在 [-1, 1] 之间)")
    
    # 可视化三种测量结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 子图1: State 振幅（实部和虚部）
    # 显示完整量子态矢的实部和虚部分量
    axes[0].bar(range(len(state)), np.real(state), alpha=0.7, label='实部 (Real)', width=0.4)
    axes[0].bar(np.arange(len(state)) + 0.4, np.imag(state), alpha=0.7, label='虚部 (Imag)', width=0.4)
    axes[0].set_xlabel('基态索引 (State Index)')
    axes[0].set_ylabel('振幅 (Amplitude)')
    axes[0].set_title(f'qml.state() - 复数振幅 (维度={len(state)})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 子图2: Probs 概率分布
    # 显示测量各基态的概率
    axes[1].bar(range(len(probs)), probs, alpha=0.7)
    axes[1].set_xlabel('基态索引 (Basis State Index)')
    axes[1].set_ylabel('概率 (Probability)')
    axes[1].set_title(f'qml.probs() - 概率分布 (维度={len(probs)})')
    axes[1].grid(True, alpha=0.3)
    
    # 子图3: Z 期望值
    # 显示每个量子比特的 PauliZ 期望值
    axes[2].bar(range(n_qubits), z_exps, alpha=0.7, color='green')
    axes[2].set_xlabel('量子比特索引 (Qubit Index)')
    axes[2].set_ylabel('⟨Zᵢ⟩ 值')
    axes[2].set_title(f'qml.expval(PauliZ(i)) - Z 期望值 (维度={n_qubits})')
    axes[2].set_ylim([-1, 1])  # Z 期望值在 [-1, 1] 范围内
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/week0_measurement_types.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 可视化已保存: {OUTPUT_DIR}/week0_measurement_types.png")


# ============================================================================
# 第二部分：量子特征提取 + 经典机器学习
# ============================================================================

def quantum_feature_extractor(n_qubits: int):
    """
    创建量子特征提取器
    
    该函数返回一个量子电路，用于从经典数据中提取量子特征。
    通过角度编码将数据编码到量子态，然后测量 PauliZ 期望值作为特征。
    
    参数:
        n_qubits: 量子比特数量（等于输入特征维度）
    
    返回:
        量子特征提取函数，输入为数据向量和量子比特索引，输出为 Z 期望值
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def extract_features(x: np.ndarray, qubit_idx: int):
        # 使用角度编码将经典数据编码到量子态
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        # 返回第 qubit_idx 个量子比特的 PauliZ 期望值作为特征
        return qml.expval(qml.PauliZ(qubit_idx))
    
    return extract_features


def normalize_to_angle_range(X: np.ndarray) -> np.ndarray:
    """
    将输入数据归一化到 [0, π] 范围
    
    角度编码需要输入在 [0, π] 范围内，因此需要先对数据进行归一化。
    对于每个特征维度，将其线性映射到 [0, π]。
    
    参数:
        X: 输入数据，形状为 (n_samples, n_features)
    
    返回:
        归一化后的数据，每个特征的值都在 [0, π] 范围内
    """
    X = np.asarray(X, dtype=float)
    # 计算每个特征的最小值和最大值
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    # 处理常数列（范围为零的情况），避免除零错误
    X_range = np.where(X_range < 1e-12, 1.0, X_range)
    # 线性映射到 [0, π]
    X_normalized = (X - X_min) / X_range * np.pi
    return X_normalized


def extract_quantum_features(X: np.ndarray, feature_extractor, n_qubits: int) -> np.ndarray:
    """
    为所有样本提取量子特征
    
    对每个样本，通过量子电路提取 n_qubits 个量子特征（每个量子比特的 Z 期望值）。
    
    参数:
        X: 输入数据，形状为 (n_samples, n_features)
        feature_extractor: 量子特征提取函数
        n_qubits: 量子比特数量（特征维度）
    
    返回:
        量子特征矩阵，形状为 (n_samples, n_qubits)
    """
    # 首先将数据归一化到 [0, π] 范围
    X_normalized = normalize_to_angle_range(X)
    X_features = []
    # 对每个样本提取量子特征
    for x in X_normalized:
        # 提取每个量子比特的 Z 期望值作为特征
        features = [feature_extractor(x, i) for i in range(n_qubits)]
        X_features.append(features)
    return np.array(X_features)


def generate_synthetic_data(n_samples: int = 100, n_features: int = 3, seed: int = 42):
    """
    生成合成数据用于回归任务
    
    生成一个非线性的回归数据集，用于演示量子特征提取的效果。
    目标函数包含正弦、平方和交互项，可以测试量子特征的非线性表达能力。
    
    目标函数: y = 0.5 * sin(π * x₁) + 0.3 * x₂² + 0.2 * x₁ * x₃ + 0.1 * ε
    其中 ε 是随机噪声。
    
    参数:
        n_samples: 样本数量
        n_features: 特征维度
        seed: 随机种子
    
    返回:
        X: 特征矩阵，形状为 (n_samples, n_features)，范围在 [-1, 1]
        y: 目标值，形状为 (n_samples,)
    """
    np.random.seed(seed)
    # 生成随机特征，范围在 [-1, 1]
    X = np.random.rand(n_samples, n_features) * 2 - 1
    # 生成非线性目标值（包含正弦、平方、交互项和噪声）
    y = (
        0.5 * np.sin(X[:, 0] * np.pi) +  # 正弦项
        0.3 * (X[:, 1] ** 2) +            # 平方项
        0.2 * X[:, 0] * X[:, 2] +        # 交互项
        0.1 * np.random.randn(n_samples)  # 噪声项
    )
    return X, y


def demo_quantum_feature_classical_ml():
    """
    演示量子特征提取 + 经典机器学习
    
    这是一个混合量子-经典机器学习的典型流程：
    1. 使用量子电路将经典数据编码并提取量子特征
    2. 使用经典机器学习模型（如 Ridge 回归）在量子特征上进行训练和预测
    
    该方法结合了量子的非线性表达能力和经典模型的高效优化。
    """
    print("\n" + "=" * 70)
    print("第二部分：量子特征提取 + 经典机器学习")
    print("=" * 70)
    
    n_samples = 100
    n_features = 3
    n_qubits = n_features  # 量子比特数等于特征维度
    
    # 准备数据（生成合成数据并划分训练集和测试集）
    X, y = generate_synthetic_data(n_samples=n_samples, n_features=n_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\n数据准备:")
    print(f"  训练集: {X_train.shape[0]} 个样本")
    print(f"  测试集: {X_test.shape[0]} 个样本")
    
    # 量子特征提取：使用量子电路将经典数据转换为量子特征
    extract_features = quantum_feature_extractor(n_qubits)
    X_train_quantum = extract_quantum_features(X_train, extract_features, n_qubits)
    X_test_quantum = extract_quantum_features(X_test, extract_features, n_qubits)
    
    print(f"\n量子特征提取:")
    print(f"  特征维度: {X_train_quantum.shape[1]} (n_qubits = {n_qubits})")
    print(f"  特征范围: [{X_train_quantum.min():.4f}, {X_train_quantum.max():.4f}]")
    
    # 标准化特征（使各特征的均值为0，标准差为1）
    # 标准化有助于提高模型训练稳定性
    scaler = StandardScaler()
    X_train_quantum_scaled = scaler.fit_transform(X_train_quantum)
    X_test_quantum_scaled = scaler.transform(X_test_quantum)
    
    # 训练经典机器学习模型（Ridge 回归）
    # Ridge 回归是带 L2 正则化的线性回归，alpha 控制正则化强度
    model = Ridge(alpha=0.1)
    model.fit(X_train_quantum_scaled, y_train)
    
    # 进行预测
    y_train_pred = model.predict(X_train_quantum_scaled)
    y_test_pred = model.predict(X_test_quantum_scaled)
    
    # 评估模型性能（使用 R² 分数）
    # R² 分数衡量模型解释目标变量方差的比例，越接近 1 越好
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n模型性能:")
    print(f"  训练集 R²: {train_r2:.4f}")
    print(f"  测试集 R²: {test_r2:.4f}")
    
    # 对比：使用原始特征（不使用量子特征提取）
    # 这样可以比较量子特征是否带来性能提升
    scaler_raw = StandardScaler()
    X_train_raw_scaled = scaler_raw.fit_transform(X_train)
    X_test_raw_scaled = scaler_raw.transform(X_test)
    model_raw = Ridge(alpha=0.1)
    model_raw.fit(X_train_raw_scaled, y_train)
    y_test_pred_raw = model_raw.predict(X_test_raw_scaled)
    test_r2_raw = r2_score(y_test, y_test_pred_raw)
    
    print(f"\n对比（原始特征）:")
    print(f"  测试集 R²: {test_r2_raw:.4f}")
    
    # 可视化：对比量子特征和原始特征的预测效果
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Parity plot（一致性图）- 量子特征
    # 横轴是真实值，纵轴是预测值，理想情况下点应该落在红色对角线上
    y_min = min(y_test.min(), y_test_pred.min())
    y_max = max(y_test.max(), y_test_pred.max())
    axes[0].scatter(y_test, y_test_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    axes[0].plot([y_min, y_max], [y_min, y_max], 'r--', linewidth=2, label='理想预测线')
    axes[0].set_xlabel('真实值 (True Value)')
    axes[0].set_ylabel('预测值 (Predicted Value)')
    axes[0].set_title(f'量子特征 (测试集 R² = {test_r2:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Parity plot - 原始特征
    axes[1].scatter(y_test, y_test_pred_raw, alpha=0.6, s=50, edgecolors='black', linewidth=0.5, color='orange')
    axes[1].plot([y_min, y_max], [y_min, y_max], 'r--', linewidth=2, label='理想预测线')
    axes[1].set_xlabel('真实值 (True Value)')
    axes[1].set_ylabel('预测值 (Predicted Value)')
    axes[1].set_title(f'原始特征 (测试集 R² = {test_r2_raw:.3f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/week0_quantum_feature_classical_ml.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 可视化已保存: {OUTPUT_DIR}/week0_quantum_feature_classical_ml.png")


# ============================================================================
# 第三部分：单特征数据的量子特征提取和升维
# ============================================================================

def generate_single_feature_data(n_samples: int = 100, seed: int = 42):
    """
    生成单特征数据
    
    用于演示如何将 1 维特征扩展为多维量子特征。
    目标函数包含正弦和平方项。
    
    参数:
        n_samples: 样本数量
        seed: 随机种子
    
    返回:
        X: 特征矩阵，形状为 (n_samples, 1)，范围在 [-1, 1]
        y: 目标值，形状为 (n_samples,)
    """
    np.random.seed(seed)
    # 生成 1 维特征，范围在 [-1, 1]
    X = np.random.rand(n_samples, 1) * 2 - 1
    # 生成非线性目标值
    y = (
        0.5 * np.sin(X[:, 0] * np.pi) +  # 正弦项
        0.3 * (X[:, 0] ** 2) +            # 平方项
        0.1 * np.random.randn(n_samples)  # 噪声项
    )
    return X, y


def demo_single_feature_expansion():
    """
    演示单特征数据的量子特征提取和升维
    
    当输入数据只有 1 维特征时，可以通过多种方法将其扩展为多维量子特征：
    1. 使用单个量子比特测量多个可观测量（Z, X, Y）
    2. 数据重上传：通过不同的旋转角度多次编码同一数据
    3. 数据重上传：使用不同的编码方式
    4. 数据重上传：在编码和测量之间加入变分层
    
    这些方法可以将 1 维特征扩展为多维特征，从而增加模型的表达能力。
    """
    print("\n" + "=" * 70)
    print("第三部分：单特征数据的量子特征提取和升维")
    print("=" * 70)
    
    X, y = generate_single_feature_data(n_samples=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\n原始数据:")
    print(f"  特征维度: {X.shape[1]} (1 维)")
    print(f"  样本数: {X.shape[0]}")
    
    # 方法1: 单量子比特 + 多个可观测量 (Z, X, Y)
    # 使用同一个量子态，但测量不同的可观测量（PauliZ, PauliX, PauliY）
    # 这样可以获得 3 个不同的特征值
    print(f"\n方法1: 单量子比特 + 多个可观测量 (Z, X, Y)")
    n_qubits = 1
    dev = qml.device("default.qubit", wires=n_qubits)
    
    # 定义三个不同的测量电路，分别测量 Z, X, Y 期望值
    @qml.qnode(dev)
    def extract_z_feature(x):
        # 使用 Y 旋转编码数据
        qml.AngleEmbedding(x, wires=[0], rotation="Y")
        # 测量 PauliZ 期望值
        return qml.expval(qml.PauliZ(0))
    
    @qml.qnode(dev)
    def extract_x_feature(x):
        # 使用 Y 旋转编码数据
        qml.AngleEmbedding(x, wires=[0], rotation="Y")
        # 测量 PauliX 期望值
        return qml.expval(qml.PauliX(0))
    
    @qml.qnode(dev)
    def extract_y_feature(x):
        # 使用 Y 旋转编码数据
        qml.AngleEmbedding(x, wires=[0], rotation="Y")
        # 测量 PauliY 期望值
        return qml.expval(qml.PauliY(0))
    
    # 归一化数据到 [0, π] 范围
    X_min, X_max = X.min(), X.max()
    X_normalized = (X - X_min) / (X_max - X_min + 1e-12) * np.pi
    
    # 提取 Z, X, Y 三个特征
    X_quantum_method1 = []
    for x in X_normalized:
        z_val = extract_z_feature(x)  # PauliZ 期望值
        x_val = extract_x_feature(x)  # PauliX 期望值
        y_val = extract_y_feature(x)  # PauliY 期望值
        X_quantum_method1.append([z_val, x_val, y_val])
    X_quantum_method1 = np.array(X_quantum_method1)
    
    print(f"  升维效果: 1 → {X_quantum_method1.shape[1]} ({X_quantum_method1.shape[1]}×)")
    
    # 方法2a: 数据重上传 - 不同的旋转角度
    # 多次编码同一数据，每次使用不同的旋转角度
    # 这样可以获得多个不同的特征值
    print(f"\n方法2a: 数据重上传 - 不同的旋转角度")
    n_layers = 3
    
    @qml.qnode(dev)
    def extract_features_rotations(x, layer_idx):
        # 初始编码
        qml.AngleEmbedding(x, wires=[0], rotation="Y")
        # 额外的旋转，角度与层索引相关
        # 这样每层会产生不同的量子态和测量结果
        qml.RY(x * (layer_idx + 1) * 0.5, wires=0)
        return qml.expval(qml.PauliZ(0))
    
    X_quantum_method2a = []
    for x in X_normalized:
        features = [extract_features_rotations(x, i) for i in range(n_layers)]
        X_quantum_method2a.append(features)
    X_quantum_method2a = np.array(X_quantum_method2a)
    
    print(f"  升维效果: 1 → {X_quantum_method2a.shape[1]} ({X_quantum_method2a.shape[1]}×)")
    
    # 方法2b: 数据重上传 - 不同的编码方式
    # 在不同层使用不同的旋转轴（X, Y, Z）进行编码
    # 这样可以捕获数据在不同方向上的特征
    print(f"\n方法2b: 数据重上传 - 不同的编码方式")
    
    @qml.qnode(dev)
    def extract_features_encodings(x, layer_idx):
        # 根据层索引选择不同的旋转轴
        if layer_idx == 0:
            qml.AngleEmbedding(x, wires=[0], rotation="Y")  # Y 轴旋转
        elif layer_idx == 1:
            qml.AngleEmbedding(x, wires=[0], rotation="X")  # X 轴旋转
        else:
            qml.AngleEmbedding(x, wires=[0], rotation="Z")  # Z 轴旋转
        return qml.expval(qml.PauliZ(0))
    
    X_quantum_method2b = []
    for x in X_normalized:
        features = [extract_features_encodings(x, i) for i in range(n_layers)]
        X_quantum_method2b.append(features)
    X_quantum_method2b = np.array(X_quantum_method2b)
    
    print(f"  升维效果: 1 → {X_quantum_method2b.shape[1]} ({X_quantum_method2b.shape[1]}×)")
    
    # 方法2c: 数据重上传 - 变分层
    # 在编码和测量之间加入可学习的变分层（旋转门）
    # 变分层包含可训练的参数，可以在训练过程中优化
    print(f"\n方法2c: 数据重上传 - 变分层")
    # 定义变分层参数（每层两个旋转角度：RY 和 RZ）
    variational_params = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    
    @qml.qnode(dev)
    def extract_features_variational(x, layer_idx):
        # 初始编码
        qml.AngleEmbedding(x, wires=[0], rotation="Y")
        # 应用变分层（可学习的旋转门）
        params = variational_params[layer_idx]
        qml.RY(params[0], wires=0)  # Y 轴旋转
        qml.RZ(params[1], wires=0)  # Z 轴旋转
        return qml.expval(qml.PauliZ(0))
    
    X_quantum_method2c = []
    for x in X_normalized:
        features = [extract_features_variational(x, i) for i in range(n_layers)]
        X_quantum_method2c.append(features)
    X_quantum_method2c = np.array(X_quantum_method2c)
    
    print(f"  升维效果: 1 → {X_quantum_method2c.shape[1]} ({X_quantum_method2c.shape[1]}×)")
    
    # 可视化四种不同的升维方法
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 方法1: Z+X+Y（单量子比特测量三个可观测量）
    axes[0, 0].scatter(X_train, X_quantum_method1[:len(X_train), 0], alpha=0.6, label='⟨Z⟩', s=30)
    axes[0, 0].scatter(X_train, X_quantum_method1[:len(X_train), 1], alpha=0.6, label='⟨X⟩', s=30)
    axes[0, 0].scatter(X_train, X_quantum_method1[:len(X_train), 2], alpha=0.6, label='⟨Y⟩', s=30)
    axes[0, 0].set_xlabel('原始特征 (Original Feature)')
    axes[0, 0].set_ylabel('量子特征值 (Quantum Feature Value)')
    axes[0, 0].set_title('方法1: Z+X+Y (1→3 维)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 方法2a: 不同旋转角度（数据重上传）
    for i in range(n_layers):
        axes[0, 1].scatter(X_train, X_quantum_method2a[:len(X_train), i], alpha=0.6, label=f'层 {i+1}', s=30)
    axes[0, 1].set_xlabel('原始特征 (Original Feature)')
    axes[0, 1].set_ylabel('量子特征值 (Quantum Feature Value)')
    axes[0, 1].set_title('方法2a: 不同旋转角度 (1→3 维)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 方法2b: 不同编码方式（数据重上传）
    for i in range(n_layers):
        axes[1, 0].scatter(X_train, X_quantum_method2b[:len(X_train), i], alpha=0.6, label=f'层 {i+1}', s=30)
    axes[1, 0].set_xlabel('原始特征 (Original Feature)')
    axes[1, 0].set_ylabel('量子特征值 (Quantum Feature Value)')
    axes[1, 0].set_title('方法2b: 不同编码方式 (1→3 维)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 方法2c: 变分层（数据重上传）
    for i in range(n_layers):
        axes[1, 1].scatter(X_train, X_quantum_method2c[:len(X_train), i], alpha=0.6, label=f'层 {i+1}', s=30)
    axes[1, 1].set_xlabel('原始特征 (Original Feature)')
    axes[1, 1].set_ylabel('量子特征值 (Quantum Feature Value)')
    axes[1, 1].set_title('方法2c: 变分层 (1→3 维)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/week0_single_feature_expansion.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 可视化已保存: {OUTPUT_DIR}/week0_single_feature_expansion.png")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数
    
    运行所有演示：
    1. 三种测量返回类型演示
    2. 量子特征提取 + 经典机器学习演示
    3. 单特征数据的量子特征提取和升维演示
    """
    print("=" * 70)
    print("Week 1 - 量子机器学习基础综合演示")
    print("=" * 70)
    print(f"\n所有输出图片将保存到: {OUTPUT_DIR}/ 文件夹")
    
    # 第一部分：三种测量返回类型
    # 演示 state(), probs(), expval() 三种不同的测量方式
    demo_measurement_types(n_qubits=3)
    
    # 第二部分：量子特征提取 + 经典机器学习
    # 演示如何使用量子电路提取特征，然后用经典模型进行预测
    demo_quantum_feature_classical_ml()
    
    # 第三部分：单特征数据的量子特征提取和升维
    # 演示如何将 1 维特征扩展为多维量子特征
    demo_single_feature_expansion()
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print(f"\n所有输出文件:")
    print(f"  - {OUTPUT_DIR}/week0_measurement_types.png")
    print(f"  - {OUTPUT_DIR}/week0_quantum_feature_classical_ml.png")
    print(f"  - {OUTPUT_DIR}/week0_single_feature_expansion.png")
    print(f"\n详细文档请参考相关文档")


if __name__ == "__main__":
    main()

