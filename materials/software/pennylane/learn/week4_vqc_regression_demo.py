"""
Week 4 - VQC（变分量子电路）回归演示 + 经典基线

目标：
  在小规模低维数据（d=2..5）上训练一个小而稳定的 VQC 回归器，
  并与经典基线（Ridge、RBF-KRR）进行对比。

为什么这样设计（与 Week2 的贫瘠高原问题相关）：
  - 浅层电路（n_layers 小）：避免贫瘠高原问题
  - 局部测量（<Z0>），可选平均 <Zi>：减少梯度消失
  - 参数少，简单纠缠（环形 CNOT）：提高训练稳定性

运行（在 qml_torch 环境中）：
  conda activate qml_torch
  python week4_vqc_regression_demo.py

可选配置：
  - 设置 DATA_CSV / FEATURE_COLS / TARGET_COL 以使用您的吸附数据集。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# -----------------------------
# 0) 数据配置（在此编辑）
# -----------------------------
DATA_CSV: Optional[str] = None  # 例如: "/Users/shl/nvidia/QML/my_adsorption.csv"
FEATURE_COLS: Optional[List[str]] = None  # 例如: ["f1", "f2", "f3"]
TARGET_COL: Optional[str] = None  # 例如: "E_ads"


# -----------------------------
# 1) 数据工具函数
# -----------------------------
def load_or_make_demo_data(n: int = 200, d: int = 5, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载或生成演示数据
    
    生成包含非线性项的合成回归数据，用于测试 VQC 回归器的性能。
    
    参数:
        n: 样本数量
        d: 特征维度
        seed: 随机种子
    
    返回:
        X: 特征矩阵，形状为 (n, d)
        y: 目标值，形状为 (n,)
    """
    rng = np.random.default_rng(seed)
    # 生成随机特征矩阵（标准正态分布）
    X = rng.normal(size=(n, d))
    # 非线性目标函数（包含正弦、平方、线性、余弦交互项和噪声）
    y = (
        0.7 * np.sin(X[:, 0])                    # 正弦项
        + 0.3 * (X[:, 1] ** 2)                   # 平方项
        - 0.2 * X[:, 2]                          # 线性项
        + 0.1 * np.cos(X[:, 3] * X[:, 4])        # 余弦交互项
        + 0.05 * rng.normal(size=n)              # 噪声项
    )
    return X, y


def _load_csv(path: str, feature_cols: List[str], target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(path)
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    return X, y


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), scaler


def to_angle_range_trainfit(X_train: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    使用训练集的最小值和最大值将特征映射到 [0, π] 范围
    
    角度编码需要输入在 [0, π] 范围内。使用训练集的统计信息避免数据泄露。
    
    参数:
        X_train: 训练集特征（用于拟合最小值和最大值）
        X: 要变换的特征矩阵（可以是训练集或测试集）
    
    返回:
        变换后的特征矩阵，每维特征的值都在 [0, π] 范围内
    """
    mn = X_train.min(axis=0)  # 每维特征的最小值
    mx = X_train.max(axis=0)  # 每维特征的最大值
    denom = np.where((mx - mn) < 1e-12, 1.0, (mx - mn))  # 处理常数列
    out = (X - mn) / denom * np.pi  # 线性映射到 [0, π]
    out = np.where((mx - mn) < 1e-12, 0.0, out)  # 常数列设为 0
    return out


# -----------------------------
# 2) VQC 模型
# -----------------------------
def make_vqc(n_qubits: int, n_layers: int, measurement: str = "local"):
    """
    创建变分量子电路（VQC）回归器
    
    该函数返回一个前向传播函数，用于 VQC 回归。
    
    参数:
        n_qubits: 量子比特数量
        n_layers: 变分层数量（建议较小，避免贫瘠高原问题）
        measurement: 测量方式
            - "local": 测量 <Z0>（第一个量子比特的 Z 期望值）
            - "mean_local": 测量所有量子比特的 Z 期望值的平均值
    
    返回:
        前向传播函数 forward(x, weights, a, b)，返回预测值
        - x: 输入数据点（已缩放到 [0, π]）
        - weights: 变分层权重，形状为 (n_layers, n_qubits, 2)
        - a, b: 线性输出层的参数（y = a * <Z> + b）
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    def entangle_ring():
        """
        环形纠缠模式
        
        使用 CNOT 门在相邻量子比特之间创建环形纠缠。
        这种方式简单高效，适合小规模电路。
        """
        # 相邻量子比特之间的 CNOT
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        # 如果量子比特数 > 2，添加最后一个量子比特到第一个量子比特的连接（形成环）
        if n_qubits > 2:
            qml.CNOT(wires=[n_qubits - 1, 0])

    @qml.qnode(dev, interface="autograd")
    def circuit(x, weights):
        """
        变分量子电路
        
        步骤：
        1. 角度编码：将输入数据编码到量子态
        2. 变分层：多层可学习的旋转门和纠缠门
        3. 测量：返回可观测量（Z 期望值）
        """
        # 1. 数据编码：使用角度编码将输入特征编码到量子态
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        # 2. 变分层：多层可学习的旋转门和纠缠门
        for l in range(n_layers):
            # 单量子比特旋转门（每层每个量子比特有两个旋转角：RY 和 RZ）
            for q in range(n_qubits):
                qml.RY(weights[l, q, 0], wires=q)  # Y 轴旋转
                qml.RZ(weights[l, q, 1], wires=q)  # Z 轴旋转
            # 纠缠层：使用环形 CNOT 模式创建纠缠
            entangle_ring()

        # 3. 测量：根据测量模式返回不同的可观测量
        if measurement == "local":
            return qml.expval(qml.PauliZ(0))  # 测量第一个量子比特的 Z 期望值
        if measurement == "mean_local":
            # 测量所有量子比特的 Z 期望值（返回列表）
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        raise ValueError(f"未知的测量方式: {measurement}")

    def forward(x: np.ndarray, weights, a, b) -> float:
        """
        前向传播函数
        
        执行量子电路计算，然后应用线性输出层。
        """
        out = circuit(x, weights)
        # 如果是平均测量模式，计算平均值
        if measurement == "mean_local":
            out = pnp.stack(out).mean()
        # 应用线性输出层：y = a * <Z> + b
        return a * out + b

    return forward


@dataclass
class FoldMetrics:
    rmse: float
    mae: float
    r2: float


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> FoldMetrics:
    return FoldMetrics(
        rmse=math.sqrt(mean_squared_error(y_true, y_pred)),
        mae=mean_absolute_error(y_true, y_pred),
        r2=r2_score(y_true, y_pred),
    )


def summarize(name: str, ms: Sequence[FoldMetrics]) -> str:
    rmse = np.array([m.rmse for m in ms])
    mae = np.array([m.mae for m in ms])
    r2 = np.array([m.r2 for m in ms])
    return (
        f"{name}: RMSE {rmse.mean():.4f}±{rmse.std():.4f} | "
        f"MAE {mae.mean():.4f}±{mae.std():.4f} | "
        f"R2 {r2.mean():.4f}±{r2.std():.4f}"
    )


def train_vqc_regressor(
    X_train_angles: np.ndarray,
    y_train: np.ndarray,
    n_layers: int,
    measurement: str,
    n_steps: int = 200,
    lr: float = 0.05,
    seed: int = 0,
):
    """
    训练 VQC 回归器
    
    使用全批次训练和 PennyLane 的 AdamOptimizer（autograd 后端）。
    同时训练电路权重和线性输出层参数 (a, b)。
    
    参数:
        X_train_angles: 训练集特征（已缩放到 [0, π] 范围），形状为 (n_train, n_qubits)
        y_train: 训练集目标值，形状为 (n_train,)
        n_layers: 变分层数量
        measurement: 测量方式（"local" 或 "mean_local"）
        n_steps: 训练步数
        lr: 学习率
        seed: 随机种子（用于初始化权重）
    
    返回:
        weights: 训练后的电路权重，形状为 (n_layers, n_qubits, 2)
        a: 线性输出层参数 a
        b: 线性输出层参数 b
        forward: 前向传播函数
    """
    rng = np.random.default_rng(seed)
    n_qubits = X_train_angles.shape[1]

    # 初始化权重（小值有助于可训练性，避免梯度消失）
    weights = pnp.array(rng.normal(scale=0.1, size=(n_layers, n_qubits, 2)), requires_grad=True)
    a = pnp.array(1.0, requires_grad=True)  # 线性输出层参数 a（初始化为 1）
    b = pnp.array(0.0, requires_grad=True)  # 线性输出层参数 b（初始化为 0）

    # 创建 VQC 前向传播函数
    forward = make_vqc(n_qubits=n_qubits, n_layers=n_layers, measurement=measurement)
    # 使用 Adam 优化器
    opt = qml.AdamOptimizer(stepsize=lr)

    # 将数据转换为 PennyLane 数组（不需要梯度）
    X_train_angles_p = pnp.array(X_train_angles, requires_grad=False)
    y_train_p = pnp.array(y_train, requires_grad=False)

    def loss_fn(w, aa, bb):
        """损失函数（均方误差）"""
        # 对所有训练样本进行预测
        preds = [forward(X_train_angles_p[i], w, aa, bb) for i in range(len(X_train_angles_p))]
        preds = pnp.stack(preds)
        # 计算均方误差
        return pnp.mean((preds - y_train_p) ** 2)

    # 训练循环
    for _ in range(n_steps):
        # AdamOptimizer.step 返回更新后的可训练参数（不返回损失，除非使用 step_and_cost）
        weights, a, b = opt.step(loss_fn, weights, a, b)

    return weights, a, b, forward


def predict_vqc(X_angles: np.ndarray, weights, a, b, forward) -> np.ndarray:
    """
    使用训练好的 VQC 进行预测
    
    参数:
        X_angles: 输入特征（已缩放到 [0, π] 范围），形状为 (n_samples, n_qubits)
        weights: 训练好的电路权重
        a, b: 训练好的线性输出层参数
        forward: 前向传播函数
    
    返回:
        预测值数组，形状为 (n_samples,)
    """
    preds = [forward(X_angles[i], weights, a, b) for i in range(len(X_angles))]
    return np.asarray(pnp.stack(preds), dtype=float)


def main():
    smoke = os.environ.get("QML_SMOKE", "").strip() not in ("", "0", "false", "False")
    n_splits = 3 if smoke else 5
    # Data
    if DATA_CSV is None:
        if smoke:
            X, y = load_or_make_demo_data(n=48, d=5, seed=0)
            print("QML_SMOKE=1: small synthetic set (48x5), 3-fold CV.")
        else:
            X, y = load_or_make_demo_data(n=200, d=5, seed=0)
            print("Using synthetic demo regression data (set DATA_CSV to use your adsorption dataset).")
    else:
        assert FEATURE_COLS is not None and TARGET_COL is not None, "Set FEATURE_COLS and TARGET_COL for CSV."
        X, y = _load_csv(DATA_CSV, FEATURE_COLS, TARGET_COL)
        print(f"Loaded CSV: {DATA_CSV}")
        print(f"X shape={X.shape}, y shape={y.shape}")

    n, d = X.shape
    n_qubits = d

    # Baselines
    ridge = Ridge(alpha=1.0)
    rbfkrr = KernelRidge(alpha=1e-2, kernel="rbf", gamma=1.0 / d)

    # VQC settings (small & stable)
    n_layers = 2
    measurement = "local"  # try "mean_local" later

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    ridge_ms: List[FoldMetrics] = []
    rbf_ms: List[FoldMetrics] = []
    vqc_ms: List[FoldMetrics] = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # scale (fair)
        X_tr_s, X_te_s, _scaler = scale_features(X_tr, X_te)

        # ridge
        ridge.fit(X_tr_s, y_tr)
        y_hat_ridge = ridge.predict(X_te_s)
        ridge_ms.append(metrics(y_te, y_hat_ridge))

        # rbf-krr
        rbfkrr.fit(X_tr_s, y_tr)
        y_hat_rbf = rbfkrr.predict(X_te_s)
        rbf_ms.append(metrics(y_te, y_hat_rbf))

        # vqc: map to angles using train only
        X_tr_a = to_angle_range_trainfit(X_tr_s, X_tr_s)
        X_te_a = to_angle_range_trainfit(X_tr_s, X_te_s)

        w, a, b, forward = train_vqc_regressor(
            X_tr_a,
            y_tr,
            n_layers=n_layers,
            measurement=measurement,
            n_steps=200,
            lr=0.05,
            seed=fold,
        )
        y_hat_vqc = predict_vqc(X_te_a, w, a, b, forward)
        vqc_ms.append(metrics(y_te, y_hat_vqc))

        print(f"Fold {fold}/{n_splits}")
        print(f"  Ridge: {ridge_ms[-1]}")
        print(f"  RBF-KRR: {rbf_ms[-1]}")
        print(f"  VQC(n_layers={n_layers}, meas={measurement}): {vqc_ms[-1]}")

    print("\n" + "=" * 70)
    print(f"{n_splits}-fold CV summary")
    print(summarize("Ridge", ridge_ms))
    print(summarize("RBF-KRR", rbf_ms))
    print(summarize("VQC", vqc_ms))


# ============================================================================
# VQC Regression with PyTorch (from quantum_ml_consolidated.py)
# ============================================================================

def train_vqc_regressor_pytorch(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_qubits: int,
    n_layers: int = 2,
    n_epochs: int = 50,
    learning_rate: float = 0.01,
    batch_size: int = 32
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    训练VQC回归器（使用PyTorch，从quantum_ml_consolidated.py）
    
    详细说明：参见 quantum_ml_complete_guide.md 第九部分
    
    注意：
        - 目标值会自动缩放到 [-1, 1] 范围（因为量子输出通常在 [-1, 1]）
        - 使用week2_quantum_neural_network.py中的QuantumClassifier
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import MinMaxScaler
    
    # 导入QuantumClassifier（需要week2_quantum_neural_network.py在同一目录）
    try:
        from week2_quantum_neural_network import QuantumClassifier
    except ImportError:
        print("警告: 无法导入QuantumClassifier，请确保week2_quantum_neural_network.py在同一目录")
        return {}, np.array([])
    
    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # 标准化目标值到 [-1, 1] 范围（因为量子输出通常在 [-1, 1]）
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1)
    
    # 创建VQC模型
    model = QuantumClassifier(n_qubits=n_qubits, n_layers=n_layers)
    
    # 定义损失函数和优化器（回归使用MSE）
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 训练模型
    print("Training VQC Regressor (PyTorch)...")
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(batch_x)
            
            # 计算损失
            loss = criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
    
    # 评估模型
    print("Evaluating VQC Regressor...")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        y_pred_scaled = test_outputs.squeeze().numpy()
    
    # 反缩放预测值
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # 计算指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    
    print(f"Test Metrics: MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")
    
    return metrics, y_pred


if __name__ == "__main__":
    main()


