"""
Week 3 - 量子核方法（分类与回归）

本脚本包含：
1. 量子核岭回归（Quantum Kernel Ridge Regression）
2. 使用量子核 SVM 进行 MNIST 分类
3. 使用 VQC 进行 MNIST 分类

本脚本适用于以下场景：
  - 小数据集（N ~ 200）
  - 低维特征（d = 2..5）
  - 回归目标（例如，吸附能）
  - 图像分类（MNIST）

运行（在 qml_torch 环境中）：
  conda activate qml_torch
  python week3_quantum_kernel_ridge_demo.py

可选配置：
  - 如果您有 CSV 文件，设置 DATA_CSV 为其路径，并设置 FEATURE_COLS / TARGET_COL。

输出：
  - 交叉验证指标打印到标准输出
  - 保存一致性图 PNG（最后一个折的数据）
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# -----------------------------
# 0) 数据配置（在此编辑）
# -----------------------------
DATA_CSV: Optional[str] = None  # 例如: "/Users/shl/nvidia/QML/my_adsorption.csv"
FEATURE_COLS: Optional[List[str]] = None  # 例如: ["f1", "f2", "f3"]
TARGET_COL: Optional[str] = None  # 例如: "E_ads"
SAVE_PLOT: bool = True  # 是否保存图表


# -----------------------------
# 1) 量子核定义
# -----------------------------
def make_quantum_kernel(n_qubits: int):
    """
    创建量子核函数
    
    量子核函数通过测量两个数据点之间的量子态重叠来计算相似度。
    核值 K(x1, x2) = |⟨φ(x1)|φ(x2)⟩|²，其中 φ(x) 是特征映射。
    
    这里使用的特征映射是角度编码（AngleEmbedding），核函数通过投影到 |00...0⟩ 基态来计算。
    
    参数:
        n_qubits: 量子比特数量
    
    返回:
        核函数，接受两个数据点 x1 和 x2，返回它们之间的核值（相似度）
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x1: np.ndarray, x2: np.ndarray):
        # 特征映射 U(x1): 使用角度编码（Y 旋转）将 x1 编码到量子态
        qml.AngleEmbedding(x1, wires=range(n_qubits), rotation="Y")
        # 应用 U(x2)^\dagger: 对 x2 应用特征映射的共轭转置
        # 这相当于计算 ⟨φ(x1)|φ(x2)⟩
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation="Y")
        # 测量投影到 |00...0⟩ 基态的概率
        # 这给出了 |⟨φ(x1)|φ(x2)⟩|² 的值
        return qml.expval(qml.Projector([0] * n_qubits, wires=range(n_qubits)))

    def kernel(x1: np.ndarray, x2: np.ndarray) -> float:
        """核函数：计算两个数据点之间的相似度"""
        return float(kernel_circuit(x1, x2))

    return kernel


def compute_kernel_matrix(X1: np.ndarray, X2: np.ndarray, kernel_fn) -> np.ndarray:
    """
    计算核矩阵
    
    核矩阵 K 的元素 K[i, j] = kernel(x1_i, x2_j) 表示数据点 x1_i 和 x2_j 之间的相似度。
    
    参数:
        X1: 第一个数据集，形状为 (n1, d)
        X2: 第二个数据集，形状为 (n2, d)
        kernel_fn: 核函数，接受两个数据点，返回相似度值
    
    返回:
        核矩阵 K，形状为 (n1, n2)
    """
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2), dtype=float)
    # 计算所有数据点对之间的核值
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_fn(X1[i], X2[j])
    return K


# -----------------------------
# 2) 工具函数
# -----------------------------
@dataclass
class FoldMetrics:
    """
    交叉验证折的评估指标
    
    存储一个交叉验证折的评估指标：
    - rmse: 均方根误差（Root Mean Squared Error）
    - mae: 平均绝对误差（Mean Absolute Error）
    - r2: R² 分数（决定系数）
    """
    rmse: float  # 均方根误差
    mae: float   # 平均绝对误差
    r2: float    # R² 分数


def summarize_metrics(ms: Sequence[FoldMetrics]) -> str:
    """
    汇总多个折的评估指标
    
    计算所有折的指标的平均值和标准差，用于评估模型的整体性能。
    
    参数:
        ms: 多个折的评估指标列表
    
    返回:
        格式化的字符串，包含 RMSE、MAE 和 R² 的平均值 ± 标准差
    """
    rmse = np.array([m.rmse for m in ms])
    mae = np.array([m.mae for m in ms])
    r2 = np.array([m.r2 for m in ms])
    return (
        f"RMSE: {rmse.mean():.4f} ± {rmse.std():.4f} | "
        f"MAE: {mae.mean():.4f} ± {mae.std():.4f} | "
        f"R2: {r2.mean():.4f} ± {r2.std():.4f}"
    )


def load_or_make_demo_data(n: int = 200, d: int = 5, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载或生成演示数据
    
    如果 DATA_CSV 未设置，则生成合成回归数据。
    目标函数包含非线性项（正弦、平方、余弦和交互项），用于测试模型的非线性拟合能力。
    
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
    # 非线性合成目标函数（用于回归任务验证）
    # 包含正弦、平方、线性、余弦交互项和噪声
    y = (
        0.7 * np.sin(X[:, 0])                    # 正弦项（非线性）
        + 0.3 * (X[:, 1] ** 2)                   # 平方项（非线性）
        - 0.2 * X[:, 2]                          # 线性项
        + 0.1 * np.cos(X[:, 3] * X[:, 4])        # 余弦交互项（非线性）
        + 0.05 * rng.normal(size=n)              # 噪声项
    )
    return X, y


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    标准化特征
    
    使用训练集的均值和标准差对训练集和测试集进行标准化。
    这确保了测试集的标准化使用的是训练集的统计信息，避免数据泄露。
    
    参数:
        X_train: 训练集特征，形状为 (n_train, d)
        X_test: 测试集特征，形状为 (n_test, d)
    
    返回:
        标准化后的训练集和测试集特征
    """
    scaler = StandardScaler().fit(X_train)  # 只在训练集上拟合
    return scaler.transform(X_train), scaler.transform(X_test)  # 对训练集和测试集进行变换

def fit_angle_mapper(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    在训练集上拟合每维特征的最小值和最大值
    
    用于将特征缩放到 [0, π] 范围（角度编码所需）。
    只在训练集上拟合，以避免数据泄露。
    
    参数:
        X_train: 训练集特征，形状为 (n_train, d)
    
    返回:
        (最小值数组, 最大值数组)，每个都是形状为 (d,) 的数组
    """
    X_train = np.asarray(X_train, dtype=float)
    mn = X_train.min(axis=0)  # 每维特征的最小值
    mx = X_train.max(axis=0)  # 每维特征的最大值
    return mn, mx


def to_angle_range_with(mn: np.ndarray, mx: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    使用预拟合的最小值和最大值将特征变换到 [0, π] 范围
    
    角度编码需要输入在 [0, π] 范围内，因此需要先进行缩放。
    使用训练集的最小值和最大值，确保测试集使用相同的缩放参数。
    
    参数:
        mn: 每维特征的最小值（从训练集拟合得到）
        mx: 每维特征的最大值（从训练集拟合得到）
        X: 要变换的特征矩阵（可以是训练集或测试集）
    
    返回:
        变换后的特征矩阵，每维特征的值都在 [0, π] 范围内
    """
    X = np.asarray(X, dtype=float)
    # 计算范围（分母），处理常数列（范围为零的情况）
    denom = np.where((mx - mn) < 1e-12, 1.0, (mx - mn))
    # 线性映射到 [0, π]
    out = (X - mn) / denom * np.pi
    # 如果某个特征在训练集中是常数（范围为零），则设置为 0（没有旋转信息）
    out = np.where((mx - mn) < 1e-12, 0.0, out)
    return out


def _load_csv(path: str, feature_cols: List[str], target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    # Prefer pandas (available in qml_torch), but keep a fallback.
    try:
        import pandas as pd

        df = pd.read_csv(path)
        X = df[feature_cols].to_numpy(dtype=float)
        y = df[target_col].to_numpy(dtype=float)
        return X, y
    except Exception:
        import csv

        X_rows = []
        y_rows = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                X_rows.append([float(row[c]) for c in feature_cols])
                y_rows.append(float(row[target_col]))
        return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=float)


def _save_parity_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_png: str) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=25, alpha=0.8, edgecolor="black", linewidth=0.3)
    plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


# -----------------------------
# 3) 主函数：交叉验证对比
# -----------------------------
def main():
    """
    主函数：对比量子核岭回归和 RBF 核岭回归
    
    使用 5 折交叉验证对比两种方法的性能：
    1. 量子核岭回归（Q-KRR）：使用量子核函数
    2. RBF 核岭回归（RBF-KRR）：使用经典 RBF 核函数
    
    输出交叉验证的平均指标和标准差。
    """
    smoke = os.environ.get("QML_SMOKE", "").strip() not in ("", "0", "false", "False")
    n_splits = 3 if smoke else 5
    # 加载数据
    if DATA_CSV is None:
        # 如果没有指定 CSV 文件，使用合成数据
        if smoke:
            X, y = load_or_make_demo_data(n=48, d=5, seed=0)
            print("QML_SMOKE=1: 小样本快速验证（48×5，3 折 CV）。")
        else:
            X, y = load_or_make_demo_data(n=200, d=5, seed=0)
            print("使用合成演示回归数据（设置 DATA_CSV 以使用您的吸附数据集）。")
    else:
        # 从 CSV 文件加载数据
        assert FEATURE_COLS is not None and TARGET_COL is not None, "为 CSV 设置 FEATURE_COLS 和 TARGET_COL。"
        X, y = _load_csv(DATA_CSV, FEATURE_COLS, TARGET_COL)
        print(f"已加载 CSV: {DATA_CSV}")
        print(f"X 形状={X.shape}, y 形状={y.shape}")

    n, d = X.shape
    n_qubits = d  # 对于 d=2..5，这是最简单的映射（每个特征对应一个量子比特）
    qkernel = make_quantum_kernel(n_qubits=n_qubits)  # 创建量子核函数

    # 模型（简单、稳定的默认参数）
    # 您可以稍后调整 alpha/gamma，但首先获得正确的基线。
    qkrr = KernelRidge(alpha=1e-2, kernel="precomputed")  # 量子核岭回归，使用预计算的核矩阵
    rbfkrr = KernelRidge(alpha=1e-2, kernel="rbf", gamma=1.0 / d)  # RBF 核岭回归，gamma 使用默认值 1/d

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    q_metrics: List[FoldMetrics] = []  # 量子核岭回归的指标
    rbf_metrics: List[FoldMetrics] = []  # RBF 核岭回归的指标
    last_fold_parity = None  # 保存最后一个折的结果用于绘制一致性图

    # 遍历每个折
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]  # 划分训练集和测试集
        y_tr, y_te = y[tr_idx], y[te_idx]

        # 标准化特征（两种模型都使用，确保公平对比）
        X_tr_s, X_te_s = scale_features(X_tr, X_te)

        # 经典基线：RBF 核岭回归
        rbfkrr.fit(X_tr_s, y_tr)
        y_hat_rbf = rbfkrr.predict(X_te_s)
        rbf_metrics.append(
            FoldMetrics(
                rmse=math.sqrt(mean_squared_error(y_te, y_hat_rbf)),  # 均方根误差
                mae=mean_absolute_error(y_te, y_hat_rbf),              # 平均绝对误差
                r2=r2_score(y_te, y_hat_rbf),                          # R² 分数
            )
        )

        # 量子核岭回归：计算核矩阵
        # 首先将特征缩放到 [0, π] 范围（角度编码所需）
        mn, mx = fit_angle_mapper(X_tr_s)  # 在训练集上拟合最小值和最大值
        X_tr_q = to_angle_range_with(mn, mx, X_tr_s)  # 变换训练集
        X_te_q = to_angle_range_with(mn, mx, X_te_s)  # 变换测试集

        # 计算核矩阵
        # K_tr: 训练集内所有样本对之间的核值（形状: (n_train, n_train)）
        # K_te: 测试集与训练集之间所有样本对之间的核值（形状: (n_test, n_train)）
        K_tr = compute_kernel_matrix(X_tr_q, X_tr_q, qkernel)
        K_te = compute_kernel_matrix(X_te_q, X_tr_q, qkernel)

        # 训练和预测
        qkrr.fit(K_tr, y_tr)  # 在核矩阵上训练
        y_hat_q = qkrr.predict(K_te)  # 使用核矩阵进行预测
        last_fold_parity = (y_te, y_hat_rbf, y_hat_q, fold)  # 保存最后一个折的结果
        q_metrics.append(
            FoldMetrics(
                rmse=math.sqrt(mean_squared_error(y_te, y_hat_q)),
                mae=mean_absolute_error(y_te, y_hat_q),
                r2=r2_score(y_te, y_hat_q),
            )
        )

        # 打印当前折的结果
        print(f"折 {fold}/{n_splits}")
        print(f"  RBF-KRR: {rbf_metrics[-1]}")
        print(f"  Q-KRR:   {q_metrics[-1]}")

    print("\n" + "=" * 70)
    print(f"{n_splits}-fold CV summary")
    print(f"RBF-KRR: {summarize_metrics(rbf_metrics)}")
    print(f"Q-KRR:   {summarize_metrics(q_metrics)}")

    if SAVE_PLOT and last_fold_parity is not None:
        y_te, y_hat_rbf, y_hat_q, fold = last_fold_parity
        _save_parity_plot(
            y_te,
            y_hat_rbf,
            title=f"RBF-KRR parity (fold {fold})",
            out_png="week3_parity_rbf_krr.png",
        )
        _save_parity_plot(
            y_te,
            y_hat_q,
            title=f"Quantum-KRR parity (fold {fold})",
            out_png="week3_parity_quantum_krr.png",
        )
        print("\nSaved parity plots:")
        print("  - week3_parity_rbf_krr.png")
        print("  - week3_parity_quantum_krr.png")


# ============================================================================
# MNIST Classification Module (from quantum_ml_consolidated.py)
# ============================================================================

def load_mnist_data(
    n_qubits: int,
    digits: Tuple[str, str] = ('3', '6'),
    n_samples_per_class: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载并预处理 MNIST 数据
    
    详细说明：参见 quantum_ml_complete_guide.md 第九部分
    """
    from sklearn.datasets import fetch_openml
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    print("Loading MNIST data...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target
    
    # 过滤指定数字
    mask = (y == digits[0]) | (y == digits[1])
    X_filtered, y_filtered = X[mask], y[mask]
    y_filtered = np.where(y_filtered == digits[0], 0, 1).astype(int)
    
    # PCA 降维
    pca = PCA(n_components=n_qubits)
    X_reduced = pca.fit_transform(X_filtered)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)
    
    # 平衡采样（如果需要）
    if n_samples_per_class is not None:
        X_train_list, y_train_list = [], []
        for label in [0, 1]:
            idx = np.where(y_filtered == label)[0][:n_samples_per_class]
            X_train_list.append(X_scaled[idx])
            y_train_list.append(y_filtered[idx])
        X_scaled = np.concatenate(X_train_list, axis=0)
        y_filtered = np.concatenate(y_train_list, axis=0)
    
    # 缩放到 [0, π] 范围（必须，AngleEmbedding 的要求）
    def scale_to_0_pi(x):
        x = np.asarray(x, dtype=float)
        mn = x.min()
        mx = x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn) * np.pi
    
    X_quantum = scale_to_0_pi(X_scaled)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_quantum, y_filtered, test_size=test_size, random_state=random_state, stratify=y_filtered
    )
    
    print(f"Train shapes: {X_train.shape}, Test shapes: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def compute_kernel_matrix_parallel(
    X1: np.ndarray,
    X2: np.ndarray,
    kernel_fn,
    n_workers: int = 4
) -> np.ndarray:
    """
    并行计算核矩阵
    
    详细说明：参见 quantum_ml_complete_guide.md 第九部分
    """
    from joblib import Parallel, delayed
    import itertools
    
    n1, n2 = X1.shape[0], X2.shape[0]
    
    # 并行计算所有元素
    results = Parallel(n_jobs=n_workers, verbose=1)(
        delayed(kernel_fn)(X1[i], X2[j])
        for i, j in itertools.product(range(n1), range(n2))
    )
    
    # 重构矩阵
    K = np.array(results).reshape(n1, n2)
    return K


def demo_mnist_classification():
    """演示MNIST分类（量子核SVM）"""
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    print("\n" + "=" * 70)
    print("MNIST Classification Demo (Quantum Kernel SVM)")
    print("=" * 70)
    
    n_qubits = 4
    n_samples_per_class = 100  # 每类100个样本（加快演示速度）
    
    # 加载MNIST数据
    X_train, X_test, y_train, y_test = load_mnist_data(
        n_qubits=n_qubits,
        digits=('3', '6'),
        n_samples_per_class=n_samples_per_class
    )
    
    print(f"\n数据加载完成:")
    print(f"  训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"  类别分布: 训练集 {np.bincount(y_train)}, 测试集 {np.bincount(y_test)}")
    
    # 量子核SVM
    print("\n" + "-" * 70)
    print("训练量子核SVM...")
    print("-" * 70)
    
    kernel_fn = make_quantum_kernel(n_qubits)
    
    print("计算训练集核矩阵...")
    K_train = compute_kernel_matrix_parallel(X_train, X_train, kernel_fn, n_workers=4)
    
    print("计算测试集核矩阵...")
    K_test = compute_kernel_matrix_parallel(X_test, X_train, kernel_fn, n_workers=4)
    
    print("训练SVM...")
    svm = SVC(kernel='precomputed')
    svm.fit(K_train, y_train)
    
    y_pred_svm = svm.predict(K_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    
    print(f"\n量子核SVM准确率: {accuracy_svm:.4f}")
    print(f"混淆矩阵:\n{confusion_matrix(y_test, y_pred_svm)}")
    
    print("\n" + "=" * 70)
    print("MNIST分类演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    # 运行回归任务
    main()
    
    # 运行MNIST分类任务（可选，取消注释以运行）
    # demo_mnist_classification()


