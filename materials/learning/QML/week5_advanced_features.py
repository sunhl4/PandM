"""
Week 5 - 高级量子特性

本脚本包含高级量子特性：
1. 贫瘠高原（Barren Plateaus）分析
2. 块编码（Block Encoding）演示

运行：
  python3 week5_advanced_features.py
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def analyze_barren_plateaus(
    n_qubits: int,
    n_layers: int = 2,
    mode: str = "local"
) -> float:
    """
    分析 Barren Plateaus 现象
    
    Barren Plateaus：
        在深层量子神经网络中，损失函数的梯度可能指数级小，
        导致训练困难。
    
    参数：
        n_qubits: 量子比特数
        n_layers: 变分层数
        mode: 测量模式
            - "local": 测量 <Z_0>
            - "global": 测量 <Z_0 ⊗ Z_1 ⊗ ... ⊗ Z_{n-1}>
            - "local_vector": 测量所有 <Z_i> 的平均值
    
    返回：
        平均梯度绝对值
    
    详细说明：参见 quantum_measurement_modes_explained.md
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    
    # 创建可观测量（根据测量模式选择不同的可观测量）
    if mode == "local":
        # 局部测量：只测量第一个量子比特的 Z
        obs = qml.PauliZ(0)
    elif mode == "global":
        # 全局测量：测量所有量子比特的 Z 的乘积（Z_0 ⊗ Z_1 ⊗ ... ⊗ Z_{n-1}）
        obs = qml.PauliZ(0)
        for i in range(1, n_qubits):
            obs = obs @ qml.PauliZ(i)  # 张量积
    elif mode == "local_vector":
        # 局部向量测量：测量所有量子比特的 Z，返回列表
        obs = [qml.PauliZ(i) for i in range(n_qubits)]
    else:
        raise ValueError(f"未知的测量模式: {mode}")
    
    @qml.qnode(dev, interface="autograd")
    def circuit(weights):
        """
        变分量子电路
        
        使用强纠缠层（StronglyEntanglingLayers）作为可学习层。
        这种层包含单量子比特旋转和强纠缠门，可能导致贫瘠高原问题。
        """
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        # 根据测量模式返回不同的测量结果
        if mode == "local_vector":
            return [qml.expval(o) for o in obs]  # 返回列表
        return qml.expval(obs)  # 返回单个值
    
    # 随机初始化权重（较大的标准差可能导致贫瘠高原）
    weights = pnp.random.normal(
        loc=0.0, scale=1.0,
        size=(n_layers, n_qubits, 3),  # StronglyEntanglingLayers 需要 3 个角度
        requires_grad=True
    )
    
    # 计算梯度（用于分析贫瘠高原）
    def scalar_out(w):
        """将输出转换为标量（如果是向量则取平均）"""
        out = circuit(w)
        if mode == "local_vector":
            out = pnp.stack(out).mean()  # 如果是向量，取平均值
        return out
    
    # 计算损失函数关于权重的梯度
    grad = qml.grad(scalar_out)(weights)
    # 返回平均梯度绝对值（梯度越小，说明越接近贫瘠高原）
    return float(pnp.mean(pnp.abs(grad)))


def block_encoding_demo():
    """
    块编码（Block Encoding）演示
    
    Block Encoding 是量子线性代数中的关键技术，用于将经典矩阵编码到量子电路中。
    它在量子算法（如量子线性系统求解器、量子机器学习）中具有重要应用。
    
    原理：
        使用 LCU（Linear Combination of Unitaries，单位算符线性组合）方法：
        1. 将矩阵分解为 Pauli 字符串的线性组合
        2. 使用 PREP + SELECT + PREP† 模式实现
    
    流程：
        - PREP：准备叠加态（系数对应的量子态）
        - SELECT：受控应用单位算符（根据控制比特选择不同的单位算符）
        - PREP†：应用 PREP 的共轭转置（"取消" 初始叠加态）
    
    这样可以将经典矩阵 A 编码为量子算符，使得 ⟨0|U|0⟩ = A（在某个子空间中）。
    """
    # 定义要编码的 4x4 矩阵
    a, b = 0.36, 0.64
    A = np.array([
        [a, 0, 0, b],
        [0, -a, b, 0],
        [0, b, a, 0],
        [b, 0, 0, -a]
    ])
    
    # 将矩阵分解为 Pauli 字符串的线性组合
    # LCU = Linear Combination of Unitaries
    LCU = qml.pauli_decompose(A)
    LCU_coeffs, LCU_ops = LCU.terms()  # 系数和对应的 Pauli 算符
    
    # 归一化的系数平方根（用于准备叠加态）
    # 这些系数决定了在 PREP 步骤中不同基态的概率振幅
    alphas = (np.sqrt(LCU_coeffs) / np.linalg.norm(np.sqrt(LCU_coeffs)))
    
    # 创建量子设备（需要 3 个量子比特：1 个控制比特 + 2 个数据比特）
    dev = qml.device("default.qubit", wires=3)
    
    # 重新标记量子比特（将 Pauli 算符映射到正确的量子比特）
    # 控制比特在 wire 0，数据比特在 wire 1 和 2
    unitaries = [qml.map_wires(op, {0: 1, 1: 2}) for op in LCU_ops]
    
    @qml.qnode(dev)
    def lcu_circuit():
        """
        LCU 电路实现
        
        实现 Linear Combination of Unitaries 算法，用于块编码。
        """
        # PREP 步骤：准备叠加态
        # 将系数 alphas 编码到控制比特的量子态中
        qml.StatePrep(alphas, wires=0)
        
        # SELECT 步骤：受控应用单位算符
        # 根据控制比特的状态，选择性地应用不同的单位算符（Pauli 算符）
        qml.Select(unitaries, control=0)
        
        # PREP† 步骤：应用 PREP 的共轭转置
        # 这相当于"取消"初始的叠加态，实现块编码
        qml.adjoint(qml.StatePrep)(alphas, wires=0)
        return qml.state()
    
    # 执行电路并获取输出量子态
    state = lcu_circuit()
    print("LCU 系数 (LCU coefficients):", np.round(LCU_coeffs, 6))
    print("alphas (归一化的系数平方根):", np.round(alphas, 6))
    print("输出量子态（前 8 个振幅）:", np.round(state[:8], 6))
    
    return state


def main():
    """
    主函数：演示所有高级功能
    
    1. 贫瘠高原分析：分析不同量子比特数和测量模式下的梯度大小
    2. 块编码演示：演示如何使用 LCU 方法将经典矩阵编码到量子电路中
    """
    print("=" * 70)
    print("Week 5 - 高级量子特性")
    print("=" * 70)
    
    # 1. 贫瘠高原（Barren Plateaus）分析
    # 贫瘠高原问题：在深层量子神经网络中，损失函数的梯度可能指数级小，导致训练困难
    print("\n" + "=" * 70)
    print("第一部分：贫瘠高原（Barren Plateaus）分析")
    print("=" * 70)
    print("分析不同量子比特数和测量模式下的平均梯度大小。")
    print("梯度越小，说明越接近贫瘠高原，训练越困难。")
    
    # 测试不同的量子比特数和测量模式
    for n_q in [4, 6, 8]:
        for mode in ["local", "global", "local_vector"]:
            grad_mag = analyze_barren_plateaus(n_q, n_layers=2, mode=mode)
            print(f"  {n_q} 量子比特，2层，{mode}测量: 平均梯度 = {grad_mag:.6f}")
    
    # 2. 块编码（Block Encoding）演示
    print("\n" + "=" * 70)
    print("第二部分：块编码（Block Encoding）演示")
    print("=" * 70)
    print("演示如何使用 LCU 方法将经典矩阵编码到量子电路中。")
    block_encoding_demo()
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

