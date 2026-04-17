"""
Week 2 - 量子神经网络（QNN）模块

本脚本包含量子神经网络的实现：
1. QuantumLayer: 基本量子层（基于循环的处理方式）
2. QuantumLayerBatch: 支持批量处理的量子层
3. QuantumNeuralNetwork: 完整的混合量子-经典神经网络
4. QuantumClassifier: 简单的量子分类器（用于二分类）

运行：
  python3 week2_quantum_neural_network.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml


class QuantumLayer(nn.Module):
    """
    量子层（PyTorch Module）
    
    功能：
        将经典数据编码到量子态，通过变分量子电路处理，然后测量得到输出。
    
    架构：
        1. 数据编码层：AngleEmbedding（将特征映射为旋转角度）
        2. 变分层：多层旋转门 + 纠缠门
        3. 测量层：测量 Z 方向期望值
    
    参数：
        n_qubits: 量子比特数
        n_layers: 变分层数
        device: 量子设备名称（"default.qubit" 或 "lightning.gpu"）
    
    注意：
        当前实现使用循环逐个处理样本。如需批量处理，使用 QuantumLayerBatch。
    """
    
    def __init__(self, n_qubits: int, n_layers: int, device: str = "default.qubit"):
        super(QuantumLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # 创建 PennyLane 量子设备
        self.dev = qml.device(device, wires=n_qubits)
        
        # 定义可训练参数
        # 形状：(n_layers, n_qubits, 2) - 每层每个量子比特2个旋转角（RY, RZ）
        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 2, dtype=torch.float32) * 0.1
        )
        
        # 定义量子电路
        @qml.qnode(self.dev, interface="torch")
        def quantum_circuit(inputs, weights):
            """
            量子电路定义
            
            步骤：
                1. 角度编码：将输入特征编码为旋转角度
                2. 变分层：多层旋转门和纠缠门
                3. 测量：返回第一个量子比特的 Z 期望值
            """
            # 数据编码
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            
            # 变分层
            for layer in range(n_layers):
                # 单量子比特旋转门
                for qubit in range(n_qubits):
                    qml.RY(weights[layer, qubit, 0], wires=qubit)
                    qml.RZ(weights[layer, qubit, 1], wires=qubit)
                
                # 纠缠层（使用 CNOT 门）相邻量子比特之间使用 CNOT 门进行纠缠
                if layer < n_layers - 1:
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
            
            # 测量第一个量子比特的 Z 期望值
            # 注意：虽然只测量第一个量子比特，但所有量子比特都通过 CNOT 纠缠参与了计算
            # 对于单输出任务（回归、二分类），这通常足够。如需多输出，可测量多个量子比特
            return qml.expval(qml.PauliZ(0))
        
        self.quantum_circuit = quantum_circuit
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 输入张量，形状 (batch_size, n_qubits)
        
        返回：
            输出张量，形状 (batch_size, 1)
        """
        # 确保输入维度正确
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # 循环处理每个样本（如需批量处理，使用 QuantumLayerBatch）
        outputs = []
        for i in range(x.shape[0]):
            output = self.quantum_circuit(x[i], self.weights)
            outputs.append(output)
        
        outputs = torch.stack(outputs)
        if outputs.dim() == 0:
            outputs = outputs.unsqueeze(0)
        
        return outputs.float()


class QuantumLayerBatch(nn.Module):
    """
    量子层（支持批量处理版本）
    
    与 QuantumLayer 的区别：
        - 支持批量处理：可以一次性处理整个批次的数据，无需循环
        - 使用 batch_params=True 启用 PennyLane 的批量处理功能
        - 性能可能更好（特别是使用 GPU 时）
    
    注意：
        - 需要设备支持批量处理（default.qubit 和 lightning.gpu 都支持）
        - 批量处理时，inputs 参数可以是 (batch_size, n_qubits) 形状
        - weights 参数必须是单个值（不批量），所有样本共享相同的权重
    """
    
    def __init__(self, n_qubits: int, n_layers: int, device: str = "default.qubit"):
        super(QuantumLayerBatch, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # 创建 PennyLane 量子设备
        self.dev = qml.device(device, wires=n_qubits)
        
        # 定义可训练参数
        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 2, dtype=torch.float32) * 0.1
        )
        
        # 定义量子电路（启用批量处理）
        # 关键：batch_params=True 允许 inputs 参数是批量数据
        @qml.qnode(self.dev, interface="torch", batch_params=True)
        def quantum_circuit(inputs, weights):
            """
            量子电路定义（支持批量处理）
            
            参数：
                inputs: 输入特征
                    - 单个样本：形状 (n_qubits,)
                    - 批量样本：形状 (batch_size, n_qubits)
                weights: 可训练权重，单个值 (n_layers, n_qubits, 2)
                    - 注意：weights 不会被批量处理，所有样本共享相同的权重
            
            返回：
                - 单个样本时：标量
                - 批量样本时：(batch_size,) 形状的张量
            """
            # 数据编码（支持批量）
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            
            # 变分层
            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.RY(weights[layer, qubit, 0], wires=qubit)
                    qml.RZ(weights[layer, qubit, 1], wires=qubit)
                
                # 纠缠层（使用 CNOT 门）
                if layer < n_layers - 1:
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
            
            # 测量第一个量子比特的 Z 期望值
            return qml.expval(qml.PauliZ(0))
        
        self.quantum_circuit = quantum_circuit
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（批量处理版本）
        
        参数：
            x: 输入张量，形状 (batch_size, n_qubits)
        
        返回：
            输出张量，形状 (batch_size, 1)
        
        优势：
            - 一次性处理整个批次，无需循环
            - 可能更快（特别是使用 GPU 时）
            - 代码更简洁
        """
        # 确保输入维度正确
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # 批量处理：一次性处理所有样本
        outputs = self.quantum_circuit(x, self.weights)
        
        # 确保输出形状正确
        if outputs.dim() == 0:
            outputs = outputs.unsqueeze(0)
        elif outputs.dim() == 1:
            outputs = outputs.unsqueeze(-1)
        
        return outputs.float()


class QuantumNeuralNetwork(nn.Module):
    """
    完整的量子神经网络模型（经典-量子-经典混合架构）
    
    架构：
        1. 经典预处理层：对输入特征进行非线性变换
        2. 量子层：执行量子计算（编码 + 变分层 + 测量）
        3. 经典分类层：将量子输出映射到类别
    
    工作流程：
        输入数据 → 经典预处理层 → 量子层 → 经典分类层 → 输出类别
    
    参数：
        n_qubits: 量子比特数（等于输入特征维度）
        n_layers: 变分层数
        n_classes: 分类类别数（默认2）
    
    详细说明：参见 quantum_ml_complete_guide.md 第九部分
    """
    
    def __init__(self, n_qubits: int, n_layers: int, n_classes: int = 2):
        super(QuantumNeuralNetwork, self).__init__()
        
        # 经典预处理层（可选，用于特征变换）
        self.preprocess = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.Tanh(),
            nn.Linear(n_qubits, n_qubits),
            nn.Tanh()
        )
        
        # 量子层：执行量子计算（编码 + 变分 + 测量）
        self.quantum_layer = QuantumLayer(n_qubits, n_layers)
        
        # 经典分类层：将量子输出映射到类别
        self.classifier = nn.Linear(1, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        流程：
            1. 经典预处理：对输入特征进行非线性变换
            2. 量子计算：将预处理后的特征编码到量子态，执行量子计算，测量
            3. 经典分类：将量子输出映射到类别
        
        参数：
            x: 输入张量，形状 (batch_size, n_qubits)
        
        返回：
            输出张量，形状 (batch_size, n_classes)
        """
        # 经典预处理 → 量子计算 → 经典分类
        x = self.preprocess(x)
        quantum_out = self.quantum_layer(x)
        if quantum_out.dim() == 1:
            quantum_out = quantum_out.unsqueeze(1)
        out = self.classifier(quantum_out)
        
        return out


class QuantumClassifier(nn.Module):
    """
    简单的量子分类器（用于二分类）
    
    这是一个更简单的版本，直接使用 VQC 进行分类。
    使用 StronglyEntanglingLayers 作为可学习层。
    """
    
    def __init__(self, n_qubits: int, n_layers: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # 创建量子设备
        dev = qml.device("default.qubit", wires=n_qubits)
        
        # 定义量子电路
        @qml.qnode(dev, interface="torch")
        def quantum_net(inputs, weights):
            """
            量子网络
            
            步骤：
                1. 数据编码：AngleEmbedding
                2. 可学习层：StronglyEntanglingLayers
                3. 测量：返回第一个量子比特的 Z 期望值
            """
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))
        
        self.quantum_net = quantum_net
        
        # 定义可学习权重
        # 形状：(n_layers, n_qubits, 3) - StronglyEntanglingLayers 需要3个角度
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 输入张量，形状 (batch_size, n_qubits)
        
        返回：
            输出张量，形状 (batch_size, 1)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        outputs = []
        for i in range(x.shape[0]):
            output = self.quantum_net(x[i], self.weights)
            outputs.append(output)
        
        return torch.stack(outputs)


def demo_quantum_layers():
    """演示量子层的使用"""
    print("=" * 70)
    print("Week 2 - Quantum Neural Network Demo")
    print("=" * 70)
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_qubits = 4
    n_layers = 2
    batch_size = 8
    
    # 创建测试数据（已缩放到 [0, π]）
    X = torch.rand(batch_size, n_qubits) * np.pi
    
    print(f"\n测试数据形状: {X.shape}")
    print(f"数据范围: [{X.min():.3f}, {X.max():.3f}]")
    
    # 1. QuantumLayer（循环版本）
    print("\n" + "-" * 70)
    print("1. QuantumLayer (循环版本)")
    print("-" * 70)
    quantum_layer = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)
    output1 = quantum_layer(X)
    print(f"输出形状: {output1.shape}")
    print(f"输出范围: [{output1.min():.3f}, {output1.max():.3f}]")
    
    # 2. QuantumLayerBatch（批量版本）
    print("\n" + "-" * 70)
    print("2. QuantumLayerBatch (批量版本)")
    print("-" * 70)
    quantum_layer_batch = QuantumLayerBatch(n_qubits=n_qubits, n_layers=n_layers)
    output2 = quantum_layer_batch(X)
    print(f"输出形状: {output2.shape}")
    print(f"输出范围: [{output2.min():.3f}, {output2.max():.3f}]")
    
    # 3. QuantumNeuralNetwork（完整模型）
    print("\n" + "-" * 70)
    print("3. QuantumNeuralNetwork (完整模型)")
    print("-" * 70)
    qnn = QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=n_layers, n_classes=2)
    output3 = qnn(X)
    print(f"输出形状: {output3.shape}")
    print(f"输出范围: [{output3.min():.3f}, {output3.max():.3f}]")
    
    # 4. QuantumClassifier（简单分类器）
    print("\n" + "-" * 70)
    print("4. QuantumClassifier (简单分类器)")
    print("-" * 70)
    qclassifier = QuantumClassifier(n_qubits=n_qubits, n_layers=n_layers)
    output4 = qclassifier(X)
    print(f"输出形状: {output4.shape}")
    print(f"输出范围: [{output4.min():.3f}, {output4.max():.3f}]")
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo_quantum_layers()

