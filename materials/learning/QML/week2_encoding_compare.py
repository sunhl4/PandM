"""
Week 2 - 编码方式对比（角度编码 vs 幅度编码 vs 基态编码）

本脚本对比三种主要的量子编码方式：
1. AngleEmbedding（角度编码）：输入维度约等于量子比特数，适合 2-5 个特征
2. AmplitudeEmbedding（幅度编码）：需要长度为 2^n 的向量（或填充），将数据编码到量子态振幅中
3. BasisEmbedding（基态编码）：离散/二进制编码，不适合连续原始特征

运行：
  python3 week2_encoding_compare.py

从本脚本中应该学习到：
  - AngleEmbedding：输入维度约等于量子比特数，对于 2-5 个特征来说最直观
  - AmplitudeEmbedding：需要长度为 2^n 的向量（或填充），将数据编码到振幅中
  - BasisEmbedding：离散/二进制编码，不适合连续原始特征

可视化内容：
  - 输入数据可视化
  - 概率分布对比
  - Z 期望值对比
  - 编码效率对比
  - 多维编码分析
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def _scale_to_0_pi(x: np.ndarray) -> np.ndarray:
    """
    将向量按特征缩放到 [0, π] 范围
    
    角度编码需要输入在 [0, π] 范围内，因此需要先进行缩放。
    对每个特征维度分别进行线性映射。
    
    参数:
        x: 输入向量，可以是 1D 或 2D 数组
    
    返回:
        缩放后的向量，每个特征的值都在 [0, π] 范围内
    """
    x = np.asarray(x, dtype=float)
    mn = x.min()
    mx = x.max()
    # 处理常数列（范围为零的情况）
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    # 线性映射到 [0, π]
    return (x - mn) / (mx - mn) * np.pi


def _pad_to_pow2(x: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    将 1D 向量填充到长度为 2^n
    
    幅度编码需要输入长度为 2^n 的向量，因此对于非 2 的幂次长度的向量需要填充。
    
    参数:
        x: 输入向量（1D 或可以展平的数组）
    
    返回:
        (填充后的向量, 需要的量子比特数)
    """
    x = np.asarray(x, dtype=float).ravel()  # 展平为 1D 数组
    d = x.shape[0]  # 原始长度
    # 计算需要的量子比特数：ceil(log2(d))
    n_qubits = int(math.ceil(math.log2(max(d, 1))))
    dim = 2**n_qubits  # 目标长度（2 的幂次）
    # 创建零向量并填充原始数据
    padded = np.zeros(dim, dtype=float)
    padded[:d] = x
    return padded, n_qubits


def _topk_probs(probs: np.ndarray, k: int = 8) -> List[Tuple[str, float]]:
    """
    获取概率最大的 k 个基态及其概率
    
    用于可视化时只显示概率最大的几个基态，而不是所有基态。
    
    参数:
        probs: 概率分布数组，长度为 2^n_qubits
        k: 返回的前 k 个最大概率的基态数量
    
    返回:
        列表，每个元素是 (基态二进制字符串, 概率值) 的元组
    """
    probs = np.asarray(probs, dtype=float).ravel()
    # 找到概率最大的 k 个索引
    idx = np.argsort(probs)[::-1][:k]
    # 计算量子比特数
    n_qubits = int(round(math.log2(len(probs))))
    out: List[Tuple[str, float]] = []
    for i in idx:
        # 将索引转换为二进制字符串（如 "000", "001", "010" 等）
        bitstr = format(i, f"0{n_qubits}b")
        out.append((bitstr, float(probs[i])))
    return out


def _z_expectations(qnode, x: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    计算所有量子比特的 Z 期望值
    
    通过复用相同的态准备电路，计算每个量子比特的 PauliZ 期望值。
    这样可以高效地获取所有量子比特的 Z 期望值作为特征。
    
    参数:
        qnode: 量子节点（量子电路），接受 (x, idx) 参数，返回第 idx 个量子比特的 Z 期望值
        x: 输入数据向量
        n_qubits: 量子比特数量
    
    返回:
        所有量子比特的 Z 期望值数组，形状为 (n_qubits,)
    """
    exps = []
    # 对每个量子比特计算 Z 期望值
    for i in range(n_qubits):
        exps.append(qnode(x, i))
    return np.asarray(exps, dtype=float)


@dataclass(frozen=True)
class EncodingResult:
    """
    编码结果数据类
    
    存储一种编码方式的结果，包括：
    - name: 编码方式名称
    - n_qubits: 使用的量子比特数
    - probs: 概率分布（各基态被测量到的概率）
    - z_exps: 所有量子比特的 Z 期望值
    """
    name: str  # 编码方式名称
    n_qubits: int  # 使用的量子比特数
    probs: np.ndarray  # 概率分布，形状为 (2^n_qubits,)
    z_exps: np.ndarray  # Z 期望值，形状为 (n_qubits,)


def angle_encoding(x: np.ndarray, rotation: str = "Y") -> EncodingResult:
    """
    角度编码（Angle Encoding）
    
    原理：将每个特征值映射为量子比特的旋转角度。
    对于 n 个特征，需要 n 个量子比特。每个量子比特独立旋转，不产生纠缠。
    
    参数：
        x: 输入特征向量（长度 = n_qubits）
        rotation: 旋转轴（"X", "Y", "Z"），默认 "Y"
    
    返回：
        EncodingResult 对象
    """
    x = _scale_to_0_pi(x)
    n_qubits = len(x)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def probs_circuit(inputs: np.ndarray):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation=rotation)
        return qml.probs(wires=range(n_qubits))

    @qml.qnode(dev)
    def z_circuit(inputs: np.ndarray, idx: int):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation=rotation)
        return qml.expval(qml.PauliZ(idx))

    probs = probs_circuit(x)
    z_exps = _z_expectations(z_circuit, x, n_qubits)
    return EncodingResult(name=f"AngleEmbedding({rotation})", n_qubits=n_qubits, probs=probs, z_exps=z_exps)


def amplitude_encoding(x: np.ndarray, normalize: bool = True) -> EncodingResult:
    """
    幅度编码（Amplitude Encoding）
    
    原理：将特征值直接映射为量子态的振幅。
    对于长度为 2^n 的特征向量，只需要 n 个量子比特。这是最紧凑的编码方式（对数资源）。
    
    参数：
        x: 输入特征向量（长度应为 2^n，否则会自动填充）
        normalize: 是否归一化（确保概率和为1），默认 True
    
    返回：
        EncodingResult 对象
    """
    padded, n_qubits = _pad_to_pow2(x)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def probs_circuit(inputs: np.ndarray):
        # NOTE: normalize=True makes it a valid quantum state even after padding.
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=normalize, pad_with=0.0)
        return qml.probs(wires=range(n_qubits))

    @qml.qnode(dev)
    def z_circuit(inputs: np.ndarray, idx: int):
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=normalize, pad_with=0.0)
        return qml.expval(qml.PauliZ(idx))

    probs = probs_circuit(padded)
    z_exps = _z_expectations(z_circuit, padded, n_qubits)
    return EncodingResult(name="AmplitudeEmbedding(+pad)", n_qubits=n_qubits, probs=probs, z_exps=z_exps)


def basis_encoding_from_thresholds(x: np.ndarray) -> EncodingResult:
    """
    基态编码演示：通过阈值化将连续特征转换为二进制位
    
    注意：这不是用于实际工作的正确方法，但有助于理解约束：
    BasisEmbedding 期望整数/二进制位（离散值），而不是连续值。
    
    这里我们通过阈值化将连续值转换为二进制位：
    - 大于中位数的值变为 1
    - 小于等于中位数的值变为 0
    
    参数:
        x: 输入特征向量（连续值）
    
    返回:
        EncodingResult 对象
    """
    x = np.asarray(x, dtype=float).ravel()
    n_qubits = len(x)
    # 使用中位数作为阈值，将连续值转换为二进制位
    bits = (x > np.median(x)).astype(int)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def probs_circuit(inputs: np.ndarray):
        qml.BasisEmbedding(inputs, wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))

    @qml.qnode(dev)
    def z_circuit(inputs: np.ndarray, idx: int):
        qml.BasisEmbedding(inputs, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(idx))

    probs = probs_circuit(bits)
    z_exps = _z_expectations(z_circuit, bits, n_qubits)
    return EncodingResult(name="BasisEmbedding(thresholded)", n_qubits=n_qubits, probs=probs, z_exps=z_exps)


def visualize_encoding_comparison(all_results: List[Tuple[int, List[EncodingResult]]]):
    """
    可视化编码方式对比
    
    生成多个图表，对比不同编码方式在不同维度下的性能：
    1. 概率分布对比（针对特定维度）
    2. 编码效率对比（量子比特数和态空间维度）
    3. Z 期望值热图（所有维度）
    4. 概率分布对比（所有维度）
    5. 编码方式特性雷达图
    
    参数:
        all_results: [(dim, [EncodingResult]), ...] 所有维度的编码结果
                    每个元素是一个元组：(输入维度, 该维度下的所有编码结果列表)
    """
    print("\n生成可视化图表...")
    
    # ========================================================================
    # 图 1: 不同编码方式的概率分布对比（针对特定维度）
    # ========================================================================
    dim_to_visualize = 3  # 选择维度 3 进行详细可视化（可以修改为其他维度）
    dim_results = next((r for d, r in all_results if d == dim_to_visualize), None)
    
    if dim_results:
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # 获取输入数据（使用固定随机种子以确保可重复性）
        np.random.seed(0)
        x = np.random.normal(size=(dim_to_visualize,))
        x_scaled = _scale_to_0_pi(x)  # 缩放到 [0, π] 范围（用于角度编码）
        
        # 子图 1: 输入数据可视化
        # 显示原始输入数据（未缩放）
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(range(len(x)), x, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('特征索引 (Feature Index)')
        ax1.set_ylabel('特征值 (Feature Value)')
        ax1.set_title(f'输入数据 (d={dim_to_visualize})')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5)  # 零线
        # 在每个柱子上标注数值
        for i, val in enumerate(x):
            ax1.text(i, val, f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=9)
        
        # 子图 2: 缩放后的数据（用于 AngleEmbedding）
        # 显示缩放到 [0, π] 范围后的数据
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(range(len(x_scaled)), x_scaled, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('特征索引 (Feature Index)')
        ax2.set_ylabel('缩放后的值 [0, π]')
        ax2.set_title('缩放后的数据（用于角度编码）')
        ax2.set_ylim([0, np.pi])  # Y 轴限制在 [0, π]
        ax2.grid(True, alpha=0.3, axis='y')
        # 在每个柱子上标注数值
        for i, val in enumerate(x_scaled):
            ax2.text(i, val, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 子图 3: Z 期望值对比
        # 对比不同编码方式下各量子比特的 Z 期望值
        ax3 = fig.add_subplot(gs[0, 2])
        methods = [r.name for r in dim_results]
        
        # 找到最大的量子比特数（用于对齐不同编码方式的柱状图）
        max_qubits = max(len(r.z_exps) for r in dim_results)
        
        # 为每个编码方式绘制 Z 期望值
        x_pos = np.arange(max_qubits)  # X 轴位置
        width = 0.25  # 柱状图宽度
        colors = ['skyblue', 'lightgreen', 'lightcoral']  # 不同编码方式的颜色
        
        for method_idx, result in enumerate(dim_results):
            z_exps = result.z_exps
            # 如果量子比特数少于最大值，只绘制实际存在的
            n_qubits = len(z_exps)
            # 计算每个柱状图的 X 轴位置（偏移以避免重叠）
            x_positions = x_pos[:n_qubits] + method_idx * width
            ax3.bar(x_positions, z_exps, width, 
                   label=f'{result.name.split("(")[0].strip()}', 
                   alpha=0.8, color=colors[method_idx % len(colors)])
        
        ax3.set_xlabel('量子比特索引 (Qubit Index)')
        ax3.set_ylabel('⟨Zᵢ⟩ 值')
        ax3.set_title('Z 期望值对比')
        ax3.set_xticks(x_pos + width)
        ax3.set_xticklabels(range(max_qubits))
        ax3.legend(title='编码方式', fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5)  # 零线
        
        # 子图 4-6: 每种编码方式的概率分布（前 k 个最大概率的基态）
        for idx, result in enumerate(dim_results):
            ax = fig.add_subplot(gs[1, idx])
            
            # 只显示前 8 个概率最大的基态（或更少，如果总基态数少于 8）
            top_probs = _topk_probs(result.probs, k=min(8, len(result.probs)))
            bitstrs = [p[0] for p in top_probs]  # 基态二进制字符串（如 "000", "001"）
            probs_vals = [p[1] for p in top_probs]  # 对应的概率值
            
            # 绘制柱状图，使用不同的颜色
            bars = ax.bar(range(len(bitstrs)), probs_vals, alpha=0.7, 
                         color=plt.cm.viridis(np.linspace(0, 1, len(bitstrs))),
                         edgecolor='black', linewidth=0.5)
            ax.set_xlabel('基态 (Basis State)')
            ax.set_ylabel('概率 (Probability)')
            ax.set_title(f'{result.name}\n前 {len(bitstrs)} 个最大概率')
            ax.set_xticks(range(len(bitstrs)))
            # X 轴标签使用狄拉克符号表示（如 |000⟩, |001⟩）
            ax.set_xticklabels([f'|{b}⟩' for b in bitstrs], rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 在每个柱子上添加数值标签
            for i, (bar, prob) in enumerate(zip(bars, probs_vals)):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 子图 7-9: 概率分布完整图（所有基态的概率分布）
        for idx, result in enumerate(dim_results):
            ax = fig.add_subplot(gs[2, idx])
            
            # 获取所有基态及其概率
            n_states = len(result.probs)  # 基态总数（2^n_qubits）
            n_qubits = result.n_qubits
            
            # 生成所有基态的二进制字符串表示（如 "000", "001", "010", ...）
            bitstrs_all = [format(i, f"0{n_qubits}b") for i in range(n_states)]
            # 根据概率值设置颜色（归一化后使用颜色映射）
            colors = plt.cm.viridis(result.probs / result.probs.max())
            
            # 绘制所有基态的概率分布
            bars = ax.bar(range(n_states), result.probs, color=colors, 
                         edgecolor='black', linewidth=0.3, alpha=0.8)
            ax.set_xlabel('基态索引 (Basis State Index)')
            ax.set_ylabel('概率 (Probability)')
            ax.set_title(f'{result.name}\n完整概率分布')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 根据基态数量决定如何标记 X 轴
            if n_states <= 16:
                # 如果基态数较少，标记所有基态
                ax.set_xticks(range(n_states))
                ax.set_xticklabels([f'|{b}⟩' for b in bitstrs_all], rotation=90, ha='center', fontsize=7)
            else:
                # 如果基态数较多，只标记部分基态（每隔 step 个标记一个）
                step = max(1, n_states // 8)
                ax.set_xticks(range(0, n_states, step))
                ax.set_xticklabels([f'|{bitstrs_all[i]}⟩' for i in range(0, n_states, step)], 
                                 rotation=90, ha='center', fontsize=7)
        
        plt.suptitle(f'Encoding Comparison Visualization (d={dim_to_visualize})', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.savefig('week1_encoding_comparison_detailed.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✅ week1_encoding_comparison_detailed.png")
    
    # ========================================================================
    # 图 2: 不同维度下的编码效率对比
    # 对比三种编码方式在不同输入维度下所需的量子比特数和态空间维度
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 提取所有输入维度
    dims = [d for d, _ in all_results]
    # 存储不同编码方式的量子比特数和态空间维度
    angle_n_qubits = []
    amplitude_n_qubits = []
    basis_n_qubits = []
    angle_state_dims = []
    amplitude_state_dims = []
    basis_state_dims = []
    
    # 遍历所有维度的结果，提取每种编码方式的信息
    for dim, results in all_results:
        # 找到每种编码方式的结果
        angle_result = next((r for r in results if 'Angle' in r.name), None)
        amplitude_result = next((r for r in results if 'Amplitude' in r.name), None)
        basis_result = next((r for r in results if 'Basis' in r.name), None)
        
        if angle_result:
            angle_n_qubits.append(angle_result.n_qubits)
            angle_state_dims.append(2 ** angle_result.n_qubits)
        if amplitude_result:
            amplitude_n_qubits.append(amplitude_result.n_qubits)
            amplitude_state_dims.append(2 ** amplitude_result.n_qubits)
        if basis_result:
            basis_n_qubits.append(basis_result.n_qubits)
            basis_state_dims.append(2 ** basis_result.n_qubits)
    
    # 子图 1: 量子比特数对比
    # 显示不同输入维度下各编码方式所需的量子比特数
    axes[0, 0].plot(dims, angle_n_qubits, 'o-', label='角度编码 (AngleEmbedding)', linewidth=2, markersize=8)
    axes[0, 0].plot(dims, amplitude_n_qubits, 's-', label='幅度编码 (AmplitudeEmbedding)', linewidth=2, markersize=8)
    axes[0, 0].plot(dims, basis_n_qubits, '^-', label='基态编码 (BasisEmbedding)', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('输入维度 (Input Dimension)')
    axes[0, 0].set_ylabel('量子比特数 (Number of Qubits)')
    axes[0, 0].set_title('量子比特数 vs 输入维度')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(dims)
    
    # 子图 2: 量子态维度对比（对数尺度）
    # 量子态维度 = 2^n_qubits，使用对数尺度可以更清晰地显示指数增长
    axes[0, 1].semilogy(dims, angle_state_dims, 'o-', label='角度编码 (AngleEmbedding)', linewidth=2, markersize=8)
    axes[0, 1].semilogy(dims, amplitude_state_dims, 's-', label='幅度编码 (AmplitudeEmbedding)', linewidth=2, markersize=8)
    axes[0, 1].semilogy(dims, basis_state_dims, '^-', label='基态编码 (BasisEmbedding)', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('输入维度 (Input Dimension)')
    axes[0, 1].set_ylabel('量子态维度（对数尺度）')
    axes[0, 1].set_title('量子态维度 vs 输入维度')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, which='both')  # 显示主网格和次网格
    axes[0, 1].set_xticks(dims)
    
    # 子图 3: Z 期望值的范围对比
    # 计算每种编码方式下 Z 期望值的最大值和最小值的差（范围）
    # 范围越大，说明该编码方式对数据的区分度越好
    z_ranges_angle = []
    z_ranges_amplitude = []
    z_ranges_basis = []
    
    for dim, results in all_results:
        angle_result = next((r for r in results if 'Angle' in r.name), None)
        amplitude_result = next((r for r in results if 'Amplitude' in r.name), None)
        basis_result = next((r for r in results if 'Basis' in r.name), None)
        
        # 计算每种编码方式的 Z 期望值范围（最大值 - 最小值）
        if angle_result:
            z_ranges_angle.append(np.max(angle_result.z_exps) - np.min(angle_result.z_exps))
        if amplitude_result:
            z_ranges_amplitude.append(np.max(amplitude_result.z_exps) - np.min(amplitude_result.z_exps))
        if basis_result:
            z_ranges_basis.append(np.max(basis_result.z_exps) - np.min(basis_result.z_exps))
    
    # 绘制分组柱状图
    axes[1, 0].bar([d - 0.2 for d in dims], z_ranges_angle, width=0.2, 
                   label='角度编码 (AngleEmbedding)', alpha=0.8, color='skyblue')
    axes[1, 0].bar(dims, z_ranges_amplitude, width=0.2, 
                   label='幅度编码 (AmplitudeEmbedding)', alpha=0.8, color='lightgreen')
    axes[1, 0].bar([d + 0.2 for d in dims], z_ranges_basis, width=0.2, 
                   label='基态编码 (BasisEmbedding)', alpha=0.8, color='lightcoral')
    axes[1, 0].set_xlabel('输入维度 (Input Dimension)')
    axes[1, 0].set_ylabel('Z 期望值范围')
    axes[1, 0].set_title('Z 期望值范围对比')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_xticks(dims)
    
    # 子图 4: 概率分布的熵（信息量）
    # 香农熵衡量概率分布的信息量，熵越大说明分布越均匀，包含的信息越多
    entropies_angle = []
    entropies_amplitude = []
    entropies_basis = []
    
    # 定义熵计算函数（香农熵）
    def entropy(probs):
        """计算概率分布的香农熵"""
        probs = probs[probs > 0]  # 只考虑非零概率（避免 log(0)）
        return -np.sum(probs * np.log2(probs))
    
    for dim, results in all_results:
        angle_result = next((r for r in results if 'Angle' in r.name), None)
        amplitude_result = next((r for r in results if 'Amplitude' in r.name), None)
        basis_result = next((r for r in results if 'Basis' in r.name), None)
        
        # 计算每种编码方式的概率分布熵
        if angle_result:
            entropies_angle.append(entropy(angle_result.probs))
        if amplitude_result:
            entropies_amplitude.append(entropy(amplitude_result.probs))
        if basis_result:
            entropies_basis.append(entropy(basis_result.probs))
    
    # 绘制熵随输入维度变化的曲线
    axes[1, 1].plot(dims, entropies_angle, 'o-', label='角度编码 (AngleEmbedding)', linewidth=2, markersize=8)
    axes[1, 1].plot(dims, entropies_amplitude, 's-', label='幅度编码 (AmplitudeEmbedding)', linewidth=2, markersize=8)
    axes[1, 1].plot(dims, entropies_basis, '^-', label='基态编码 (BasisEmbedding)', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('输入维度 (Input Dimension)')
    axes[1, 1].set_ylabel('香农熵 (bits)')
    axes[1, 1].set_title('概率分布熵')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(dims)
    
    plt.suptitle('Encoding Efficiency Comparison Across Dimensions', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('week1_encoding_efficiency_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ week1_encoding_efficiency_comparison.png")
    
    # ========================================================================
    # 图 3: Z 期望值热图（所有维度和所有编码方式）
    # 使用热图显示不同输入维度和量子比特索引下的 Z 期望值
    # ========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    encoding_names = ['AngleEmbedding', 'AmplitudeEmbedding', 'BasisEmbedding']
    
    # 为每种编码方式创建热图
    for enc_idx, enc_name in enumerate(encoding_names):
        z_matrix = []
        max_qubits = 0
        
        # 收集所有维度的 Z 期望值
        for dim, results in all_results:
            # 找到对应编码方式的结果
            result = next((r for r in results if enc_name.split('(')[0].strip() in r.name), None)
            if result:
                z_vals = list(result.z_exps)
                max_qubits = max(max_qubits, len(z_vals))  # 记录最大量子比特数
                z_matrix.append(z_vals)
        
        # 填充到相同长度（用 NaN 填充较短的向量，使矩阵形状一致）
        for row in z_matrix:
            while len(row) < max_qubits:
                row.append(np.nan)
        
        z_matrix = np.array(z_matrix)
        
        # 绘制热图，使用 coolwarm 颜色映射（蓝色表示负值，红色表示正值）
        im = axes[enc_idx].imshow(z_matrix, cmap='coolwarm', aspect='auto', 
                                  vmin=-1, vmax=1, interpolation='nearest')
        axes[enc_idx].set_xlabel('量子比特索引 (Qubit Index)')
        axes[enc_idx].set_ylabel('输入维度 (Input Dimension)')
        axes[enc_idx].set_title(f'{enc_name}\nZ 期望值热图')
        axes[enc_idx].set_yticks(range(len(dims)))
        axes[enc_idx].set_yticklabels(dims)
        axes[enc_idx].set_xticks(range(max_qubits))
        axes[enc_idx].set_xticklabels(range(max_qubits))
        
        # 在每个单元格中添加数值标签
        for i in range(len(dims)):
            for j in range(max_qubits):
                if not np.isnan(z_matrix[i, j]):
                    # 根据 Z 值的大小选择文字颜色（大值时用白色，小值时用黑色）
                    axes[enc_idx].text(j, i, f'{z_matrix[i, j]:.2f}', 
                                      ha='center', va='center', 
                                      color='white' if abs(z_matrix[i, j]) > 0.5 else 'black',
                                      fontsize=8)
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[enc_idx], label='⟨Zᵢ⟩ 值')
    
    plt.suptitle('Z Expectation Values Heatmap Across Dimensions', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('week1_encoding_z_expectations_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ week1_encoding_z_expectations_heatmap.png")
    
    # ========================================================================
    # 图 4: 概率分布对比（所有维度）
    # 为每个输入维度显示三种编码方式的概率分布
    # ========================================================================
    n_dims = len(all_results)
    fig, axes = plt.subplots(n_dims, 3, figsize=(18, 5 * n_dims))
    if n_dims == 1:
        axes = axes.reshape(1, -1)  # 如果只有一个维度，需要重新调整 axes 形状
    
    # 遍历每个输入维度
    for row_idx, (dim, results) in enumerate(all_results):
        # 遍历该维度下的每种编码方式
        for col_idx, result in enumerate(results):
            ax = axes[row_idx, col_idx]
            
            # 显示前 k 个最大概率的基态（最多 16 个）
            top_probs = _topk_probs(result.probs, k=min(16, len(result.probs)))
            bitstrs = [p[0] for p in top_probs]  # 基态二进制字符串
            probs_vals = [p[1] for p in top_probs]  # 对应的概率值
            
            # 绘制柱状图
            bars = ax.bar(range(len(bitstrs)), probs_vals, alpha=0.7,
                         color=plt.cm.viridis(np.linspace(0, 1, len(bitstrs))),
                         edgecolor='black', linewidth=0.5)
            ax.set_xlabel('基态 (Basis State)')
            ax.set_ylabel('概率 (Probability)')
            title = f'{result.name}\nd={dim}, n_qubits={result.n_qubits}'
            ax.set_title(title, fontsize=10)
            ax.set_xticks(range(len(bitstrs)))
            # X 轴标签使用狄拉克符号表示
            ax.set_xticklabels([f'|{b}⟩' for b in bitstrs], rotation=45, ha='right', fontsize=7)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 在子图上添加 Z 期望值信息（显示在顶部）
            z_info = ', '.join([f'⟨Z_{i}⟩={v:.2f}' for i, v in enumerate(result.z_exps)])
            ax.text(0.5, 0.95, z_info, transform=ax.transAxes, 
                   ha='center', va='top', fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Probability Distributions Comparison Across Dimensions and Encoding Methods', 
                fontsize=16, fontweight='bold', y=0.998)
    plt.savefig('week1_encoding_probability_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ week1_encoding_probability_distributions.png")
    
    # ========================================================================
    # 图 5: 编码方式特性对比雷达图
    # 使用雷达图从多个维度（易用性、输入灵活性、态空间、信息容量、硬件效率）对比三种编码方式
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 定义评估维度
    categories = ['易用性 (Ease of Use)', '输入灵活性 (Input Flexibility)', 
                 '态空间 (State Space)', '信息容量 (Information Capacity)', 
                 '硬件效率 (Hardware Efficiency)']
    n_categories = len(categories)
    
    # 评分（1-5，5 为最好）
    # 角度编码：易用性高，灵活性好，但态空间和信息容量中等，硬件效率高
    angle_scores = [5, 4, 3, 3, 5]
    # 幅度编码：易用性中等，灵活性较差，但态空间和信息容量最高，硬件效率中等
    amplitude_scores = [3, 2, 5, 5, 3]
    # 基态编码：易用性低，灵活性最低，态空间和信息容量低，硬件效率较高
    basis_scores = [2, 1, 2, 2, 4]
    
    # 计算雷达图的角度（均匀分布在圆周上）
    angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
    angles += angles[:1]  # 闭合（添加第一个角度到末尾）
    
    # 为每个编码方式的评分也添加第一个值到末尾（用于闭合图形）
    angle_scores += angle_scores[:1]
    amplitude_scores += amplitude_scores[:1]
    basis_scores += basis_scores[:1]
    
    # 绘制三种编码方式的雷达图
    ax.plot(angles, angle_scores, 'o-', linewidth=2, label='角度编码 (AngleEmbedding)', color='skyblue')
    ax.fill(angles, angle_scores, alpha=0.25, color='skyblue')
    
    ax.plot(angles, amplitude_scores, 's-', linewidth=2, label='幅度编码 (AmplitudeEmbedding)', color='lightgreen')
    ax.fill(angles, amplitude_scores, alpha=0.25, color='lightgreen')
    
    ax.plot(angles, basis_scores, '^-', linewidth=2, label='基态编码 (BasisEmbedding)', color='lightcoral')
    ax.fill(angles, basis_scores, alpha=0.25, color='lightcoral')
    
    # 设置标签和刻度
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=9)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title('编码方式特性对比\n（雷达图）', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig('week1_encoding_characteristics_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ week1_encoding_characteristics_radar.png")
    
    print("\n所有可视化图表已生成完成！")


def main() -> None:
    """
    主函数
    
    测试不同输入维度下三种编码方式的表现，并生成可视化图表。
    """
    np.random.seed(0)  # 设置随机种子以确保可重复性

    print("Week 2 - 编码方式对比")
    print("=" * 60)
    
    all_results = []  # 存储所有结果用于可视化
    
    # 测试不同的输入维度（2, 3, 4, 5）
    for d in [2, 3, 4, 5]:
        # 生成随机输入数据（d 维）
        x = np.random.normal(size=(d,))
        print(f"\n输入维度 d={d}")
        print(f"x = {np.round(x, 4)}")

        # 对同一输入数据使用三种不同的编码方式
        results: List[EncodingResult] = [
            angle_encoding(x),              # 角度编码
            amplitude_encoding(x),          # 幅度编码
            basis_encoding_from_thresholds(x),  # 基态编码（通过阈值化）
        ]
        
        all_results.append((d, results))  # 保存该维度的所有结果

        # 打印每种编码方式的结果
        for r in results:
            print("\n" + "-" * 60)
            print(f"{r.name}")
            print(f"量子比特数 = {r.n_qubits}   态空间维度 = {2 ** r.n_qubits}")
            print(f"<Z_i> = {np.round(r.z_exps, 4)}")
            print("前 k 个最大概率的基态:")
            # 显示前 8 个最大概率的基态及其概率
            for bitstr, p in _topk_probs(r.probs, k=min(8, len(r.probs))):
                print(f"  |{bitstr}⟩ : {p:.6f}")

    print("\n完成。")
    
    # 生成所有可视化图表
    visualize_encoding_comparison(all_results)


if __name__ == "__main__":
    main()



