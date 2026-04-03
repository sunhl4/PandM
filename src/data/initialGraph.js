/**
 * Initial graph data seeded from d:\Yaozheng\QuantumChemistry workspace.
 *
 * Node schema:
 *   id          – unique string
 *   label       – display label (Chinese OK)
 *   type        – 'root' | 'module' | 'category' | 'topic' | 'leaf'
 *   color       – optional override (uses type default otherwise)
 *   description – one-line summary
 *   content     – markdown string shown in NodePanel
 *   tags        – string[]
 *   links       – { label, url }[]
 *
 * Edge schema:
 *   source / target – node ids
 */

export const INITIAL_GRAPH = {
  nodes: [
    /* ─── ROOT ─────────────────────────────────────────────── */
    {
      id: 'root',
      label: '知识宇宙',
      type: 'root',
      description: '工作计划与学习资料的量子知识图谱',
      content: `# 知识宇宙

这是你的个人知识管理系统。

两大核心模块：
- **工作计划** — 研究路线、技术主线、近期任务
- **学习资料** — 经典方法与量子计算的交叉学科知识库

点击任意节点展开分支，双击聚焦子图。`,
      tags: ['系统', '导航'],
      links: [],
    },

    /* ─── MODULE 1: 工作计划 ───────────────────────────────── */
    {
      id: 'work-plan',
      label: '工作计划',
      type: 'module',
      description: '研究路线图与技术攻关计划',
      content: `# 工作计划

> 统一逻辑：**基础映射 → 少比特分子算法 → 量子优势判断 → SQD/量子数据路线 → 催化与力场应用**

## 四条核心主线

| 主线 | 核心问题 |
|------|---------|
| 少比特应用线 | 比特不多时量子计算还有没有价值？|
| 量子优势线 | 何时、何种体系、何种资源下优于经典？|
| 化学应用线 | 强关联、过渡金属活性中心的量子解法 |
| 量子数据线 | 比特串分布作为"量子数据"与AI协作 |

## 六个核心文件

- \`README.md\` — 全局结构与学习路线
- \`Phase2/02_Qiskit_Nature_H2_LiH.ipynb\` — 分子哈密顿量与少比特化学
- \`Phase3/02_量子优势分析.ipynb\` — 量子优势的多维判断
- \`SQD.md\` — 量子数据主线
- \`Phase3/01_进阶算法综述.ipynb\` — ADAPT-VQE/SQD/SKQD关系
- \`Phase4\` — 催化与力场两大应用故事`,
      tags: ['规划', '路线图'],
      links: [],
    },
    {
      id: 'wp-fewqubit',
      label: '少比特应用线',
      type: 'category',
      description: '验证算法、强关联局域问题、active space工作流',
      content: `# 少比特应用线

**核心问题**: 比特不多的时候，量子计算还有没有价值？

## 价值所在

- 验证算法正确性
- 刻画强关联局域问题
- 形成 active space 工作流
- 构建量子数据和建立混合量子经典管线

## 典型体系

| 体系 | 规模 | 意义 |
|------|------|------|
| H₂ | 2 qubits | 最小化学基准 |
| LiH | 4 qubits | 键解离验证 |
| 2-site Hubbard | 2 qubits | 强关联模型 |
| Fe-N4 toy model | 4 qubits | 催化活性中心 |

少比特阶段的真正价值是**找准未来会放大的方法和数据路线**。`,
      tags: ['NISQ', '少比特', 'active space'],
      links: [
        { label: 'Phase2: H2/LiH notebook', url: 'https://nbviewer.org/github/sunhl4/PandM/blob/main/materials/Phase2_CoreAlgorithms/02_Qiskit_Nature_H2_LiH.ipynb' },
      ],
    },
    {
      id: 'wp-advantage',
      label: '量子优势线',
      type: 'category',
      description: '精度、体系、资源、经典baseline综合判断',
      content: `# 量子优势线

量子优势不是"是否比经典快"这么单一，而是**在什么精度、什么体系、什么资源预算、什么经典baseline下更值得**。

## 近期机会 (NISQ)

- Shallow circuit 算法
- 采样型方法 (SQD)
- 量子嵌入 (DMET + VQE)
- 量子数据与 QML

## 长期优势 (FT)

- QPE 对 FCI 级问题多项式复杂度优势
- 强关联大体系、FCI 级精度
- 需要大量物理比特与纠错开销

## 关键参考

- 量子优势分析 notebook（Phase3）
- Towards Quantum Advantage in Chemistry（文献翻译）`,
      tags: ['量子优势', 'NISQ', '容错'],
      links: [
        { label: '量子优势分析 notebook', url: 'https://nbviewer.org/github/sunhl4/PandM/blob/main/materials/Phase3_AdvancedMethods/02_量子优势分析.ipynb' },
      ],
    },
    {
      id: 'wp-chemistry',
      label: '化学应用线',
      type: 'category',
      description: '强关联体系与工业催化、力场开发',
      content: `# 化学应用线

化学天然是量子多体问题，**问题本身与量子计算语言高度同构**。

## 关键特征体系

- 强关联体系、近简并态
- 自旋态竞争
- 断键成键过程
- 过渡金属活性中心（TM oxide, Fe-N4, Co-N4）

## 两个应用故事

### 非均相催化
\`Fe-N4 / Co-N4 / TM oxide / active space / embedding\`

### 力场开发
\`PES数据 → ReaxFF / NNP / QML力场\`

经典近似方法（DFT、CCSD(T)）在这些体系最脆弱，而工业最在意。`,
      tags: ['催化', '力场', '强关联', 'DMET'],
      links: [
        { label: '非均相催化 notebook', url: 'https://nbviewer.org/github/sunhl4/PandM/blob/main/materials/Phase4_Applications/01_非均相催化量子计算.ipynb' },
        { label: '力场开发 notebook', url: 'https://nbviewer.org/github/sunhl4/PandM/blob/main/materials/Phase4_Applications/02_量子计算辅助力场开发.ipynb' },
      ],
    },
    {
      id: 'wp-qdata',
      label: '量子数据线',
      type: 'category',
      description: 'SQD比特串分布作为量子数据与AI协作',
      content: `# 量子数据线

**核心问题**: 公司为什么会关心 SQD 和量子测量数据？

## SQD 工作原理

\`bitstring → 合法组态 → 投影子空间 → 经典对角化\`

量子设备产出的不是单纯一个能量数字，而是：
- 比特串分布
- 组态样本
- 时间演化后的采样快照

## 量子数据的价值

这些样本不是噪声垃圾，而是能被**恢复、筛选、投影、特征化**的"量子数据"。

最容易与经典 AI 形成**协作**，而不是简单替代经典 AI。

## 参考材料

- \`SQD.md\` — 核心原理
- \`ADAPT-VQE.md\` — 自适应 ansatz
- \`sqd_nat_chem_2024\` — 复现代码`,
      tags: ['SQD', '量子数据', 'QML', 'SKQD'],
      links: [
        { label: 'SQD.md', url: 'https://github.com/sunhl4/PandM/blob/main/materials/SQD.md' },
      ],
    },
    {
      id: 'wp-pipeline',
      label: 'DFT-QC流水线',
      type: 'topic',
      description: 'dft_qc_pipeline工程实现',
      content: `# DFT-QC Pipeline

**工程**: \`dft_qc_pipeline\`

## 架构

\`\`\`
core/          # pipeline.py, config.py
embedding/     # dmet.py — 量子嵌入
hamiltonian/   # builder.py — 哈密顿量构建
postprocessing/ # ml_export.py, inter_fragment_estimate.py
\`\`\`

## 典型工作流

1. DFT 计算全体系（PySCF）
2. DMET 分割活性空间
3. 构建活性空间哈密顿量
4. VQE/SQD 求解
5. 结果导出

## 示例 Notebooks

- \`01_H2_minimal.ipynb\`
- \`02_N2_multisolver.ipynb\`
- \`03_FeN4_DMET_SQD.ipynb\``,
      tags: ['DMET', 'DFT', 'VQE', 'pipeline'],
      links: [],
    },
    {
      id: 'wp-bench',
      label: '基准测试平台',
      type: 'topic',
      description: 'quantum_chem_bench: HF–FCI vs VQE/QPE/SQD',
      content: `# Quantum Chemistry Benchmark

**工程**: \`quantum_chem_bench\`

## 支持的方法

| 类型 | 方法 |
|------|------|
| 经典 | HF, CCSD, FCI |
| 量子 | VQE, QPE, SQD, ADAPT-VQE, QSE |

## 测试体系

- H₂ (STO-3G)
- LiH (STO-3G)
- N₂ (6-31G) — 解离曲线

## YAML 配置驱动

\`\`\`yaml
molecule: h2
basis: sto-3g
methods: [hf, vqe, sqd]
\`\`\``,
      tags: ['benchmark', 'VQE', 'QPE', 'SQD'],
      links: [],
    },

    /* ─── MODULE 2: 学习资料 ───────────────────────────────── */
    {
      id: 'learning',
      label: '学习资料',
      type: 'module',
      description: '经典方法与量子计算交叉学科知识库',
      content: `# 学习资料

系统化的学习体系，覆盖两大方向：

## 经典方法
- **计算化学** — DFT, CCSD(T), 分子轨道理论
- **分子动力学** — 力场, AIMD, 增强采样
- **机器学习** — NNP, 图神经网络, 迁移学习

## 量子计算（交叉方向）
- **QC × 计算化学** — VQE, QPE, 量子嵌入
- **QC × 分子动力学** — 量子增强采样
- **QC × 机器学习** — QML, 量子核方法`,
      tags: ['学习', '知识库'],
      links: [],
    },

    /* ─── 经典方法 ──────────────────────────────────────────── */
    {
      id: 'classical',
      label: '经典方法',
      type: 'category',
      description: '计算化学、分子动力学、机器学习',
      content: `# 经典方法

量子计算的基础与对照组。理解经典方法的能力边界，才能找准量子计算的切入点。`,
      tags: ['经典', '基础'],
      links: [],
    },
    {
      id: 'comp-chem',
      label: '计算化学',
      type: 'category',
      description: 'DFT, HF, CCSD, PySCF, 分子轨道理论',
      content: `# 计算化学

## 核心理论层次

\`\`\`
HF → MP2 → CCSD → CCSD(T) → FCI
DFT (LDA → GGA → Hybrid → meta-GGA)
\`\`\`

## 关键概念

- **Born-Oppenheimer 近似** — 电子与核运动分离
- **基组** (STO-3G, 6-31G, cc-pVDZ)
- **活性空间** (CASSCF, CASPT2, NEVPT2)
- **嵌入方法** (DMET, DFT/WF)

## 工具

- **PySCF** — Python从头算框架
- **Psi4** — 高级量化软件
- **ORCA** — 多参考方法

## 学习材料

Phase1 & Phase2 notebooks 涵盖 HF、DFT 基础到量子计算接口。`,
      tags: ['DFT', 'HF', 'CCSD', 'PySCF', 'active space'],
      links: [
        { label: 'Phase1: 量子计算基础与概念映射', url: 'https://nbviewer.org/github/sunhl4/PandM/blob/main/materials/Phase1_Fundamentals/01_量子计算基础与概念映射.ipynb' },
        { label: 'Phase2: VQE原理与实现', url: 'https://nbviewer.org/github/sunhl4/PandM/blob/main/materials/Phase2_CoreAlgorithms/01_VQE原理与实现.ipynb' },
      ],
    },
    {
      id: 'md',
      label: '分子动力学',
      type: 'category',
      description: 'MD模拟、力场、AIMD、增强采样',
      content: `# 分子动力学

## 方法谱系

- **经典 MD** — 力场驱动（AMBER, CHARMM, ReaxFF）
- **AIMD** — DFT/QM 力驱动（Born-Oppenheimer MD）
- **增强采样** — metadynamics, umbrella sampling, REMD

## 力场类型

| 类型 | 代表 | 特点 |
|------|------|------|
| 分析力场 | AMBER, CHARMM | 快速，有限精度 |
| 反应力场 | ReaxFF | 处理化学反应 |
| 神经网络力场 | DeepMD, NequIP | 高精度+效率 |

## 与量子计算的连接

- 量子精度 PES 数据 → 训练 NNP
- 量子增强采样
- 量子嵌入 QM/MM`,
      tags: ['MD', 'ReaxFF', 'NNP', 'AIMD', 'sampling'],
      links: [],
    },
    {
      id: 'ml-classical',
      label: '机器学习',
      type: 'category',
      description: 'NNP、图神经网络、主动学习、迁移学习',
      content: `# 机器学习（计算化学方向）

## 主要应用

- **神经网络势函数 (NNP)** — DeepMD-kit, NequIP, MACE
- **分子性质预测** — HOMO/LUMO, 溶剂化能
- **图神经网络 (GNN)** — SchNet, DimeNet, PaiNN
- **主动学习** — 不确定性采样构建训练集

## 输入表示

- ACSF / SOAP 描述符
- 等变图神经网络 (E(3)-equivariant)
- 原子坐标 + 元素类型

## 量子精度数据生成

\`\`\`
CCSD(T) / VQE → PES points → NNP training
\`\`\`

这是量子计算近期最清晰的工业价值路径之一。`,
      tags: ['NNP', 'GNN', 'DeepMD', '主动学习'],
      links: [],
    },

    /* ─── 量子计算 ──────────────────────────────────────────── */
    {
      id: 'quantum-comp',
      label: '量子计算',
      type: 'category',
      description: '量子计算与经典方法的三条交叉路线',
      content: `# 量子计算（交叉方向）

## 三条交叉路线

量子计算与经典计算化学、分子动力学、机器学习的三大交叉：

1. **QC × 计算化学** — 直接求解电子结构问题
2. **QC × 分子动力学** — 量子增强采样与量子力场
3. **QC × 机器学习** — 量子核方法与量子模型

## 当前阶段

NISQ 时代：50–1000 个含噪比特
- 浅电路变分算法（VQE, ADAPT-VQE）
- 采样型方法（SQD）
- 量子-经典混合工作流`,
      tags: ['量子计算', 'NISQ', '交叉方向'],
      links: [],
    },
    {
      id: 'qc-chem',
      label: 'QC × 计算化学',
      type: 'category',
      description: 'VQE, QPE, ADAPT-VQE, SQD, DMET嵌入',
      content: `# 量子计算 × 计算化学

**核心目标**: 用量子计算机直接求解强关联电子结构问题。

## 算法谱系

### NISQ 算法
- **VQE** (Variational Quantum Eigensolver) — 变分本征值求解
- **ADAPT-VQE** — 自适应 ansatz 构建
- **SQD** (Sample-based Quantum Diagonalization) — 采样量子对角化
- **SKQD** — 对称约束版本

### 容错算法
- **QPE** (Quantum Phase Estimation) — 精确相位估计

## 量子嵌入

\`DFT → DMET → 活性空间哈密顿量 → VQE/SQD\`

## 学习路线（4阶段）

| 阶段 | 内容 | Notebook |
|------|------|---------|
| Phase 1 | 量子比特、门、测量 | 基础概念映射 |
| Phase 2 | VQE 实现, H₂/LiH | 核心算法 |
| Phase 3 | ADAPT-VQE, SQD, 量子优势 | 进阶方法 |
| Phase 4 | 催化、力场应用 | 应用落地 |`,
      tags: ['VQE', 'ADAPT-VQE', 'SQD', 'QPE', 'DMET'],
      links: [
        { label: 'Phase1: 量子计算基础', url: 'https://nbviewer.org/github/sunhl4/PandM/blob/main/materials/Phase1_Fundamentals/01_量子计算基础与概念映射.ipynb' },
        { label: 'Phase2: VQE原理与实现', url: 'https://nbviewer.org/github/sunhl4/PandM/blob/main/materials/Phase2_CoreAlgorithms/01_VQE原理与实现.ipynb' },
        { label: 'Phase3: 进阶算法综述', url: 'https://nbviewer.org/github/sunhl4/PandM/blob/main/materials/Phase3_AdvancedMethods/01_进阶算法综述.ipynb' },
      ],
    },
    {
      id: 'qc-vqe',
      label: 'VQE 原理与实现',
      type: 'topic',
      description: 'Variational Quantum Eigensolver — 变分量子本征值求解',
      content: `# VQE — Variational Quantum Eigensolver

## 核心思路

利用变分原理：$\\langle \\psi(\\theta) | H | \\psi(\\theta) \\rangle \\geq E_0$

通过优化参数 $\\theta$ 最小化期望值来近似基态能量。

## 算法流程

\`\`\`
初始参数 θ
  ↓
准备量子态 |ψ(θ)⟩ （ansatz 电路）
  ↓
测量 Pauli 字符串 ⟨H⟩
  ↓
经典优化器更新 θ (COBYLA, BFGS, Adam)
  ↓
收敛 → 得到近似基态能量
\`\`\`

## Ansatz 类型

- **UCCSD** — 化学启发，参数少但深度大
- **HEA** — 硬件高效，浅但表达力有限
- **ADAPT** — 自适应，按需添加算子

## Qiskit 实现

Phase2 Notebook: \`01_VQE原理与实现.ipynb\``,
      tags: ['VQE', 'ansatz', 'UCCSD', 'Qiskit'],
      links: [
        { label: 'VQE原理与实现.ipynb', url: 'https://nbviewer.org/github/sunhl4/PandM/blob/main/materials/Phase2_CoreAlgorithms/01_VQE原理与实现.ipynb' },
      ],
    },
    {
      id: 'qc-sqd',
      label: 'SQD / SKQD',
      type: 'topic',
      description: 'Sample-based Quantum Diagonalization',
      content: `# SQD — Sample-based Quantum Diagonalization

## 核心思路

\`比特串 → 合法组态 → 投影子空间 → 经典对角化\`

## 工作流程

1. 量子电路产生比特串分布
2. 筛选满足粒子数/自旋对称性的比特串
3. 投影到这些组态张成的子空间
4. 在子空间内做经典稀疏对角化（Lanczos）

## 优势

- 对硬件噪声更鲁棒（比 VQE）
- 输出"量子数据"可用于 AI/ML
- 可用 SKQD 引入更多对称性约束

## 参考

- \`SQD.md\` 技术文档
- Nature Chemistry 2024 SQD 论文复现`,
      tags: ['SQD', 'SKQD', '量子数据', '对角化'],
      links: [
        { label: 'SQD.md', url: 'https://github.com/sunhl4/PandM/blob/main/materials/SQD.md' },
      ],
    },
    {
      id: 'qc-adaptvqe',
      label: 'ADAPT-VQE',
      type: 'topic',
      description: '自适应构建ansatz，最小参数数量',
      content: `# ADAPT-VQE

自适应变分量子本征值求解器。

## 核心思路

不固定 ansatz 结构，而是**贪心地从算子库中选择梯度最大的算子**逐步添加。

## 算子库

- 费米子自旋算子池 (FGSD)
- Pauli 字符串算子池

## 迭代流程

\`\`\`
while not converged:
    计算所有算子的梯度 |∂E/∂θ|
    选择梯度最大的算子 Aₖ
    将 exp(iθₖAₖ) 添加到 ansatz
    优化所有参数
\`\`\`

## 优势

- 参数数量远少于 UCCSD
- 理论上可达 FCI 精度
- 适应性强

## 参考

\`ADAPT-VQE.md\` 详细推导`,
      tags: ['ADAPT-VQE', 'ansatz', '变分'],
      links: [
        { label: 'ADAPT-VQE.md', url: 'https://github.com/sunhl4/PandM/blob/main/materials/ADAPT-VQE.md' },
      ],
    },
    {
      id: 'qc-md',
      label: 'QC × 分子动力学',
      type: 'category',
      description: '量子增强采样、量子力场数据生成',
      content: `# 量子计算 × 分子动力学

## 主要方向

### 1. 量子精度 PES 数据生成
用量子计算机计算高精度势能面，训练神经网络力场：
\`VQE/SQD PES → NNP训练集 → MD模拟\`

### 2. 量子增强采样
- 量子退火 (QA) 辅助构象搜索
- 量子 Monte Carlo
- 量子随机游走

### 3. QM/MM with quantum solver
活性区用量子计算机，环境用 MM 力场。

## 现状与展望

近期最可行路径：**量子计算生成高质量训练数据** → 经典 NNP 用于 MD。

直接量子增强 MD 需要等容错量子计算机。`,
      tags: ['MD', 'QM/MM', '量子采样', 'NNP'],
      links: [],
    },
    {
      id: 'qc-ml',
      label: 'QC × 机器学习',
      type: 'category',
      description: 'QML、量子核方法、变分量子分类器',
      content: `# 量子计算 × 机器学习

## 主要方向

### 量子核方法
- 量子特征映射 $\\phi(x) \\to |\\phi(x)\\rangle$
- 量子核矩阵 $k(x,x') = |\\langle\\phi(x)|\\phi(x')\\rangle|^2$
- 量子 SVM

### 变分量子分类器 (VQC)
- 数据编码层 + 变分层
- 参数通过梯度优化

### 量子生成模型
- QGAN (量子生成对抗网络)
- 量子玻尔兹曼机

### 量子数据 → 经典ML
- SQD 比特串分布作为特征
- 量子涨落特征

## 参考框架

- **PennyLane** — 量子机器学习
- **Qiskit Machine Learning**
- \`quantum_chem_bench\` — 基准测试`,
      tags: ['QML', '量子核', 'VQC', 'PennyLane'],
      links: [],
    },

    /* ─── Phase1 基础 ─────────────────────────────────────── */
    {
      id: 'phase1',
      label: 'Phase1: 基础',
      type: 'topic',
      description: '量子计算基础概念、Qiskit入门',
      content: `# Phase 1 — 量子计算基础

## 学习目标

建立量子计算与量子化学的概念映射，掌握 Qiskit 基础操作。

## 内容覆盖

### 量子比特与量子门
- 量子态、叠加态、纠缠
- 基本量子门 (H, X, CNOT, Rz)
- 量子电路构建

### 量子化学概念映射
- 哈密顿量 → 量子算符
- 波函数 → 量子态
- 期望值 → 测量

## Notebooks

1. \`01_量子计算基础与概念映射.ipynb\`
2. \`02_Qiskit入门实践.ipynb\``,
      tags: ['基础', 'Qiskit', '量子比特', 'Phase1'],
      links: [
        { label: '量子计算基础与概念映射', url: 'https://nbviewer.org/github/sunhl4/PandM/blob/main/materials/Phase1_Fundamentals/01_量子计算基础与概念映射.ipynb' },
        { label: 'Qiskit入门实践', url: 'https://nbviewer.org/github/sunhl4/PandM/blob/main/materials/Phase1_Fundamentals/02_Qiskit入门实践.ipynb' },
      ],
    },

    /* ─── 文献资料 ────────────────────────────────────────── */
    {
      id: 'literature',
      label: '文献资料',
      type: 'category',
      description: '翻译文献与精读笔记',
      content: `# 文献资料

## 已翻译文献

| 文献 | 类型 | 状态 |
|------|------|------|
| RevModPhys 92, 015003 | 综述 | ✅ 翻译完成 |
| Quantum Chemistry in the Age of QC | 综述 | ✅ 翻译完成 |
| Towards Quantum Advantage in Chemistry | 展望 | ✅ 翻译完成 |

## 精读问答

每篇文献配套 QA 文件，按统一模板记录：
- 核心贡献
- 方法解析
- 关键公式
- 与工作的关联

## 问答记录

\`学习问答记录.md\` — 累计 1500+ 行交互式学习记录`,
      tags: ['文献', '翻译', '精读', 'QA'],
      links: [],
    },
  ],

  /* ─── EDGES ─────────────────────────────────────────────── */
  edges: [
    { source: 'root', target: 'work-plan' },
    { source: 'root', target: 'learning' },

    // 工作计划分支
    { source: 'work-plan', target: 'wp-fewqubit' },
    { source: 'work-plan', target: 'wp-advantage' },
    { source: 'work-plan', target: 'wp-chemistry' },
    { source: 'work-plan', target: 'wp-qdata' },
    { source: 'work-plan', target: 'wp-pipeline' },
    { source: 'work-plan', target: 'wp-bench' },

    // 学习资料分支
    { source: 'learning', target: 'classical' },
    { source: 'learning', target: 'quantum-comp' },
    { source: 'learning', target: 'literature' },

    // 经典方法
    { source: 'classical', target: 'comp-chem' },
    { source: 'classical', target: 'md' },
    { source: 'classical', target: 'ml-classical' },

    // 量子计算
    { source: 'quantum-comp', target: 'qc-chem' },
    { source: 'quantum-comp', target: 'qc-md' },
    { source: 'quantum-comp', target: 'qc-ml' },

    // QC×计算化学 子节点
    { source: 'qc-chem', target: 'phase1' },
    { source: 'qc-chem', target: 'qc-vqe' },
    { source: 'qc-chem', target: 'qc-sqd' },
    { source: 'qc-chem', target: 'qc-adaptvqe' },

    // 跨模块连接（虚线）
    { source: 'wp-qdata', target: 'qc-sqd', type: 'cross' },
    { source: 'wp-chemistry', target: 'qc-chem', type: 'cross' },
    { source: 'wp-pipeline', target: 'qc-chem', type: 'cross' },
    { source: 'ml-classical', target: 'qc-ml', type: 'cross' },
    { source: 'md', target: 'qc-md', type: 'cross' },
  ],
}

export const NODE_TYPE_CONFIG = {
  root: { color: '#00d4ff', radius: 36, glowColor: 'rgba(0,212,255,0.6)' },
  module: { color: '#8b5cf6', radius: 28, glowColor: 'rgba(139,92,246,0.5)' },
  category: { color: '#10b981', radius: 22, glowColor: 'rgba(16,185,129,0.4)' },
  topic: { color: '#f97316', radius: 16, glowColor: 'rgba(249,115,22,0.35)' },
  leaf: { color: '#f472b6', radius: 12, glowColor: 'rgba(244,114,182,0.3)' },
}
