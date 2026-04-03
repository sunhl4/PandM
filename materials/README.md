# 量子计算化学学习工作区

## 背景与定位

本工作区为有计算化学博士背景（DFT/ReaxFF/ML/MD）的研究人员设计，系统学习量子计算在计算化学中的应用。

## 目录结构

本文件夹与同级目录 `dft_qc_pipeline/`（软件包）并列；工作区根目录另有 [`../README.md`](../README.md) 总索引。

```
LearningPlan/
├── README.md                    ← 工作区总索引（软件 + 资料并列说明）
├── dft_qc_pipeline/             ← DFT+QC 管线代码与示例 Notebook
└── learning_materials/          ← 本文件夹：学习计划与资料
    ├── README.md                ← 本文件：导航与概览
    ├── 文献补充_计算化学与量子计算.md
    ├── Phase1_Fundamentals/
    │   ├── 01_量子计算基础与概念映射.ipynb
    │   └── 02_Qiskit入门实践.ipynb
    ├── Phase2_CoreAlgorithms/
    │   ├── 01_VQE原理与实现.ipynb
    │   └── 02_Qiskit_Nature_H2_LiH.ipynb
    ├── Phase3_AdvancedMethods/
    │   ├── 01_进阶算法综述.ipynb
    │   └── 02_量子优势分析.ipynb
    └── Phase4_Applications/
        ├── 01_非均相催化量子计算.ipynb
        └── 02_量子计算辅助力场开发.ipynb
```

## 环境配置

```bash
# 创建专用conda环境
conda create -n qc_chem python=3.11
conda activate qc_chem

# 核心量子计算工具
pip install qiskit==1.3.1
pip install qiskit-aer==0.15.0
pip install qiskit-ibm-runtime==0.29.0

# 量子化学扩展
pip install qiskit-nature==0.7.2
pip install qiskit-nature-pyscf==0.4.0

# 经典量子化学后端（Linux/WSL 可直接 pip；原生 Windows 勿直接执行下一行，见下方「Windows 说明」）
pip install pyscf==2.6.2

# 量子机器学习
pip install qiskit-machine-learning==0.7.2

# 进阶插件（第三阶段）
pip install qiskit-addon-sqd==0.8.0

# 科学计算与可视化
pip install numpy scipy matplotlib pandas jupyter
```

### Windows 说明：`pyscf` 与 `qiskit-nature-pyscf`

PyPI 上的 **PySCF 通常不提供 Windows 预编译 wheel**，`pip` 会下载源码并调用 **CMake + MSVC（NMake）** 编译。若日志里出现 **`cmake -SC:\Users\...`**（少了空格，把 `-S` 和 `C:` 粘在一起），或紧跟着 **`CMake Error at CMakeLists.txt:1` / `Parse error` / `got unquoted argument with text "b"`**，说明踩中了上游 `setup.py` 的 CMake 命令写法问题；**不要反复裸跑 `pip install pyscf`**，请改用下面的 `install_pyscf_windows.py`。若修补后仍提示 **`nmake` 找不到 / `CMAKE_C_COMPILER not set`**，说明本机未安装或未在 **VS 本机工具命令行** 中使用 **Visual Studio Build Tools（含“使用 C++ 的桌面开发”）**。

**推荐**：在 **WSL2（Ubuntu）或 Linux** 下创建同一 `qc_chem` 环境再执行上述 `pip`（Linux 上常有 PySCF 的 manylinux wheel，可避免本地编译）。若必须留在原生 Windows，请先安装 VS Build Tools，再重试；conda 的 `win-64` 频道通常也 **没有** `pyscf` 现成包。

**原生 Windows 自动修补安装**：在已激活的 `qc_chem` 中进入本文件夹 `learning_materials`，执行 `python install_pyscf_windows.py`（可选：`python install_pyscf_windows.py 2.12.1` 固定版本）。脚本会下载 PySCF 源码、把错误的 `cmake -SC:\...` 改成正确的 `-S` / `-B` 参数，再调用 `pip install`；**编译阶段仍依赖本机 MSVC**。

### WSL2（Ubuntu）可重复安装 PySCF

在 WSL 里进入本目录（路径示例：`cd /mnt/d/Yaozheng/QuantumChemistry/LearningPlan/learning_materials`），执行：

```bash
chmod +x install_pyscf_wsl.sh
conda activate qc_chem
./install_pyscf_wsl.sh          # PyPI 最新版
./install_pyscf_wsl.sh 2.6.2    # 与上文 pip 列表一致的固定版本
```

脚本会用 `apt` 安装编译依赖（OpenBLAS、CMake、build-essential 等），再对**当前**解释器执行 `pip install pyscf`。可选环境变量：`SKIP_APT=1`（已装依赖）、`CREATE_VENV=1`（在本目录创建 `.venv_wsl_pyscf`）、`PYTHON=python3.11`（指定解释器）。

在 **Windows PowerShell** 里不要直接 `./install_pyscf_wsl.sh`（那是 Bash，且默认当前目录下也没有这个文件）。任选其一：

- 已打开 **WSL 终端**：`cd` 到本目录后执行上面的 `./install_pyscf_wsl.sh`。
- 留在 **PowerShell**：先 `conda activate` 在 WSL 里自己做；或在本目录执行  
  `powershell -ExecutionPolicy Bypass -File .\learning_materials\install_pyscf_wsl.ps1 2.6.2`  
  （在 `LearningPlan` 根目录执行；或先 `cd learning_materials` 再执行 `.\install_pyscf_wsl.ps1`。内部会调用 `wsl.exe`，要求本机已安装 WSL2 + Ubuntu）。

## 学习路径总览

```
阶段一（4-6周）
  ↓ RevModPhys.92.015003 第I-III章
  ↓ IBM Quantum Learning: Basics of Quantum Information
  ↓ Phase1 两个Notebook
  
阶段二（6-8周）
  ↓ acs.chemrev.8b00803 第IV-VI章
  ↓ RevModPhys 第IV-VI章
  ↓ science.abd3880 （Science Perspective）
  ↓ Phase2 两个Notebook
  
阶段三（8-12周）
  ↓ Quantum_Advantage_in_Computational_Chemistry.pdf
  ↓ 2512.13657v2.pdf（iQCC）
  ↓ 2508.02578v2.pdf（SKQD）
  ↓ Phase3 两个Notebook
  
阶段四（持续）
  ↓ Phase4 两个Notebook
  ↓ 结合自身专业发表研究成果
```

## 配套文献（本目录已有）

| 文件 | 内容 | 对应阶段 |
|------|------|----------|
| `RevModPhys.92.015003.pdf` | 量子计算化学权威综述（RMP） | 阶段一+二 |
| `acs.chemrev.8b00803.pdf` | 量子计算时代的量子化学（Chem.Rev.） | 阶段一+二 |
| `science.abd3880.pdf` | Google VQE实验的Science视角 | 阶段二 |
| `Quantum_Advantage_in_Computational_Chemistry.pdf` | 量子优势时间线分析 | 阶段三 |
| `2512.13657v2.pdf` | iQCC工业应用（OLED分子） | 阶段三 |
| `2508.02578v2.pdf` | SKQD最新算法（可证明收敛） | 阶段三 |

**索引（表外补充）**：多参考/活性空间、ML 力场（Behler–Parrinello、NequIP、MACE 等）、催化量子计算、QPE/VQE 经典论文与 PySCF 文献入口，见同目录 **[文献补充_计算化学与量子计算.md](文献补充_计算化学与量子计算.md)**（与上表互补，不重复已有 PDF）。

## 2508.02578v2（SKQD / SqDRIFT）精读与实现路径

下列内容与 [Phase3_AdvancedMethods/01_进阶算法综述.ipynb](Phase3_AdvancedMethods/01_进阶算法综述.ipynb) 中 **§3 与新增「3.1 精读与实现路径」** 一致，便于脱离 Notebook 查阅。

**arXiv（与本地 `2508.02578v2.pdf` 对应）**：https://arxiv.org/abs/2508.02578

### 论文建议阅读顺序

1. **摘要 + 引言**：在 NISQ 上求基态的动机；与 VQE、纯采样对角化相比多出来的东西是什么。  
2. **预备知识与 Krylov 思路**：为何用酉演化 $e^{-iHt}$ 生成 Krylov 型子空间，而不是直接在哈密顿量 $H$ 上重复作用。  
3. **SKQD 主流程**：参考态、时间步长 $\Delta t$、Krylov 维数 $r$、**量子采样**如何进入子空间构造、经典侧如何投影并对角化。  
4. **SqDRIFT**：用 **qDRIFT**（随机哈密顿量模拟）实现/近似时间演化；与 Trotter 相比的电路深度与误差界。  
5. **主要定理**：可证明收敛的条件（与 QPE / 稀疏性等假设的对应）。  
6. **数值实验**：多环芳烃（PAH）等——与有机/共轭体系、方法尺度感相关。

### 与本计划其他部分的衔接

| 已有内容 | 与 SKQD 的关系 |
|----------|----------------|
| Phase2：Qiskit Nature / PySCF | 提供**同一套**分子泡利哈密顿量，作为 SKQD 输入 |
| Phase3 §2 SQD | 强调**采样 + 小子空间经典对角化**；SKQD 再叠 **时间演化 + Krylov** |
| Phase3 §3 下一格代码 | **KQD**：无采样、无 qDRIFT 的精确小矩阵演示，只用于**收敛直觉**，不是论文完整 SKQD |

### 代码与工具路径（由易到难）

| 步骤 | 目标 | 工具 / 入口 |
|:---:|:---|:---|
| **A** | 在态矢量上精确做 $e^{-iHt}$，观察 Krylov 子空间能量随维数 $m$ 收敛 | 同上 Notebook **KQD 代码格**；或 `Statevector` + `scipy.linalg.expm` |
| **B** | 把 $e^{-iHt}$ 变成**量子线路**（Trotter / 高阶积公式等） | Qiskit 文档中 Hamiltonian simulation / `TimeEvolutionProblem` 等（随版本检索） |
| **C** | **采样型**子空间 + IBM 生态中与 SKQD 命名的教程 | **IBM Quantum Learning**：https://learning.quantum.ibm.com/（站内搜 SKQD）；Qiskit 文档 https://docs.quantum.ibm.com/（站内搜 *Sample-based Krylov*） |
| **D** | **SQD**（与 SKQD 相近但不同） | `pip install qiskit-addon-sqd` + 官方示例；写清与 SKQD 差在哪一步 |
| **E** | **SqDRIFT** 实现级细节 | 以 **2508.02578v2** 正文算法与附录为主；若作者公开代码则对照实现 |

### 建议完成的自检清单

- [ ] 在 PDF 上标出：**定理假设** ↔ 你熟悉的 Davidson/Lanczos 或 FCI 子空间条件。  
- [ ] 用 H₂（或 4 比特模型）说明：**采样子空间维数 $K$** 与能量误差的定性关系。  
- [ ] 半页笔记：**SQD vs SKQD**（哪一步引入 Krylov / 时间演化 / qDRIFT）。

## 关键外部资源

- IBM Quantum Learning: https://learning.quantum.ibm.com/
- Qiskit Nature 文档: https://qiskit-community.github.io/qiskit-nature/
- IBM Quantum 平台（免费设备访问）: https://quantum.ibm.com/
- InQuanto（Quantinuum工业软件）: https://docs.quantinuum.com/inquanto/
- arXiv 量子计算化学跟踪: https://arxiv.org/list/quant-ph/recent
