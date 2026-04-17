# PandM · Materials 学习工作区

> 三大主目录：**学习资料** / **工作计划** / **软件工程**

---

## 目录结构

```
materials/
├── 学习资料/                          # 六个学习方向
│   ├── 经典计算化学/                  # DFT, CCSD, 分子动力学, PySCF
│   ├── 量子计算/                      # 量子比特, 量子门, Qiskit
│   │   └── Phase1_Fundamentals/
│   │       ├── 01_量子计算基础与概念映射.ipynb
│   │       └── 02_Qiskit入门实践.ipynb
│   ├── 机器学习/                      # NNP, GNN, 主动学习
│   ├── 量子计算×计算化学/             # 路径：learning/quantum-chem/（量子计算相关计算化学学习资料）
│   │   ├── learning-ms/               # VQE、进阶算法、优势分析等 Notebook + 第4周书面稿
│   │   ├── QC_learn_unified/          # QC-learn + learning-ms 书面稿单文件合订（见 README_unified.md）
│   │   └── literature/                # 译文、PDF、SQD/ADAPT 笔记
│   │       ├── 学习问答记录.md        # 含：工作流标准、文献补充索引、附录模板、问答条目
│   │       ├── 学习问答记录.html      # Pandoc + MathJax（本目录 python regen_qa_html.py）
│   │       └── regen_qa_html.py
│   ├── 量子计算×机器学习/             # QML, 量子核方法, VQC
│   ├── 量子计算×分子动力学/           # 量子增强采样, 量子力场
│
├── 工作计划/                          # 两个工作方向
│   ├── 量子计算×计算化学/
│   │   └── 工作计划.md               # 完整调研报告（10部分）
│   └── 量子计算×机器学习/
│
├── 软件工程/                          # 三个工程方向
│   ├── 量子计算×计算化学/
│   │   ├── Phase4_Applications/
│   │   │   ├── 01_非均相催化量子计算.ipynb
│   │   │   └── 02_量子计算辅助力场开发.ipynb
│   │   ├── install_pyscf_windows.py
│   │   ├── install_pyscf_wsl.ps1
│   │   ├── install_pyscf_wsl.sh
│   │   └── regen_qa_html.py
│   ├── 量子计算×机器学习/
│   └── 量子计算×分子动力学/
│
└── README.md                          # 本文件
```

---

## 三大模块说明

### 1. 学习资料

系统化知识库，覆盖六个方向：

| 分支 | 核心内容 |
|------|---------|
| 经典计算化学 | HF, DFT, CCSD, 分子动力学, PySCF/ORCA |
| 量子计算 | 量子比特, 量子门, Qiskit, NISQ现状 |
| 机器学习 | NNP (DeepMD/MACE), GNN, 主动学习 |
| QC × 计算化学 | VQE, SQD/SKQD, ADAPT-VQE, DMET嵌入 |
| QC × 机器学习 | QML, 量子核, VQC, PennyLane |
| QC × 分子动力学 | 量子精度PES, QM/MM, 量子增强采样 |

### 2. 工作计划

研究路线图与技术攻关规划：

| 方向 | 核心主线 |
|------|---------|
| QC × 计算化学 | 少比特应用线、量子优势线、化学应用线（催化活性位） |
| QC × 机器学习 | 量子数据线（SQD→AI）、QML力场方向 |

> 完整调研报告（含PPT大纲、里程碑、问答题库）：`工作计划/量子计算×计算化学/工作计划.md`

### 3. 软件工程

工程实现与工具链：

| 方向 | 工程内容 |
|------|---------|
| QC × 计算化学 | DFT-QC Pipeline, 量子化学基准平台, Phase4应用Notebooks |
| QC × 机器学习 | QML工具链, 量子数据处理流水线 |
| QC × 分子动力学 | NNP训练增强, QM/MM接口, PES数据生成 |

---

## 学习路径建议

```
经典计算化学 (理论基础)
    ↓
量子计算 (Phase1: 基础)
    ↓
QC×计算化学 (Phase2: VQE → Phase3: SQD/ADAPT-VQE)
    ↓                    ↓
工作计划/QC×计算化学    软件工程/QC×计算化学
(研究规划)              (Phase4: 催化/力场应用)
    ↓
QC×机器学习 → 工作计划/QC×ML
    ↓
QC×分子动力学 → 软件工程/QC×MD
```

---

## 环境搭建

### PySCF 安装

**Windows**：
```bash
python 软件工程/量子计算×计算化学/install_pyscf_windows.py
```

**WSL**：
```bash
bash 软件工程/量子计算×计算化学/install_pyscf_wsl.sh
```

### Qiskit 环境

```bash
pip install qiskit qiskit-nature qiskit-aer
pip install qiskit-addon-sqd       # SQD方法
pip install pennylane               # QML方向
```

---

## 外部资源

| 资源 | 链接 |
|------|------|
| IBM Quantum | https://quantum.ibm.com |
| Qiskit Nature | https://qiskit-community.github.io/qiskit-nature |
| InQuanto | https://docs.quantinuum.com/inquanto |
| PySCF | https://pyscf.org |
| PennyLane | https://pennylane.ai |
| DeepMD-kit | https://github.com/deepmodeling/deepmd-kit |
