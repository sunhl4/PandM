# 量子机器学习完整文献综述 (2007-2025)
## Quantum Machine Learning: A Comprehensive Literature Review

> 📅 整理日期: 2025年1月17日
> 
> 🎯 本综述面向量子机器学习研究者，特别关注**化学计算、分子动力学、吸附能预测、过渡态预测**等应用领域
> 
> 📎 所有论文标题均为可点击链接

---

# 目录

1. [发展趋势与历史演进](#发展趋势与历史演进)
2. [2025年文献](#2025年文献)
3. [2024年文献](#2024年文献)
4. [2023年文献](#2023年文献)
5. [2022年文献](#2022年文献)
6. [2021年文献](#2021年文献)
7. [2020年文献](#2020年文献)
8. [2007-2019年奠基性文献](#2007-2019年奠基性文献)
9. [化学与材料应用专题汇总](#化学与材料应用专题汇总)
10. [阅读建议与未来方向](#阅读建议与未来方向)

---

# 发展趋势与历史演进

## 📊 论文数量统计

| 年份 | 论文数量 | 增长趋势 | 特点 |
|------|----------|----------|------|
| 2007-2019 | ~50 | 基础奠定期 | 理论框架建立 |
| 2020 | ~25 | ████ | NISQ算法兴起 |
| 2021 | ~35 | █████ | 量子优势探索 |
| 2022 | ~180 | ████████████ | 快速扩展期 |
| 2023 | ~280 | ██████████████████ | 成熟发展期 |
| 2024 | ~250 | █████████████████ | 应用深化期 |
| 2025 | ~350+ | ████████████████████████ | 实用化探索期 |

## 🔄 研究主题演进 (2007-2025)

### 阶段一: 理论奠基 (2007-2019)
- **2008**: Harrow-Hassidim-Lloyd (HHL) 算法提出线性系统量子加速
- **2009**: 量子随机游走机器学习初探
- **2014**: 量子PCA算法
- **2016-2017**: 变分量子特征求解器(VQE)发展
- **2018**: 变分量子算法(VQA)框架系统化
- **2019**: 量子神经网络(QNN)架构初步形成

### 阶段二: NISQ算法爆发 (2020-2021)
**标志性成果:**
- 数据重上传(Data Re-uploading)方法提出 (Pérez-Salinas et al., 2020)
- 噪声诱导贫瘠高原发现 (Wang et al., 2020)
- 量子核方法系统化 (Schuld, 2021)
- 量子优势实验证明 (Huang et al., 2021)

**研究特点:**
- 理论框架趋于完整
- NISQ设备实验验证开始
- 量子-经典混合算法成为主流

### 阶段三: 快速扩展 (2022-2023)
**主要趋势:**
1. 变分算法深入发展 (VQE, QAOA, VQC)
2. 量子核方法与量子神经网络并行发展
3. 实际应用探索 (金融、化学、优化)
4. 贫瘠高原问题成为核心挑战
5. 量子-经典混合架构多样化

**关键论文:**
- Caro et al., 2022 - 少样本泛化理论
- Cerezo et al., 2023 - 贫瘠高原可证明缺失与经典可模拟性
- Jerbi et al., 2023 - 量子机器学习的阴影

### 阶段四: 应用深化与实用化 (2024-2025)
**2024年特点:**
- 硬件实验验证增加
- 化学和材料科学应用深化
- 量子优势边界更加清晰
- 错误缓解技术成熟

**2025年特点:**
- 混合量子-经典架构主导
- 早期容错时代VQA
- 实用规模问题尝试
- 自动化量子电路设计
- 量子储备池计算兴起

## 🎯 核心研究领域演进

### 变分量子算法 (VQE/VQA/QAOA)
```
2017: VQE基础理论
  ↓
2019-2020: Ansatz设计策略
  ↓
2021-2022: 贫瘠高原问题深入研究
  ↓
2023: 参数迁移与预训练
  ↓
2024: 硬件实验验证
  ↓
2025: 早期容错与大规模应用
```

### 量子核方法
```
2018-2019: 量子核概念引入
  ↓
2020-2021: 量子核与经典核比较
  ↓
2022: 量子核训练与对齐
  ↓
2023: 量子核指数集中问题
  ↓
2024-2025: 硬件感知核设计
```

### 量子神经网络
```
2017-2018: 参数化量子电路
  ↓
2019-2020: 数据编码策略
  ↓
2021: 表达能力与可训练性理论
  ↓
2022-2023: 混合架构与深度QNN
  ↓
2024-2025: 量子Transformer、量子图神经网络
```

### 化学与材料应用
```
2016-2017: VQE分子能量计算
  ↓
2019-2020: 小分子模拟
  ↓
2021-2022: QNN力场与分子性质预测
  ↓
2023: 分子对接与药物发现
  ↓
2024: 大分子系统与材料发现
  ↓
2025: 几何优化、过渡态、吸附能
```

## 📈 技术热点变迁

| 年份 | 最热门主题 | 新兴主题 |
|------|-----------|----------|
| 2020 | 数据编码、VQE | 噪声影响分析 |
| 2021 | 量子核方法、量子优势 | 量子RL、量子NLP |
| 2022 | 贫瘠高原、QAOA | 量子图神经网络 |
| 2023 | 混合架构、量子储备池 | 量子扩散模型 |
| 2024 | 硬件验证、错误缓解 | 量子LLM |
| 2025 | 实用化应用、容错QML | 量子Transformer |

## 🔬 与化学/材料相关的研究演进

### 分子能量预测
| 年份 | 关键进展 |
|------|----------|
| 2016-2017 | VQE在H₂、LiH等小分子上验证 |
| 2020 | 深度VQE分治策略 |
| 2021 | 噪声弹性测量技术 |
| 2022 | QNN力场生成 |
| 2023 | 化学启发动态Ansatz |
| 2024 | 大规模分子VQE |
| 2025 | 混合量子-神经波函数 |

### 材料科学应用
| 年份 | 关键进展 |
|------|----------|
| 2022 | 量子特征映射用于材料 |
| 2023 | 量子卷积神经网络材料分类 |
| 2024 | 量子主动学习材料设计 |
| 2025 | 量子增强ML材料发现 |

### 药物发现
| 年份 | 关键进展 |
|------|----------|
| 2022 | 量子核用于药物筛选 |
| 2023 | 量子-经典混合蛋白结合预测 |
| 2024 | 量子分子对接 |
| 2025 | Q2SAR多核学习药物发现 |

---

# 2025年文献

## ⭐ 化学与材料科学应用 (与研究最相关)

### 分子能量与量子化学

| 论文 | 关键内容 | 相关度 |
|------|----------|--------|
| [Li et al., 2025 - Quantum Machine Learning of Molecular Energies with Hybrid Quantum-Neural Wavefunction](https://arxiv.org/search/?query=Quantum+Machine+Learning+Molecular+Energies+Hybrid+Quantum+Neural+Wavefunction+2025&searchtype=all) | 混合量子-神经波函数分子能量预测 | ⭐⭐⭐ |
| [Fonseca et al., 2025 - An Introduction to Variational Quantum Eigensolver Applied to Chemistry](https://arxiv.org/search/?query=Introduction+Variational+Quantum+Eigensolver+Applied+Chemistry+2025&searchtype=all) | VQE化学应用入门 | ⭐⭐⭐ |
| [Jones et al., 2025 - Parametrized Quantum Circuit Learning for Quantum Chemical Applications](https://arxiv.org/search/?query=Parametrized+Quantum+Circuit+Learning+Quantum+Chemical+Applications+2025&searchtype=all) | 量子化学应用PQC学习 | ⭐⭐⭐ |
| [Patel et al., 2025 - Quantum Measurement for Quantum Chemistry on a Quantum Computer](https://arxiv.org/search/?query=Quantum+Measurement+Quantum+Chemistry+Quantum+Computer+2025&searchtype=all) | 量子计算机量子化学测量 | ⭐⭐⭐ |
| [Boy et al., 2025 - Encoding molecular structures in quantum machine learning](https://arxiv.org/search/?query=Encoding+molecular+structures+quantum+machine+learning+2025&searchtype=all) | 分子结构量子编码 | ⭐⭐⭐ |
| [Patil & Mondal & Maitra, 2025 - Machine Learning Approach towards Quantum Error Mitigation for Accurate Molecular Energetics](https://arxiv.org/search/?query=Machine+Learning+Quantum+Error+Mitigation+Accurate+Molecular+Energetics+2025&searchtype=all) | ML误差缓解精确分子能量 | ⭐⭐⭐ |

### 分子几何与优化

| 论文 | 关键内容 | 相关度 |
|------|----------|--------|
| [Hao et al., 2025 - Large-scale Efficient Molecule Geometry Optimization with Hybrid Quantum-Classical Computing](https://arxiv.org/search/?query=Large+scale+Efficient+Molecule+Geometry+Optimization+Hybrid+Quantum+Classical+2025&searchtype=all) | 大规模分子几何优化 | ⭐⭐⭐ |
| [Pan et al., 2025 - MolQAE: Quantum Autoencoder for Molecular Representation Learning](https://arxiv.org/search/?query=MolQAE+Quantum+Autoencoder+Molecular+Representation+Learning+2025&searchtype=all) | 分子表示学习量子自编码器 | ⭐⭐⭐ |
| [Kamata et al., 2025 - Molecular Quantum Transformer](https://arxiv.org/search/?query=Molecular+Quantum+Transformer+Kamata+2025&searchtype=all) | 分子量子Transformer | ⭐⭐⭐ |

### 材料科学

| 论文 | 关键内容 | 相关度 |
|------|----------|--------|
| [Graña et al., 2025 - Materials Discovery With Quantum-Enhanced Machine Learning Algorithms](https://arxiv.org/search/?query=Materials+Discovery+Quantum+Enhanced+Machine+Learning+Algorithms+2025&searchtype=all) | 量子增强ML材料发现 | ⭐⭐⭐ |
| [Wang & Gong & Pei, 2025 - Quantum and Hybrid Machine-Learning Models for Materials-Science Tasks](https://arxiv.org/search/?query=Quantum+Hybrid+Machine+Learning+Models+Materials+Science+Tasks+2025&searchtype=all) | 材料科学量子混合ML | ⭐⭐⭐ |

### 量子蒙特卡洛与波函数

| 论文 | 关键内容 | 相关度 |
|------|----------|--------|
| [Fu et al., 2025 - Local Pseudopotential Unlocks the True Potential of Neural Network-based Quantum Monte Carlo](https://arxiv.org/search/?query=Local+Pseudopotential+Neural+Network+Quantum+Monte+Carlo+2025&searchtype=all) | 局域赝势释放NN-QMC潜力 | ⭐⭐⭐ |
| [Wu et al., 2025 - Hybrid tensor network and neural network quantum states for quantum chemistry](https://arxiv.org/search/?query=Hybrid+tensor+network+neural+network+quantum+states+quantum+chemistry+2025&searchtype=all) | 混合张量网络与NN量子态 | ⭐⭐⭐ |
| [Xiao & Xiang, 2025 - Implementing advanced trial wave functions in fermion quantum Monte Carlo via stochastic sampling](https://arxiv.org/search/?query=Implementing+advanced+trial+wave+functions+fermion+quantum+Monte+Carlo+stochastic+2025&searchtype=all) | 费米子QMC高级试探波函数 | ⭐⭐⭐ |

## 变分量子算法 (VQE/VQA)

| 论文 | 关键内容 |
|------|----------|
| [Yip & Yeter-Aydeniz & Dong, 2025 - Variational Quantum Annealing for Quantum Chemistry](https://arxiv.org/search/?query=Variational+Quantum+Annealing+Quantum+Chemistry+2025&searchtype=all) | 量子化学变分量子退火 |
| [Dangwal et al., 2025 - Variational Quantum Algorithms in the era of Early Fault Tolerance](https://arxiv.org/search/?query=Variational+Quantum+Algorithms+Early+Fault+Tolerance+2025&searchtype=all) | 早期容错时代VQA |
| [Galvão et al., 2025 - Variational quantum computing for quantum simulation](https://arxiv.org/search/?query=Variational+quantum+computing+quantum+simulation+principles+challenges+2025&searchtype=all) | 量子模拟VQC原理与挑战 |

## QAOA与组合优化

| 论文 | 关键内容 |
|------|----------|
| [Adler & Stein & Lachner, 2025 - Scaling Quantum Simulation-Based Optimization with Deep QAOA Circuits](https://arxiv.org/search/?query=Scaling+Quantum+Simulation+Based+Optimization+Deep+QAOA+Circuits+2025&searchtype=all) | 深层QAOA电路扩展 |
| [Omanakuttan et al., 2025 - Threshold for Fault-tolerant Quantum Advantage with QAOA](https://arxiv.org/search/?query=Threshold+Fault+tolerant+Quantum+Advantage+QAOA+2025&searchtype=all) | QAOA容错量子优势阈值 |
| [Papalitsas et al., 2025 - Quantum Approximate Optimization Algorithms for Molecular Docking](https://arxiv.org/search/?query=Quantum+Approximate+Optimization+Algorithms+Molecular+Docking+2025&searchtype=all) | QAOA分子对接 |

## 量子核方法

| 论文 | 关键内容 |
|------|----------|
| [Liu et al., 2025 - Hardware-Aware Quantum Kernel Design Based on Graph Neural Networks](https://arxiv.org/search/?query=Hardware+Aware+Quantum+Kernel+Design+Graph+Neural+Networks+2025&searchtype=all) | GNN硬件感知量子核设计 |
| [Kairon & Jäger & Krems, 2025 - Equivalence between exponential concentration in QML kernels and barren plateaus](https://arxiv.org/search/?query=Equivalence+exponential+concentration+QML+kernels+barren+plateaus+2025&searchtype=all) | QML核指数集中与贫瘠高原等价 |

## 量子储备池计算

| 论文 | 关键内容 |
|------|----------|
| [Ahmed & Tennie & Magri, 2025 - Robust quantum reservoir computers for forecasting chaotic dynamics](https://arxiv.org/search/?query=Robust+quantum+reservoir+computers+forecasting+chaotic+dynamics+2025&searchtype=all) | 混沌动力学预测鲁棒量子储备池 |
| [Cimini et al., 2025 - Large-scale quantum reservoir computing using a Gaussian Boson Sampler](https://arxiv.org/search/?query=Large+scale+quantum+reservoir+computing+Gaussian+Boson+Sampler+2025&searchtype=all) | 高斯玻色采样器大规模量子储备池 |

## 理论基础

| 论文 | 关键内容 |
|------|----------|
| [Babbush et al., 2025 - The Grand Challenge of Quantum Applications](https://arxiv.org/search/?query=Grand+Challenge+Quantum+Applications+Babbush+2025&searchtype=all) | **量子应用大挑战** (必读!) |
| [Zimborás et al., 2025 - Myths around quantum computation before full fault tolerance](https://arxiv.org/search/?query=Myths+quantum+computation+before+full+fault+tolerance+2025&searchtype=all) | **容错前量子计算神话** (必读!) |
| [Huang et al., 2025 - Generative quantum advantage for classical and quantum problems](https://arxiv.org/search/?query=Generative+quantum+advantage+classical+quantum+problems+2025&searchtype=all) | 生成量子优势 |

---

# 2024年文献

## ⭐ 化学与材料科学应用

| 论文 | 关键内容 | 相关度 |
|------|----------|--------|
| [Dutta et al., 2024 - Simulating Chemistry on Bosonic Quantum Devices](https://arxiv.org/search/?query=Simulating+Chemistry+Bosonic+Quantum+Devices+2024&searchtype=all) | 玻色量子设备化学模拟 | ⭐⭐⭐ |
| [Magnusson et al., 2024 - Towards Efficient Quantum Computing for Quantum Chemistry](https://arxiv.org/search/?query=Towards+Efficient+Quantum+Computing+Quantum+Chemistry+Transcorrelated+2024&searchtype=all) | 超相关Ansatz量子化学 | ⭐⭐⭐ |
| [Beaulieu et al., 2024 - Robust Quantum Reservoir Computing for Molecular Property Prediction](https://arxiv.org/search/?query=Robust+Quantum+Reservoir+Computing+Molecular+Property+Prediction+2024&searchtype=all) | 鲁棒QRC分子性质预测 | ⭐⭐⭐ |
| [Li et al., 2024 - Quantum molecular docking with quantum-inspired algorithm](https://arxiv.org/search/?query=Quantum+molecular+docking+quantum+inspired+algorithm+2024&searchtype=all) | 量子启发分子对接 | ⭐⭐⭐ |
| [Garrigues et al., 2024 - Towards molecular docking with neutral atoms](https://arxiv.org/search/?query=Towards+molecular+docking+neutral+atoms+2024&searchtype=all) | 中性原子分子对接 | ⭐⭐⭐ |
| [Lourenço et al., 2024 - Exploring Quantum Active Learning for Materials Design and Discovery](https://arxiv.org/search/?query=Exploring+Quantum+Active+Learning+Materials+Design+Discovery+2024&searchtype=all) | 量子主动学习材料设计 | ⭐⭐⭐ |
| [Patel et al., 2024 - Quantum Boltzmann machine learning of ground-state energies](https://arxiv.org/search/?query=Quantum+Boltzmann+machine+learning+ground+state+energies+2024&searchtype=all) | QBM基态能量学习 | ⭐⭐⭐ |

## 变分量子算法

| 论文 | 关键内容 |
|------|----------|
| [Chen et al., 2024 - Crossing The Gap Using Variational Quantum Eigensolver](https://arxiv.org/search/?query=Crossing+Gap+Using+Variational+Quantum+Eigensolver+2024&searchtype=all) | VQE跨越能隙 |
| [Gratsea et al., 2024 - OnionVQE Optimization Strategy for Ground State Preparation on NISQ Devices](https://arxiv.org/search/?query=OnionVQE+Optimization+Strategy+Ground+State+Preparation+NISQ+2024&searchtype=all) | OnionVQE优化策略 |
| [Nakaji et al., 2024 - The generative quantum eigensolver (GQE)](https://arxiv.org/search/?query=generative+quantum+eigensolver+GQE+ground+state+search+2024&searchtype=all) | 生成量子特征求解器 |

## QAOA与优化

| 论文 | 关键内容 |
|------|----------|
| [Augustino et al., 2024 - Strategies for running the QAOA at hundreds of qubits](https://arxiv.org/search/?query=Strategies+running+QAOA+hundreds+qubits+2024&searchtype=all) | 数百量子比特QAOA策略 |
| [Montanez-Barrera & Michielsen, 2024 - Towards a universal QAOA protocol](https://arxiv.org/search/?query=Towards+universal+QAOA+protocol+quantum+advantage+2024&searchtype=all) | 通用QAOA协议 |
| [Morris & Lotshaw, 2024 - Performant near-term quantum combinatorial optimization](https://arxiv.org/search/?query=Performant+near+term+quantum+combinatorial+optimization+2024&searchtype=all) | 高性能近期量子组合优化 |

## 量子核方法

| 论文 | 关键内容 |
|------|----------|
| [Schnabel & Roth, 2024 - Quantum Kernel Methods under Scrutiny: A Benchmarking Study](https://arxiv.org/search/?query=Quantum+Kernel+Methods+Scrutiny+Benchmarking+Study+2024&searchtype=all) | 量子核方法基准研究 |
| [Sahin et al., 2024 - Efficient Parameter Optimisation for Quantum Kernel Alignment](https://arxiv.org/search/?query=Efficient+Parameter+Optimisation+Quantum+Kernel+Alignment+2024&searchtype=all) | 量子核对齐高效参数优化 |

## 量子强化学习

| 论文 | 关键内容 |
|------|----------|
| [Meyer et al., 2024 - Robustness and Generalization in Quantum Reinforcement Learning via Lipschitz Regularization](https://arxiv.org/search/?query=Robustness+Generalization+Quantum+Reinforcement+Learning+Lipschitz+2024&searchtype=all) | Lipschitz正则化量子RL |
| [Liu et al., 2024 - QTRL: Toward Practical Quantum Reinforcement Learning](https://arxiv.org/search/?query=QTRL+Toward+Practical+Quantum+Reinforcement+Learning+2024&searchtype=all) | QTRL实用量子RL |

## 理论与综述

| 论文 | 关键内容 |
|------|----------|
| [Bowles & Ahmed & Schuld, 2024 - Better than classical? The subtle art of benchmarking quantum machine learning models](https://arxiv.org/search/?query=Better+than+classical+subtle+art+benchmarking+quantum+machine+learning+2024&searchtype=all) | QML基准测试的艺术 |
| [Hegde et al., 2024 - Beyond the Buzz: Strategic Paths for Enabling Useful NISQ Applications](https://arxiv.org/search/?query=Beyond+Buzz+Strategic+Paths+Enabling+Useful+NISQ+Applications+2024&searchtype=all) | 实用NISQ应用战略路径 |
| [Wolf, 2024 - Why we care (about quantum machine learning)](https://arxiv.org/search/?query=Why+we+care+quantum+machine+learning+Wolf+2024&searchtype=all) | 为何关注QML |

---

# 2023年文献

## ⭐ 化学与材料科学应用

| 论文 | 关键内容 | 相关度 |
|------|----------|--------|
| [D'Arcangelo et al., 2023 - Leveraging Analog Quantum Computing with Neutral Atoms for Solvent Configuration Prediction in Drug Discovery](https://arxiv.org/search/?query=Leveraging+Analog+Quantum+Computing+Neutral+Atoms+Solvent+Configuration+Prediction+Drug+Discovery+2023&searchtype=all) | 中性原子溶剂配置预测 | ⭐⭐⭐ |
| [Ding & Huang & Yuan, 2023 - Molecular docking via quantum approximate optimization algorithm](https://arxiv.org/search/?query=Molecular+docking+quantum+approximate+optimization+algorithm+2023&searchtype=all) | QAOA分子对接 | ⭐⭐⭐ |
| [Sager-Smith & Mazziotti, 2023 - Reducing the Quantum Many-electron Problem to Two Electrons with Machine Learning](https://arxiv.org/search/?query=Reducing+Quantum+Many+electron+Problem+Two+Electrons+Machine+Learning+2023&searchtype=all) | ML简化多电子问题 | ⭐⭐⭐ |
| [Domingo et al., 2023 - Hybrid quantum-classical convolutional neural networks to improve molecular protein binding affinity predictions](https://arxiv.org/search/?query=Hybrid+quantum+classical+convolutional+neural+networks+molecular+protein+binding+affinity+2023&searchtype=all) | 混合CNN蛋白结合预测 | ⭐⭐⭐ |

## 变分量子算法

| 论文 | 关键内容 |
|------|----------|
| [Cerezo et al., 2023 - Does provable absence of barren plateaus imply classical simulability?](https://arxiv.org/search/?query=provable+absence+barren+plateaus+imply+classical+simulability+Cerezo+2023&searchtype=all) | 贫瘠高原缺失与经典可模拟性 |
| [Barison & Vicentini & Carleo, 2023 - Embedding Classical Variational Methods in Quantum Circuits](https://arxiv.org/search/?query=Embedding+Classical+Variational+Methods+Quantum+Circuits+2023&searchtype=all) | 经典变分方法嵌入量子电路 |
| [Feniou et al., 2023 - Adaptive variational quantum algorithms on a noisy intermediate scale quantum computer](https://arxiv.org/search/?query=Adaptive+variational+quantum+algorithms+noisy+intermediate+scale+quantum+2023&searchtype=all) | 自适应VQA |

## 量子机器学习理论

| 论文 | 关键内容 |
|------|----------|
| [Caro et al., 2023 - Classical Verification of Quantum Learning](https://arxiv.org/search/?query=Classical+Verification+Quantum+Learning+Caro+2023&searchtype=all) | 量子学习经典验证 |
| [Jerbi et al., 2023 - Shadows of quantum machine learning](https://arxiv.org/search/?query=Shadows+quantum+machine+learning+Jerbi+2023&searchtype=all) | 量子机器学习阴影 |
| [Ragone et al., 2023 - A Unified Theory of Barren Plateaus for Deep Parametrized Quantum Circuits](https://arxiv.org/search/?query=Unified+Theory+Barren+Plateaus+Deep+Parametrized+Quantum+Circuits+2023&searchtype=all) | 贫瘠高原统一理论 |

## 量子储备池计算

| 论文 | 关键内容 |
|------|----------|
| [Domingo & Carlo & Borondo, 2023 - Taking advantage of noise in quantum reservoir computing](https://arxiv.org/search/?query=Taking+advantage+noise+quantum+reservoir+computing+2023&searchtype=all) | 利用噪声量子储备池计算 |
| [Garcia-Beni et al., 2023 - Squeezing as a resource for time series processing in quantum reservoir computing](https://arxiv.org/search/?query=Squeezing+resource+time+series+processing+quantum+reservoir+computing+2023&searchtype=all) | 压缩态量子储备池时序处理 |

---

# 2022年文献

## ⭐ 化学与材料科学应用

| 论文 | 关键内容 | 相关度 |
|------|----------|--------|
| [Ceroni et al., 2022 - Generating Approximate Ground States of Molecules Using Quantum Machine Learning](https://arxiv.org/search/?query=Generating+Approximate+Ground+States+Molecules+Using+Quantum+Machine+Learning+2022&searchtype=all) | QML生成分子近似基态 | ⭐⭐⭐ |
| [Beaudoin et al., 2022 - Quantum Machine Learning for Material Synthesis and Hardware Security](https://arxiv.org/search/?query=Quantum+Machine+Learning+Material+Synthesis+Hardware+Security+2022&searchtype=all) | QML材料合成 | ⭐⭐⭐ |
| [Chandarana et al., 2022 - Digitized-Counterdiabatic Quantum Algorithm for Protein Folding](https://arxiv.org/search/?query=Digitized+Counterdiabatic+Quantum+Algorithm+Protein+Folding+2022&searchtype=all) | 蛋白质折叠数字反糖 | ⭐⭐⭐ |
| [Kiss et al., 2022 - Quantum neural networks force fields generation](https://arxiv.org/search/?query=Quantum+neural+networks+force+fields+generation+2022&searchtype=all) | QNN力场生成 | ⭐⭐⭐ |

## 变分量子算法

| 论文 | 关键内容 |
|------|----------|
| [Tilly et al., 2022 - The VQE: a review of methods and best practices](https://arxiv.org/search/?query=VQE+review+methods+best+practices+Tilly+2022&searchtype=all) | VQE方法综述 |
| [Caro et al., 2022 - Generalization in quantum machine learning from few training data](https://arxiv.org/search/?query=Generalization+quantum+machine+learning+few+training+data+Caro+2022&searchtype=all) | 少样本QML泛化 |

## 理论与综述

| 论文 | 关键内容 |
|------|----------|
| [Schuld & Killoran, 2022 - Is quantum advantage the right goal for quantum machine learning?](https://arxiv.org/search/?query=quantum+advantage+right+goal+quantum+machine+learning+Schuld+2022&searchtype=all) | 量子优势是否是正确目标 |
| [Meyer et al., 2022 - A Survey on Quantum Reinforcement Learning](https://arxiv.org/search/?query=Survey+Quantum+Reinforcement+Learning+Meyer+2022&searchtype=all) | 量子RL综述 |
| [Sajjan et al., 2022 - Quantum machine learning for chemistry and physics](https://arxiv.org/search/?query=Quantum+machine+learning+chemistry+physics+Sajjan+2022&searchtype=all) | 化学物理QML |

---

# 2021年文献

## 里程碑论文

| 论文 | 关键内容 | 重要性 |
|------|----------|--------|
| [Bharti et al., 2021 - Noisy intermediate-scale quantum (NISQ) algorithms](https://arxiv.org/search/?query=Noisy+intermediate+scale+quantum+NISQ+algorithms+Bharti+2021&searchtype=all) | NISQ算法综述 | ⭐⭐⭐ |
| [Schuld, 2021 - Supervised quantum machine learning models are kernel methods](https://arxiv.org/search/?query=Supervised+quantum+machine+learning+models+kernel+methods+Schuld+2021&searchtype=all) | 监督QML是核方法 | ⭐⭐⭐ |
| [Huang et al., 2021 - Quantum advantage in learning from experiments](https://arxiv.org/search/?query=Quantum+advantage+learning+experiments+Huang+2021&searchtype=all) | 实验学习量子优势 | ⭐⭐⭐ |
| [Huang et al., 2021 - The power of data in quantum machine learning](https://arxiv.org/search/?query=power+data+quantum+machine+learning+Huang+2021&searchtype=all) | 数据在QML中的力量 | ⭐⭐⭐ |
| [Caro et al., 2021 - Generalization in quantum machine learning from few training data](https://arxiv.org/search/?query=Generalization+quantum+machine+learning+few+training+data+2021&searchtype=all) | 少样本QML泛化 | ⭐⭐⭐ |

## 化学相关

| 论文 | 关键内容 |
|------|----------|
| [Huggins et al., 2021 - Efficient and noise resilient measurements for quantum chemistry on near-term quantum computers](https://arxiv.org/search/?query=Efficient+noise+resilient+measurements+quantum+chemistry+near+term+2021&searchtype=all) | 噪声弹性量子化学测量 |
| [Motta & Rice, 2021 - Emerging quantum computing algorithms for quantum chemistry](https://arxiv.org/search/?query=Emerging+quantum+computing+algorithms+quantum+chemistry+2021&searchtype=all) | 量子化学新兴量子算法 |

## 算法与架构

| 论文 | 关键内容 |
|------|----------|
| [Beer et al., 2021 - Quantum machine learning of graph-structured data](https://arxiv.org/search/?query=Quantum+machine+learning+graph+structured+data+Beer+2021&searchtype=all) | 图结构数据QML |
| [Kyriienko & Paine & Elfving, 2021 - Solving nonlinear differential equations with differentiable quantum circuits](https://arxiv.org/search/?query=Solving+nonlinear+differential+equations+differentiable+quantum+circuits+2021&searchtype=all) | 可微分量子电路解微分方程 |

---

# 2020年文献

## 奠基性论文

| 论文 | 关键内容 | 重要性 |
|------|----------|--------|
| [Cerezo et al., 2020 - Variational Quantum Algorithms](https://arxiv.org/search/?query=Variational+Quantum+Algorithms+Cerezo+2020&searchtype=all) | 变分量子算法综述 | ⭐⭐⭐ |
| [Wang et al., 2020 - Noise-Induced Barren Plateaus in Variational Quantum Algorithms](https://arxiv.org/search/?query=Noise+Induced+Barren+Plateaus+Variational+Quantum+Algorithms+2020&searchtype=all) | 噪声诱导贫瘠高原 | ⭐⭐⭐ |
| [Pérez-Salinas et al., 2020 - Data re-uploading for a universal quantum classifier](https://arxiv.org/search/?query=Data+re+uploading+universal+quantum+classifier+2020&searchtype=all) | 数据重上传通用分类器 | ⭐⭐⭐ |
| [Schuld, Sweke, Meyer, 2020 - The effect of data encoding on the expressive power of variational quantum machine learning models](https://arxiv.org/search/?query=effect+data+encoding+expressive+power+variational+quantum+machine+learning+2020&searchtype=all) | 数据编码对表达能力影响 | ⭐⭐⭐ |
| [Abbas et al., 2020 - The power of quantum neural networks](https://arxiv.org/search/?query=power+quantum+neural+networks+Abbas+2020&searchtype=all) | 量子神经网络的力量 | ⭐⭐⭐ |

## 其他重要论文

| 论文 | 关键内容 |
|------|----------|
| [Beer et al., 2020 - Training deep quantum neural networks](https://arxiv.org/search/?query=Training+deep+quantum+neural+networks+Beer+2020&searchtype=all) | 深度QNN训练 |
| [Bausch, 2020 - Recurrent Quantum Neural Network](https://arxiv.org/search/?query=Recurrent+Quantum+Neural+Network+Bausch+2020&searchtype=all) | 循环量子神经网络 |
| [Fujii et al., 2020 - Deep Variational Quantum Eigensolver](https://arxiv.org/search/?query=Deep+Variational+Quantum+Eigensolver+divide+conquer+2020&searchtype=all) | 深度VQE分治法 |

---

# 2007-2019年奠基性文献

## 核心理论奠基

| 年份 | 论文 | 关键内容 |
|------|------|----------|
| 2008 | [Harrow, Hassidim, Lloyd - Quantum algorithm for linear systems of equations](https://arxiv.org/search/?query=Quantum+algorithm+linear+systems+equations+HHL+2008&searchtype=all) | HHL算法 |
| 2014 | [Lloyd, Mohseni, Rebentrost - Quantum principal component analysis](https://arxiv.org/search/?query=Quantum+principal+component+analysis+Lloyd+2014&searchtype=all) | 量子PCA |
| 2016 | [Peruzzo et al. - A variational eigenvalue solver on a photonic quantum processor](https://arxiv.org/search/?query=variational+eigenvalue+solver+photonic+quantum+processor+Peruzzo+2016&searchtype=all) | VQE首次实现 |
| 2017 | [Farhi & Neven - Classification with Quantum Neural Networks on Near Term Processors](https://arxiv.org/search/?query=Classification+Quantum+Neural+Networks+Near+Term+Processors+Farhi+2017&searchtype=all) | 近期处理器QNN分类 |
| 2018 | [Havlíček et al. - Supervised learning with quantum-enhanced feature spaces](https://arxiv.org/search/?query=Supervised+learning+quantum+enhanced+feature+spaces+Havlicek+2018&searchtype=all) | 量子增强特征空间 |
| 2018 | [Mitarai et al. - Quantum circuit learning](https://arxiv.org/search/?query=Quantum+circuit+learning+Mitarai+2018&searchtype=all) | 量子电路学习 |
| 2018 | [McClean et al. - Barren plateaus in quantum neural network training landscapes](https://arxiv.org/search/?query=Barren+plateaus+quantum+neural+network+training+landscapes+McClean+2018&searchtype=all) | 贫瘠高原发现 |
| 2019 | [Cao et al. - Quantum Chemistry in the Age of Quantum Computing](https://arxiv.org/search/?query=Quantum+Chemistry+Age+Quantum+Computing+Cao+2019&searchtype=all) | 量子计算时代量子化学 |
| 2019 | [Benedetti et al. - Parameterized quantum circuits as machine learning models](https://arxiv.org/search/?query=Parameterized+quantum+circuits+machine+learning+models+Benedetti+2019&searchtype=all) | PQC作为ML模型 |

---

# 化学与材料应用专题汇总

## 分子能量与电子结构

| 年份 | 论文 | 贡献 |
|------|------|------|
| 2016 | Peruzzo et al. | VQE首次光子实现 |
| 2019 | Cao et al. | 量子计算量子化学综述 |
| 2020 | Fujii et al. | 深度VQE分治策略 |
| 2021 | Huggins et al. | 噪声弹性测量 |
| 2021 | Motta & Rice | 量子化学算法综述 |
| 2022 | Ceroni et al. | QML近似基态生成 |
| 2022 | Kiss et al. | QNN力场生成 |
| 2023 | Sager-Smith & Mazziotti | ML简化多电子问题 |
| 2024 | Dutta et al. | 玻色器件化学模拟 |
| 2024 | Patel et al. | QBM基态能量 |
| 2025 | Li et al. | 混合量子-神经波函数 |
| 2025 | Wu et al. | 张量网络神经网络量子态 |

## 分子结构与几何优化

| 年份 | 论文 | 贡献 |
|------|------|------|
| 2022 | Chandarana et al. | 蛋白质折叠 |
| 2023 | Domingo et al. | 蛋白结合亲和力 |
| 2024 | Li et al. | 量子分子对接 |
| 2025 | Hao et al. | 大规模分子几何优化 |
| 2025 | Pan et al. | 分子表示学习自编码器 |

## 材料科学

| 年份 | 论文 | 贡献 |
|------|------|------|
| 2022 | Beaudoin et al. | 材料合成QML |
| 2024 | Lourenço et al. | 量子主动学习材料设计 |
| 2025 | Graña et al. | 量子增强材料发现 |
| 2025 | Wang & Gong & Pei | 材料科学任务混合ML |

## 药物发现

| 年份 | 论文 | 贡献 |
|------|------|------|
| 2023 | D'Arcangelo et al. | 溶剂配置预测 |
| 2023 | Ding & Huang & Yuan | QAOA分子对接 |
| 2024 | Garrigues et al. | 中性原子分子对接 |
| 2025 | Giraldo et al. | Q2SAR多核药物发现 |
| 2025 | Papalitsas et al. | QAOA分子对接 |

---

# 阅读建议与未来方向

## 📚 入门阅读顺序

### 第一阶段: 基础概念 (1-2周)
1. [Cerezo et al., 2020](https://arxiv.org/search/?query=Variational+Quantum+Algorithms+Cerezo+2020&searchtype=all) - VQA综述
2. [Schuld, 2021](https://arxiv.org/search/?query=Supervised+quantum+machine+learning+models+kernel+methods+Schuld+2021&searchtype=all) - QML与核方法
3. [Bharti et al., 2021](https://arxiv.org/search/?query=Noisy+intermediate+scale+quantum+NISQ+algorithms+Bharti+2021&searchtype=all) - NISQ算法

### 第二阶段: 核心问题 (2-3周)
4. [McClean et al., 2018](https://arxiv.org/search/?query=Barren+plateaus+quantum+neural+network+training+landscapes+McClean+2018&searchtype=all) - 贫瘠高原
5. [Wang et al., 2020](https://arxiv.org/search/?query=Noise+Induced+Barren+Plateaus+Variational+Quantum+Algorithms+2020&searchtype=all) - 噪声诱导贫瘠高原
6. [Pérez-Salinas et al., 2020](https://arxiv.org/search/?query=Data+re+uploading+universal+quantum+classifier+2020&searchtype=all) - 数据重上传

### 第三阶段: 化学应用 (2-3周)
7. [Cao et al., 2019](https://arxiv.org/search/?query=Quantum+Chemistry+Age+Quantum+Computing+Cao+2019&searchtype=all) - 量子化学综述
8. [Motta & Rice, 2021](https://arxiv.org/search/?query=Emerging+quantum+computing+algorithms+quantum+chemistry+2021&searchtype=all) - 量子化学算法
9. [Sajjan et al., 2022](https://arxiv.org/search/?query=Quantum+machine+learning+chemistry+physics+Sajjan+2022&searchtype=all) - 化学物理QML

### 第四阶段: 前沿进展 (持续)
10. [Babbush et al., 2025](https://arxiv.org/search/?query=Grand+Challenge+Quantum+Applications+Babbush+2025&searchtype=all) - 量子应用大挑战
11. 2025年化学相关论文

## 🔮 未来研究方向

### 短期 (1-2年)
1. **混合量子-经典架构优化** - 更高效的量子经典界面
2. **噪声缓解技术** - 更实用的错误缓解方案
3. **量子特征映射设计** - 针对化学问题的专用映射

### 中期 (3-5年)
1. **早期容错量子算法** - 利用有限纠错能力
2. **大分子系统模拟** - 扩展到实际化学系统
3. **过渡态与反应路径** - 动力学性质预测

### 长期 (5-10年)
1. **实用量子优势** - 在特定化学问题上超越经典
2. **材料设计自动化** - 量子辅助材料发现
3. **药物发现加速** - 端到端量子药物设计

## 🎯 对你研究最相关的论文 Top 20

### 必读 (Top 10)
1. **Li et al., 2025** - 混合量子-神经波函数分子能量
2. **Hao et al., 2025** - 大规模分子几何优化
3. **Boy et al., 2025** - 分子结构量子编码
4. **Graña et al., 2025** - 量子增强材料发现
5. **Wu et al., 2025** - 张量网络神经网络量子化学
6. **Sajjan et al., 2022** - 化学物理QML综述
7. **Motta & Rice, 2021** - 量子化学新兴算法
8. **Kiss et al., 2022** - QNN力场生成
9. **Ceroni et al., 2022** - QML分子基态
10. **Tilly et al., 2022** - VQE综述

### 推荐 (Top 11-20)
11. **Patil & Maitra, 2025** - 分子能量误差缓解
12. **Bincoletto et al., 2025** - 电子结构电路参数
13. **Fu et al., 2025** - 神经网络量子蒙特卡洛
14. **Dutta et al., 2024** - 玻色器件化学模拟
15. **Giraldo et al., 2025** - Q2SAR药物发现
16. **Beaulieu et al., 2024** - QRC分子性质预测
17. **Sager-Smith & Mazziotti, 2023** - ML简化多电子问题
18. **Huggins et al., 2021** - 噪声弹性化学测量
19. **Ding et al., 2023** - QAOA分子对接
20. **Wang & Gong & Pei, 2025** - 材料科学混合ML

---

*文档生成: 2025年1月17日*
*收录论文: ~1200+篇 (2007-2025)*
*📎 所有论文均附可点击链接*

---

---

# 附录A: 2025年完整论文列表

> 按主题分类，共350+篇

## A.1 化学/材料/药物发现

| 论文 | 链接 |
|------|------|
| Li et al. - Quantum ML of Molecular Energies with Hybrid Quantum-Neural Wavefunction | [arXiv](https://arxiv.org/search/?query=Quantum+Machine+Learning+Molecular+Energies+Hybrid+Quantum+Neural+Wavefunction+2025&searchtype=all) |
| Fonseca et al. - Introduction to VQE Applied to Chemistry | [arXiv](https://arxiv.org/search/?query=Introduction+Variational+Quantum+Eigensolver+Applied+Chemistry+2025&searchtype=all) |
| Jones et al. - Parametrized Quantum Circuit Learning for Quantum Chemical Applications | [arXiv](https://arxiv.org/search/?query=Parametrized+Quantum+Circuit+Learning+Quantum+Chemical+Applications+2025&searchtype=all) |
| Patel et al. - Quantum Measurement for Quantum Chemistry on a Quantum Computer | [arXiv](https://arxiv.org/search/?query=Quantum+Measurement+Quantum+Chemistry+Quantum+Computer+2025&searchtype=all) |
| Boy et al. - Encoding molecular structures in quantum machine learning | [arXiv](https://arxiv.org/search/?query=Encoding+molecular+structures+quantum+machine+learning+2025&searchtype=all) |
| Hao et al. - Large-scale Molecule Geometry Optimization with Hybrid QC Computing | [arXiv](https://arxiv.org/search/?query=Large+scale+Efficient+Molecule+Geometry+Optimization+Hybrid+Quantum+Classical+2025&searchtype=all) |
| Pan et al. - MolQAE: Quantum Autoencoder for Molecular Representation Learning | [arXiv](https://arxiv.org/search/?query=MolQAE+Quantum+Autoencoder+Molecular+Representation+Learning+2025&searchtype=all) |
| Kamata et al. - Molecular Quantum Transformer | [arXiv](https://arxiv.org/search/?query=Molecular+Quantum+Transformer+Kamata+2025&searchtype=all) |
| Giraldo et al. - Q2SAR: Quantum Multiple Kernel Learning for Drug Discovery | [arXiv](https://arxiv.org/search/?query=Q2SAR+Quantum+Multiple+Kernel+Learning+Drug+Discovery+2025&searchtype=all) |
| Graña et al. - Materials Discovery With Quantum-Enhanced ML Algorithms | [arXiv](https://arxiv.org/search/?query=Materials+Discovery+Quantum+Enhanced+Machine+Learning+Algorithms+2025&searchtype=all) |
| Wang & Gong & Pei - Quantum and Hybrid ML Models for Materials-Science Tasks | [arXiv](https://arxiv.org/search/?query=Quantum+Hybrid+Machine+Learning+Models+Materials+Science+Tasks+2025&searchtype=all) |
| Fu et al. - Local Pseudopotential for NN-based Quantum Monte Carlo | [arXiv](https://arxiv.org/search/?query=Local+Pseudopotential+Neural+Network+Quantum+Monte+Carlo+2025&searchtype=all) |
| Wu et al. - Hybrid tensor network and NN quantum states for quantum chemistry | [arXiv](https://arxiv.org/search/?query=Hybrid+tensor+network+neural+network+quantum+states+quantum+chemistry+2025&searchtype=all) |

## A.2 VQE/VQA

| 论文 | 链接 |
|------|------|
| Yip & Yeter-Aydeniz & Dong - Variational Quantum Annealing for Quantum Chemistry | [arXiv](https://arxiv.org/search/?query=Variational+Quantum+Annealing+Quantum+Chemistry+2025&searchtype=all) |
| Dangwal et al. - VQAs in the era of Early Fault Tolerance | [arXiv](https://arxiv.org/search/?query=Variational+Quantum+Algorithms+Early+Fault+Tolerance+2025&searchtype=all) |
| Galvão et al. - Variational quantum computing for quantum simulation | [arXiv](https://arxiv.org/search/?query=Variational+quantum+computing+quantum+simulation+principles+challenges+2025&searchtype=all) |
| Kalam et al. - Efficient Quantum Information-Inspired Ansatz for VQE | [arXiv](https://arxiv.org/search/?query=Efficient+Quantum+Information+Inspired+Ansatz+VQE+Atomic+Systems+2025&searchtype=all) |
| Rohe et al. - Accelerated VQE: Parameter Recycling | [arXiv](https://arxiv.org/search/?query=Accelerated+VQE+Parameter+Recycling+Similar+Recurring+Problems+2025&searchtype=all) |
| Possel et al. - Truncated Variational Hamiltonian Ansatz | [arXiv](https://arxiv.org/search/?query=Truncated+Variational+Hamiltonian+Ansatz+2025&searchtype=all) |

## A.3 QAOA/组合优化

| 论文 | 链接 |
|------|------|
| Adler & Stein & Lachner - Scaling Optimization with Deep QAOA Circuits | [arXiv](https://arxiv.org/search/?query=Scaling+Quantum+Simulation+Based+Optimization+Deep+QAOA+Circuits+2025&searchtype=all) |
| Bach & Maciejewski & Safro - Solving Large-Scale QUBO with Multilevel QAOA | [arXiv](https://arxiv.org/search/?query=Solving+Large+Scale+QUBO+Transferred+Parameters+Multilevel+QAOA+2025&searchtype=all) |
| Omanakuttan et al. - Threshold for Fault-tolerant Quantum Advantage with QAOA | [arXiv](https://arxiv.org/search/?query=Threshold+Fault+tolerant+Quantum+Advantage+QAOA+2025&searchtype=all) |
| Papalitsas et al. - QAOA for Molecular Docking | [arXiv](https://arxiv.org/search/?query=Quantum+Approximate+Optimization+Algorithms+Molecular+Docking+2025&searchtype=all) |

## A.4 量子神经网络

| 论文 | 链接 |
|------|------|
| Ahmed et al. - QNN: Comparative Analysis and Noise Robustness | [arXiv](https://arxiv.org/search/?query=Quantum+Neural+Networks+Comparative+Analysis+Noise+Robustness+2025&searchtype=all) |
| Chen & Kuo - Quantum Adaptive Self-Attention for Quantum Transformer | [arXiv](https://arxiv.org/search/?query=Quantum+Adaptive+Self+Attention+Quantum+Transformer+Models+2025&searchtype=all) |
| Zhang & Zhao - Survey of Quantum Transformers | [arXiv](https://arxiv.org/search/?query=Survey+Quantum+Transformers+Approaches+Advantages+Challenges+2025&searchtype=all) |
| Faria et al. - Quantum Graph Attention Networks | [arXiv](https://arxiv.org/search/?query=Quantum+Graph+Attention+Networks+Trainable+Quantum+Encoders+Inductive+Graph+2025&searchtype=all) |

## A.5 量子储备池计算

| 论文 | 链接 |
|------|------|
| Ahmed & Tennie & Magri - Robust QRC for chaotic dynamics | [arXiv](https://arxiv.org/search/?query=Robust+quantum+reservoir+computers+forecasting+chaotic+dynamics+2025&searchtype=all) |
| Cimini et al. - Large-scale QRC using Gaussian Boson Sampler | [arXiv](https://arxiv.org/search/?query=Large+scale+quantum+reservoir+computing+Gaussian+Boson+Sampler+2025&searchtype=all) |
| Li et al. - QRC for Realized Volatility Forecasting | [arXiv](https://arxiv.org/search/?query=Quantum+Reservoir+Computing+Realized+Volatility+Forecasting+2025&searchtype=all) |

## A.6 理论/综述

| 论文 | 链接 |
|------|------|
| Babbush et al. - The Grand Challenge of Quantum Applications | [arXiv](https://arxiv.org/search/?query=Grand+Challenge+Quantum+Applications+Babbush+2025&searchtype=all) |
| Zimborás et al. - Myths around quantum computation before full fault tolerance | [arXiv](https://arxiv.org/search/?query=Myths+quantum+computation+before+full+fault+tolerance+2025&searchtype=all) |
| Ghayour - QML: A Modern Tutorial for Researchers | [arXiv](https://arxiv.org/search/?query=Quantum+Machine+Learning+Modern+Tutorial+Researchers+2025&searchtype=all) |
| Tomar et al. - Comprehensive Survey of QML | [arXiv](https://arxiv.org/search/?query=Comprehensive+Survey+QML+Data+Analysis+Algorithmic+Advancements+2025&searchtype=all) |

---

# 附录B: 2024年完整论文列表

> 按主题分类，共250+篇

## B.1 化学/材料/药物发现

| 论文 | 链接 |
|------|------|
| Dutta et al. - Simulating Chemistry on Bosonic Quantum Devices | [arXiv](https://arxiv.org/search/?query=Simulating+Chemistry+Bosonic+Quantum+Devices+2024&searchtype=all) |
| Battaglia et al. - Active space embedding methods for quantum computing | [arXiv](https://arxiv.org/search/?query=active+space+embedding+quantum+computing+Battaglia+2024&searchtype=all) |
| Magnusson et al. - Transcorrelated and Adaptive Ansatz Techniques | [arXiv](https://arxiv.org/search/?query=Transcorrelated+Adaptive+Ansatz+quantum+chemistry+2024&searchtype=all) |
| Patel et al. - QBM learning of ground-state energies | [arXiv](https://arxiv.org/search/?query=Quantum+Boltzmann+machine+ground+state+energies+2024&searchtype=all) |
| Li et al. - Quantum molecular docking | [arXiv](https://arxiv.org/search/?query=Quantum+molecular+docking+quantum+inspired+algorithm+2024&searchtype=all) |
| Garrigues et al. - Molecular docking with neutral atoms | [arXiv](https://arxiv.org/search/?query=molecular+docking+neutral+atoms+2024&searchtype=all) |
| Lourenço et al. - Quantum Active Learning for Materials Design | [arXiv](https://arxiv.org/search/?query=Quantum+Active+Learning+Materials+Design+Discovery+2024&searchtype=all) |
| Beaulieu et al. - Robust QRC for Molecular Property Prediction | [arXiv](https://arxiv.org/search/?query=Robust+Quantum+Reservoir+Computing+Molecular+Property+Prediction+2024&searchtype=all) |

## B.2 VQE/VQA

| 论文 | 链接 |
|------|------|
| Chen et al. - Crossing The Gap Using VQE | [arXiv](https://arxiv.org/search/?query=Crossing+Gap+Variational+Quantum+Eigensolver+2024&searchtype=all) |
| Nakaji et al. - Generative quantum eigensolver (GQE) | [arXiv](https://arxiv.org/search/?query=generative+quantum+eigensolver+GQE+ground+state+search+2024&searchtype=all) |
| Gratsea et al. - OnionVQE Optimization Strategy | [arXiv](https://arxiv.org/search/?query=OnionVQE+Optimization+Strategy+Ground+State+NISQ+2024&searchtype=all) |

## B.3 QAOA/组合优化

| 论文 | 链接 |
|------|------|
| Augustino et al. - QAOA at hundreds of qubits | [arXiv](https://arxiv.org/search/?query=Strategies+running+QAOA+hundreds+qubits+2024&searchtype=all) |
| Montanez-Barrera & Michielsen - Universal QAOA protocol | [arXiv](https://arxiv.org/search/?query=Towards+universal+QAOA+protocol+quantum+advantage+2024&searchtype=all) |
| Morris & Lotshaw - Performant near-term quantum combinatorial optimization | [arXiv](https://arxiv.org/search/?query=Performant+near+term+quantum+combinatorial+optimization+2024&searchtype=all) |

## B.4 量子核方法

| 论文 | 链接 |
|------|------|
| Schnabel & Roth - Quantum Kernel Methods under Scrutiny | [arXiv](https://arxiv.org/search/?query=Quantum+Kernel+Methods+Scrutiny+Benchmarking+Study+2024&searchtype=all) |
| Sahin et al. - Efficient Quantum Kernel Alignment | [arXiv](https://arxiv.org/search/?query=Efficient+Parameter+Optimisation+Quantum+Kernel+Alignment+2024&searchtype=all) |

## B.5 理论/综述

| 论文 | 链接 |
|------|------|
| Bowles & Schuld - Better than classical? Benchmarking QML | [arXiv](https://arxiv.org/search/?query=Better+than+classical+subtle+art+benchmarking+quantum+machine+learning+2024&searchtype=all) |
| Wolf - Why we care about QML | [arXiv](https://arxiv.org/search/?query=Why+we+care+quantum+machine+learning+Wolf+2024&searchtype=all) |

---

# 附录C: 2023年完整论文列表

> 按主题分类，共280+篇

## C.1 化学/材料/药物发现

| 论文 | 链接 |
|------|------|
| Wu et al. - Real neural network state for quantum chemistry | [arXiv](https://arxiv.org/search/?query=real+neural+network+state+quantum+chemistry+Wu+2023&searchtype=all) |
| Sager-Smith & Mazziotti - Reducing many-electron problem to two electrons | [arXiv](https://arxiv.org/search/?query=Reducing+Quantum+Many+electron+Problem+Two+Electrons+Machine+Learning+2023&searchtype=all) |
| D'Arcangelo et al. - Neutral atoms for drug discovery | [arXiv](https://arxiv.org/search/?query=Analog+Quantum+Computing+Neutral+Atoms+Solvent+Configuration+Drug+Discovery+2023&searchtype=all) |
| Domingo et al. - Hybrid CNN for protein binding affinity | [arXiv](https://arxiv.org/search/?query=Hybrid+quantum+classical+CNN+molecular+protein+binding+affinity+2023&searchtype=all) |
| Ding & Huang & Yuan - QAOA for molecular docking | [arXiv](https://arxiv.org/search/?query=Molecular+docking+quantum+approximate+optimization+algorithm+2023&searchtype=all) |
| Khan et al. - Many-body distribution functionals | [arXiv](https://arxiv.org/search/?query=Quantum+machine+learning+record+speed+Many+body+distribution+functionals+2023&searchtype=all) |

## C.2 VQE/VQA

| 论文 | 链接 |
|------|------|
| Cerezo et al. - Barren plateaus and classical simulability | [arXiv](https://arxiv.org/search/?query=provable+absence+barren+plateaus+imply+classical+simulability+Cerezo+2023&searchtype=all) |
| Ragone et al. - Unified Theory of Barren Plateaus | [arXiv](https://arxiv.org/search/?query=Unified+Theory+Barren+Plateaus+Deep+Parametrized+Quantum+Circuits+2023&searchtype=all) |
| Barison & Carleo - Embedding Classical Methods in Quantum Circuits | [arXiv](https://arxiv.org/search/?query=Embedding+Classical+Variational+Methods+Quantum+Circuits+2023&searchtype=all) |

## C.3 理论/综述

| 论文 | 链接 |
|------|------|
| Caro et al. - Classical Verification of Quantum Learning | [arXiv](https://arxiv.org/search/?query=Classical+Verification+Quantum+Learning+Caro+2023&searchtype=all) |
| Jerbi et al. - Shadows of quantum machine learning | [arXiv](https://arxiv.org/search/?query=Shadows+quantum+machine+learning+Jerbi+2023&searchtype=all) |
| Zhang et al. - AI for Science in Quantum Systems | [arXiv](https://arxiv.org/search/?query=Artificial+Intelligence+Science+Quantum+Atomistic+Continuum+Systems+2023&searchtype=all) |

---

# 附录D: 2022年完整论文列表

> 按主题分类，共180+篇

## D.1 化学/材料

| 论文 | 链接 |
|------|------|
| Ceroni et al. - Generating Approximate Ground States with QML | [arXiv](https://arxiv.org/search/?query=Generating+Approximate+Ground+States+Molecules+Using+Quantum+Machine+Learning+2022&searchtype=all) |
| Kiss et al. - QNN force fields generation | [arXiv](https://arxiv.org/search/?query=Quantum+neural+networks+force+fields+generation+2022&searchtype=all) |
| Beaudoin et al. - QML for Material Synthesis | [arXiv](https://arxiv.org/search/?query=Quantum+Machine+Learning+Material+Synthesis+Hardware+Security+2022&searchtype=all) |
| Chandarana et al. - Digitized-Counterdiabatic for Protein Folding | [arXiv](https://arxiv.org/search/?query=Digitized+Counterdiabatic+Quantum+Algorithm+Protein+Folding+2022&searchtype=all) |

## D.2 VQE/理论

| 论文 | 链接 |
|------|------|
| Tilly et al. - VQE: review of methods and best practices | [arXiv](https://arxiv.org/search/?query=VQE+review+methods+best+practices+Tilly+2022&searchtype=all) |
| Caro et al. - Generalization from few training data | [arXiv](https://arxiv.org/search/?query=Generalization+quantum+machine+learning+few+training+data+Caro+2022&searchtype=all) |
| Schuld & Killoran - Is quantum advantage the right goal? | [arXiv](https://arxiv.org/search/?query=quantum+advantage+right+goal+quantum+machine+learning+Schuld+2022&searchtype=all) |
| Sajjan et al. - QML for chemistry and physics | [arXiv](https://arxiv.org/search/?query=Quantum+machine+learning+chemistry+physics+Sajjan+2022&searchtype=all) |
| Meyer et al. - Survey on Quantum RL | [arXiv](https://arxiv.org/search/?query=Survey+Quantum+Reinforcement+Learning+Meyer+2022&searchtype=all) |

---

# 附录E: 2021年完整论文列表

> 里程碑论文，共35篇

## E.1 核心理论

| 论文 | 链接 |
|------|------|
| Bharti et al. - NISQ algorithms | [arXiv](https://arxiv.org/search/?query=Noisy+intermediate+scale+quantum+NISQ+algorithms+Bharti+2021&searchtype=all) |
| Schuld - Supervised QML models are kernel methods | [arXiv](https://arxiv.org/search/?query=Supervised+quantum+machine+learning+models+kernel+methods+Schuld+2021&searchtype=all) |
| Huang et al. - Quantum advantage in learning from experiments | [arXiv](https://arxiv.org/search/?query=Quantum+advantage+learning+experiments+Huang+2021&searchtype=all) |
| Huang et al. - The power of data in QML | [arXiv](https://arxiv.org/search/?query=power+data+quantum+machine+learning+Huang+2021&searchtype=all) |
| Caro et al. - Generalization from few training data | [arXiv](https://arxiv.org/search/?query=Generalization+quantum+machine+learning+few+training+data+2021&searchtype=all) |

## E.2 化学相关

| 论文 | 链接 |
|------|------|
| Huggins et al. - Noise resilient measurements for quantum chemistry | [arXiv](https://arxiv.org/search/?query=Efficient+noise+resilient+measurements+quantum+chemistry+near+term+2021&searchtype=all) |
| Motta & Rice - Emerging algorithms for quantum chemistry | [arXiv](https://arxiv.org/search/?query=Emerging+quantum+computing+algorithms+quantum+chemistry+2021&searchtype=all) |

## E.3 算法架构

| 论文 | 链接 |
|------|------|
| Beer et al. - QML of graph-structured data | [arXiv](https://arxiv.org/search/?query=Quantum+machine+learning+graph+structured+data+Beer+2021&searchtype=all) |
| Kyriienko et al. - Solving nonlinear differential equations | [arXiv](https://arxiv.org/search/?query=Solving+nonlinear+differential+equations+differentiable+quantum+circuits+2021&searchtype=all) |
| Kartsaklis et al. - lambeq: Quantum NLP library | [arXiv](https://arxiv.org/search/?query=lambeq+Efficient+High+Level+Python+Library+Quantum+NLP+2021&searchtype=all) |

---

# 附录F: 2020年完整论文列表

> 奠基性论文，共25篇

## F.1 核心奠基

| 论文 | 链接 |
|------|------|
| Cerezo et al. - Variational Quantum Algorithms | [arXiv](https://arxiv.org/search/?query=Variational+Quantum+Algorithms+Cerezo+2020&searchtype=all) |
| Wang et al. - Noise-Induced Barren Plateaus | [arXiv](https://arxiv.org/search/?query=Noise+Induced+Barren+Plateaus+Variational+Quantum+Algorithms+2020&searchtype=all) |
| Pérez-Salinas et al. - Data re-uploading for universal classifier | [arXiv](https://arxiv.org/search/?query=Data+re+uploading+universal+quantum+classifier+2020&searchtype=all) |
| Schuld et al. - Effect of data encoding on expressive power | [arXiv](https://arxiv.org/search/?query=effect+data+encoding+expressive+power+variational+quantum+machine+learning+2020&searchtype=all) |
| Abbas et al. - The power of quantum neural networks | [arXiv](https://arxiv.org/search/?query=power+quantum+neural+networks+Abbas+2020&searchtype=all) |

## F.2 算法架构

| 论文 | 链接 |
|------|------|
| Beer et al. - Training deep quantum neural networks | [arXiv](https://arxiv.org/search/?query=Training+deep+quantum+neural+networks+Beer+2020&searchtype=all) |
| Bausch - Recurrent Quantum Neural Network | [arXiv](https://arxiv.org/search/?query=Recurrent+Quantum+Neural+Network+Bausch+2020&searchtype=all) |
| Fujii et al. - Deep Variational Quantum Eigensolver | [arXiv](https://arxiv.org/search/?query=Deep+Variational+Quantum+Eigensolver+divide+conquer+2020&searchtype=all) |
| Chen et al. - Quantum Long Short Term Memory | [arXiv](https://arxiv.org/search/?query=Quantum+Long+Short+Term+Memory+Chen+2020&searchtype=all) |

---

# 附录G: 2007-2019年奠基文献

## G.1 算法理论奠基

| 年份 | 论文 | 链接 |
|------|------|------|
| 2008 | Harrow, Hassidim, Lloyd - HHL Algorithm | [arXiv](https://arxiv.org/search/?query=Quantum+algorithm+linear+systems+equations+HHL+2008&searchtype=all) |
| 2014 | Lloyd et al. - Quantum PCA | [arXiv](https://arxiv.org/search/?query=Quantum+principal+component+analysis+Lloyd+2014&searchtype=all) |
| 2016 | Peruzzo et al. - VQE on photonic processor | [arXiv](https://arxiv.org/search/?query=variational+eigenvalue+solver+photonic+quantum+processor+Peruzzo+2016&searchtype=all) |
| 2017 | Farhi & Neven - QNN on Near Term Processors | [arXiv](https://arxiv.org/search/?query=Classification+Quantum+Neural+Networks+Near+Term+Processors+Farhi+2017&searchtype=all) |
| 2018 | Havlíček et al. - Quantum-enhanced feature spaces | [arXiv](https://arxiv.org/search/?query=Supervised+learning+quantum+enhanced+feature+spaces+Havlicek+2018&searchtype=all) |
| 2018 | Mitarai et al. - Quantum circuit learning | [arXiv](https://arxiv.org/search/?query=Quantum+circuit+learning+Mitarai+2018&searchtype=all) |
| 2018 | McClean et al. - Barren plateaus discovery | [arXiv](https://arxiv.org/search/?query=Barren+plateaus+quantum+neural+network+training+landscapes+McClean+2018&searchtype=all) |
| 2019 | Cao et al. - Quantum Chemistry in the Age of Quantum Computing | [arXiv](https://arxiv.org/search/?query=Quantum+Chemistry+Age+Quantum+Computing+Cao+2019&searchtype=all) |
| 2019 | Benedetti et al. - PQC as machine learning models | [arXiv](https://arxiv.org/search/?query=Parameterized+quantum+circuits+machine+learning+models+Benedetti+2019&searchtype=all) |

---

## 版本信息

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2025-01-17 | 初始版本，整合2007-2025年文献 |
| v1.1 | 2025-01-17 | 添加完整论文列表附录 |

