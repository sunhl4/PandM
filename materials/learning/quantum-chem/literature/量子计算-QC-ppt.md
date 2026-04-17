# 量子计算在计算化学领域的应用

2026-04  
汇报人：G1 组-孙宏亮  

---

## 1. 计算化学为什么值得做量子计算

分子电子结构问题的二次量子化哈密顿量（Second Quantization）：

$$
\hat{H} = \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} g_{pqrs} a_p^\dagger a_q^\dagger a_s a_r
$$

相关能定义：

$$
E_{\mathrm{corr}} = E_{\mathrm{exact}} - E_{\mathrm{HF}}
$$

- 经典计算的瓶颈不是"小分子完全算不动"，而是**强关联、多参考、近简并、过渡金属活性中心、自旋竞争**等场景。[1][2]
- 这些场景正是催化、激发态、断键和材料缺陷问题中最难且最有产业价值的部分。[1][3]

> 关键点：DFT 的系统性偏差恰好集中在量子计算最有优势的区域。

---

## 2. 为什么离子阱硬件适合切入

| 特性 | 离子阱 | 超导 | 化学任务需求 |
|---|---|---|---|
| **连通拓扑** | 全连通 | 近邻 | 非局域费米子激发 |
| **双门保真度** | > 99.5% | ~99% | 深线路变分优化 |
| **读出误差** | < 1% | ~1-2% | 测量密集型算法（SQD） |
| **相干时间** | 秒级 | 微秒级 | 长迭代变分序列 |

对于少量子比特规模，更现实的目标是：

- 小活性空间的高可信原型验证
- 嵌入式量子求解器（DMET+SQD）
- 算法/硬件联合 benchmark 服务

这一路线与计算化学行业落地逻辑一致：先解决最难的局部电子相关问题，再与经典工作流拼接。[1][3][4]

---

## 3. 量子计算化学方法全景图（2026）

| 方法族 | 代表方法 | 典型化学任务 | 近中期判断 |
|---|---|---|---|
| 变分基态算法 | `VQE`, `UCCSD`, `HEA`, `k-UpCCGSD`, `ADAPT-VQE`, `OO-VQE` | 基态能量、解离曲线、强关联小体系 | 高 |
| 激发态 / 光谱 | `VQD`, `QSE`, `qEOM`, `SSVQE` | 激发态、吸收发射、圆锥交叉 | 中 |
| 时间演化 / 动力学 | `Trotter`, `QITE`, `VarQITE`, real-time simulation | 反应动力学、非绝热过程、振电耦合 | 中 |
| 子空间 / 量子中心 | `SQD`, `SKQD`, quantum Krylov | 谱性质、采样恢复能量、低深度求解 | 高 |
| 嵌入 / 降阶 | `DMET`, `AVAS`, `CAS-DMET`, `TC-VQE`, downfolding | 催化位点、材料缺陷、局域强关联 | 高 |
| 容错电子结构 | `QPE`, qubitization, `LCU` | FCI 级精度、资源评估 | 中长期高 |
| 误差缓解 / 测量 | `ZNE`, `PEC`, `CDR`, classical shadows | 噪声机可信计算、测量降本 | 高 |
| QML for chemistry | `QSVM`, `QNN`, kernels | 性质预测、AI 力场、筛选 | 中高 |

说明：对当前硬件最现实的主线仍是"活性空间 + 变分/子空间方法 + 经典嵌入"。[1][2][3]

---

## 4. 从电子结构到量子线路

标准工作流：

1. 经典端生成积分与轨道（PySCF / VASP / Wannier90）
2. 选定活性空间（AVAS / IAO 局域化）
3. 构造费米子哈密顿量
4. 做费米子到量子比特映射（JW / BK / Parity + tapering）
5. 在量子硬件上求解（VQE / ADAPT-VQE / SQD）
6. 与经典结果对照并回写工作流

在工程上，真正决定能否落地的不是单一算法，而是整条工作流是否可复现、可 benchmark、可解释。[1][2]

---

## 5. Jordan-Wigner、Bravyi-Kitaev 与 Parity 映射

Jordan-Wigner (`JW`) 是最直接的费米子到量子比特映射；Bravyi-Kitaev (`BK`) 在更新与奇偶信息编码上做折中；Parity 映射更便于与对称性约减结合。[5][6][7]

$$
a_j^\dagger =
\left(\prod_{k=0}^{j-1} Z_k\right)\frac{X_j - iY_j}{2},
\qquad
a_j =
\left(\prod_{k=0}^{j-1} Z_k\right)\frac{X_j + iY_j}{2}
$$

| 映射 | 特点 | 适用语境 |
|---|---|---|
| `JW` | 最直观，易解释与调试 | 教学、首版验证 |
| `BK` | 平衡局域性与更新成本 | 近端量子化学 |
| `Parity + tapering` | 易结合守恒量减比特 | 资源受限硬件 |

---

## 6. `VQE`：Variational Quantum Eigensolver

`VQE`（变分量子本征求解器）最小化参数化量子态的能量期望值：[8][9]

$$
E(\theta) = \min_{\theta}\, \frac{\langle \Phi_0 | U^\dagger(\theta)\, \hat{H}\, U(\theta) | \Phi_0 \rangle}{\langle \Phi_0 | U^\dagger(\theta)\, U(\theta) | \Phi_0 \rangle}
$$

**UCCSD Ansatz**（Unitary Coupled Cluster Singles and Doubles）：

$$
U(\theta) = \exp\!\bigl(\hat{T}(\theta) - \hat{T}^\dagger(\theta)\bigr), \quad \hat{T} = \hat{T}_1 + \hat{T}_2
$$

常见 ansatz 对比：

| Ansatz | 可解释性 | 线路深度 | 适用场景 |
|---|---|---|---|
| `UCCSD` | 强，化学意义明确 | 深 | 精度优先 |
| `HEA` | 弱 | 浅 | 硬件友好 |
| `k-UpCCGSD` | 中 | 中 | 深度-精度折中 |
| `OO-VQE` | 强 | 深 | 同时优化轨道与参数 |

---

## 7. `ADAPT-VQE`：自适应变分路线

`ADAPT-VQE`（Adaptive Derivative-Assembled Pseudo-Trotter VQE）通过逐步从算符池中加入最重要的激发算符，缓解固定 ansatz 带来的冗余深度问题。[10]

每轮按**对易子梯度**选取算符：

$$
g_i = \left|\langle \psi | [\hat{H},\, \hat{A}_i] | \psi \rangle\right|
$$

选取 $g_i$ 最大的算符 $\hat{A}_i$ 加入 ansatz，直到梯度收敛。

对化学应用的意义：

- 更贴近问题结构，往往比盲目堆叠 `HEA` 更节省门数
- 适合过渡金属活性位这类需要"按需长大"的问题
- `qubit-ADAPT` 可直接在量子比特算符池中构造 ansatz
- 相干时间受限的离子阱上，门数节省直接转化为精度提升

---

## 8. 激发态与动力学方法不能忽略

除了基态，量子化学还关心激发态、光谱和动力学：

- `VQD`：Variational Quantum Deflation，用于逐个求低激发态
- `QSE`：Quantum Subspace Expansion，在基态附近扩展子空间
- `qEOM`：quantum Equation-of-Motion，把经典 EOM 思路搬到量子态上
- `SSVQE`：Subspace-Search VQE，同时优化多个正交态
- `QITE / VarQITE`：Quantum Imaginary Time Evolution / Variational QITE
- real-time simulation：用于反应动力学、振电耦合、非绝热过程

​	激发态与动力学已经是重要主线。[1][2][3]

---

## 9. `SQD / SKQD`：值得重点关注的量子中心路线

近期更值得重视的是 sample-based / subspace-based 路线。[11][12]

**SQD（Sample-Based Quantum Diagonalization）**：在采样组态子空间 $\mathcal{V} = \{|c_i\rangle\}$ 内解广义本征值问题：

$$
\sum_j H_{ij}\, v_j = E \sum_j S_{ij}\, v_j, \quad H_{ij} = \langle c_i | \hat{H} | c_j \rangle
$$

**SKQD（Sample-Based Krylov Quantum Diagonalization）**：引入时间演化态族，系统性捕捉激发态信息：
$$
\mathcal{K} = \bigl\{\, e^{-i\hat{H}t_n}\,|\psi_0\rangle \,\bigr\}, \quad |\psi(t_n)\rangle \approx \sum_k \alpha_k\, |c_k\rangle
$$

现实意义：

- 不要求过深线路，更容易与误差缓解和经典后处理结合
- 很适合作为 少比特硬件的化学 benchmark 主线

---

## 10. 嵌入式量子化学是更现实的产业接口

**DMET（Density Matrix Embedding Theory）**——Schmidt 分解将环境压缩为 Bath 轨道：
$$
|\Psi\rangle = \sum_i \lambda_i\, |f_i\rangle_{\mathrm{frag}} \otimes |b_i\rangle_{\mathrm{bath}}
$$

核心：将 100+ 轨道问题压缩为 4～8 轨道量子子问题。[13]

**AVAS（Automated Valence Active Space）**——基于原子轨道投影算符 $\hat{P}$ 自动选择活性空间：
$$
\mathbf{D}_{\mathrm{active}} = \mathbf{P}\, \mathbf{D}_{\mathrm{total}}\, \mathbf{P}
$$

适合过渡金属催化中心（如 Fe-N4）的轨道筛选。[14]

其他嵌入形式：`CAS-DMET`、downfolding、Projector embedding

这条路线与异相催化尤其契合：大体系主体仍用 `DFT / HF / Wannier / 局域化`，最关键的金属中心、吸附位、自旋竞争片段交给量子求解。[15]

![多尺度量子-经典分区示意图（Fe-N4 活性位）](heterogeneous_catalysis_quantum_classical.png)

> 三层分区策略：最内层（橙色）= 强关联量子区域（DMET+SQD/VQE，~10 qubits）；中间层（蓝色）= 经典 DFT/HF 环境；外层（灰色）= 分子力学/力场。右侧为 Fe 3d 活性空间对应的 10 量子比特线路示意——与公司当前硬件规模直接对应。

---

## 11. 面向公司当前硬件的优先发展方向

建议按"四层递进"来布局：

1. **量子化学 benchmark 层**  
   `H2 / LiH / BeH2 / 小活性空间 N2`，建立硬件可重复基线

2. **嵌入式催化原型层**  
   单活性位、单吸附物、单反应步，验证 `DMET / AVAS + VQE / SQD`

3. **误差与编译协同层**  
   把映射、tapering、编译、测量分组、误差缓解串成统一流程(不懂)

4. **行业解决方案层**  
   向云服务端向客户提供"化学 benchmark 服务 + 场景原型"

这比直接追求大体系全量子求解更符合当前资源条件。[2][3][4][11]

---

## 12. `dft_qc_pipeline` 架构详解

项目目标：DFT/HF + 嵌入 + 局部哈密顿量 + 量子求解器

**模块化工作流**：

```
YAML Config → PySCF Backend (HF/DFT) → DMET/AVAS 嵌入
           → Hamiltonian Builder (JW/Parity mapping)
           → 量子求解器 (VQE / ADAPT-VQE / SQD)
           → 后处理 (1-RDM / 片段能量 / ML export)
```

**核心工程特性**：

- **注册表机制**：`@registry.register("ion_trap")` 一行接入新硬件后端
- **自动活性空间**：集成 AVAS 与 IAO 局域化。
- **多片段自洽**：支持 DMET 1-RDM 匹配循环（Schmidt bath 自洽迭代）
- **VASP 接口**：预留 Wannier90 stub，可提取周期性体系局域轨道

工程依据：`dft_qc_pipeline/README.md`，`examples/03_FeN4_DMET_SQD.ipynb`

---
## 13. `quantum_chem_bench`：H₂ 全方法 benchmark 结果

H₂ (STO-3G, R = 0.735 Å) 全方法 benchmark——经典 HF→FCI 与五种量子求解器能量误差对比（对数坐标，参考化学精度线 1.59 mHa = 1 kcal/mol）。

![H₂ All Methods Energy Error](fig1_h2_all_methods.png)

**严格溯源数据表**（PySCF 2.12.1 独立验证，`H 0 0 0; H 0 0 0.735 Å`，STO-3G 基组）：

| 方法 | 能量 (Ha) | 误差 vs FCI (mHa) | 数据来源 |
|---|---:|---:|---|
| `HF` | −1.1169989968 | +20.3070 | PySCF `scf.RHF` |
| `MP2` | −1.1300208767 | +7.2852 | PySCF `mp.MP2` |
| `CISD` | −1.1373060358 | **0.0000** | PySCF `ci.CISD`（2e 体系精确等价 FCI）|
| `CCSD` | −1.1373061934 | −0.0002 | PySCF `cc.CCSD`（非变分，略低于 FCI）|
| `FCI` | −1.1373060358 | 0.0000 | PySCF `fci.FCI` |
| `VQE-UCCSD` | −1.1373060358 | 0.0000 | Qiskit statevector + SLSQP |
| `VQE-HEA` | −1.1373060347 | +0.0000011 | Qiskit statevector + COBYLA |
| `ADAPT-VQE` | −1.1373060358 | 0.0000 | Qiskit AdaptVQE（fallback UCCSD）|
| `QPE (ideal)` | −1.1373060358 | 0.0000 | Qiskit QPE 理想极限（精确对角化）|
| `SQD` | −1.1373060358 | 0.0000 | qiskit-addon-sqd 0.12.x，5000 shots |

**配置溯源**：`quantum_chem_bench/configs/h2_sto3g.yaml`，parity mapper + Z2 对称性约减（4 自旋轨道 → 2 qubit）。[8][9][10][11]

---

## 14. `quantum_chem_bench`：LiH 势能面基准

LiH (STO-3G，4e/4o 活性空间，4 qubits)——经典与量子方法精度-成本对比。

> **图待补充**：运行 `quantum_chem_bench/examples/02_LiH_benchmark.ipynb` 。

工程依据：`quantum_chem_bench/examples/02_LiH_benchmark.ipynb`

---

## 15. `dft_qc_pipeline`：Fe-N4 催化活性位案例

Fe-N4 单原子催化剂（Fe 3d 活性空间，5 轨道 6 电子）——方法精度对比。

- DFT 方法在自旋态预测上容易出错，DMET+SQD 可给出可靠自旋基态

工程入口：`dft_qc_pipeline/examples/03_FeN4_DMET_SQD.ipynb`。[13][14]

---

## 16. 应用：多相催化量子增强工作流

将量子计算与 VASP/PySCF 经典工作流结合，解决经典 DFT 最大误差来源：

**完整流程**：

1. **VASP**：周期性 DFT 计算表面吸附结构与初始电子密度
2. **Wannier90**：提取活性位局域 Wannier 函数（d 轨道）
3. **DMET / AVAS**：构建嵌入哈密顿量，压缩为量子子问题
4. **SKQD（离子阱）**：求解强关联局域能级与 1-RDM
5. **回写**：将量子精度 1-RDM 反馈至催化机理判断或参数校准

**核心价值**：解决多相催化中的**自旋交叉**与**过渡态近简并**问题——这是目前经典 VASP 计算最大的系统性误差来源。

- 不需要整体系上量子，只需最关键的金属中心片段

[3][4][13][15]

---

## 17. 应用：量子数据驱动的 AI 

量子计算机不只是求解器，更是**高价值量子数据发生器**：

$$
\mathcal{D}_\mathrm{quantum} = \bigl\{\, (\mathbf{R}_i,\; E_i,\; \nabla E_i,\; \rho_i) \,\bigr\}
$$

其中 $\rho_i$ 为 SQD 产出的 1-RDM，包含经典方法无法给出的多参考关联信息。

**应用路径**：

- **数据增强**：用量子精度修正 DFT 训练集，训练更准确的神经网络势（NNP）
- **量子核方法（QML）**：利用量子特征映射定义分子相似度核，用于催化活性筛选
- **混合势能面**：量子求解强关联区域，经典方法覆盖大范围构型空间

**战略价值**：

- 量子数据是稀缺且高价值的资产，先发布局可形成数据壁垒
- 与 AI 结合后，单次量子计算的价值被放大数量级

---

## 18. 公司落地路线建议

**近期具体抓手**：

- 活性位嵌入 + `SQD / ADAPT-VQE` + 小规模 benchmark + 实机验证
- 构建 `dft_qc_pipeline` 和 `quantum_chem_bench` 的硬件接入口
- 推进 SKQD / QITE / real-time simulation 等算法预研

---

## 19. 结论

### 当前阶段最现实的价值

量子计算在化学中的现实价值，不在于立刻全面替代经典，而在于：

1. **聚焦强关联局域问题**（DMET + AVAS + 活性位）
2. **构建嵌入式混合量子经典工作流**（VASP → Wannier → DMET → SQD）
3. **沉淀可被 AI 放大的高价值量子数据**（量子精度 1-RDM 训练集）

### 对当前公司最值得优先投入的两条主线

- **主线 A：强关联活性位建模**（Fe-N4 / FeMoco 类问题）
- **主线 B：量子化学 benchmark 与量子数据能力**（量子精度数据资产化）

---

## 20. 参考文献

[1] Sam McArdle et al. Quantum computational chemistry. *Rev. Mod. Phys.* **92**, 015003 (2020).

[2] Jules Tilly et al. The Variational Quantum Eigensolver: a review of methods and best practices. *Phys. Rep.* **986**, 1 (2022).

[3] Yudong Cao et al. Quantum Chemistry in the Age of Quantum Computing. *Chem. Rev.* **119**, 10856 (2019).

[4] Colin D. Bruzewicz et al. Trapped-Ion Quantum Computing: Progress and Challenges. *Appl. Phys. Rev.* **6**, 021314 (2019).

[5] P. Jordan, E. Wigner. Uber das Paulische Aquivalenzverbot. *Z. Phys.* **47**, 631 (1928).

[6] J. T. Seeley, M. J. Richard, P. J. Love. The Bravyi-Kitaev transformation for quantum computation of electronic structure. *J. Chem. Phys.* **137**, 224109 (2012).

[7] S. Bravyi et al. Tapering off qubits to simulate fermionic Hamiltonians. arXiv:1701.08213 (2017).

[8] A. Peruzzo et al. A variational eigenvalue solver on a photonic quantum processor. *Nat. Commun.* **5**, 4213 (2014).

[9] J. Romero et al. Strategies for quantum computing molecular energies using the unitary coupled cluster ansatz. *Quantum Sci. Technol.* **4**, 014008 (2019).

[10] H. R. Grimsley et al. An adaptive variational algorithm for exact molecular simulations on a quantum computer. *Nat. Commun.* **10**, 3007 (2019).

[11] J. Robledo-Moreno et al. Chemistry beyond the scale of exact diagonalization on a quantum-centric supercomputer. *Science Advances* **11**, eadu9991 (2025).

[12] J. Yu et al. Quantum-Centric Algorithm for Sample-Based Krylov Diagonalization. arXiv:2501.09702 (2025).

[13] G. Knizia, G. K.-L. Chan. Density Matrix Embedding: A Simple Alternative to Dynamical Mean-Field Theory. *Phys. Rev. Lett.* **109**, 186404 (2012).

[14] E. R. Sayfutyarova et al. Automated Construction of Molecular Active Spaces from Atomic Valence Orbitals. *J. Chem. Theory Comput.* **13**, 4063 (2017).

[15] S. Battaglia et al. A general framework for active space embedding methods with applications in quantum computing. *npj Comput. Mater.* **10**, 297 (2024).

[16] W. Dobrautz et al. Toward Real Chemical Accuracy on Current Quantum Hardware Through the Transcorrelated Method. *J. Chem. Theory Comput.* **20**, 4146 (2024).

[17] C. Di Paola et al. Platinum-based catalysts for oxygen reduction reaction simulated with a quantum computer. *npj Comput. Mater.* **10**, 285 (2024).
