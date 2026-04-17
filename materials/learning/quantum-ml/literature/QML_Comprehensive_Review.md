# 量子机器学习研究综述：从理论基础到前沿进展（详细版）

> 基于 Zhao et al. arXiv:2604.07639 (2026) 及相关文献的系统性深度梳理

---

## 目录

1. [摘要](#摘要)
2. [**第一章　核心论文完整技术解析：Zhao et al. (2026)**](#第一章)
   - [1.1 研究背景与问题的精确定位](#第一章)
   - [1.2 数据模型：分层数据生成过程（Section C）](#第一章)
   - [1.3 主定理精确陈述（Section II）](#第一章)
   - [1.4 量子 Oracle 草图：核心算法（Section D）](#第一章)
   - [1.5 干涉型经典影子（Interferometric Classical Shadows）](#第一章)
   - [1.6 经典难度证明：NOPE 框架（Section E）](#第一章)
   - [1.7 三类应用的完整量子算法（Section F）](#第一章)
   - [1.8 数值实验详细分析（Appendix A）](#第一章)
   - [1.9 论文的理论意义与开放问题（Section IV）](#第一章)
   - [1.10 量子算法总览：从 Oracle 草图到完整 ML 流水线](#第一章)
3. [第二章 量子线性代数（QLAS）路线：从 HHL 到 QSVT](#第二章)
4. [第三章 变分量子算法（VQA）与 Barren Plateaus](#第三章)
5. [第四章 量子核方法与去量子化](#第四章)
6. [第五章 量子学习理论：优势的严格证明体系](#第五章)
7. [第六章 经典影子与量子态层析](#第六章)
8. [第七章 量子流式算法与空间复杂度分离](#第七章)
9. [第八章 应用领域、生成模型与前沿方向](#第八章)
10. [第九章 核心开放问题与研究展望](#第九章)
11. [第十章 量子随机游走、LCU 与量子蒙特卡洛](#第十章)
12. [第十一章 量子优化算法深度解析](#第十一章)
13. [第十二章 量子误差缓解技术全景](#第十二章)
14. [第十三章 量子自然语言处理（QNLP）](#第十三章)
15. [第十四章 量子金融与量子强化学习](#第十四章)
16. [第十五章 等变量子网络与几何量子机器学习](#第十五章)
17. [第十六章 量子硬件与工程实现路径](#第十六章)
18. [参考文献](#参考文献)

---

<a name="摘要"></a>
## 摘要

本文以 Zhao、Zlokapa、Neven、Babbush、Preskill、McClean 和 Huang 于 2026 年发表的工作《Exponential quantum advantage in processing massive classical data》(arXiv:2604.07639) 为核心，系统梳理量子机器学习（QML）领域自 2009 年至今的主要研究脉络。

全文涵盖：量子线性代数（QLAS）路线、变分量子算法（VQA）与 Barren Plateaus、量子核方法与去量子化、量子学习理论与严格优势证明、经典影子与量子态层析、量子流式算法与空间复杂度，以及各应用领域的前沿进展。

重点阐明本文（Zhao et al. 2026）在「经典数据流式处理 + 空间有界量子机」模型下，如何通过**量子 Oracle 草图**（Quantum Oracle Sketching）与**干涉型经典影子**（Interferometric Classical Shadows）两大技术，在不依赖 QRAM 和计算复杂性猜想的前提下，证明了对海量经典数据进行分类、降维和线性系统求解时存在的**指数级量子空间优势**，并给出了在 IMDb 情感分析与单细胞 RNA 测序数据上的数值验证。

本版本在原版基础上大幅扩写，每一核心技术均包含：思想直觉、数学原理、关键公式推导、电路/伪代码实现、与相关工作的联系。

---

<a name="第一章"></a>
## 第一章　核心论文完整技术解析：Zhao et al. (2026)

> **论文全称**：Exponential quantum advantage in processing massive classical data  
> **作者**：Haimeng Zhao, Alexander Zlokapa, Hartmut Neven, Ryan Babbush, John Preskill, Jarrod R. McClean, Hsin-Yuan Huang  
> **机构**：Caltech / Google Quantum AI / MIT / Oratomic  
> **发表**：arXiv:2604.07639v1, 2026 年 4 月 8 日  
> **篇幅**：144 页，包含主文（Section I–IV）及附录 A–F

本章对这篇论文进行**完整、不简化**的逐节技术解析，覆盖所有 144 页的核心内容，包括：严格的数学定义、定理的精确陈述、证明思路与关键步骤、电路构造与算法伪代码，以及与现有文献的关系。

---

### 1.1　研究背景与问题的精确定位

#### 1.1.1　量子优势的三重障碍

量子计算的「有用优势」长期以来局限于密码分析（Shor 算法 [Shor 1994]）和量子模拟（Lloyd 1996）两个领域。对于更广泛的**经典数据处理与机器学习**场景，障碍来自三个相互关联的技术问题：

**障碍一：数据加载瓶颈（Data Loading Bottleneck）**

量子算法（如 HHL [2009]、量子 PCA [Lloyd et al. 2014]、量子推荐系统 [Kerenidis-Prakash 2017]）的加速依赖于对数据的**叠加查询**：

$$O_f: |x\rangle \to (-1)^{f(x)}|x\rangle \quad \text{或} \quad O_A: |i\rangle|j\rangle|0\rangle \to |i\rangle|j\rangle|A_{ij}\rangle$$

而现实世界只提供**逐一经典采样** $z_t \sim \mathcal{D}$，无法直接构造上述叠加查询。

**障碍二：QRAM 的规模代价（QRAM Overhead）**

解决方案之一是量子随机访问存储器（QRAM）。物理实现上最自然的是**桶形路由树**（bucket-brigade tree, Giovannetti-Lloyd-Maccone 2008 [Ref 13]）：
- 一棵深度为 $\lceil \log_2 N \rceil$ 的二叉树，每个内部节点是三态量子路由器（left/right/wait）；
- $N$ 个叶节点各存一个数据项；
- 单次查询电路深度 $O(\log N)$，但需维护 $O(N)$ 个路由器的量子相干性；
- 容错实现的物理量子比特数：$O(N \cdot \mathrm{polylog}(N))$（每个路由器需要纠错）；
- Jaques & Rattew（Quantum, 2025 [Ref 16]）的 resource accounting 表明：维护 QRAM 所需的经典控制基础设施（$O(N)$ 个经典处理器）通常已足以直接经典并行求解目标问题。

**障碍三：去量子化（Dequantization）**

Tang（2019–2023 [Ref 29–31]）揭示了上述量子 ML 算法的加速来自**采样查询（Sample-and-Query，SQ）模型**这一强输入假设，等价于 QRAM：

> **SQ oracle**：对向量 $v \in \mathbb{R}^n$，允许：
> (a) 以概率 $v_i^2 / \|v\|^2$ 采样下标 $i$；  
> (b) 查询 $\|v\|^2$ 的近似值；  
> (c) 给定 $i$，查询 $v_i$ 的精确值。

在此模型下，Tang 设计的经典随机算法（quantum-inspired algorithms）可在 $O(\mathrm{poly}(n/\varepsilon))$ 时间内完成 HHL、PCA、推荐等任务，将量子优势从指数降至多项式。共同的原因：两类方法都隐含 $\Omega(N)$ 的空间开销（QRAM 节点数或 SQ 预处理数据结构）。

**障碍四：读出瓶颈（Readout Bottleneck）**

即使量子算法制备了 $|x\rangle \propto A^{-1}|b\rangle$，要提取解的 $N$ 个分量也需要 $\Omega(N/\varepsilon^2)$ 次测量（Holevo 界 [Holevo 1973 [Ref 63]]：$n$ 量子比特一次测量最多提取 $n$ 经典比特）。

#### 1.1.2　Zhao et al. (2026) 的核心创新：改变资源度量轴

本文的核心洞见是：从「计算时间」改为「**内存空间**」来衡量复杂度，在这一新维度下论证量子优势：

**新框架的精确定义**：

| 符号 | 含义 |
|:---:|:---|
| $N$ | 问题规模（线性系统维度、样本数或特征维度） |
| $M$ | 流式样本总数（sample complexity） |
| $S_Q$ | 量子机规模 = 量子工作寄存器的逻辑量子比特数 |
| $S_C$ | 经典机规模 = 经典工作内存的比特数（浮点数个数） |
| $\tau$ | 刷新时间（refreshing time）：超出此窗口的样本统计独立 |
| $R$ | 重复数（repetition number）：$\tau$ 窗口内任一数据条目的最大期望重复次数 |

**主框架定理（Theorem 7，见 1.7 节）**：
$$M_{\mathcal{C}} \cdot S_{\mathcal{C}} \geq \Omega(N \cdot Q_C) \quad \text{（对任意经典学习算法 $\mathcal{C}$）}$$

量子机用 $M_Q = \tilde{O}(NQ^2/\varepsilon)$ 样本、$S_Q = \mathrm{poly}(\log N)$ 空间完成任务；若 $Q_C/Q^2 \gg 1$，则经典机必须以更大的空间-样本乘积为代价。

#### 1.1.3　与去量子化的精确关系

去量子化依赖 SQ oracle（需要 $\Omega(N)$ 空间预处理）；而本文的流式模型：
- **无 SQ oracle**：每步只看一个采样 $(x_t, f(x_t))$，看完即丢；
- **硬内存上界**：$S_C = O(N^{0.99}) \ll N$，无法容纳 Tang 类算法需要的数据结构；
- 两者讨论的是**完全不同的资源假设**，而非相互推翻。

#### 1.1.4　如何理解 $S_C = O(N^{0.99})$ 与 Theorem 7 在说什么

**读法先明：这不是「经典机很弱」的稻草人设定**  
$S_C = O(N^{0.99})$ **并不是**在说：故意给经典机一个「可笑的小内存」所以它当然失败。恰恰相反：$N^{0.99}$ 随 $N$ 增长**极大**，与线性规模 $\Theta(N)$ 只相差因子 $N^{0.01}$，是**几乎线性**但仍**严格次线性**的内存上界——大到大体上「很像能装很多东西」，却**仍不足以**存放规模为 $\Omega(N)$ 的**完整**数据对象或 Tang 式 SQ 预处理结构（那需要真正的 $\Omega(N)$ 空间）。Theorem 7 一类结果的力度在于：在此设定下，即便允许流式样本数 $M$ **任意增大**（论文对应叙述为「无论多少样本仍无法完成」），在**信息论 / 通信复杂度**意义下经典流式机仍无法解决 NOPE/Forrelation 类任务；这才与仅需

$$
S_Q = \mathrm{poly}(\log N)
$$

的量子工作空间形成**尖锐分离**。要点是「流式观测的信息能否被压进这份**几乎**线性、却**装不下全集**的经典状态里」，而不是「经典侧内存小到不公平」。

**（1）符号 $S_C = O(N^{0.99})$ 的字面意思**  
$S_C$ 表示经典**工作内存**能长期保存的比特数（或等价地，分支程序每层的状态数上界 $2^{S_C}$）。  
记号

$$
S_C = O(N^{0.99})
$$

表示：存在常数 $C, N_0$，使得对所有足够大的 $N$，都有

$$
S_C \;\le\; C \cdot N^{0.99}.
$$

也就是说，内存随 $N$ 增长，但**比线性慢**：指数 $0.99 < 1$，故

$$
\frac{N^{0.99}}{N} = N^{-0.01} \xrightarrow[N\to\infty]{} 0,
$$

即 $N^{0.99}$ 渐近地远小于 $N$（记作 $N^{0.99} = o(N)$）。论文中更一般的写法常是 $N^{1-\zeta}$（任意小的 $\zeta > 0$）；取 $0.99$ 只是选一个**具体的** $1 - \zeta$（此处 $\zeta = 0.01$），便于叙述「几乎像 $\Theta(N)$ 那么大，但仍**装不下**规模为 $\Omega(N)$ 的全表或 SQ 结构」。

**（2）为什么要强调「小于 $N$」**  
若数据有 $\Theta(N)$ 个独立条目，**光把全集存进 RAM** 就需要 $\Omega(N)$ 比特（或单元）。因此：

- $S_C = O(N^{0.99})$ $\Rightarrow$ **不可能**在内存里保留完整数据或 Tang 式 SQ 预处理结构（通常要 $\Omega(N)$ 空间）；
- Theorem 7 进一步说明：在**流式**设定下，即便允许样本数 $M$ 很大，**只要 $S_C$ 仍保持这种次线性上界**，经典机仍无法完成 NOPE/Forrelation 类任务；而量子机可用 $S_Q = \mathrm{poly}(\log N)$ 的工作空间与 $\tilde{O}(N)$ 样本完成。对比的是**空间**与**流式信息论限制**，不是「经典时间一定比量子慢」。

**（3）Theorem 7 的直觉：双人通信画一张图**  
把长度为 $M$ 的流式样本**按奇偶位置切开**：

- 玩家 **Alice** 只看奇数时刻的样本，**Bob** 只看偶数时刻的样本；
- 两人各自只有**本地内存**（规模与 $S_C$ 相当），目标是合作算出原流式问题的答案。

若存在「$M$ 个样本、内存 $S_C$」的经典流式算法，则可改写为两人**多轮通信**的协议：每步把当前内存状态当作可传输的信息。**内存越小，每步能传给对方的新信息越少**；而 NOPE 类问题的**经典通信/查询下界**要求总通信量足够大。把这两边合起来，就得到形如

$$
M_{\mathcal{C}} \cdot S_{\mathcal{C}} \;\gtrsim\; N \cdot Q_C
$$

的**乘积下界**：样本少就必须内存大，内存小就必须样本极多（在相应任务上甚至仍不够）。$S_C = O(N^{0.99})$ 刻画的是「**渐近上仍低于**存全数据所需的 $\Omega(N)$，却已是 $o(N)$ 中相当『慷慨』的一档」——不是极小常数内存，而是**故意紧贴着**「差一点点就能装下全集」的边界，以凸显信息论分离而非欺负经典机。

---

### 1.2　数据模型：分层数据生成过程（Section C）

#### 1.2.1　流式机器的形式模型

**经典学习算法（Definition C.1）**：大小为 $S$ 的经典学习算法被定义为有向图，节点分为 $M+1$ 层（层 $0$ 到层 $M$）：
- 每层至多 $2^S$ 个节点（每个节点对应一个 $S$ 比特的内存状态）；
- 层 $0$ 只有一个根节点；
- 层 $M$ 的节点为叶节点，每个叶节点携带输出 $h_v$；
- 每个节点有 $|\mathcal{I}|$ 条出边，标记为所有可能的输入 $I \in \mathcal{I}$。

计算过程：从根节点出发，对每个时间步的输入 $I_t \in \mathcal{I}$ 跟随对应出边，最终到达叶节点输出结果。

等价于**分支程序**（branching program）——空间有界计算的最一般（非均匀）模型。

**量子学习算法（Definition C.2）**：大小为 $S$ 的量子学习算法由以下组成：
- 初始 $S$ 量子比特状态 $\rho_0$；
- $M$ 组量子信道集合 $(\mathcal{C}_0, \ldots, \mathcal{C}_{M-1})$，每组 $\mathcal{C}_i = \{C_i^I : I \in \mathcal{I}\}$（$S$ 量子比特上的信道）；
- 最终 POVM 测量 $\{M_h\}$（$\sum_h M_h = I$）。

计算过程：
$$\rho_M = C_{M-1}^{I_{M-1}} \circ \cdots \circ C_0^{I_0}(\rho_0), \quad \text{输出 } h \text{ 的概率为 } \mathrm{tr}(M_h \rho_M)$$

#### 1.2.2　分层数据生成过程（HDGP, Hierarchical Data Generation Process）

为精确刻画真实数据的时间相关性，定义：

**定义（HDGP, Section C.1）**：$l$ 层分层数据生成过程 $\mathcal{D}$ 由以下递归结构给出：
$$\mathcal{D} = (D_0 \xrightarrow{\alpha_1} \times T_1 \; D_1 \xrightarrow{\alpha_2} \times T_2 \; \cdots \xrightarrow{\alpha_l} \times T_l \; z)$$

- $D_0$ 是根情境（root situation）的分布；
- $\alpha_k$ 是第 $k$ 层情境，由 $D_{k-1}$ 抽取；
- 每个 $\alpha_k$ 重复使用 $T_k$ 步；
- 最终数据点 $z$ 在 $T_l \cdots T_1$ 步的刷新时间 $\tau_\mathcal{D} = T_l \cdots T_1$ 内以边际分布 $p(z)$ 采样。

**刷新时间 $\tau$** = 超出此时间窗口的样本在统计上近似独立。

**重复数 $R$**（Def C.3）：
$$R_\mathcal{D} = \max_{z \in \mathcal{Z}} \sum_{i=1}^{\tau_\mathcal{D}} \left( \Pr[z_i = z \mid z_1 = z] - \Pr[z_i = z] \right)$$

**Lemma C.1（共情境增强相关性）**：对任意两时刻 $i, j$ 和任意 $z$：
$$\Pr[z_j = z \mid z_i = z] \geq \Pr[z_j = z]$$

**证明思路**：设共享情境为 $\sigma$，条件独立性给出 $\Pr[z_j = z \mid z_i = z] = \mathbb{E}_\sigma[q_\sigma^2(z)] / p(z)$，再由 Jensen 不等式 $\mathbb{E}[q_\sigma^2] \geq (\mathbb{E}[q_\sigma])^2 = p(z)^2$ 得证。

**Lemma C.4（重复数界定方差）**：对任意 $z$，频率 $m_z = \frac{1}{M}\sum_{t=1}^M \mathbf{1}[z_t = z]$ 满足：
$$\mathrm{Var}[m_z] \leq \frac{p(z) R_\mathcal{D}}{M}$$

**证明**：将方差按相关深度 $K(i,j)$ 展开，利用 Lemma C.1 和 C.2 的单调性估计各协方差，求和得 $\mathrm{Var}[m_z] \leq p(z) R_\mathcal{D} / M$。这是整个算法分析的基础：它表明 $M = O(R_\mathcal{D} / p(z) \varepsilon^2)$ 样本足以将 $m_z$ 的方差控制在 $\varepsilon^2$ 内。

**Lemma C.5（重复数界定条件漂移）**：在已处理 $z_1, \ldots, z_{t-1}$ 的条件下，新一轮数据 $m_z(t)$ 的期望漂移满足：
$$\mathbb{E}_{z_1,\ldots,z_{t-1}}\left[\max_{z \in \mathcal{Z}} \left|\mathbb{E}[m_z(t) \mid z_1,\ldots,z_{t-1}] - \mathbb{E}[m_z(t)]\right|\right] \leq \frac{\sqrt{p_{\max} |\mathcal{Z}| R_\mathcal{D}}}{M}$$

这个引理控制了**偏差项**（bias term），是推广到相关数据时的关键。

#### 1.2.3　三类应用任务的数据模型

| 任务 | 样本格式 $z_t$ | 从何处均匀采样 |
|:---|:---|:---|
| 线性系统 $A\vec{x}=\vec{b}$ | $(i, j, A_{ij}, k, b_k)$ | $(i,j)$ 从非零元均匀采样；$k$ 从 $[N]$ 均匀采样 |
| 分类（LS-SVM） | $(i, \vec{x}_i, y_i)$ | $i$ 从 $[N]$ 均匀采样 |
| 降维（PCA） | $(i, \vec{x}_i)$ | $i$ 从 $[N]$ 均匀采样 |

---

### 1.3　主定理精确陈述（Section II）

本节给出所有主定理的精确量化陈述。

#### 1.3.1　线性系统（Theorem 1 & 2，Section F.1）

**问题设定**：设 $A \in \mathbb{R}^{N \times N}$ 是 $s$-稀疏（每行/列至多 $s$ 个非零元），条件数 $\kappa$；$\vec{b} \in \mathbb{R}^N$。设 $M \in \mathbb{R}^{N \times N}$ 是稀疏正定矩阵（「感兴趣的分量」矩阵，如 $M=I$ 时目标为 $\|\vec{x}^*\|^2$）。

**任务**：从流式样本中估计二次型
$$\Phi = \vec{x}^{*\top} M \vec{x}^* = \vec{b}^\top (A^\top A)^{-1} M (A^\top A)^{-1} \vec{b}$$
至 $\varepsilon$ 绝对误差。

**Theorem 1（静态线性系统）**：
- **量子算法**：$S_Q = \mathrm{poly}(\log N, s, \log \kappa, \log(1/\varepsilon))$ 个逻辑量子比特；使用 $M = \tilde{O}(N s^2 \kappa^4 / \varepsilon^2)$ 样本；以至少 $2/3$ 的概率输出 $\varepsilon$-近似值。
- **经典下界**：任何大小 $S_C = O(N^{0.99})$ 的经典算法无法完成此任务（即使给予无限时间）。

**Theorem 2（动态线性系统）**：设 $A, \vec{b}$ 按 HDGP 变化（刷新时间 $\tau = \tilde{O}(N)$，$R=O(1)$），但 $\Phi$ 近似不变（变化 $\leq \varepsilon$）：
- **量子算法**：同上，仍用 $\tilde{O}(N)$ 样本完成；
- **经典下界**：任何 $S_C = O(N^{0.99})$ 的经典算法需要超多项式 $\mathrm{superpoly}(N)$ 样本。

#### 1.3.2　二分类（Theorem 3 & 4，Section F.2）

**问题设定**：特征矩阵 $X \in \mathbb{R}^{N \times D}$（$N$ 个样本，$D$ 维特征），每行 $s$-稀疏；标签向量 $\vec{y} \in \{-1,+1\}^N$。设正则化后的数据矩阵
$$\tilde{X} = \begin{pmatrix} X \\ \sqrt{\lambda} I_D \end{pmatrix} \in \mathbb{R}^{(N+D) \times D}$$
条件数为 $\kappa$（由 $\ell_2$ 正则化保证有界）。

**LS-SVM 权重向量**：
$$\vec{w}^* = (X^\top X + \lambda I_D)^{-1} X^\top \vec{y} = \tilde{X}^\dagger \tilde{\vec{y}}, \quad \tilde{\vec{y}} = (\vec{y}^\top, \vec{0}^\top)^\top$$

其中 $\tilde{X}^\dagger$ 是 Moore-Penrose 伪逆。

**任务**：给定稀疏测试向量 $\vec{x}' \in \mathbb{R}^D$（$s'$-稀疏，间隔（margin）下界 $\gamma > 0$），预测符号 $\mathrm{sign}(\vec{x}' \cdot \vec{w}^*)$。

**Theorem 3（静态分类）**：
- **量子算法**：$S_Q = \mathrm{poly}(\log D, \log N, s, s', \log \kappa, \log(1/\gamma\varepsilon))$ 量子比特；$M = \tilde{O}(N s^2 \kappa^4 / \varepsilon^2)$ 样本；以至少 $2/3$ 概率输出正确预测。
- **经典下界**：任何 $S_C = O(D^{0.99})$ 的经典机（无论样本多少）无法完成此任务。

**Theorem 4（动态分类）**：训练集每 $\tau = \tilde{O}(N)$ 步刷新，但分类边界近似不变：经典机需 $\mathrm{superpoly}(N)$ 样本。

#### 1.3.3　主成分分析与降维（Theorem 5 & 6，Section F.3）

**问题设定**：数据矩阵 $X \in \mathbb{R}^{N \times D}$，每行 $s$-稀疏；协方差矩阵 $\Sigma = X^\top X / N$；谱间隔 $\Delta > 0$（最大最小特征值之差）；引导向量 $\vec{g} \in \mathbb{R}^D$ 满足 $|\langle \vec{g}, \vec{w}_1 \rangle| \geq \chi > 0$（$\vec{w}_1$ 是最大特征向量）。

**任务**：给定稀疏测试向量 $\vec{x}' \in \mathbb{R}^D$，估计低维表示 $\xi(\vec{x}') = \langle \vec{x}', \vec{w}_1 \rangle^2$ 至 $\varepsilon$ 误差。

**Theorem 5（静态 PCA）**：
- **量子算法**：$S_Q = \mathrm{poly}(\log D, \log N, s, s', 1/\chi, 1/\Delta, \log(1/\varepsilon))$ 量子比特；$M = \tilde{O}(N s^2 / (\chi^2 \Delta^2 \varepsilon^2))$ 样本；以至少 $2/3$ 概率成功。
- **经典下界**：任何 $S_C = O(D^{0.99})$ 的经典机（无论样本多少）无法完成此任务。

**Theorem 6（动态 PCA）**：同样可处理演化的协方差矩阵（主成分近似不变），经典机需 $\mathrm{superpoly}(N)$ 样本。

#### 1.3.4　空间分离的数量级（图 2 的实验验证）

| 数据集 | $D$（特征维度） | 量子机规模（逻辑比特） | 经典流式算法规模 | 空间优势 |
|:---|:---:|:---:|:---:|:---:|
| IMDb 情感分析 | $\approx 89{,}000$ | $\leq 55$ | $\geq 89{,}000$ 浮点数 | $> 10^4 \times$ |
| PBMC68k scRNA-seq PCA | $\approx 30{,}000$ | $\leq 50$ | $\geq 30{,}000$ 浮点数 | $> 10^4 \times$ |
| 20newsgroup 话题分析 | $\approx 50{,}000$ | $\leq 52$ | $\geq 50{,}000$ 浮点数 | $> 10^4 \times$ |
| Dorothea 药物发现 | $\approx 100{,}000$ | $\leq 57$ | $\geq 100{,}000$ 浮点数 | $> 10^6 \times$ |

量子机内存规模公式（具体见 Appendix A）：
$$S_Q^{\text{LS-SVM}} = 2\lceil\log_2(N+2D)\rceil + \lceil\log_2(s+1)\rceil + 4 \quad \text{（逻辑量子比特）}$$
$$S_Q^{\text{PCA}} = 2\lceil\log_2(N+D)\rceil + \lceil\log_2 s\rceil + 4 \quad \text{（逻辑量子比特）}$$

---

### 1.4　量子 Oracle 草图：核心算法（Section D）

#### 1.4.1　问题的直觉与困难

**目标**：从 $M$ 个流式样本 $(x_t, f(x_t))$ 中，构造近似相位 oracle $O_f: |x\rangle \to (-1)^{f(x)}|x\rangle$，以便代入任何量子查询算法。

**关键工具**：量子时间演化。若定义对角哈密顿量
$$H_f = \sum_{x \in [N]} f(x)|x\rangle\langle x|$$
则
$$e^{-iH_f\pi} = \sum_x e^{-i\pi f(x)}|x\rangle\langle x| = \sum_x (-1)^{f(x)}|x\rangle\langle x| = O_f$$

**困难**：$H_f$ 需要存储 $N$ 个系数——正是我们要避免的。

**朴素方法（为何失败）**：接收到样本 $(x_t, f(x_t))$ 时，施加对应的随机哈密顿量 $h_{x_t} = f(x_t)|x_t\rangle\langle x_t|$。总演化为：
$$V_{\text{naive}} = \exp\!\left(i\sum_t h_{x_t}/M\right)$$

问题：随机哈密顿量模拟（Randomized Hamiltonian Simulation）的分析（Campbell 2019, Chen-Huang-Kueng-Tropp 2021 [Ref 85,86]）表明，一般情形的误差为：
$$\varepsilon_{\text{naive}} \sim \frac{N^2}{M}$$

要使 $\varepsilon \leq \varepsilon_0$，需要 $M \geq N^2/\varepsilon_0$——这消耗的样本使量子优势全无（单次 oracle 查询就需 $N^2$ 样本，而经典算法只需 $O(1)$）。

#### 1.4.2　规避去相干的关键洞见：正交子空间

**Theorem D.12（相位 oracle 草图，IID 情形）**：设 $f: [N] \to \{0,1\}$，样本 $z_t = (x_t, f(x_t))$ IID 从均匀分布 $p(x) = 1/N$ 采样。对每个样本施加：
$$V_t = \exp\!\left(i\frac{\pi N}{M} f(x_t) |x_t\rangle\langle x_t|\right)$$

则 $M \geq 2N/\varepsilon$ 样本足以保证：
$$\left\|\mathbb{E}[V_M \cdots V_1] - O_f\right\|_\diamond \leq \varepsilon$$

**证明的关键**：

**步骤 1（乘积展开）**：由于 $\{|x\rangle\langle x|\}_{x \in [N]}$ 是相互正交的投影算符，所有 $V_t$ **对易**：
$$V_M \cdots V_1 = \exp\!\left(i\pi N \sum_t f(x_t)|x_t\rangle\langle x_t|/M\right) = \sum_{x=1}^N \exp\!\left(i\pi N m_x f(x)\right)|x\rangle\langle x|$$

其中 $m_x = \frac{1}{M}\sum_t \mathbf{1}[x_t = x]$ 是样本 $x$ 的经验频率。

**步骤 2（集中不等式）**：由大数定律，$m_x \to p(x) = 1/N$，故 $\pi N m_x \to \pi$，即：
$$\mathbb{E}[V_M \cdots V_1] = \sum_x \exp(i\pi N \mathbb{E}[m_x] f(x))|x\rangle\langle x| = \sum_x (-1)^{f(x)}|x\rangle\langle x| = O_f$$

这是期望的目标算符。

**步骤 3（误差分析，核心引理 D.7）**：对随机对角酉算符 $e^{iX}$（$X = \sum_x \tau_x |x\rangle\langle x|$，$\tau_x$ 是随机变量），有：
$$\left\|e^{i\mathbb{E}[X]} - \mathbb{E}[e^{iX}]\right\| \leq \frac{1}{2}\max_x \mathrm{Var}[\tau_x]$$

**证明**：利用酉矩阵的 spectral norm 和对角情形的逐分量估计：
$$\left|e^{i\mathbb{E}[\tau_x]} - \mathbb{E}[e^{i\tau_x}]\right| = \left|\mathbb{E}[e^{i\tau_x} - e^{i\mathbb{E}[\tau_x]}]\right| \leq \frac{1}{2}\mathrm{Var}[\tau_x]$$
（用 $|e^{ia} - e^{ib}| \leq |a-b|$ 和 Jensen 不等式）。

**步骤 4（代入方差上界）**：$\tau_x = \pi N m_x f(x)$，故 $\mathrm{Var}[\tau_x] = \pi^2 N^2 f(x)^2 \mathrm{Var}[m_x]$。由 Bernoulli 分布：$\mathrm{Var}[m_x] = \frac{p(x)(1-p(x))}{M} \leq \frac{p(x)}{M} = \frac{1}{NM}$。

故 $\mathrm{Var}[\tau_x] \leq \pi^2 N^2 / (NM) = \pi^2 N / M$，

误差 $\leq \frac{\pi^2 N}{2M}$，取 $M = \pi^2 N / \varepsilon$ 即可。

**与朴素方法的对比**：
| 方法 | 误差 | 所需样本数 $M$ |
|:---|:---:|:---:|
| 一般随机 Hamiltonian 模拟 | $\sim N^2/M$ | $\Omega(N^2/\varepsilon)$ |
| 量子 oracle 草图（正交子空间） | $\sim N/M$ | $\Theta(N/\varepsilon)$ |

**本质原因**：不同 $x$ 的哈密顿量 $h_x = f(x)|x\rangle\langle x|$ 作用于**相互正交**的子空间，误差不会跨不同基矢积累！一般随机 Hamiltonian 模拟的误差来自交叉项（不同 Hamiltonian 的非对易性），而在量子 oracle 草图中这些交叉项精确为零。

#### 1.4.3　钻石距离界（Theorem D.2）

**关键引理**：对酉信道（unitary channel）$U: \rho \mapsto U\rho U^\dagger$ 和 $V: \rho \mapsto V\rho V^\dagger$：
$$\frac{1}{4}\|U - \mathbb{E}[V]\|_\diamond \leq \|U - \mathbb{E}[V]\|$$

其中 $\|A\|_\diamond = \sup_{\rho} \|A(\rho)\|_1$ 是钻石范数，$\|A\|$ 是算符范数（最大奇异值）。

因此对期望酉算符的算符范数控制等价于（在常数因子内）控制信道的钻石距离——这是量子算法分析的标准误差度量。

#### 1.4.4　样本复杂度的优化性（Theorem D.13）

**定理陈述**：任何量子算法，若以至多 $M$ 个独立样本 $(x_t, f(x_t))$ 构造相位 oracle 的 $\varepsilon$-近似（钻石距离），则必须有 $M = \Omega(N/\varepsilon)$。

**证明思路**（信息论下界）：利用 Born 规则的平方根关系——量子振幅与概率之间的关系是平方根，因此将 $M$ 个概率事件（每个样本的 $1/N$ 概率）累积为 $O(1)$ 精度的振幅，最少需要 $M \sim N$ 次。

更精确地，考虑 Forrelation 任务（后见 1.7 节）：量子算法对 oracle 做单次查询（$Q=1$），需要 $M = \Omega(N/\varepsilon)$ 样本才能将 $O_f$ 近似到 $\varepsilon$ 精度。这等价于将 $N$ 个比特的 Boolean 函数压缩到 $\mathrm{poly}(\log N)$ 个量子比特中，所需测量次数由信息论下界给出。

**推论：多次查询的总样本数**：若算法需要 $Q$ 次 oracle 查询，每次精度 $\varepsilon/Q$，则总样本数为：
$$M_{\text{total}} = Q \cdot \Theta\!\left(\frac{N}{\varepsilon/Q}\right) = \Theta\!\left(\frac{NQ^2}{\varepsilon}\right) \tag{QOS-1}$$

Q² 的依赖性是**不可避免**的，反映了 Born 规则的平方根：将 $p(x) = 1/N$ 精度的统计信息转换为振幅精度 $\varepsilon/Q$ 需要 $O(N/(\varepsilon/Q)^2)$ 次测量，而一个查询利用了 $N$ 个位置同时，需要 $O(N \cdot 1/(\varepsilon/Q)) = O(NQ/\varepsilon)$ 次采样，$Q$ 次总共 $O(NQ^2/\varepsilon)$。

#### 1.4.5　推广到相关数据（Theorem D.16）

**定理陈述**：设数据由重复数 $R$ 的 HDGP 生成。则：
$$M \geq \frac{t^2 p_{\max} + 2t\sqrt{p_{\max}|\mathcal{X}|}}{\varepsilon} R_\mathcal{D}$$

样本（从任意起始时刻 $t_0$ 开始）足以保证：
$$\mathbb{E}_{z_1,\ldots,z_{t_0-1}}\left\|\mathbb{E}[V_{t_0+M-1}\cdots V_{t_0} \mid z_1,\ldots,z_{t_0-1}] - U\right\|_\diamond \leq \varepsilon$$

其中**条件期望钻石距离**（conditional expected diamond distance）是正确的误差度量——它对已处理的历史数据条件化，防止过去的随机性污染当前轮次的保证。

**证明结构**：误差分解为两部分：
$$\varepsilon \leq \underbrace{\varepsilon_{\text{variance}}}_{\text{当前轮内统计波动}} + \underbrace{\varepsilon_{\text{bias}}}_{\text{历史数据引入的期望偏移}}$$

- **方差项**：用 Lemma C.4 得 $\mathrm{Var}[m_x] \leq p(x)R_\mathcal{D}/M$，代入 Lemma D.7，得 $\varepsilon_{\text{variance}} \leq t^2 p_{\max} R_\mathcal{D} / (2M)$；
- **偏差项**：用 Lemma C.5 得 $\max_x |\mathbb{E}[m_x \mid z_{<t_0}] - \mathbb{E}[m_x]| \leq \sqrt{p_{\max}|\mathcal{X}|R_\mathcal{D}}/M$，代入 $|e^{ia} - e^{ib}| \leq |a-b|$，得 $\varepsilon_{\text{bias}} \leq 2t\sqrt{p_{\max}|\mathcal{X}|R_\mathcal{D}}/M$。

**Lemma D.17（误差积累的线性化）**：对由相关随机变量 $Z_1,\ldots,Z_Q$ 参数化的 $Q$ 个量子信道：
$$\left\|\mathbb{E}[V^{(Q)} \circ C^{(Q)} \circ \cdots \circ V^{(1)} \circ C^{(1)}] - U^{(Q)} \circ C^{(Q)} \circ \cdots \circ U^{(1)} \circ C^{(1)}\right\|_\diamond \leq \sum_{j=1}^Q \mathbb{E}_{Z_1,\ldots,Z_{j-1}}\left\|\mathbb{E}[V^{(j)} \mid Z_1,\ldots,Z_{j-1}] - U^{(j)}\right\|_\diamond$$

**证明**：展开为望远镜求和（telescoping sum），利用钻石范数的次乘性和三角不等式。

这个引理意义重大：即使数据有任意相关性，**总误差仍线性积累**，等于各步误差之和——就像 IID 情形一样。

#### 1.4.6　处理未知非均匀分布（Lemma D.18）

当边际分布 $p(x)$ 未知（且可能非均匀）时，无法直接设定 $\tau = \pi N$。解决方案是使用**量子奇异值变换（QSVT）应用阈值函数**：

**构造步骤**：
1. 用 Theorem D.16 构造 $U(t) = \sum_x e^{ip(x)f(x)t}|x\rangle\langle x|$（令 $t = 1/p_{\max}$，则指数 $\in [0,1]$）；
2. 引入辅助比特 $a$，构造 $W = S_a X_a H_a (\text{c}_a U^\dagger) X_a (\text{c}_a U) H_a$（Hadamard 测试变体），使得 $\langle 0_a|W|0_a\rangle = \sin(\Lambda)$，其中 $\Lambda = \mathrm{diag}(p(x)f(x)/p_{\max})$；
3. 用 QSVT（Theorem D.8）实现阈值函数 $P(\sin(\cdot))$：$P(w) \approx +1$ 当 $w \in [\sin(p_{\min}/p_{\max}), 1]$（对应 $f(x)=1$），$P(w) \approx -1$ 当 $w \approx 0$（对应 $f(x)=0$）；
4. 所用多项式次数 $d = O(\log(1/\varepsilon)/\sin(p_{\min}/p_{\max})) = O(p_{\max}/p_{\min} \cdot \log(1/\varepsilon))$；
5. 总样本数：$M = O(p_{\max}\sqrt{p_{\max}|\mathcal{X}|}/p_{\min}^2 \cdot R_\mathcal{D} \log^2(1/\varepsilon)/\varepsilon)$。

对均匀分布（$p_{\max} = p_{\min} = 1/N$），简化为 $M = O(N R_\mathcal{D} \log^2(1/\varepsilon)/\varepsilon)$。

#### 1.4.7　线性代数原语（Section D.5）

量子 oracle 草图最终需要推广到向量和矩阵的数据结构，以支持线性代数算法。

**稀疏矩阵的 oracle（Lemma D.19）**：设 $A \in \mathbb{R}^{N \times N}$ 是行 $s_r$-稀疏、列 $s_c$-稀疏的矩阵。定义三种稀疏 oracle：
- **元素 oracle**：$O_A^{\text{ele}}: |i\rangle|j\rangle|0\rangle \to |i\rangle|j\rangle|A_{ij}\rangle$（返回矩阵元）；
- **行指标 oracle**：$O_A^{\text{ind,row}}: |i\rangle|k\rangle \to |i\rangle|j(i,k)\rangle$（返回第 $i$ 行第 $k$ 个非零元的列下标）；
- **列指标 oracle**：$O_A^{\text{ind,col}}: |j\rangle|k\rangle \to |j\rangle|i(j,k)\rangle$（类似）。

**构造方法**：

对元素 oracle，利用 Theorem D.16：
- 每次观察样本 $(i_t, j_t, A_{i_t j_t})$，施加相位旋转 $\exp(i\pi N_{\text{nnz}} A_{i_t j_t} |i_t\rangle\langle i_t| \otimes |j_t\rangle\langle j_t| / M)$；
- 不同 $(i,j)$ 对的哈密顿量作用于正交子空间，误差 $O(N_{\text{nnz}}/M)$；
- 所需样本：$M = O(RN_{\text{nnz}}\min(s_r,s_c)/\varepsilon)$；
- 内存：$2\lceil\log_2 N\rceil + b$ 量子比特（$b$ 位存储矩阵元的二进制表示）。

对行指标 oracle，构造更复杂（涉及 in-place 二分搜索），需要额外 $\lceil\log_2 s\rceil + 2$ 辅助比特，样本数同阶。

**量子态草图（Quantum State Sketching，Theorem D.24）**：设 $\vec{b} \in \mathbb{R}^N$ 为待制备的向量，从向量数据生成过程（$z_t = (j_t, b_{j_t})$）采样。目标：制备量子态 $|b\rangle = \vec{b}/\|\vec{b}\|$（单位化）。

构造分为两步：
1. **Hadamard 变换预处理**：对样本做随机 Hadamard 旋转 $h: [N] \to \{0,1\}$，构造变换后向量 $\vec{b}'$（Hadamard 变换 $H^{\otimes n}$ 加随机相位 $(-1)^{h(j)}$）——目的是使 $\vec{b}'$ 近乎均匀；
2. **均匀分布的 oracle 草图**：用 Theorem D.16 的正交子空间技巧，以 $M_0$ 个样本构造 $\mathrm{diag}(\vec{b}')$ 的相位 oracle（等价于 $\vec{b}'$ 的 block encoding 核心部分）；
3. **反归一化**：用 QSVT 的 $\arcsin$ 多项式近似将 $\mathrm{diag}(\vec{b}')$ 的奇异值变换为归一化版本；
4. **反 Hadamard 变换**恢复原始 $|b\rangle$。

总样本数：$M = \tilde{O}(RN\|b\|_\infty^2/(\|\vec{b}\|_2^2\varepsilon^2))$。

---

### 1.5　干涉型经典影子（Interferometric Classical Shadows，Section F.2 的 Theorem F.16）

#### 1.5.1　读出问题的精确陈述

量子算法制备了 $\mathrm{poly}(\log D)$ 量子比特的量子态 $|\psi\rangle$（编码了 LS-SVM 的权重向量方向或 PCA 的主成分方向）。读出任务：对给定稀疏测试向量 $\vec{x}' \in \mathbb{R}^D$（$s'$-稀疏），估计**带符号内积** $\langle \psi | \vec{x}'\rangle$（包含符号信息，这是分类任务中决定预测标签的关键）。

**问题**：Holevo 界表明 $n = \mathrm{poly}(\log D)$ 量子比特的单次测量只能提取 $n$ 比特经典信息——不足以存储 $D$ 维向量。

#### 1.5.2　Hadamard 测试

**基本电路**：
```
|0⟩_a ─── H ─── ●─────────── H ─── Measure
                 │
|ψ⟩   ────────── U ──────────────── |ψ⟩
```

施加受控酉算符 $U$ 后，辅助比特的测量期望值为：
$$\mathbb{E}[\text{Measure}] = \mathrm{Re}\langle \psi | U | \psi \rangle$$

类似地，在 Hadamard 后插入 $S^\dagger$ 门（相位门）可得虚部 $\mathrm{Im}\langle \psi | U | \psi \rangle$。

#### 1.5.3　经典影子（Classical Shadows，Huang-Kueng-Preskill 2020 [Ref 74]）

标准经典影子技术：对量子态 $|\psi\rangle$，随机选择酉算符 $U_i$（从某集合采样），测量 $U_i|\psi\rangle$，得到经典快照 $\hat{\rho}_i = U_i^\dagger |b_i\rangle\langle b_i| U_i$（其中 $b_i$ 是测量结果）。利用多个快照的均值估计期望值 $\mathrm{tr}(O|\psi\rangle\langle\psi|)$。

**关键属性**：使用 Clifford 随机电路时，$T$ 个快照可以预测 $2^O(T \log T)$ 个迹 $1$ 可观测量（offline prediction），每个预测精度为 $\varepsilon$，成功概率 $1-\delta$，所需快照数 $T = O(\log(1/\delta) \cdot \|O\|_{\text{shadow}}^2 / \varepsilon^2)$。

#### 1.5.4　干涉型经典影子（Theorem F.16 的完整陈述）

**目标**：估计 $\langle \psi | \vec{x}' \rangle$，其中 $|\psi\rangle$ 是 $\mathrm{poly}(\log D)$ 量子比特的量子态，$\vec{x}'$ 是 $s'$-稀疏向量（测试时给出，训练时不需要知道）。

**算法**：

**训练阶段**（在量子态 $|\psi\rangle$ 上进行 $T$ 次独立实验）：

1. 从 Clifford 群均匀随机采样酉算符 $U_i$（$i = 1, \ldots, T$）；
2. 构造 Hadamard 测试电路：
   ```
   |0⟩_a ─── H ─── ●───────── H ─── Measure → bit_i
                    │
   |ψ⟩   ─────────── U_i ──────────
   ```
3. 测量辅助比特，得到比特 $b_i \in \{0,1\}$；
4. 经典后处理：每个 $(U_i, b_i)$ 对构成一个「干涉型经典快照」；
5. 存储所有 $T$ 个 $(U_i, b_i)$ 对作为经典影子（使用 $O(T \cdot n)$ 经典存储，$n = \mathrm{poly}(\log D)$）。

**预测阶段**（纯经典计算，给定稀疏测试向量 $\vec{x}'$）：

对于稀疏向量 $\vec{x}' = \sum_{k \in \text{supp}} x'_k |k\rangle$（$|\text{supp}| = s'$），计算：
$$\hat{\xi}(\vec{x}') = \frac{1}{T}\sum_{i=1}^T (2b_i - 1) \langle\vec{x}'| U_i^\dagger |0\rangle$$

这是 $\mathrm{Re}\langle\psi|(\vec{x}' \cdot |\cdot\rangle)|0\rangle$ 的无偏估计（通过 Hadamard 测试的期望值计算）。

**样本复杂度**（Theorem F.16 的精确陈述）：
$$T = O\!\left(\frac{s' \log(D/\varepsilon) + \log(1/\delta)}{\varepsilon^2}\right)$$

次量子电路执行，可估计任意 $s'$-稀疏测试向量的内积至 $\varepsilon$ 误差，成功概率 $\geq 1-\delta$。

**对任意数量测试向量**：同一组 $T$ 个快照可用于预测无限多个测试向量（离线/offline 预测），每个预测的精度 $\varepsilon$ 和失败概率 $\delta/K$（$K$ 个测试）。

#### 1.5.5　与 Holevo 界的相容性

Holevo 界：$n$ 量子比特最多携带 $n$ 比特经典信息。

**为何不矛盾**：

1. 干涉型经典影子**不是在提取量子态的所有信息**，而是在估计特定结构的线性函数；
2. 稀疏测试向量 $\vec{x}'$（$s'$ 个非零分量）在 $D$ 维空间中的「有效自由度」仅为 $s' \log(D/s')$，远小于 $D$；
3. 每次 Hadamard 测试只提取 1 比特，但它是对特定子空间的投影——利用了测试向量的稀疏结构；
4. $T = O(s' \log D / \varepsilon^2)$ 次测量提取了 $T$ 比特信息，这**足以**（在稀疏结构约束下）精确估计与 $s'$ 个非零坐标相关的内积，与 $D$ 的大小无关。

---

### 1.6　经典难度证明：NOPE 框架（Section E）

#### 1.6.1　NOPE 问题的形式定义（Section E.1）

**噪声 Oracle 属性估计（Noisy Oracle Property Estimation，NOPE）**：

**设置**：
- $f: [N] \to \{0,1\}$ 是未知 Boolean 函数；
- 目标属性 $P: ([N] \to \{0,1\}) \to \mathbb{R}$（例如 Forrelation 值）；
- 数据：$M$ 个噪声样本 $(x_t, f(x_t) \oplus \eta_t)$，其中 $\eta_t \sim \text{Bern}(q)$（独立翻转噪声，$q < 1/2$）；
- 任务：以 $2/3$ 的概率输出 $P(f)$ 的 $\varepsilon$-近似。

**复杂度参数**：
- $Q$：完成此任务所需的**量子**查询复杂度（即量子算法对完美 oracle $O_f$ 的查询次数）；
- $Q_C$：完成此任务所需的**经典随机化**查询复杂度（对完美 oracle 的查询次数）。

#### 1.6.2　核心定理：样本-空间乘积下界（Theorem E.2）

**Theorem E.2（精确陈述）**：设 NOPE 任务的量子查询复杂度为 $Q$ 而经典查询复杂度为 $Q_C$。设 $\mathcal{L}$ 是大小 $S$ 的经典学习算法，以 $M$ 个样本完成静态 NOPE 任务（$R=1$）。则：
$$M \cdot S \geq \Omega\!\left(\frac{N Q_C}{\mathrm{polylog}(N)}\right)$$

**推论**：若量子机用 $M_Q = \Theta(NQ^2/\varepsilon)$ 样本（由 Oracle 草图最优性给出，公式 QOS-1），任意经典机用同等样本 $M_C = M_Q$，则：
$$S_C \geq \Omega\!\left(\frac{Q_C}{Q^2 \cdot \mathrm{polylog}(N)}\right)$$

当 $Q_C = \Omega(N^{1-\zeta})$（Forrelation）且 $Q = O(1)$，得 $S_C \geq \Omega(N^{1-\zeta})$，而 $S_Q = O(\log N)$——**指数空间分离**。

#### 1.6.3　Query-to-Communication Lifting（Section E.3）

证明 Theorem E.2 的核心技术是**查询-通信归约**（Query-to-Communication Lifting），来自 Chattopadhyay et al. 2021 [Ref 92]，本文做了关键推广。

**基本思想**：

1. **编码**：设 $g: \{0,1\}^b \times \{0,1\}^b \to \{0,1\}$ 是一个「小差异」（low-discrepancy）两方函数（噪声编码函数）。对任意 $o \in \{0,1\}^N$，定义噪声版 oracle：

$$f(x) = g(y^{(0)}_x, y^{(1)}_\alpha)$$

其中 $y^{(0)} \in \{0,1\}^{N \times b}$ 和 $y^{(1)} \in \{0,1\}^{N \times b}$ 是对应的两方输入，$\alpha \in \{0,1\}$ 是当前情境。

2. **分布数据生成过程 $\mathcal{D}^N_{g,T}$**：情境 $\alpha$ 每隔 $T$ 步刷新，每步均匀采样 $x \in [N]$，观察 $(x, g(y^{(0)}_x, y^{(1)}_\alpha))$——这正是有噪声 NOPE 的数据流。

3. **通信模拟（Theorem E.25）**：任意经典学习算法 $\mathcal{L}$（大小 $S$，样本 $M$）可以被一个**随机化并行决策树** $\mathcal{A}$（查询复杂度 $Q_{\mathcal{A}} = O(MS/(Tb))$）以 $2^{-\eta b/8}$ 误差模拟。

4. **密度恢复技术（Density Restoring Partition，Theorem E.21）**：维护一个「矩形猜测」$\mathcal{Y}^{(0)} \times \mathcal{Y}^{(1)}$（两方输入的笛卡尔积），利用低差异编码保证猜测集的密度始终足够高。关键量：**缺陷**（deficiency）$D_\infty(Y^{(0)}, Y^{(1)}, \rho) = 2^{b|{\rm free}(\rho)|} - H_\infty(Y^{(0)}_{{\rm free}(\rho)}) - H_\infty(Y^{(1)}_{{\rm free}(\rho)})$，其中 $H_\infty$ 是最小熵。算法确保任何时刻缺陷有界，从而模拟误差有界。

5. **查询下界应用**：决策树 $\mathcal{A}$ 用 $Q_{\mathcal{A}}$ 次查询就能估计 $P(f)$，但经典查询复杂度下界要求 $Q_{\mathcal{A}} \geq Q_C / \mathrm{polylog}(N)$，故：

$$\frac{MS}{Tb} \geq \frac{Q_C}{\mathrm{polylog}(N)}$$

取 $T = N$，$b = O(\log N)$（典型参数），得 $MS \geq \Omega(NQ_C/\mathrm{polylog}(N))$。

#### 1.6.4　学习版 XOR 引理（Learning XOR Lemma，Section E.4）

**标准 XOR 引理（Yao 1982 [Ref 97]）**：若一个函数 $f$ 具有优势 $\varepsilon$（即计算 $f$ 的成功概率为 $1/2 + \varepsilon$），则计算 $f$ 的 $T$ 次独立实例的 XOR 的成功概率至多为 $2^{-\Omega(T)}$。

**流式 XOR 引理（Assadi-N 2021 [Ref 99]）**：在流式模型中类似结论成立，但证明利用了数据的**对抗性排序**，不适用于随机采样情形。

**本文的学习版 XOR 引理（Theorem E.30，精确陈述）**：考虑**动态 NOPE** 任务：$\log_2 N$ 个 NOPE 实例 $(f^{(1)}, \ldots, f^{(\log N)})$ 串行呈现，每个持续 $\tau = \tilde{O}(N)$ 时间步，目标是跟踪每一个实例的属性值。

**结论**：若经典机大小 $S = O(N^{0.99})$，则对任意 $l$：
$$\Pr[\text{经典机在第 } l \text{ 个实例上成功}] \leq \frac{1}{2} + N^{-\Omega(1)}$$

即成功概率严格低于 $1/2 + N^{-\Omega(1)}$，对任意实例均成立（均匀失败）。

**推论（超多项式样本下界）**：结合混合参数（hybrid argument），若经典机内存 $S = O(N^{0.99})$，则对任意 $\mathrm{poly}(N)$ 的样本数，成功概率严格低于 $2/3$（所需阈值），即样本复杂度下界为 $\mathrm{superpoly}(N)$。

**证明的去随机化技术（Section E.4.d）**：
Yao 的 XOR 引理证明通常假设独立随机输入，但流式学习模型中不同时间步的数据来自同一分布——需要去随机化。本文开发了**有界差异去随机化**（bounded-discrepancy derandomization）技术：
- 将随机哈希函数替换为 $2k$-wise 独立哈希（$k = O(\log N)$）；
- 利用矩估计（moment bounds）代替完全独立性，具体地：$2k$ 阶矩与完全随机情形一致，Markov 不等式给出概率上界；
- 使得 XOR 引理的证明不需要完全随机的 oracle 函数，只需计算高效可实现的伪随机函数。

#### 1.6.5　Forrelation 作为 NOPE 的极端实例（Section E.5.a）

**Forrelation 问题**（Aaronson-Ambainis 2015 [Ref 95], Bansal-Sinha 2021 [Ref 96]）：

给定 $f, g: \{0,1\}^n \to \{-1,+1\}$，计算：
$$\Phi(f,g) = \frac{1}{2^n}\sum_{x \in \{0,1\}^n} \hat{f}(x) g(x)$$

其中 $\hat{f}(x) = 2^{-n/2}\sum_y (-1)^{x \cdot y} f(y)$ 是 $f$ 的（归一化）Hadamard 变换。

**查询复杂度分析**：
- **量子**：$Q = O(1)$（常数次 Hadamard 测试）：制备 $|f\rangle = \frac{1}{\sqrt{2^n}}\sum_y f(y)|y\rangle$（需 oracle 查询一次），施加 $H^{\otimes n}$，测量并与 $g$ 对比。
- **经典**：$Q_C = \Omega(N^{1-\zeta})$ 对任意 $\zeta > 0$（Bansal-Sinha 2021 的 $k$-forrelation 结果）。这是**量子与经典查询复杂度的最优分离**。

**代入 Theorem E.2**：任意经典机（$M = \tilde{O}(N)$ 样本）必须有：
$$S_C \geq \Omega\!\left(\frac{Q_C}{Q^2}\right) = \Omega\!\left(\frac{N^{1-\zeta}}{1}\right) = \Omega(N^{1-\zeta})$$

而量子机：$S_Q = O(\log N)$——空间分离比为 $\Omega(N^{1-\zeta}/\log N)$，即**指数级空间分离**（以比特数衡量）。

#### 1.6.6　从 NOPE 到实际应用的归约（Section E.5.b–e）

将 Forrelation-NOPE 的经典难度推广到线性系统、分类、PCA 三类实际任务，需要证明这三类任务具有与 NOPE 相同难度。本文用 BQP-hardness 的构造完成这一目标。

**线性系统的归约（Section E.5.c）**：
- 利用矩阵求逆的 BQP-hardness（Harrow-Hassidim-Lloyd 2009 [Ref 118]）：任何解线性系统的算法都可以模拟量子电路；
- 构造一族线性系统 $\{(A^{(f)}, \vec{b}^{(f)})\}_{f}$（由函数 $f$ 参数化），使得估计 $\Phi^{(f)} = \vec{x}^{*\top}M\vec{x}^*$ 等价于估计 Forrelation 属性 $P(f)$；
- 因此线性系统任务对 $O(N^{0.99})$ 大小的经典机同样困难。

**分类的归约（Section E.5.d）**：
- LS-SVM 的权重向量 $\vec{w}^* = (X^\top X + \lambda I)^{-1}X^\top\vec{y}$ 包含矩阵求逆——与 HHL 等价；
- 构造数据矩阵 $X^{(f)}$ 和标签向量 $\vec{y}^{(f)}$ 使得分类任务等价于 Forrelation-NOPE；
- 对稀疏测试向量的预测符号正是 NOPE 属性的一位编码。

**PCA 的归约（Section E.5.e，使用 Feynman-Kitaev 哈密顿量构造）**：
- PCA 需要计算 $X^\top X$ 的最大特征向量——等价于基态制备；
- 利用修改版 Feynman-Kitaev 线路哈密顿量（circuit Hamiltonian, [Ref 164]）构造一族数据矩阵，使得最大特征向量的分量编码了量子线路的输出；
- 维度归约任务的预测值等价于 BQP 计算结果，故对经典机同样困难。

---

### 1.7　三类应用的完整量子算法（Section F）

#### 1.7.1　线性系统的量子算法（Section F.1）

**完整算法（Algorithm F.1，定理 F.3）**：

**输入**：流式样本 $\{(i_t, j_t, A_{i_t j_t}, k_t, b_{k_t})\}$

**第一阶段：Oracle 构建（$M = \tilde{O}(NQ^2/\varepsilon)$ 样本）**

1. 用量子态草图（Theorem D.24）构造增广矩阵 $\tilde{A} = \begin{pmatrix} A \\ \sqrt{\lambda}I \end{pmatrix}$ 的 block encoding $U_{\tilde{A}}$：
   - 对列下标寄存器，用 in-place 二分搜索确定第 $k$ 个非零元的位置；
   - 样本数：$M_1 = \tilde{O}(N s^2/\varepsilon_1)$。
2. 用量子态草图构造向量 $\vec{b}$ 的状态制备酉算符 $U_{|b\rangle}$（制备 $U_{|b\rangle}|0\rangle = |b\rangle = \vec{b}/\|\vec{b}\|$）：
   - 对 $(k_t, b_{k_t})$ 施加 Hadamard 变换 + 相位旋转；
   - 样本数：$M_2 = \tilde{O}(N/\varepsilon_2)$。

**第二阶段：量子线性系统求解（Costa et al. 2022 [Ref 82] + QSVT）**

对 block encoding $U_{\tilde{A}}$ 应用 QSVT 实现多项式近似 $p(x) \approx 1/x$（截断奇异值多项式，次数 $d = O(\kappa \log(\kappa/\varepsilon))$）：

$$|\psi^*\rangle \propto \tilde{A}^\dagger (\tilde{A}\tilde{A}^\dagger)^{-1} |\tilde{y}\rangle = (\tilde{A}^\top\tilde{A})^{-1}\tilde{A}^\top|\tilde{y}\rangle \propto \vec{w}^*$$

实际上这等价于 $|(X^\top X + \lambda I)^{-1}X^\top \vec{y}\rangle$，即 LS-SVM 权重向量的量子态。

**电路结构**：
```
[Oracle Sketch 1: U_A] ─────────────────────────────────┐
[Oracle Sketch 2: U_|b⟩] ──────────────────────────────┐│
                                                         ││
|0⟩ ─ QSVT(p(x)=1/x, degree d, via U_A) ──────── |ψ*⟩ ││
|0⟩ ─ State Prep (U_|b⟩) ───────────────────────────────┘│
(ancilla) ──────────────────────────────────────────────┘
```

**第三阶段：读出（Interferometric Classical Shadow）**

对量子态 $|\psi^*\rangle$ 应用干涉型经典影子（Theorem F.16），估计二次型 $\vec{x}^{*\top} M \vec{x}^*$：
- 对矩阵 $M$（实验指定），制备 $|M\vec{x}^*\rangle \propto M|\psi^*\rangle$（用 $M$ 的 block encoding + QSVT）；
- 用 Hadamard 测试测量 $\langle\psi^*|M|\psi^*\rangle$；
- 重复 $T = O(1/\varepsilon^2)$ 次取均值。

**总量子比特数**：$S_Q = 2\lceil\log_2 N\rceil + \lceil\log_2 s\rceil + O(\log(\kappa/\varepsilon))$（与 $N$ 呈对数关系）。

**总样本数**：$M = \tilde{O}(N s^2 \kappa^4 / \varepsilon^2)$（来自 $Q = O(\kappa)$ 次 oracle 查询，每次需 $\Theta(N/(\varepsilon/Q))$ 样本）。

#### 1.7.2　LS-SVM 分类的量子算法（Section F.2）

**完整算法（Theorem F.11 的实现细节）**：

**预处理（一次性，与测试向量无关）**：

1. 构造增广数据矩阵 $\tilde{X} = \begin{pmatrix}X \\ \sqrt{\lambda}I\end{pmatrix} \in \mathbb{R}^{(N+D)\times D}$ 的 block encoding（$D$ 是特征维度，$s$ 是行稀疏度）；
2. 构造标签向量 $\tilde{\vec{y}} = (\vec{y}^\top, \vec{0}_D^\top)^\top$ 的状态制备酉算符；
3. 用 QSVT 计算 $|\tilde{X}^\dagger \tilde{\vec{y}}\rangle \propto |\vec{w}^*\rangle$（LS-SVM 权重向量的量子态）——这是实现矩阵伪逆的 QSVT 过程；
4. **量子态的样本量**：$M = \tilde{O}(N s^2 \kappa^4 / \varepsilon^2)$；
5. **干涉型经典影子训练**：制备 $T = O(s' \log(D/\varepsilon)/\varepsilon^2)$ 个快照，存储经典影子模型。

**推断阶段（纯经典，无需量子计算）**：

给定稀疏测试向量 $\vec{x}'$（$s'$-稀疏），用经典影子模型计算 $\hat{y} = \mathrm{sign}(\hat{\langle\vec{x}'|\vec{w}^*\rangle})$。

**核心定理 F.16（干涉型经典影子的精确保证）**：

设 $|\psi^*\rangle$ 是制备好的权重向量量子态，稀疏度 $s' = O(\mathrm{polylog}(D))$，间隔 $\gamma > 0$。则存在 $T = O(s' \log(D/\varepsilon)/\varepsilon^2)$ 使得：
$$\Pr\!\left[\hat{y} \neq y^*\right] \leq \frac{1}{3}$$

对任意 $s'$-稀疏且间隔 $\geq \gamma$ 的测试向量 $\vec{x}'$（其中 $\varepsilon \leq \gamma/2$）。

#### 1.7.3　主成分分析的量子算法（Section F.3）

**完整算法（Theorem F.21 的实现细节）**：

**构造协方差矩阵的 block encoding**：

数据矩阵 $X \in \mathbb{R}^{N \times D}$（$s$-稀疏）的协方差矩阵 $\Sigma = X^\top X / N$。Block encoding 的构造：
1. 构造 $X$ 的稀疏 oracle（$O_X^{\text{ele}}$ 和 $O_X^{\text{ind,row}}$，来自 Lemma D.19–D.21）；
2. 利用公式 $\Sigma = X^\top X / N$，通过 LCU（Linear Combination of Unitaries）构造 $\Sigma$ 的 block encoding。

**基态制备（Lin-Tong 算法 [Ref 84]）**：

设 $\Sigma = \sum_j \sigma_j^2 |u_j\rangle\langle u_j|$（SVD，$\sigma_1 \geq \sigma_2 \geq \cdots$），$\vec{w}_1 = |u_1\rangle$ 是最大特征向量（即第一主成分）。

1. 准备引导态 $|\phi_0\rangle = \vec{g}/\|\vec{g}\|$（由引导向量 $\vec{g}$ 给出，满足 $|\langle\phi_0|u_1\rangle|^2 = \chi^2 > 0$）；
2. 构造滤波多项式 $p_{\text{filter}}(x)$：在谱间隔 $\Delta > 0$ 的帮助下，实现：
$$p_{\text{filter}}(x) \approx \begin{cases} 1 & x \geq \sigma_1^2 - \Delta/2 \\ 0 & x \leq \sigma_2^2 + \Delta/2 \end{cases}$$
所需多项式次数：$d = O(1/(\chi\Delta) \cdot \log(1/\varepsilon))$；
3. 应用 QSVT 实现 $p_{\text{filter}}(\Sigma)$，在引导态 $|\phi_0\rangle$ 上：
$$p_{\text{filter}}(\Sigma)|\phi_0\rangle \approx \chi |u_1\rangle \Rightarrow |u_1\rangle + O(\varepsilon) \text{ (after normalization)}$$
4. 振幅放大（oblivious amplitude amplification）将成功概率从 $\chi^2$ 提升至 $\Omega(1)$，额外乘以 $O(1/\chi)$ 次查询。

**总代价**：
- 对 $\Sigma$ 的 block encoding 查询：$O(d/\chi) = O(1/(\chi^2\Delta)\log(1/\varepsilon))$ 次；
- 每次查询对数据生成的样本需求（来自 QOS-1 公式）：$\tilde{O}(Ns^2/\varepsilon_{\text{per-query}})$；
- 总样本数：$M = \tilde{O}(Ns^2/(\chi^2\Delta^2\varepsilon^2))$。

**预测阶段（干涉型经典影子）**：

对制备好的 $|\vec{w}_1\rangle$ 应用 Theorem F.16，对任意稀疏测试向量 $\vec{x}'$ 估计低维表示 $\xi(\vec{x}') = \langle\vec{x}'|\vec{w}_1\rangle^2$。

---

### 1.8　数值实验详细分析（Appendix A）

#### 1.8.1　四个数据集

| 数据集 | 任务 | $N$（样本数） | $D$（特征数，截断前） | 方法 |
|:---|:---:|:---:|:---:|:---|
| IMDb 情感分析 | 二分类 | $25{,}000$ | $89{,}527$ (TF-IDF) | LS-SVM ($\lambda = 10$) |
| PBMC68k scRNA-seq | PCA | $68{,}450$ | $32{,}738$ (基因) | 第一主成分 |
| 20newsgroup 话题分析 | 二分类 | $18{,}846$ | $50{,}000$+ (TF-IDF) | LS-SVM ($\lambda = 1$) |
| Dorothea 药物发现 | 二分类 | $1{,}150$ | $100{,}000$ (化合物特征) | LS-SVM ($\lambda = 200$) |

#### 1.8.2　算法对比与内存计算

**四种算法**：
1. **量子 oracle 草图（本文，橙色）**：内存公式 $S_Q^{\text{LS-SVM}} = 2\lceil\log_2(N+2D)\rceil + \lceil\log_2(s+1)\rceil + 4$（逻辑量子比特）；
2. **经典稀疏矩阵算法（灰色）**：下界 $S \geq N_{\text{nnz}}$（非零元个数，需存整个数据矩阵）；
3. **QRAM 量子算法（灰色）**：同经典稀疏矩阵，$S \geq N_{\text{nnz}}$（QRAM 需存储所有数据）；
4. **经典流式算法（蓝色）**：下界 $S \geq D$（需存整个权重向量 $\vec{w} \in \mathbb{R}^D$）。

**性能度量**：
- 分类：5-fold 交叉验证准确率（在随机类别对上平均）；
- 降维：截断后第一主成分的解释方差比（divided by untruncated baseline）。

**维度截断**：通过最小文档频率（minimal document frequency）截断稀有特征，描绘内存-性能权衡曲线。

#### 1.8.3　量子 oracle 草图的基准测试（图 3）

对四类 oracle 进行数值验证（使用 JAX 实现 [Ref 102]）：

**Boolean 函数相位 oracle**：
- 维度范围：$N \in [10^2, 10^3]$；
- 样本数范围：$M \in [10^5, 10^8]$；
- 误差度量：算符范数误差 $\|\mathbb{E}[V] - O_f\|$；
- 拟合结果：$\varepsilon = c_1 \cdot N / M$（$c_1 \approx \pi^2/2$，RMS 相对误差 $< 3\%$），完美符合理论预测。

**向量状态制备酉**：
- 归一化目标：$\|\vec{v}\|_2 = 1/(5\arcsin(1)) \approx 0.127$（便于 QSVT 的 $\arcsin$ 实现）；
- 误差度量：Euclidean norm $\|\mathbb{E}[V|0\rangle] - |v\rangle\|$；
- 拟合：$\varepsilon \sim c_2 \cdot N / M$（$c_2 \approx 1/\|\vec{v}\|_2$）。

**稀疏矩阵元素 oracle**：
- 矩阵维度：$100 \times 100$；
- 非零元数范围：$N_{\text{nnz}} \in [250, 2000]$；
- 样本数：$M \in [10^5, 10^8]$；
- 拟合：$\varepsilon \sim c_3 \cdot N_{\text{nnz}} / M$。

**稀疏行指标 oracle**：
- 行稀疏度固定为 $s = 8$；
- 矩阵维度：$50 \times 50$ 至 $500 \times 500$；
- 拟合：$\varepsilon \sim c_4 \cdot N / M$。

所有拟合的 RMS 相对误差均 $< 3\%$，实验完全符合理论预测，验证了量子 oracle 草图的正确性和参数的准确性。

---

### 1.9　论文的理论意义与开放问题（Section IV）

#### 1.9.1　无条件性的重要含义

**定理的无条件性**：Theorem E.2（样本-空间乘积下界）是**信息论**的——它不依赖任何计算复杂度猜想，仅基于：
1. 量子力学的正确性（Born 规则）；
2. 经典查询复杂度的已知下界（Forrelation 的 $Q_C = \Omega(N^{1-\zeta})$）；
3. 通信复杂度的有限性（有界信息传输）。

**强过 BPP ≠ BQP**：即使在极端情形 BPP = BQP（多项式时间量子 = 多项式时间经典），本文的**空间**优势仍然成立——量子空间 $O(\log N)$ vs 经典空间 $\Omega(N^{1-\zeta})$。

#### 1.9.2　Bell 型实验的类比

| Bell 不等式 | 本文的空间-样本分离 |
|:---|:---|
| 经典局域隐变量：相关函数 $\leq 2\sqrt{2}$ | 经典流式机：$M \cdot S \geq \Omega(NQ_C)$ |
| 量子力学：最大 CHSH 值 $= 2\sqrt{2}$ | 量子机：$M_Q \cdot S_Q = \tilde{O}(NQ^2 \log N)$ |
| 实验验证 [Aspect 1982] | 近期实验 [Niroula et al. 2025, Kretschmer et al. 2025] |
| 证伪：颠覆量子非局域性 | 证伪：颠覆量子力学的 Hilbert 空间指数维度 |

2025 年的两个实验（trapped-ion 量子流式算法实现 [Niroula et al. arXiv:2511.03689]，和无条件分离的演示 [Kretschmer et al. arXiv:2509.07255]）提供了支持性但尚不完整的实验证据。

#### 1.9.3　开放问题

1. **运行时优化**：量子 oracle 草图的主要运行时为 $\tilde{O}(N)$（数据加载），后续处理仅需 $\mathrm{poly}(\log N)$。由于草图操作主要由**对易操作**组成，存在大规模并行化的机会；

2. **变分量子增强**：将量子 oracle 草图与参数化量子电路（VQA/VQE）结合，通过训练变分参数进一步优化经典模型；

3. **更广泛的应用**：ODE/PDE 求解、大规模优化（凸/非凸）、信号处理、通信——任何具有超二次查询分离的问题都可能获益；

4. **去量子化的边界**：本文结果表明，即使线性量子加速（多项式）的情形，若内存瓶颈严重，量子机仍可有指数空间优势——这为重新审视被「去量子化」的算法提供了新视角；

5. **硬件设计**：量子 oracle 草图的并行性和对逻辑量子比特数的极低要求（$< 100$ 逻辑比特）使其成为容错量子计算早期阶段的理想候选任务。

---

### 1.10　量子算法总览：从 Oracle 草图到完整 ML 流水线

下图总结了完整的量子 ML 流水线（以 LS-SVM 分类为例）：

```
┌─────────────────────────────────────────────────────────────────┐
│  阶段 1: 数据加载（量子 Oracle 草图）                            │
│  流式样本 z_1,...,z_M ──► 量子态旋转序列 V_1,...,V_M            │
│  内存: O(log D) 量子比特 ──► block encoding U_X                 │
│  样本数: M = Õ(N·Q²/ε)，其中 Q = O(κ) 查询次数               │
└────────────────────────────────┬───────────────────────────────┘
                                 │
┌────────────────────────────────▼───────────────────────────────┐
│  阶段 2: 量子线性代数（QSVT 框架）                              │
│  block encoding U_X + U_|y⟩                                    │
│  QSVT(p(x)=x†, 次数 d=O(κ log κ/ε)) ──► |ψ*⟩ ∝ X†(XX†+λI)⁻¹y│
│  内存: 同阶（复用量子比特）                                     │
│  门数: O(d) × cost(oracle) = O(κ log(κ/ε) · s log D)          │
└────────────────────────────────┬───────────────────────────────┘
                                 │
┌────────────────────────────────▼───────────────────────────────┐
│  阶段 3: 读出（干涉型经典影子）                                  │
│  T = O(s' log(D/ε)/ε²) 次 Hadamard 测试                       │
│  输出: T 个比特 b_1,...,b_T（经典影子模型，存储 O(T·log D) 比特）│
└────────────────────────────────┬───────────────────────────────┘
                                 │
┌────────────────────────────────▼───────────────────────────────┐
│  阶段 4: 推断（纯经典）                                         │
│  给定稀疏测试 x'，计算 ŷ = sign(Σ_i (2b_i-1)⟨x'|U_i†|0⟩)    │
│  内存: O(s') 浮点数                                             │
│  时间: O(T·s') 浮点运算                                         │
└─────────────────────────────────────────────────────────────────┘
```

**总内存对比**：
- 量子机（阶段 1–3）：$S_Q = 2\lceil\log_2(N+2D)\rceil + O(\log s) + O(1)$ 个**逻辑量子比特**
- 经典流式机（完成同等任务）：$S_C \geq D$（需存权重向量 $\vec{w}^* \in \mathbb{R}^D$）
- **分离比**：$S_C / S_Q \geq D / (2\log_2(2D)) \approx D/(2\log D)$，即**接近指数级**


---

<a name="第二章"></a>
## 第二章　量子线性代数（QLAS）路线：从 HHL 到 QSVT

### 2.1　HHL 算法——量子机器学习的原点（2009）

#### 问题设定

给定 $N \times N$ 厄米（Hermitian）稀疏线性方程组 $A\vec{x} = \vec{b}$，其中：
- $A$ 是 $s$-稀疏（每行最多 $s$ 个非零元），条件数 $\kappa = \lambda_{\max}/\lambda_{\min}$；
- 向量 $\vec{b}$ 以量子态 $|b\rangle = \sum_i b_i / \|\vec{b}\| \cdot |i\rangle$ 给出。

目标：制备量子态 $|x\rangle \propto A^{-1}|b\rangle$，并对其进行测量以估计某个期望值 $\langle x|M|x\rangle$。

#### HHL 算法步骤

**核心工具**：量子相位估计（Quantum Phase Estimation, QPE）

设 $A$ 的谱分解为 $A = \sum_j \lambda_j |u_j\rangle\langle u_j|$，$|b\rangle = \sum_j \beta_j |u_j\rangle$（展开在 $A$ 的本征基上），则：

$$A^{-1}|b\rangle = \sum_j \frac{\beta_j}{\lambda_j} |u_j\rangle$$

**步骤 1**：制备 $|b\rangle$（利用 QRAM）。

**步骤 2**：通过哈密顿量模拟 $e^{iAt}$（$t = O(\kappa / \varepsilon)$ 时间步），利用 QPE 估计本征值 $\lambda_j$：

$$|b\rangle|0\rangle \xrightarrow{\text{QPE}} \sum_j \beta_j |u_j\rangle|\tilde{\lambda}_j\rangle$$

其中 $|\tilde{\lambda}_j\rangle$ 是本征值 $\lambda_j$ 的二进制表示（精度 $\varepsilon$）。

**步骤 3**：对本征值寄存器引入辅助比特，实现**受控旋转**：

$$\sum_j \beta_j |u_j\rangle|\tilde{\lambda}_j\rangle|0\rangle \xrightarrow{R_y} \sum_j \beta_j |u_j\rangle|\tilde{\lambda}_j\rangle\!\left(\frac{C}{\tilde{\lambda}_j}|0\rangle + \sqrt{1 - \frac{C^2}{\tilde{\lambda}_j^2}}|1\rangle\right)$$

其中 $C = O(1/\kappa)$ 是正规化常数（确保旋转角 $\arcsin(C/\tilde{\lambda}_j) \in [0, \pi/2]$）。

**步骤 4**：对辅助比特测量，后选 $|0\rangle$ 结果（成功概率 $\approx C^2/\kappa^2$），得到：

$$\frac{1}{\|A^{-1}|b\rangle\|} \sum_j \beta_j \frac{C}{\lambda_j} |u_j\rangle = C|x\rangle$$

**步骤 5**：利用振幅放大（amplitude amplification）将成功概率提升至 $\Omega(1)$，引入额外 $O(\kappa)$ 因子。

#### 复杂度分析

**量子复杂度**：

$$T_{\text{HHL}} = O\!\left(s^2 \kappa^2 \log^3(N) / \varepsilon\right)$$

（主要代价来自 QPE 的哈密顿量模拟步骤：$e^{iAt}$ 需 $t = O(\kappa)$，每步 $O(s \log N / \varepsilon)$ 门数）。

**经典复杂度**（对比）：
- 直接法（Gaussian 消元）：$O(N^3)$；
- 共轭梯度法（稀疏情形）：$O(Ns\kappa\sqrt{\kappa})$；
- Randomized Kaczmarz 法：$O(Ns/\varepsilon^2)$。

**「指数加速」的条件**：$\log^3(N)$ vs. $O(N \cdot \mathrm{poly}(\kappa))$——仅当 $N$ 极大、$\kappa$ 适中时成立。

#### HHL 的两个隐性假设（Aaronson 2015 批判）

**假设 1（输入模型）**：$|b\rangle$ 已经制备好了。

实际上，将经典向量 $\vec{b}$ 转换为量子态 $|b\rangle = \sum_i b_i/\|\vec{b}\||i\rangle$ 需要 QRAM，而这等价于解决了 $\Theta(N)$ 的存储问题。

**假设 2（输出模型）**：输出是量子态 $|x\rangle$。

直接读出 $\vec{x}^*$ 的所有 $N$ 个分量需要 $O(N/\varepsilon^2)$ 次测量，与经典方法相比无优势。仅当只需估计单个期望值 $\langle x|M|x\rangle$（$M$ 是稀疏矩阵或可高效测量的可观测量时），才有优势。

---

### 2.2　量子相位估计（QPE）：HHL 的核心子程序

#### 思想直觉

设 $U = e^{iAt}$ 是酉算符，具有本征值 $e^{i\phi_j}$（$\phi_j = \lambda_j t$）和本征态 $|u_j\rangle$。QPE 的目标是在辅助寄存器中写入 $\phi_j$ 的 $m$ 位二进制近似。

#### 电路构造

```
辅助寄存器（m 量子比特）：
|0⟩ ─── H ─── ●── ... ── QFT†── |φ_0⟩
|0⟩ ─── H ─── ●── ... ── QFT†── |φ_1⟩
...
|0⟩ ─── H ─── ●── ... ── QFT†── |φ_{m-1}⟩

目标寄存器：
|u_j⟩ ── U^1 ── U^2 ── U^4 ── ... ── U^{2^{m-1}} ── |u_j⟩
```

具体地：
1. 辅助寄存器初始化为 $|0\rangle^{\otimes m}$，施加 Hadamard 门得到均匀叠加；
2. 对 $k = 0, 1, \ldots, m-1$，以第 $k$ 个辅助比特控制施加 $U^{2^k}$；
3. 对辅助寄存器施加逆量子傅里叶变换 $\text{QFT}^\dagger$；
4. 测量辅助寄存器，以高概率得到 $\tilde{\phi}_j \approx \phi_j / (2\pi)$ 的 $m$ 位近似。

**精度**：以概率 $\geq 1 - \delta$ 得到精度 $\varepsilon = 2^{-m}$ 的估计，需 $m = O(\log(1/\varepsilon\delta))$ 个辅助比特。

**门数**：$O(m^2)$ 个门（主要来自 QFT）加上 $\sum_{k=0}^{m-1} \text{cost}(U^{2^k})$ 的代价。

---

### 2.3　量子奇异值变换（QSVT）——统一框架（2019）

#### 核心概念：Block Encoding

**定义**：设 $A \in \mathbb{C}^{N \times N}$，$\|A\| \leq 1$。若酉算符 $U \in \mathbb{C}^{(N+m) \times (N+m)}$ 满足：

$$(\langle 0|^{\otimes m} \otimes I) \cdot U \cdot (|0\rangle^{\otimes m} \otimes I) = A$$

即 $A$ 出现在 $U$ 的左上角 $N \times N$ 子块中，则称 $U$ 是 $A$ 的一个 $(1, m, 0)$-block encoding。

**图示**：
```
         ┌───────┐
|0^m⟩ ───┤       ├─── |0^m⟩（后选）
         │   U   │       ↓
|ψ⟩  ───┤       ├─── A|ψ⟩
         └───────┘
```

当辅助寄存器测量为 $|0^m\rangle$ 时，目标寄存器状态为 $A|\psi\rangle / \|A|\psi\rangle\|$。

**如何构造 Block Encoding**（Zhao et al. 中的具体方法）：

对稀疏矩阵 $A$，利用行/列 oracle $O_A: |i\rangle|j\rangle|0\rangle \to |i\rangle|j\rangle|A_{ij}\rangle$，构造 block encoding 的标准方法（Low & Chuang, 2019）：

```
q_row ── [O_A] ── [H^⊗m] ── ... ── [H^⊗m] ── [O_A†] ──
q_col ── [O_A] ── ...
q_val ── [Rot] ── ...
```

Zhao et al. 的量子 oracle 草图替代了 $O_A$，使得可以在不存储矩阵的情况下动态构造 block encoding。

#### 量子信号处理（QSP）

**定理（Gilyén et al. 2019）**：设 $U$ 是矩阵 $A$ 的 block encoding，$p(x)$ 是次数为 $d$ 的实多项式，$|p(x)| \leq 1$ 对 $x \in [-1,1]$。则存在量子电路 $W(U, \Phi)$（由 $d+1$ 个相位 $\Phi = (\phi_0, \ldots, \phi_d)$ 和 $O(d)$ 次 $U$ 及 $U^\dagger$ 调用构成），使得：

$$(\langle 0| \otimes I) \cdot W(U, \Phi) \cdot (|0\rangle \otimes I) = p(A)$$

其中 $p(A)$ 是将多项式作用于矩阵 $A$ 的特异值变换（Singular Value Transformation）：若 $A = \sum_j \sigma_j |u_j\rangle\langle v_j|$（SVD），则 $p(A) = \sum_j p(\sigma_j)|u_j\rangle\langle v_j|$。

**意义**：任何可以用多项式表示的矩阵函数，都可以高效量子化！

#### QSVT 统一的经典算法

| 矩阵函数 $p(x)$ | 应用 | 多项式近似 |
|:---|:---|:---|
| $p(x) = 1/x$（矩阵求逆） | HHL 线性方程组 | Chebyshev 近似，次数 $O(\kappa \log(1/\varepsilon))$ |
| $p(x) = e^{-ixt}$（矩阵指数） | 哈密顿量模拟 | Jacobi-Anger 展开，次数 $O(t + \log(1/\varepsilon))$ |
| $p(x) = \mathrm{sign}(x)$（符号函数） | 特征值过滤 | 最优多项式近似，次数 $O(\sqrt{\kappa}\log(1/\varepsilon))$ |
| $p(x) = x^k$（矩阵幂） | 幂方法（主特征向量） | 直接，次数 $k$ |
| $p(x) = \sqrt{x}$（矩阵平方根） | 量子漫步 | Chebyshev，次数 $O(\log(1/\varepsilon))$ |

#### Zhao et al. 中 QSVT 的具体应用

**应用 1：非均匀量子态草图（Quantum State Sketching）**

目标：将任意向量 $\vec{v} \in \mathbb{R}^D$（不一定稀疏）的量子态 $|v\rangle = \vec{v}/\|\vec{v}\|$ 从流式样本中制备出来。

方法：
1. 用 oracle 草图构造对角矩阵 $\hat{D} = \mathrm{diag}(v_1, \ldots, v_D)$ 的 block encoding；
2. 用 QSVT 实现 $p(x) = x / \|\vec{v}\|$（归一化）；
3. 作用于均匀叠加态 $\frac{1}{\sqrt{D}}\sum_i|i\rangle$，得到 $|v\rangle$。

**应用 2：PCA 的基态制备（Lemma F.24，基于 Lin-Tong 2020）**

目标：制备协方差矩阵 $\Sigma = X^\top X / N$ 的最大特征向量 $|\vec{w}_1\rangle$（即第一主成分）。

方法（Lin-Tong 基态制备算法）：
1. 构造 $\Sigma$ 的 block encoding $U_\Sigma$（由 oracle 草图给出）；
2. 设引导态 $|\phi_0\rangle$（满足 $|\langle\phi_0|\vec{w}_1\rangle| \geq \chi$）；
3. 用 QSVT 实现谱过滤多项式 $p_{\text{filter}}$，将 $|\phi_0\rangle$ 中 $|\vec{w}_1\rangle$ 的分量「放大」：

$$|\phi_0\rangle = \chi|\vec{w}_1\rangle + \sqrt{1-\chi^2}|\text{其他}\rangle \xrightarrow{p_{\text{filter}}} |\vec{w}_1\rangle + O(\varepsilon)$$

4. 谱过滤需要次数 $d = O(1/\chi \cdot \log(1/\varepsilon))$ 的多项式，对应 $O(d)$ 次 $U_\Sigma$ 调用；
5. 总样本数：$O(d) \times M_{\text{oracle-sketch}} = O(N/\chi\varepsilon^2 \cdot \text{polylog})$。

---

### 2.4　量子相位估计与线性系统求解的最优化

**最优量子线性系统求解器**（Costa et al. 2022 \[Ref 82\]）：

**改进 1**：将哈密顿量模拟替换为 **LCU（Linear Combination of Unitaries）**：

$$e^{iAt} = \sum_{k=0}^\infty \frac{(iAt)^k}{k!} \approx \sum_{k=0}^{K} \frac{(iAt)^k}{k!}$$

每个 $(iAt)^k$ 可以表示为 $A$ 的 block encoding 的乘积，截断后用 QSVT 实现多项式 $e^{ixt}$ 在特异值上的作用。复杂度：$O(t \log(1/\varepsilon))$（而非 HHL 的 $O(t \cdot s^2 \log(N))$）。

**改进 2**：用**离散绝热定理**（Discrete Adiabatic Theorem）替代振幅放大：

构造一系列中间矩阵 $A_0 = I \to A_1 \to \cdots \to A_K = A$，用绝热插值将 $|b\rangle$ 绝热地变换为 $A^{-1}|b\rangle$。条件数 $\kappa$ 进入绝热路径的长度，而非重复放大次数，降低了常数因子。

**最终复杂度**：

$$T_{\text{Costa}} = O\!\left(\kappa \log(\kappa / \varepsilon)\right) \times \text{cost}(U_A)$$

其中 $\text{cost}(U_A)$ 是矩阵 $A$ 的 block encoding 的门数（$O(s \log N)$）。

相比 HHL 的 $O(s^2 \kappa^2 \log^3(N) / \varepsilon)$，改进了 $O(s\kappa / \text{polylog})$ 因子。

---

<a name="第三章"></a>
## 第三章　变分量子算法（VQA）与 Barren Plateaus

### 3.1　变分量子本征解器（VQE）与 NISQ 范式（2014）

#### 问题设定

**变分原理**（Rayleigh-Ritz）：对任意归一化量子态 $|\psi\rangle$，哈密顿量 $H$ 的基态能量满足：

$$E_0 = \min_{|\psi\rangle} \langle\psi|H|\psi\rangle$$

**变分量子本征解器（VQE）** 将量子电路 $U(\vec{\theta})$ 参数化，用经典优化器最小化期望值：

$$E(\vec{\theta}) = \langle 0|U^\dagger(\vec{\theta}) H U(\vec{\theta})|0\rangle$$

#### 参数化量子电路（PQC）的结构

最常用的 PQC 结构是**层状硬件高效 ansatz（Hardware-Efficient Ansatz, HEA）**：

$$U(\vec{\theta}) = \prod_{l=1}^{L} \left(\prod_{i} R_{y}(\theta_{l,i}) \cdot \prod_{(i,j) \in E} \text{CNOT}_{ij}\right)$$

其中 $L$ 是电路层数，$E$ 是量子比特连接图（由设备物理结构决定），$R_y(\theta) = e^{-i\theta Y/2}$ 是 $y$ 轴旋转门。

**图示（4 量子比特，2 层）**：
```
q0 ─── Ry(θ00) ──●── Ry(θ10) ──●───
                  │              │
q1 ─── Ry(θ01) ──⊕──●── Ry(θ11)──⊕──●──
                     │              │
q2 ─── Ry(θ02) ──●──⊕── Ry(θ12) ──⊕──
                  │
q3 ─── Ry(θ03) ──⊕── Ry(θ13) ──────────
```

**分子哈密顿量的表示**（第二量子化，Jordan-Wigner 变换）：

对分子 $H = \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} h_{pqrs} a_p^\dagger a_q^\dagger a_r a_s$（$a_p^\dagger, a_q$ 是费米算符），

Jordan-Wigner 变换：$a_p^\dagger = \frac{X_p - iY_p}{2} \otimes \prod_{q<p} Z_q$，

转换后：$H = \sum_k c_k P_k$，其中 $P_k$ 是 Pauli 算符的张量积，$c_k$ 是实系数。

VQE 需要测量 $\langle P_k \rangle = \langle 0|U^\dagger P_k U|0\rangle$，总测量次数 $\sim O(\text{Pauli 项数})$，对 $N$ 个电子的分子约为 $O(N^4)$ 项。

#### 梯度计算：参数偏移规则（Parameter Shift Rule）

对 Pauli 旋转门 $R_\alpha(\theta) = e^{-i\theta \alpha / 2}$（$\alpha$ 是 Pauli 算符），期望值对 $\theta$ 的梯度满足：

$$\frac{\partial E}{\partial \theta} = \frac{1}{2}\left[E\!\left(\theta + \frac{\pi}{2}\right) - E\!\left(\theta - \frac{\pi}{2}\right)\right]$$

这是精确梯度（无有限差分近似误差），且只需将参数偏移 $\pm\pi/2$ 并重新执行量子电路估计期望值，无需量子内存。

对于更一般的门（如多控门），推广的参数偏移规则存在但形式更复杂（Wierichs et al. 2022）。

**完整梯度向量**：$\nabla_{\vec{\theta}} E$ 需 $2|\vec{\theta}|$ 次量子电路执行（每个参数正向和反向偏移各一次）。

---

### 3.2　QAOA：量子近似优化算法

**问题设定**：$n$ 变量二进制优化，目标函数 $C(\vec{z}) = \sum_\alpha c_\alpha \prod_{i \in \alpha} z_i$（$z_i \in \{0,1\}$）。

**QAOA 电路**（深度 $p$）：

$$|\vec{\gamma}, \vec{\beta}\rangle = \prod_{l=1}^p e^{-i\beta_l B} e^{-i\gamma_l C}|+\rangle^{\otimes n}$$

其中 $C = \sum_\alpha c_\alpha \prod_{i \in \alpha} Z_i$（问题哈密顿量），$B = \sum_i X_i$（混合器），$|+\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$。

**近似比保证**：
- 对 MaxCut，$p=1$ 的 QAOA 保证 $\geq 0.6924$ 近似比（Farhi et al. 2014）——优于最优经典局部算法，但劣于 Goemans-Williamson（$\approx 0.878$）；
- 对 $p \to \infty$，QAOA 趋于精确解（因为 $p$ 足够大时电路可表达任意酉算符）；
- **NLTS 猜想（现已证明，Anshu-Nirkhe 2023）** 表明存在具有量子 NLTS 性质的哈密顿量，为 QAOA 超越经典提供了理论可能，但具体界还未知。

---

### 3.3　Barren Plateaus：完整数学分析

#### 梯度消失的来源

**定理（McClean et al. 2018）**：设 $U(\vec{\theta})$ 从酉 2-设计（unitary 2-design）中随机采样（例如随机 Clifford 电路、足够深的随机量子电路），代价函数 $C(\vec{\theta}) = \langle 0|U^\dagger(\vec{\theta}) H U(\vec{\theta})|0\rangle$（$H$ 是全局可观测量，$\|H\| \leq 1$）。则对任意参数 $\theta_k$：

$$\mathbb{E}_{\vec{\theta}}[\partial_{\theta_k} C] = 0$$

$$\mathrm{Var}_{\vec{\theta}}[\partial_{\theta_k} C] \leq \frac{2\|H\|^2}{2^n - 1} \cdot \frac{1}{\text{poly}(d_L, d_R)}$$

其中 $d_L, d_R$ 是参数 $\theta_k$ 左侧和右侧子电路的 Hilbert 空间维数（均 $\sim 2^{n/2}$）。

**含义**：梯度方差 $\sim O(4^{-n})$，梯度幅度 $\sim O(2^{-n})$。在 $n = 50$ 量子比特时，梯度约为 $10^{-15}$，完全淹没在测量噪声中。

#### 证明草图（2-design 论证）

设参数 $\theta_k$ 是门 $W_k = e^{-i\theta_k P_k/2}$（$P_k$ 是 Pauli 算符）的旋转角。电路分为三部分：$U(\vec{\theta}) = U_R W_k U_L$。

梯度：$\partial_{\theta_k} C = \frac{i}{2}\langle 0|U_L^\dagger [P_k, U_R^\dagger H U_R] U_L|0\rangle$。

设 $G = U_R^\dagger H U_R$（经 $U_R$ 变换后的可观测量）。对 $U_R$ 在酉 2-design 上积分：

$$\mathbb{E}_{U_R}[G] = \frac{\mathrm{Tr}[H]}{2^n} I = 0 \quad \text{（若 } \mathrm{Tr}[H]=0\text{）}$$

$$\mathbb{E}_{U_R}[G_{ij} G_{kl}^*] = \frac{\delta_{il}\delta_{jk}\mathrm{Tr}[H^2] - \delta_{ij}\delta_{kl}\mathrm{Tr}[H]^2}{2^{2n}-2^n}$$

由此得方差 $\mathrm{Var}[C] = O(\mathrm{Tr}[H^2] / (2^{2n})) = O(1/4^n)$（全局 $H$ 满足 $\mathrm{Tr}[H^2] = O(2^n)$）。

#### 局部代价函数缓解 BP

**定理（Cerezo et al. 2021 \[Ref 38\]）**：若代价函数为**局部**可观测量之和 $C_{\text{local}} = \frac{1}{n}\sum_{i=1}^n \langle 0|U^\dagger Z_i U|0\rangle$（$Z_i$ 是第 $i$ 个量子比特的 $Z$ 算符），则：

$$\mathrm{Var}[\partial_{\theta_k} C_{\text{local}}] \geq \frac{1}{\text{poly}(n)} \cdot \frac{1}{4^n / \text{poly}(n)} = \frac{1}{\text{poly}(n)}$$

即梯度消失仅为多项式级别（而非指数级别），在实践中是可训练的。

**物理解释**：全局可观测量（如哈密顿量总能量）平均了所有量子比特的贡献，导致梯度「平滑掉」；局部可观测量保留了近邻量子比特间的关联梯度信号。

#### 「BP 无 ↔ 经典可模拟」困境的精确表述

**定理（Cerezo et al. 2023 \[Ref 18\]）**：在以下条件下，若 PQC 结构保证 $\mathrm{Var}[\partial_\theta C] \geq 1/\text{poly}(n)$（无 BP），则该 PQC 是经典可模拟的：

1. 该 PQC 具有有界纠缠熵（区域律，如矩阵乘积态结构）；
2. 对一维几何拓扑的电路，有界纠缠意味着 MPS 表示的 bond dimension $\leq \text{poly}(n)$，可经典模拟。

**反例的可能性**：上述定理的条件（有界纠缠熵）在二维拓扑或专用问题（如量子化学）中可能不成立，因此存在逃脱该困境的空间（第九章详述）。

---

### 3.4　梯度估计的实现细节

#### 自然梯度（Quantum Natural Gradient, QNG）

经典梯度下降在参数空间（欧式度量）进行更新：$\vec{\theta}_{t+1} = \vec{\theta}_t - \eta \nabla E$。

**问题**：参数空间的欧式度量不等于量子态 Hilbert 空间的 Fubini-Study 度量，导致优化路径在量子态几何中非最优。

**量子自然梯度（Stokes et al. 2020）**：使用 Quantum Fisher 信息矩阵（QFIM）$F$ 作为黎曼度量：

$$\vec{\theta}_{t+1} = \vec{\theta}_t - \eta F^{-1}(\vec{\theta}_t) \nabla E(\vec{\theta}_t)$$

其中 $F_{jk} = \mathrm{Re}\left[\langle\partial_{\theta_j}\psi|\partial_{\theta_k}\psi\rangle - \langle\partial_{\theta_j}\psi|\psi\rangle\langle\psi|\partial_{\theta_k}\psi\rangle\right]$。

**计算代价**：QFIM 的精确计算需 $O(|\vec{\theta}|^2)$ 次量子电路执行，对于深层 VQA 代价极高；实践中用块对角近似或随机估计。

#### SPSA（同步摄动随机近似）

完全无需梯度的黑盒优化算法：

$$\vec{\theta}_{t+1} = \vec{\theta}_t - a_t \cdot \frac{E(\vec{\theta}_t + c_t\vec{\Delta}_t) - E(\vec{\theta}_t - c_t\vec{\Delta}_t)}{2c_t} \cdot \vec{\Delta}_t^{-1}$$

其中 $\vec{\Delta}_t$ 是随机符号向量（Bernoulli $\pm 1$），只需 **2 次**电路执行即可估计伪梯度，与参数数量无关。

---

<a name="第四章"></a>
## 第四章　量子核方法与去量子化

### 4.1　量子核方法的完整理论

#### 核方法背景

**核技巧（Kernel Trick）**：对于不可线性分离的数据，将特征映射 $\phi: \mathcal{X} \to \mathcal{H}$（$\mathcal{H}$ 是（可无限维的）特征 Hilbert 空间），在特征空间中进行线性学习，等价于使用核函数：

$$k(\vec{x}, \vec{x}') = \langle\phi(\vec{x}), \phi(\vec{x}')\rangle_{\mathcal{H}}$$

**优点**：无需显式计算 $\phi(\vec{x})$，只需能计算 $k(\vec{x}, \vec{x}')$；特征空间可以是无限维（如 Gaussian 核的 RKHS）。

**支持向量机（SVM）**：最大化间隔（margin）分类超平面：

$$\min_{\vec{\alpha}} \frac{1}{2}\vec{\alpha}^\top K \vec{\alpha} - \vec{1}^\top \vec{\alpha} \quad \text{s.t.} \quad \vec{\alpha}^\top \vec{y} = 0, \; 0 \leq \alpha_i \leq C$$

其中 $K_{ij} = k(\vec{x}_i, \vec{x}_j)$（核矩阵），$\vec{\alpha}$ 是对偶变量。

#### 量子特征映射

**定义**：给定经典输入 $\vec{x} \in \mathbb{R}^d$，参数化量子电路 $U(\vec{x})$ 作用于 $|0\rangle^{\otimes n}$ 生成量子态：

$$\phi(\vec{x}) = U(\vec{x})|0\rangle^{\otimes n} \in \mathbb{C}^{2^n}$$

量子核函数：

$$k_Q(\vec{x}, \vec{x}') = |\langle 0^n | U^\dagger(\vec{x}')U(\vec{x}) | 0^n\rangle|^2$$

**典型量子特征映射**（IQP，Instantaneous Quantum Polynomial 电路）：

$$U(\vec{x}) = H^{\otimes n} \cdot \exp\left(i\sum_{j<k} x_j x_k Z_j Z_k\right) \cdot H^{\otimes n} \cdot \exp\left(i\sum_j x_j Z_j\right)$$

这对应于将 $\vec{x}$ 的所有二阶交叉项编码为量子相位，再加上一层 Hadamard（全局纠缠）。

**指数大特征空间**：$n$ 量子比特给出 $2^n$ 维特征空间。理论上，量子核可以表达 $2^n \times 2^n$ 大小的核矩阵，相当于在 $2^n$ 维空间中的内积——远超任何多项式阶经典核。

#### 量子核矩阵的估计

对训练集 $\{\vec{x}_1, \ldots, \vec{x}_N\}$，需估计 $N \times N$ 核矩阵 $K$。每个元素：

$$K_{ij} = |\langle 0^n | U^\dagger(\vec{x}_j)U(\vec{x}_i) | 0^n\rangle|^2$$

通过 Hadamard 测试电路：
```
|0⟩ ── H ──────── ●─────── H ── 测量
                  │
|0^n⟩ ── U(xi) ──U†(xj)─────── （不测量）
```

对辅助比特测量 $0$ 的概率为 $(1 + \mathrm{Re}[K_{ij}])/2$，测量 $L$ 次可估计 $K_{ij}$ 到精度 $\varepsilon \sim 1/\sqrt{L}$。

**实际挑战**：
1. 需要估计 $O(N^2)$ 个核矩阵元素，每个需 $O(1/\varepsilon^2)$ 次量子电路；
2. 总计 $O(N^2/\varepsilon^2)$ 次量子电路执行——对 $N = 10^4$, $\varepsilon = 0.01$，约 $10^{12}$ 次，远超 NISQ 能力；
3. 解决方案：采样核矩阵（只估计随机选取的 $O(N)$ 个元素），或使用 Nyström 近似。

#### Liu et al.（2021）的严格量子核优势

**构造**：基于**离散对数（DLOG）假设**，定义一个特殊的监督学习任务：

- 数据：$(\vec{x}, y)$，其中 $\vec{x} = g^r \pmod{p}$（$g$ 是 $\mathbb{Z}_p$ 的生成元，$r$ 随机），$y = f(r)$（某个简单函数）；
- 量子特征映射：$|\phi(\vec{x})\rangle = $ 量子傅里叶变换（QFT）作用于 $\vec{x}$ 的量子编码，对应在 $\mathbb{Z}_p$ 上的傅里叶基；
- 量子核：$k_Q(\vec{x}, \vec{x}') = |\langle\text{QFT}(\vec{x})|\text{QFT}(\vec{x}')\rangle|^2 = $ DFT 核。

**定理**：量子 SVM（使用上述量子核）以 $O(N)$ 个训练样本和 $O(\log N)$ 的预测时间学习该任务；任何经典分类器，若不能求解 DLOG，则需指数多训练样本。

**局限**：该任务是**人工构造的**，自然数据集中不太可能出现 DLOG 结构。

---

### 4.2　去量子化的完整理论

#### Tang（2019）的核心技术

**采样查询（SQ）模型**：对向量 $\vec{v} \in \mathbb{R}^N$，提供以下 oracle 访问：
- `sample(v)`：以概率 $v_i^2 / \|\vec{v}\|^2$ 返回下标 $i$；
- `query(v, i)`：返回近似值 $\tilde{v}_i \approx v_i$（误差 $\varepsilon$）；
- `norm(v)`：返回 $\|\vec{v}\|^2$ 的近似值。

**关键引理（FKV 引理，Frieze-Kannan-Vempala 2004）**：在 SQ 模型下，可以在 $O(k^3/\varepsilon^2)$ 时间内（与 $N$ 无关！）计算矩阵 $A = A_1 + \varepsilon_{\text{approx}} \cdot \text{noise}$ 的 $k$ 秩近似，其中 $A_1$ 是最优 $k$ 秩近似。

**Tang 2019 推荐系统算法**：

输入：用户偏好矩阵 $A \in \mathbb{R}^{m \times n}$（用户 × 物品），对用户 $i$ 的行 $A_i$ 具有 SQ oracle 访问。

输出：对用户 $i$ 的 $k$ 秩推荐（前 $k$ 个最相关物品的近似）。

```python
def dequantized_recommendation(A, i, k, epsilon):
    # 步骤 1：对行 A_i 均匀采样，利用 SQ oracle 重要性采样
    S = sample_rows(A, s=O(k/epsilon^2))  # 采样 s 行
    
    # 步骤 2：在采样子矩阵上计算近似 SVD（厄米近似）
    V = approximate_SVD(A[S,:], k)  # O(sk^2/epsilon^2) 时间
    
    # 步骤 3：将查询行 A_i 投影到近似奇异向量
    for each test item j:
        score[j] = project(A_i, V) dot e_j  # O(sk) 时间
    
    return top_k(score)
```

**时间复杂度**：$O(\text{poly}(k, 1/\varepsilon) \cdot \text{polylog}(mn))$（与 $m, n$ 成对数关系）。

量子推荐系统的「指数加速」变为对数因子（polylog），加速来源于 SQ 访问的随机采样能力，而非量子叠加本身。

#### Zhao et al. 如何规避去量子化

**关键差异**：在流式模型中：
- **没有 SQ oracle**：机器只收到一次性流式样本 $z_t \sim \mathcal{D}$，无法反复对同一样本「查询」其精确值；
- **没有重要性采样预处理**：SQ 模型假设已知 $\|\vec{v}\|^2$（需要预扫描全数据集），流式模型无此保证；
- **空间有界**：即使在处理样本时积累了部分信息，总内存量小于 $O(N^{0.99})$，无法存储 Tang 算法的 SQ 数据结构（需 $\Omega(N)$ 空间）。

因此，去量子化在流式空间有界模型下**原则上无法应用**，这正是 Zhao et al. 优势的根本来源。

---

### 4.3　量子核的可去量子化边界

#### 随机 Fourier 特征（RFF）作为量子核的经典近似

对**移位不变核**（shift-invariant kernel）$k(\vec{x}, \vec{x}') = k(\vec{x} - \vec{x}')$，Bochner 定理：

$$k(\vec{x} - \vec{x}') = \int_{\mathbb{R}^d} p(\vec{\omega}) e^{i\vec{\omega}^\top(\vec{x}-\vec{x}')} d\vec{\omega}$$

其中 $p(\vec{\omega})$ 是核的 Fourier 变换（在 RBF 核情形为高斯分布）。

**随机 Fourier 特征（RFF，Rahimi & Recht 2007）**：采样 $D$ 个随机频率 $\vec{\omega}_1, \ldots, \vec{\omega}_D \sim p(\vec{\omega})$，构造近似特征映射：

$$\phi_{\text{RFF}}(\vec{x}) = \frac{1}{\sqrt{D}}\left[\cos(\vec{\omega}_1^\top\vec{x}+b_1), \ldots, \cos(\vec{\omega}_D^\top\vec{x}+b_D)\right]^\top$$

则 $k(\vec{x},\vec{x}') \approx \phi_{\text{RFF}}(\vec{x}) \cdot \phi_{\text{RFF}}(\vec{x}')$，误差以概率 $\geq 1-\delta$ 不超过 $\varepsilon$，需 $D = O(\log(N/\delta)/\varepiv2)$ 个特征。

**当 RFF 可去量子化量子核**（arXiv:2503.23931, 2025）：

若量子核 $k_Q$ 是平移不变的，且其 Fourier 变换 $p_Q(\vec{\omega})$ 可以被 RFF 采样高效近似，则经典 RFF-SVM 可以达到与量子 SVM 相当的泛化性能，去量子化成立。

**两个必要条件**：

1. **核对齐（Kernel Alignment）**：$\sum_{ij} y_i y_j k_Q(\vec{x}_i, \vec{x}_j) \geq c \cdot \sum_{ij} y_i y_j k_{\text{RFF}}(\vec{x}_i, \vec{x}_j)$（量子核与经典近似核在训练集上的对齐程度 $\geq c > 0$）；
2. **谱浓缩（Spectral Concentration）**：$p_Q(\vec{\omega})$ 主要集中在有界区域，使得 $D = O(\text{poly}(d, 1/\varepsilon))$ 个 RFF 采样足够。

**何时量子核保留优势**：当数据具有量子特定的非平移不变结构（例如编码了量子电路的干涉项），RFF 无法表示，量子核有本质优势。区分这两种情况需要分析数据的 Fourier 谱结构。

---

<a name="第五章"></a>
## 第五章　量子学习理论：优势的严格证明体系

### 5.1　经典 PAC 学习框架回顾

#### Probably Approximately Correct（PAC）学习

**设定**：
- 假设类 $\mathcal{H}$：目标概念集合；
- 未知分布 $\mathcal{D}$：数据分布；
- 训练样本 $S = \{(\vec{x}_i, h^*(\vec{x}_i))\}_{i=1}^m$（$h^* \in \mathcal{H}$ 是目标概念）；
- 学习算法 $\mathcal{A}$：以 $S$ 为输入，输出假设 $\hat{h} \in \mathcal{H}$。

**目标**：以样本数 $m = O((\log|\mathcal{H}| + \log(1/\delta)) / \varepsilon)$，以概率 $\geq 1-\delta$ 输出 $\hat{h}$ 满足：

$$\Pr_{\vec{x} \sim \mathcal{D}}[\hat{h}(\vec{x}) \neq h^*(\vec{x})] \leq \varepsilon$$

**VC 维**：$\text{VC}(\mathcal{H})$ 是 $\mathcal{H}$ 可以打散（shatter）的最大点集大小。PAC 样本复杂度：

$$m = \Theta\!\left(\frac{\text{VC}(\mathcal{H}) + \log(1/\delta)}{\varepsilon}\right)$$

#### 量子 PAC 学习框架

**量子样本**：经典样本 $(\vec{x}, y)$ 的量子版本是量子训练态：

$$\rho_{\text{train}} = \sum_{(\vec{x}, y)} \mathcal{D}(\vec{x}) |\vec{x}, y\rangle\langle \vec{x}, y|$$

或（具有相位信息的）纯态版本：$|\psi_{\text{train}}\rangle = \sum_{\vec{x}} \sqrt{\mathcal{D}(\vec{x})} |\vec{x}\rangle|h^*(\vec{x})\rangle$。

**量子-经典样本复杂度对比**（Bshouty-Jackson 1995, Atici-Servedio 2005）：

| 学习问题 | 经典样本 | 量子样本 |
|:---|:---:|:---:|
| DNF 公式 | $O(2^n)$ | $O(n^{3/2}/\varepsilon)$（指数减少） |
| 布尔函数 $s$-稀疏 Fourier | $O(s \log n / \varepsilon^2)$ | $O(s \log(1/\varepsilon))$ |
| $k$-CNF | $O(n^k / \varepsilon)$ | $O(n^{k/2})$ |

**量子 ORACLe 访问的优势来源**：量子样本可以提供输入的叠加（即 Fourier 变换的自然实现），使得函数的「频域」结构更容易识别。

---

### 5.2　量子与经典内存学习的指数分离（Chen et al. 2022）

#### 模型设定

**n 量子比特量子态层析（Tomography）**：未知量子态 $\rho$（$n$ 量子比特），通过重复实验（每次独立复制 $\rho$）来学习 $\rho$ 的某些性质。

**两种学习者**：
1. **无量子内存（classical memory）**：每次对 $\rho$ 进行测量，立即将结果存储为经典比特，累积 $m$ 个经典测量结果后进行推断；
2. **有量子内存（quantum memory）**：可以在内存中积累多个 $\rho$ 的副本，同时对多个副本进行联合量子测量后再推断。

**结论（Chen et al. 2022 \[Ref 110\]）**：

对任何 $n$ 量子比特密度矩阵 $\rho$，预测所有单量子比特约化密度矩阵（即所有 $\rho_i = \mathrm{Tr}_{\neq i}[\rho]$）：
- **无量子内存**：需要 $m = \Omega(2^{n/2})$ 个实验；
- **有量子内存**：$m = O(n)$ 个实验。

**分离大小**：指数级！$2^{n/2}$ vs. $n$。

#### 证明的核心思路

**下界**（无量子内存）：

设测量策略是每次对一个 $\rho$ 副本施加量子测量 $\{M_a\}$，获得经典结果 $a$。所有经典信息来自 $m$ 个独立测量结果 $(a_1, \ldots, a_m)$。

关键引理（Holevo 界推广）：经过 $m$ 次单独测量，无量子内存学习者所获得的关于 $\rho$ 的互信息量不超过：

$$I(\rho; a_1, \ldots, a_m) \leq m \cdot n \quad \text{（每次至多 } n \text{ 比特）}$$

而预测所有 $n$ 个约化密度矩阵（每个需要估计 $O(1)$ 个参数）需要 $\Omega(n)$ 比特互信息——但因信息提取效率低，需要 $\Omega(2^{n/2})$ 次测量才能积累足够信息（具体由 $n$-qubit 量子态参数空间的局部几何决定）。

**上界**（有量子内存）：

量子内存学习者可以同时持有多个 $\rho$ 副本，对其施加**交换测试（SWAP test）** 或更一般的 $t$ 拷贝测试（$t$-copy test）。通过 $n$ 个精心设计的测量（每次测量 $O(1)$ 个 $\rho$ 副本），可以提取所有约化密度矩阵的信息。

---

### 5.3　量子统计查询（QSQ）框架与 Lewis et al. 2026 的结果

#### QSQ 模型

**经典统计查询（SQ）模型（Kearns 1998）**：学习算法不直接访问训练样本，而是通过以下 oracle 查询：

$$\text{STAT}(h, \tau): \text{返回 } \hat{\mu} \text{ 满足 } |\hat{\mu} - \mathbb{E}_{(\vec{x},y)\sim\mathcal{D}}[h(\vec{x},y)]| \leq \tau$$

其中容忍度 $\tau$ 是参数，查询复杂度以统计查询次数计。

SQ 学习的核心优势：在此模型下证明的学习复杂度下界对**所有随机化算法**（包括量子算法）都适用——因为量子算法也只能通过测量间接访问数据。

**量子统计查询（QSQ）模型（Arunachalam & Maity 2023）**：将经典统计查询推广到量子设定，学习者可以查询：

$$\text{QSTAT}(\mathcal{E}, \tau): \text{返回量子信道 } \hat{\mathcal{E}} \text{ 满足 } \|\hat{\mathcal{E}} - \mathbb{E}_{(\vec{x},y)\sim\mathcal{D}}[\mathcal{E}_{(\vec{x},y)}]\|_\diamond \leq \tau$$

其中 $\|\cdot\|_\diamond$ 是钻石范数（diamond norm），量子信道 $\mathcal{E}_{(\vec{x},y)}$ 编码了训练样本的量子信息。

#### Lewis, Gilboa & McClean（2026）的自然分布优势

**问题设定**：学习两层神经网络 $f(\vec{x}) = \sum_{r=1}^k a_r \sigma(\vec{w}_r^\top \vec{x})$（$k$ 个神经元），其中激活函数 $\sigma$ 是奇函数（如 ReLU）。

数据分布：$\vec{x} \sim \mathcal{N}(0, I_d)$（标准高斯分布）或更一般的广义高斯分布。

**定理（Lewis et al. 2026 \[Ref 外\]）**：

在 QSQ 框架下：
- **量子算法**：$O(d^{0.99} \cdot k^2 / \varepsilon^2)$ 次 QSQ 查询（$\tau = \varepsilon$）可以学习两层神经网络到误差 $\varepsilon$；
- **经典 SQ 算法**：任何经典 SQ 算法需要 $\Omega(d^{1.5})$ 次查询（在高斯分布上）。

当 $d = n$（参数维度 = 量子比特数），分离约为 $d^{0.5}$ 倍。但更重要的是，**这是第一个在自然分布（高斯）上对学习自然函数类（浅层神经网络）给出严格量子优势的结果**。

**证明思路**：
- 量子优势来自于量子态的叠加可以并行计算所有神经元权重的 Fourier 分量（类似量子傅里叶采样）；
- 经典下界来自 SQ 模型下高斯分布上矩函数的计算下界（Fourier 度 $\geq 3$ 的单项式需要指数次查询）。

---

### 5.4　样本复杂度与 VC 维的量子类比

#### 量子 VC 维（QVC）

**定义（Arunachalam & de Wolf 2018）**：量子假设类 $\mathcal{H}_Q$（参数化量子电路族）的量子 VC 维 $\text{QVC}(\mathcal{H}_Q)$ 定义为：

$$\text{QVC}(\mathcal{H}_Q) = \max\{m : \exists \vec{x}_1,\ldots,\vec{x}_m, \forall \vec{y} \in \{0,1\}^m, \exists h \in \mathcal{H}_Q, h(\vec{x}_i) = y_i\}$$

即 $\mathcal{H}_Q$ 可以打散的最大点集大小。

**对于量子核分类器**：$\text{QVC}(\mathcal{H}_{k_Q}) = O(n^2)$（$n$ 量子比特），对应 $2^{2n}$ 维核矩阵的秩上界。但这个维度与样本复杂度的关系仍不完全清楚。

**实用泛化界（Huang et al. 2021）**：对 $n$ 量子比特量子核 SVM，泛化误差界：

$$R[\hat{h}] \leq \hat{R}_N[\hat{h}] + O\!\left(\sqrt{\frac{n \log(Nn)}{N}}\right)$$

其中 $\hat{R}_N$ 是训练误差，$N$ 是训练样本数。这表明量子核方法在 $N \gg n$ 时具有良好的泛化性，但对于量子优势而言，需要证明经典方法无法达到同样的训练误差（即 $n$ 维问题的 SVM 边界）。

---

<a name="第六章"></a>
## 第六章　经典影子与量子态层析

### 6.1　经典影子协议的完整描述

#### 背景：量子态层析的代价

完整的量子态层析（State Tomography）需要估计密度矩阵 $\rho$（$n$ 量子比特）的所有 $4^n$ 个独立参数。精度 $\varepsilon$ 的层析：
- 经典测量：需要 $O(4^n / \varepsilon^2)$ 个独立 $\rho$ 副本；
- 最优量子策略（联合测量）：仍需 $\Omega(4^n)$ 个 $\rho$ 副本（因为参数数量本身是 $4^n$）。

**问题**：很多实际任务只需要预测 $\rho$ 的**少量**可观测量 $\{O_1, \ldots, O_K\}$（$K \ll 4^n$），完整层析是浪费的。

#### 经典影子协议（Huang, Kueng & Preskill 2020 \[Ref 74\]）

**直觉**：随机测量提供了 $\rho$ 的"快照"（shadow），足够多的快照允许预测任意可观测量，但测量次数只与**最难预测的单个可观测量**有关，而非 $K$。

**协议**（单次测量）：

1. **随机旋转**：从 ensemble $\mathcal{U}$（如随机 Clifford 群或 Pauli 测量基）均匀采样酉算符 $U$；
2. **作用**：将 $U$ 施加到 $\rho$：$U\rho U^\dagger$；
3. **测量**：在计算基 $\{|b\rangle\}$ 上测量，得到结果 $b \in \{0,1\}^n$；
4. **影子构造**：定义「影子」为：

$$\hat{\sigma} = \mathcal{M}^{-1}(U^\dagger|b\rangle\langle b|U)$$

其中 $\mathcal{M}: \rho \mapsto \mathbb{E}_{U,b}[U^\dagger|b\rangle\langle b|U]$ 是测量信道（已知的线性映射），$\mathcal{M}^{-1}$ 是其逆（若存在）或伪逆。

**关键**：$\mathbb{E}[\hat{\sigma}] = \rho$，即每个影子是 $\rho$ 的无偏估计量。

#### 影子范数（Shadow Norm）

**定义**：可观测量 $O$ 的影子范数为：

$$\|O\|_{\text{shadow}} = \max_{\text{纯态 }\sigma} \sqrt{\mathbb{E}_{U,b}[(\langle b|U O U^\dagger|b\rangle)^2 / \|\mathcal{M}^{-1}(U^\dagger|b\rangle\langle b|U)\|^2]}$$

**重要性**：使用中位平均估计器，以概率 $\geq 1-\delta$ 估计 $K$ 个可观测量 $\{O_1, \ldots, O_K\}$ 到精度 $\varepsilon$，所需影子数：

$$T = O\!\left(\frac{\log(K/\delta)}{\varepsilon^2} \cdot \max_i \|O_i\|^2_{\text{shadow}}\right)$$

**注意**：$T$ 与 $K$ 是**对数**关系（而非线性）！

#### 不同 ensemble 的影子范数

**全局随机 Clifford 影子**：对 $n$ 量子比特的任意可观测量 $O$，

$$\|O\|^2_{\text{shadow}} = 2^n \cdot \|O\|_F^2 / \|O\|^2 = O(2^n)$$

特点：可预测**任意** $K$ 个可观测量，但 $T$ 随 $n$ 指数增加。

**局部 Pauli 影子（Pauli 随机旋转）**：每个量子比特独立地随机旋转（从 $\{X, Y, Z\}$ 测量基中选一个），则对 $k$-局部可观测量（最多涉及 $k$ 个量子比特）：

$$\|O\|^2_{\text{shadow}} = 3^k \cdot \|O\|^2_{\infty}$$

特点：对 $k$-局部可观测量，$T = O(3^k \log K / \varepsilon^2)$（与 $n$ 完全无关！）。适合 Pauli 算符、局部项哈密顿量等。

**浅层随机电路影子**（Hu et al. 2025）：深度 $O(1)$ 的随机 Clifford 电路提供了两者的折中：
- 对 $k$-局部可观测量的影子范数 $\sim 2^k$（比 Pauli 影子好，因为 $3^k > 2^k$）；
- 对非局域可观测量可提供比全局 Clifford 更小的 $T$（如非局域 Pauli 算符）。

#### 影子的经典后处理

**中位平均估计器（Median of Means）**：

将 $T$ 个影子分成 $K'$ 组，每组 $T/K'$ 个影子。对每组计算均值估计：

$$\hat{o}_g(O) = \frac{K'}{T}\sum_{t \in \text{第 } g \text{ 组}} \mathrm{Tr}[O\hat{\sigma}_t]$$

最终估计取各组均值的中位数：$\hat{o}(O) = \mathrm{median}(\hat{o}_1(O), \ldots, \hat{o}_{K'}(O))$。

**Chernoff 界保证**：$K' = O(\log(K/\delta))$ 时，以概率 $\geq 1-\delta$ 对所有 $K$ 个可观测量同时成立 $|\hat{o}(O_i) - \mathrm{Tr}[O_i\rho]| \leq \varepsilon$。

---

### 6.2　干涉型经典影子的完整协议

#### 问题：线性分类器的符号提取

量子 oracle 草图制备了权重量子态 $|w^*\rangle \propto \vec{w}^*$（LS-SVM 解的量子版本）。对稀疏测试向量 $\vec{x}' = \sum_{j \in \text{supp}(\vec{x}')} x'_j |j\rangle$（支撑集大小 $s$），预测任务是：

$$\hat{y} = \mathrm{sign}\!\left(\sum_{j} x'_j \langle j | w^*\rangle\right) = \mathrm{sign}(\langle\vec{x}'|w^*\rangle)$$

需要提取内积 $\langle\vec{x}'|w^*\rangle$ 的实部符号。

#### 制备 $|\vec{x}'\rangle$ 的量子电路

对 $s$-稀疏向量 $\vec{x}' = \sum_{j=1}^s x'_j |f_j\rangle$（$f_j$ 是非零分量下标），构造酉算符 $V_{x'}$ 使得 $V_{x'}|0\rangle = |\vec{x}'\rangle / \|\vec{x}'\|$：

```
|0⟩ ──── H^{log s} ──── [稀疏态制备网络] ────
```

稀疏态制备网络：$O(s)$ 个旋转门和 CNOT 门（Shende-Markov-Bullock 分解，2006），电路深度 $O(s)$。

#### 干涉型经典影子完整电路

**单次测量电路**：

```
辅助比特  |0⟩ ─── H ────────────────── ●─────── H ─── 测量（结果 b）
                                        │
权重寄存器 |w*⟩ ─────────────────── [cV_{x'}] ──────── （不测量）
```

其中 `cV_{x'}` 是以辅助比特控制的 $V_{x'}$ 门：当辅助比特为 $|1\rangle$ 时，施加 $V_{x'}^\dagger$。

**测量统计**：

$$\Pr[b=0] = \frac{1}{2}\left(1 + \frac{\mathrm{Re}[\langle 0^n | V_{x'}^\dagger | w^*\rangle]}{\||\vec{x}'\rangle\| \cdot \||w^*\rangle\|}\right) = \frac{1 + \mathrm{Re}[\langle \vec{x}' | w^*\rangle] / \|\vec{x}'\|}{2}$$

$$\Pr[b=1] = \frac{1 - \mathrm{Re}[\langle \vec{x}' | w^*\rangle] / \|\vec{x}'\|}{2}$$

故期望（经 $T$ 次测量取平均）：

$$\hat{\mu}_{x'} = \frac{1}{T}\sum_{t=1}^T (1 - 2b_t) = \frac{\mathrm{Re}[\langle\vec{x}'|w^*\rangle]}{\|\vec{x}'\|}$$

符号 $\mathrm{sign}(\hat{\mu}_{x'}) = \mathrm{sign}(\mathrm{Re}[\langle\vec{x}'|w^*\rangle])$。

**单个测试向量所需测量次数**：$T_{x'} = O(1/\varepsilon^2)$（Hoeffding 界）。

#### 离线预测：覆盖网络论证

**挑战**：若有 $K$ 个测试向量，每个需 $O(1/\varepsilon^2)$ 次测量，总计 $O(K/\varepsilon^2)$ 次——当 $K$ 很大时不可接受。

**关键观察**：所有测试向量都是 $s$-稀疏的，在 $\ell_2$ 单位球面上的 $s$-稀疏向量集 $\mathcal{S}_s = \{\vec{x}' \in \mathbb{R}^D : \|\vec{x}'\|=1, |\text{supp}(\vec{x}')| \leq s\}$ 的 $\varepsilon_0$-covering net 大小为：

$$|\mathcal{N}_{\varepsilon_0}| \leq \binom{D}{s} \cdot (3/\varepsilon_0)^s \leq \left(\frac{3D}{s\varepsilon_0}\right)^s$$

故 $\log|\mathcal{N}_{\varepsilon_0}| = O(s \log(D/s\varepsilon_0))$。

**离线构建预测模型**（Theorem F.16 完整证明草图）：

1. 对 $\varepsilon_0$-covering net 中的每个 $\vec{x}' \in \mathcal{N}_{\varepsilon_0}$，执行 $T = O(\log(|\mathcal{N}_{\varepsilon_0}|/\delta)/\varepsilon^2)$ 次干涉型影子测量，估计 $\langle\vec{x}'|w^*\rangle$ 的实部；

2. 存储所有 $|\mathcal{N}_{\varepsilon_0}|$ 个估计值（总存储量 $O(|\mathcal{N}_{\varepsilon_0}|)$ 个实数，但实际上不需要——只需存储每个估计的符号，或直接计算）；

3. 对任意 $s$-稀疏测试向量 $\vec{x}''$，找到其在 $\mathcal{N}_{\varepsilon_0}$ 中的最近邻 $\vec{x}'$（$\|\vec{x}'' - \vec{x}'\| \leq \varepsilon_0$），用 $\mathrm{sign}(\langle\vec{x}'|w^*\rangle)$ 作为 $\mathrm{sign}(\langle\vec{x}''|w^*\rangle)$ 的预测（Lipschitz 连续性保证误差传播 $\leq \varepsilon_0$）。

**总测量次数**：

$$T_{\text{total}} = |\mathcal{N}_{\varepsilon_0}| \cdot T = O\!\left(\left(\frac{3D}{s\varepsilon_0}\right)^s \cdot \frac{s\log(D/s\varepsilon_0) + \log(1/\delta)}{\varepsilon^2}\right)$$

取 $s = O(\log D)$（对应论文的稀疏性假设 $s \leq \mathrm{poly}(\log D)$），则 $|\mathcal{N}_{\varepsilon_0}| = D^{O(\log D)} = \mathrm{poly}(D)$（准多项式），$T_{\text{total}} = \mathrm{poly}(D, 1/\varepsilon) \cdot \log(1/\delta)$。

---

<a name="第七章"></a>
## 第七章　量子流式算法与空间复杂度分离

### 7.1　通信复杂度基础

#### 双方通信模型

**设定**：Alice 持有 $x \in \mathcal{X}$，Bob 持有 $y \in \mathcal{Y}$，目标是计算函数 $f(x,y)$。通信复杂度 $\text{CC}(f)$ 是最优协议（最坏情况下）交换的比特数。

**典型例子**：

- **相等性检验（Equality）**：$f(x,y) = [x = y]$，$\text{CC}(\text{EQ}) = O(\log(1/\delta))$（随机化，利用哈希），$\text{CC}_{\text{det}}(\text{EQ}) = \Theta(n)$（确定性）；
- **内积（Inner Product）**：$f(x,y) = \langle x, y\rangle \pmod{2}$，$\text{CC}_R(\text{IP}) = \Omega(n)$（随机化下界）；
- **不相交集（Disjointness）**：$f(x,y) = [\exists i: x_i = y_i = 1]$，$\text{CC}_R(\text{DISJ}) = \Omega(n)$（著名的 Razborov 1992 下界）。

**量子通信复杂度**：Alice 和 Bob 可以交换量子比特并使用纠缠，$\text{QCC}(f) \leq \text{CC}(f)$，某些问题有指数分离（如 Raz 1999 的矩阵秩问题）。

#### Query-to-Communication Lifting 定理

**核心定理（Chattopadhyay-Mande-Sherif 2019，本文推广）**：

设函数 $f: \{0,1\}^n \to \{0,1\}$ 的确定性查询复杂度（decision tree complexity）为 $D(f)$，选取合适的「提升函数」$g: \{0,1\}^b \times \{0,1\}^b \to \{0,1\}$（如 Index 函数），则组合函数 $f \circ g^n$ 的确定性通信复杂度满足：

$$\text{CC}_{\text{det}}(f \circ g^n) = \Omega(D(f) \cdot b)$$

对随机化版本（相差 $\mathrm{poly}(n)$ 对数因子），类似结果成立。

**Zhao et al. 的应用**：将流式算法的内存下界通过提升归约到通信复杂度下界：

```
NOPE 的流式复杂度
   ↓ 归约（分配内存到 Alice/Bob）
通信复杂度问题（Forrelation 的通信版本）
   ↓ 下界（经典查询复杂度 Q_C 的 lifting）
通信复杂度下界 = Ω(N · Q_C)
   ↓ 转换
流式复杂度下界: M · S ≥ Ω(N · Q_C)
```

---

### 7.2　Forrelation 问题的深入分析

#### Forrelation 的定义与量子优势

**Forrelation 函数**：

$$\mathcal{F}(f, g) = \frac{1}{2^{n+1}}\sum_{x,y \in \{0,1\}^n} (-1)^{x \cdot y} f(x) g(y)$$

其中 $f, g: \{0,1\}^n \to \{-1,+1\}$，$x \cdot y = \sum_i x_i y_i \pmod{2}$ 是内积。

等价地，$\mathcal{F}(f,g) = \frac{1}{\sqrt{2^n}} \langle \hat{f}, g \rangle$，其中 $\hat{f}$ 是 $f$ 的 Hadamard 变换。

**量子计算**（单次 oracle 调用！）：

```
|0^n⟩ ─── H^⊗n ─── O_f ─── H^⊗n ─── O_g ─── H^⊗n ─── 测量
```

其中 $O_f: |x\rangle \to f(x)|x\rangle$（相位 oracle）。

输出概率 $\Pr[\text{测量 }0^n] = \left(\frac{1 + \mathcal{F}(f,g)}{2}\right)^2$，故 $O(1)$ 次重复即可估计 $\mathcal{F}(f,g)$（$Q = O(1)$）。

**经典下界**（Aaronson-Ambainis 2015, Bansal-Sinha 2021）：

对任意 $\delta > 0$，任何随机化算法以 $2/3$ 的概率估计 $\mathcal{F}(f,g)$ 需要：

$$Q_C(k\text{-Forrelation}) = \Omega(N^{1-1/k}) \quad (k\text{-Forrelation 的更精细版本})$$

对 $k=2$（标准 Forrelation）：$Q_C = \tilde{\Omega}(N^{1/2})$。

Bansal-Sinha（2021）证明了最优分离：$Q = O(1)$，$Q_C = \Omega(N^{1-\zeta})$（任意 $\zeta > 0$），对应 $k$-Forrelation 的 $k \to \infty$ 极限（无界多项式分离）。

#### Forrelation 在 ML 优势证明中的角色

**归约链**：

1. **Forrelation → NOPE**：将 Forrelation 的 oracle 函数 $(f, g)$ 打包为 NOPE 的 oracle $h(x,y) = $ Forrelation 的参数编码，则 NOPE 实例的 $Q_C = Q_C(\text{Forrelation})$；

2. **NOPE → 线性系统**：将线性方程组 $A\vec{x} = \vec{b}$ 的矩阵条目编码为 NOPE 的 oracle 值（$A_{ij}$ 和 $b_k$ 通过 Forrelation 的参数构造），则求解线性系统归约到 NOPE；

3. **线性系统 → LS-SVM**：LS-SVM 的岭分类器 $(X^\top X + \lambda I)^{-1}X^\top\vec{y}$ 是一个特殊的线性方程组，故归约是显然的；

4. **线性系统 → PCA**：主成分是协方差矩阵的最大特征向量，通过幂迭代（矩阵-向量乘积序列）归约到线性系统求解。

整个归约链将 Forrelation 的 $Q_C = \Omega(N^{1-\zeta})$ 传递到所有三类 ML 任务的经典难度上。

---

### 7.3　Raz 时空权衡框架的完整表述

#### 学习奇偶函数（Parity Learning）的时空下界

**问题**：未知奇偶函数 $h_s(x) = \langle s, x \rangle \pmod{2}$（$s \in \{0,1\}^n$ 是秘密向量），每次可以获得均匀随机 $x$ 和 $h_s(x)$。目标：恢复 $s$。

**经典算法**：高斯消元，$m = n$ 个样本，时间 $O(n^2)$，空间 $O(n^2)$（存储矩阵）。

**时空权衡下界（Raz 2018 \[Ref 158\]）**：

任何随机化算法以 $2/3$ 概率恢复 $s$，若时间（样本数）为 $T$ 且空间（工作内存）为 $S$，则：

$$T \cdot S \geq \Omega(n^2)$$

特别地，$S = O(n)$ 时需要 $T = \Omega(n)$（高斯消元已是最优！），而 $S = O(\sqrt{n})$ 时需要 $T = \Omega(n^{3/2})$（超过样本数的平方根）。

**证明技术**（信息理论论证）：

设算法在 $T$ 步后内存状态为 $M_T$（$S$ 比特），要恢复 $s$（$n$ 比特），需要 $H(s | M_T) \approx 0$（条件熵接近零）。

每步样本 $(x_t, h_s(x_t))$ 提供关于 $s$ 的条件互信息：

$$I(s; (x_t, h_s(x_t)) | M_{t-1}, x_1, \ldots, x_{t-1}) \leq \frac{S}{n} \cdot H(h_s(x_t))$$

（来自内存 $S$ 对于每个单独比特 $\langle s, e_j\rangle$ 所能「记住」的信息量受 $S/n$ 约束）。

积累 $T$ 步后：$I(s; M_T) \leq T \cdot O(S/n)$，而 $I(s; M_T)$ 需达到 $n$（恢复 $s$），故 $T \cdot S \geq \Omega(n^2)$。

#### 从 Raz 框架到 NOPE（Zhao et al. 推广）

**经典奇偶学习 → NOPE**：奇偶学习中，秘密向量 $s$ 定义了 oracle $f_s(x) = h_s(x)$，NOPE 是估计 $P(f_s)$（某个关于 $f_s$ 的性质）。通过将 Raz 框架中的互信息论证推广到一般 oracle 性质，得到：

$$M \cdot S \geq \Omega(N \cdot Q_C(P))$$

其中 $Q_C(P)$ 是经典查询复杂度（估计性质 $P$）。

这是 Theorem E.2（Zhao et al. 论文）的精确内容，是所有经典难度定理的统一来源。

---

<a name="第八章"></a>
## 第八章　应用领域、生成模型与前沿方向

### 8.1　量子化学：VQE 的核心应用

#### 从量子化学到 VQE 的完整对应

**电子结构哈密顿量**（Born-Oppenheimer 近似，二次量子化）：

$$H = \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} g_{pqrs} a_p^\dagger a_q^\dagger a_r a_s - E_{\text{nuc}}$$

其中 $h_{pq}$ 是单电子积分（动能 + 核-电子相互作用），$g_{pqrs}$ 是双电子积分（电子-电子排斥），$a_p^\dagger, a_q$ 是费米子产生/湮灭算符。

**Jordan-Wigner 变换**将费米子算符映射到 Pauli 算符：

$$a_j = \left(\prod_{k<j} Z_k\right) \cdot \frac{X_j + iY_j}{2}$$

对应的 Pauli 表示哈密顿量：$H = \sum_\alpha h_\alpha P_\alpha$（$P_\alpha$ 是 Pauli 积，$h_\alpha$ 是实系数），项数 $O(N^4)$（$N$ 是空间轨道数）。

**UCCSD ansatz**（化学上最自然的 VQE ansatz）：

$$U_{\text{UCCSD}}(\vec{\theta}) = e^{T(\vec{\theta}) - T^\dagger(\vec{\theta})}$$

其中 $T(\vec{\theta}) = \sum_{ia} \theta_i^a a_a^\dagger a_i + \sum_{ijab} \theta_{ij}^{ab} a_a^\dagger a_b^\dagger a_j a_i$（单激发和双激发算符之和）。

**参数数量**：对 $N$ 个轨道，$N_e$ 个电子，$O(N^4)$ 个参数（大分子时极多），是 VQE 优化的主要挑战。

#### 量子化学的 VQE 实际结果

| 分子 | 轨道数 | 量子比特数 | 化学精度（Hartree） | VQE 结果 | 需求 |
|:---|:---:|:---:|:---:|:---:|:---|
| H₂ | 4 | 4 | $10^{-3}$ | $< 10^{-4}$ Ha | 已实现（NISQ） |
| LiH | 12 | 12 | $1.6\times10^{-3}$ | $\sim 2\times10^{-4}$ Ha | 已实现（NISQ+误差缓解） |
| BeH₂ | 14 | 14 | $6.3\times10^{-4}$ | 接近 | 已实现 |
| H₂O（化学精度） | ~24 | ~24 | $1.6\times10^{-3}$ | 未达到 | 需 ~1000 逻辑比特 |
| 咖啡因 | ~50 | ~100 | $10^{-3}$ | 未实现 | 需 ~10⁴ 逻辑比特 |
| Fe-S 簇（固氮酶） | ~100 | ~200 | $10^{-3}$ | 未实现 | 需 ~10⁵ 逻辑比特 |

---

### 8.2　量子生成对抗网络（QGAN）

#### 经典 GAN 回顾

**GAN 框架**（Goodfellow et al. 2014）：

- **生成器** $G_\theta: z \to x$（潜在变量 $z$ 到数据空间 $x$ 的映射）；
- **判别器** $D_\phi: x \to [0,1]$（真实数据 vs. 生成数据的分类器）；
- **目标**：$\min_\theta \max_\phi V(G_\theta, D_\phi) = \mathbb{E}_{x\sim p_{\text{real}}}[\log D_\phi(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D_\phi(G_\theta(z)))]$。

纳什均衡时，$G_\theta$ 的分布等于真实分布 $p_{\text{real}}$。

#### QGAN 架构（Lloyd & Weedbrook 2018 \[Ref 122\]）

**量子生成器** $G_\theta = U_G(\theta)|0\rangle\langle 0|U_G^\dagger(\theta)$（参数化量子电路制备的混合态）；

**量子判别器** $D_\phi = \mathrm{Tr}[M_\phi \cdot \rho]$（量子测量，$M_\phi$ 是参数化 POVM 元素）；

**QGAN 训练循环**：

```python
for epoch in range(num_epochs):
    # 生成器更新
    rho_fake = U_G(theta) |0><0| U_G†(theta)  # 生成量子态
    loss_G = -Tr[M_phi · rho_fake]  # 生成器希望判别器输出 1
    theta -= lr * grad_theta(loss_G)
    
    # 判别器更新
    rho_real = real_quantum_state()  # 真实量子数据
    loss_D = Tr[M_phi · rho_real] - Tr[M_phi · rho_fake]  # Wasserstein 形式
    phi += lr * grad_phi(loss_D)
```

**量子优势**：$n$ 量子比特的量子态空间是 $4^n$ 维的，量子生成器可以表达经典生成器难以捕捉的分布（如量子纠缠态的分布）。对纯态 $|\psi\rangle\langle\psi|$，量子生成器天然是纯态生成器。

#### Born 机器（Born Machine，Gao et al. 2018 \[Ref 21\]）

**模型**：量子电路 $U(\theta)$ 作用于 $|0\rangle^{\otimes n}$，测量得到分布：

$$p_\theta(x) = |\langle x|U(\theta)|0\rangle|^2 \quad (x \in \{0,1\}^n)$$

这是量子测量的 Born 规则（$p_\theta$ 自动是归一化的非负分布）。

**训练目标**：最小化与目标分布 $q$ 的 KL 散度：

$$\mathcal{L}(\theta) = \text{KL}(q \| p_\theta) = \sum_x q(x) \log \frac{q(x)}{p_\theta(x)}$$

**量子优势**：Born 机器可以天然表达**量子干涉**（多路径相消或相长）产生的非经典分布，例如硬核对角（hard-core boson）分布，经典随机电路无法高效生成。

Gao et al.（2022 \[Ref 22\]）证明：对于某类**量子关联（纠缠）诱导的**分布，任何经典神经网络生成器需要多项式更多参数，而量子电路可以更紧凑地表达——这提供了参数效率上的量子优势。

---

### 8.3　量子感知机与量子神经网络

#### 量子感知机的三种流派

**流派 1：VQA 类量子神经网络（QNN）**

$$f_{\text{QNN}}(\vec{x}) = \langle 0|U^\dagger(\vec{x}, \vec{\theta}) O U(\vec{x}, \vec{\theta})|0\rangle$$

其中 $U(\vec{x}, \vec{\theta}) = V(\vec{\theta}) \cdot E(\vec{x})$（$E(\vec{x})$ 是数据编码层，$V(\vec{\theta})$ 是可训练层）。

优点：可在 NISQ 设备上实现；缺点：受 Barren Plateaus 困扰，泛化性理论不完整。

**流派 2：量子循环神经网络（QRNN）**

对时序数据 $(\vec{x}_1, \ldots, \vec{x}_T)$，量子隐变量（$n$ 量子比特密度矩阵 $\rho_t$）按以下规则演化：

$$\rho_{t+1} = \mathrm{Tr}_{\text{input}}\!\left[U(\vec{x}_t) \rho_t \otimes |\vec{x}_t\rangle\langle\vec{x}_t| U^\dagger(\vec{x}_t)\right]$$

输出 $y_t = \mathrm{Tr}[O_{\text{out}} \rho_t]$。

Anschuetz et al.（2023 \[Ref 23\]）证明了对特定序列预测任务（如量子系统演化的预测），QRNN 相比经典 RNN 有**可解释的多项式优势**：量子隐变量天然编码了量子关联，而经典隐变量无法高效模拟纠缠态的时间演化。

**流派 3：量子图神经网络（QGNN，Verdon et al. 2019）**

将图神经网络（GNN）的消息传递机制量子化：

$$\rho^{(l+1)}_i = \text{AGGREGATE}\left(\rho^{(l)}_i, \bigoplus_{j \in \mathcal{N}(i)} U_{ij}(\vec{\theta}) \rho^{(l)}_j U^\dagger_{ij}(\vec{\theta})\right)$$

适用于分子图（原子为节点，化学键为边）的性质预测。

---

### 8.4　量子计算增强传感：Heisenberg 极限

#### 量子感应的基本极限

对精密测量（如相位估计 $\phi$ 的测量误差），$N$ 个探针资源的测量精度受如下极限约束：

- **标准量子极限（SQL）**：$\Delta\phi = 1/\sqrt{N}$（经典传感，各探针独立测量，中心极限定理）；
- **Heisenberg 极限（HL）**：$\Delta\phi = 1/N$（最优量子策略，纠缠态传感）。

**纠缠增强传感方案**（如 GHZ 态）：

$$|GHZ\rangle = \frac{|0\rangle^{\otimes N} + |1\rangle^{\otimes N}}{\sqrt{2}}$$

用于相位估计：

$$|GHZ\rangle \xrightarrow{e^{i\phi Z}^{\otimes N}} \frac{|0\rangle^{\otimes N} + e^{iN\phi}|1\rangle^{\otimes N}}{\sqrt{2}}$$

Hadamard 测量后，相位分辨率为 $\Delta\phi = \pi/N$（Heisenberg 极限）。

**代价**：GHZ 态对**退相干极其敏感**——单一量子比特的错误导致整个态崩溃。Zhou et al.（2018 \[Ref 132\]）利用量子纠错保护 GHZ 态，实现了在非理想条件下仍接近 Heisenberg 极限的传感精度。

#### 量子计算增强传感（Allen et al. 2025 \[Ref 115\]）

**核心思想**：量子计算机不仅是测量设备，还可以作为**量子信号处理器**——先用量子电路处理传感信号，再测量，提取信噪比。

具体算法：
1. 初始化量子传感器（量子比特/离子/光子）；
2. 让传感器与外场（磁场、引力波、电场等）耦合 $t$ 时间（信号积累）；
3. 用量子算法（如量子相位估计）处理传感量子态；
4. 测量，输出信号估计值。

**优势**：相比经典后处理（先测量为经典比特，再估计），量子处理可以利用量子态的相干叠加同时处理所有可能的信号值，提取指数更丰富的相关信息。

---

### 8.5　单细胞转录组学：量子 PCA 的「杀手级应用」

#### scRNA-seq 数据结构

**数据矩阵** $X \in \mathbb{R}^{N \times D}$：
- $N$：细胞数（$10^3 \sim 10^6$，单批次 $\sim 10^4$）；
- $D$：基因数（人类 ~33000 蛋白质编码基因）；
- 稀疏性：每细胞平均表达 ~3000 基因（稀疏度 ~90%）；
- 目标：降维（PCA 到 50-200 维），揭示细胞类型/状态的连续谱。

**与量子 PCA 的对应**：

| scRNA-seq 参数 | 量子算法参数 |
|:---|:---|
| 细胞数 $N \sim 10^4$ | 流式样本数 $M = \tilde{O}(N) \sim 10^4$（每个样本是一个细胞的表达谱） |
| 基因维度 $D \sim 33000$ | 量子寄存器大小 $n = \log_2 D \approx 15$ 比特 |
| 稀疏度 ~90% | 稀疏假设 $s \leq \mathrm{poly}(\log D) \approx 10$（每细胞表达的非零基因数的对数） |
| PCA 降维目标（前 50 主成分） | 量子 PCA 的 $\chi \sim 1/\sqrt{50} \approx 0.14$（引导向量质量） |

**经典瓶颈**：对 $N = 10^6$，$D = 33000$ 的大规模 scRNA-seq 数据：
- 完整矩阵：$10^6 \times 33000 \times 4$ 字节 $\approx$ 120 GB；
- 经典流式 PCA（Incremental PCA）内存：$O(D \times k)$（$k$ 是保留主成分数）$\approx$ $33000 \times 50 \times 4$ 字节 $\approx$ 6 MB；
- 量子 PCA 内存：$O(\log D) = O(15)$ 量子比特 $\approx$ **20 个物理比特**（理论上！）。

论文数值实验（图 2）在 PBMC 数据集（$N \approx 3000$，$D \approx 30000$）上演示了 4 个数量级的内存优势。

---

<a name="第九章"></a>
## 第九章　核心开放问题与研究展望

### 9.1　Oracle Sketching 框架的延伸与优化

#### 非稀疏数据的 Oracle Sketching

**问题**：当前 Theorem 3-6 严格依赖稀疏性（$s \leq \mathrm{poly}(\log N)$）。对 NLP 中的稠密嵌入向量（如 BERT 的 768 维密集向量），稀疏性假设不成立。

**可能的路线**：

**路线 A：低秩 + 稀疏分解（Robust PCA）**
对数据矩阵 $X = L + S$（$L$ 低秩，$S$ 稀疏），先用经典 Robust PCA 分解（Candès et al. 2011，$O(mD)$ 时间），再对低秩部分 $L$ 使用量子 PCA。

量子优势的保留条件：$L$ 的秩 $r \leq \mathrm{poly}(\log D)$，$S$ 的稀疏度 $s \leq \mathrm{poly}(\log D)$。在 NLP 中，嵌入矩阵往往确实是近似低秩的（Aghajanyan et al. 2021 的内在维度估计）。

**路线 B：量子草图的稀疏化（Sparsification）**
将密集向量 $\vec{v}$ 通过随机投影（如 JL 引理）降维到 $s = O(\mathrm{poly}(\log D))$ 维：

$$\vec{v} \xrightarrow{R \in \mathbb{R}^{s \times D}, \text{i.i.d.} \pm 1/\sqrt{s}} R\vec{v} \quad (\text{JL 嵌入})$$

保证 $\|\vec{v}\| \approx \|R\vec{v}\|$（$\varepsilon$ 精度的 JL 保持），然后对 $R\vec{v}$ 使用量子草图。代价：需要 $O(sD)$ 次随机投影操作（或利用快速 JL 变换 $O(D \log D)$）。

**路线 C：量子随机哈希**
将输入向量随机哈希到较小维度，利用量子叠加同时处理所有哈希值，在哈希空间中完成 oracle 草图，再用量子 unHash 恢复。技术上需要量子 locality-sensitive hashing (QLSH)，是目前尚未完全解决的方向。

---

#### 超越 LS-SVM 的损失函数

当前 Theorem 3/4 仅对最小二乘损失（Ridge / LS-SVM）成立。对于其他常用损失函数：

| 损失函数 | 对应模型 | 量子草图的障碍 |
|:---|:---|:---|
| Logistic 损失 | Logistic 回归 | 需要迭代算法（梯度下降），每步需要 oracle 草图；总查询次数 $Q = O(1/\varepsilon)$，故 $M = O(NQ^2/\varepsilon) = O(N/\varepsilon^3)$——仍有优势 |
| Hinge 损失 | SVM | 对偶问题是 QP，需要量子 QP 求解器（基于 QSVT）；oracle 草图提供矩阵 block encoding |
| 交叉熵 | 多分类神经网络 | 需要多次 softmax 计算（复杂非线性操作），目前无直接量子草图对应 |
| 对比学习损失 | 表示学习 | 需要批内样本间的内积矩阵，可用量子 oracle 草图构造，但损失函数涉及归一化（softmax），复杂度不明 |

**推广路线**：将 LS-SVM 的「线性」归约链（分类 → 线性系统 → NOPE）推广到一般凸损失，关键是证明：对一般凸损失的最优解，同样存在一个 NOPE 实例，其量子-经典查询复杂度分离与 Forrelation 相同量级。

---

#### 变分 + Sketching 混合架构

**思想**：将参数化量子层 $V(\vec{\theta})$ 插入 oracle sketching 流水线，用于：
1. **预处理**：将输入数据 $|\vec{x}\rangle$ 通过 $V_{\text{enc}}(\vec{\theta}_{\text{enc}})$ 映射到更适合 oracle 的特征空间；
2. **后处理**：对 oracle 草图输出 $|w^*\rangle$ 施加 $V_{\text{dec}}(\vec{\theta}_{\text{dec}})$ 进行非线性变换；
3. **自适应采样**：用 VQE 类优化确定最优样本采样分布 $\mathcal{D}_{\text{opt}}(\vec{\theta})$，降低 oracle 草图的方差。

**理论框架**：设混合模型为 $f_{\text{hybrid}}(\vec{x}) = $ Interferometric Shadow $\circ$ Oracle Sketch $\circ V(\vec{\theta})$，则优化目标为：

$$\min_{\vec{\theta}} \mathbb{E}_{(\vec{x},y)}[\mathcal{L}(f_{\text{hybrid}}(\vec{x}; \vec{\theta}), y)]$$

其中 $\mathcal{L}$ 是任务损失函数。这是一个经典-量子混合优化问题，梯度通过参数偏移规则估计（作用于变分层），oracle 草图部分的梯度通过分析影子测量的期望值来计算。

---

### 9.2　BP 困境的出路

#### 量子纠错辅助 VQA

**思路**：使用逻辑量子比特（通过量子纠错编码，如 Steane [[7,1,3]] 码或 Surface Code）实现深层 VQA，使电路深度不受噪声诱导 BP 的限制（错误率 $\to 0$）。

**关键问题**：消除物理噪声并不能消除**结构性 BP**（来自酉 2-设计的梯度浓缩）。逃脱结构性 BP 需要 ansatz 结构利用**域知识**（domain knowledge）而非随机初始化。

#### LDPC 码 + VQA 的混合策略

**量子低密度奇偶校验码（Quantum LDPC, QLDPC）**：代码距离 $d = O(n)$，编码率 $k/n = O(1)$（相比 Surface Code 的 $k/n = O(1/n)$），实现更高效的量子纠错。

在 QLDPC 保护的逻辑空间中定义变分量子电路，天然地只探索物理上有意义（保持 LDPC 码结构）的参数空间子集，有望缩减 BP 的出现范围。

---

### 9.3　量子数据与经典数据优势的统一理论

#### 信息论框架的统一尝试

两条路线的信息论根源：

**路线 A（量子数据，Huang et al. 2022）**：优势来自量子测量的**不可克隆性**（no-cloning theorem）。经典机必须先测量量子数据，这一过程是不可逆的（投影）；而量子内存可以保留未测量的量子数据副本，在多次测量时提供更多信息。

**路线 B（经典数据，Zhao et al. 2026）**：优势来自量子叠加在**流式压缩**中的效率。经典机在处理流式数据时，内存必须存储数据的经典摘要（会丢失相位信息）；而量子内存可以将数据编码为相干叠加态，保留指数维信息。

**统一描述**（猜想）：两类优势均来自同一物理原理——**量子叠加允许同时维护多个互不相容（非对易）的信息集合**：
- 路线 A：量子内存同时维护多个不可克隆量子态的分量；
- 路线 B：量子内存同时维护数据的所有 oracle 函数值的叠加。

在信息论语言中，两者均源于量子信道的量子容量（quantum capacity）大于经典容量（Shannon capacity）的事实（在适当定义的通信模型下）。

**形式化猜想**：存在统一的量子-经典分离度量 $\Delta_Q$，满足：
1. $\Delta_Q \geq $ Bell 不等式违反量（量子非局域性）；
2. $\Delta_Q \geq $ 量子-经典通信复杂度分离；
3. $\Delta_Q \geq $ 量子学习样本复杂度分离（路线 A）；
4. $\Delta_Q \geq $ 量子流式空间复杂度分离（路线 B）。

这一猜想若成立，将为 QML 优势提供完整的「相对论前的电磁学统一」级别的理论框架。

---

### 9.4　近期实验验证路线图

#### 2026–2028 年窗口：20–40 逻辑比特的实证

**目标**：在真实量子硬件上演示 oracle sketching 对小规模 ML 任务（$N \sim 1000$，$D \sim 100$）的「量子内存优势实证」。

**关键里程碑**：

1. **硬件目标**：IBM/Google 超导量子计算机或 IonQ 离子阱，达到 30 逻辑比特（$\sim 3$ 万物理比特，表面码，物理错误率 $< 0.1\%$）；

2. **电路优化目标**：将 oracle sketching 的每个「多控相位门」（原始 $O(n)$ 个 T 门）通过 T 门蒸馏 + 专用编译器优化到 $\leq 20$ 个物理操作；

3. **小规模数据集**：构造 $N = 1024$（$n = 10$ 量子比特），$D = 512$（$\log_2 D = 9$），稀疏度 $s = 5$ 的合成测试集；量子机使用 oracle sketching 完成二分类；经典流式机使用有限内存（$S_C = O(D^{0.99}) = O(500)$ 比特）尝试完成同一任务并验证失败；

4. **统计分析**：重复 $\geq 10^3$ 次独立试验，报告 $>5\sigma$ 的统计显著分类优势。

#### 2028–2032 年窗口：50–100 逻辑比特的「端到端」演示

对接近真实应用规模的数据集（$N \sim 10^4$，$D \sim 1000$），完成与论文数值实验相当的「量子 oracle sketching vs. 经典最优流式算法」的直接比较，实现论文所预言的 $>10^4\times$ 内存优势的实验验证。

这一演示将是量子计算史上第一个对**自然 ML 任务**的**无条件**（信息论意义上）量子优势的实验证据，具有与历史上第一个 Bell 不等式测验（Aspect 1982）相当的意义。

---

<a name="参考文献"></a>
## 参考文献

### 核心论文

**\[Core\]** Zhao H., Zlokapa A., Neven H., Babbush R., Preskill J., McClean J.R., Huang H.-Y. (2026). Exponential quantum advantage in processing massive classical data. *arXiv:2604.07639v1*.

---

### 第一章 相关

**\[1\]** Feynman R.P. (1986). Quantum mechanical computers. *Foundations of Physics*, 16(6), 507–531.

**\[2\]** Shor P.W. (1994). Algorithms for quantum computation: discrete logarithms and factoring. *Proc. 35th FOCS*, 124–134.

**\[3\]** Babbush R. et al. (2025). The grand challenge of quantum applications. arXiv:2511.09124.

**\[4\]** Preskill J. (2025). Beyond NISQ: The megaquop machine. *ACM Trans. Quantum Computing*, 6(3), 1–7.

**\[8\]** Dalzell A.M. et al. (2025). *Quantum Algorithms: A Survey of Applications and End-to-end Complexities*. Cambridge University Press.

**\[11\]** Biamonte J. et al. (2017). Quantum machine learning. *Nature*, 549(7671), 195–202.

**\[12\]** Aaronson S. (2015). Read the fine print. *Nature Physics*, 11(4), 291–293.

**\[13\]** Giovannetti V., Lloyd S., Maccone L. (2008). Quantum random access memory. *PRL*, 100(16), 160501.

**\[15\]** Dalzell A.M. et al. (2025). A distillation-teleportation protocol for fault-tolerant QRAM. arXiv:2505.20265.

**\[16\]** Jaques S., Rattew A.G. (2025). QRAM: A survey and critique. *Quantum*, 9, 1922.

**\[17\]** Schuld M., Killoran N. (2022). Is quantum advantage the right goal for QML? *PRX Quantum*, 3(3), 030101.

**\[18\]** Cerezo M. et al. (2023). Does provable absence of barren plateaus imply classical simulability? arXiv:2312.09121.

---

### 第二章 相关（量子线性代数）

**\[9\]** Gilyén A., Su Y., Low G.H., Wiebe N. (2019). Quantum singular value transformation and beyond. *Proc. STOC 2019*, 193–204.

**\[10\]** Martyn J.M., Rossi Z.M., Tan A.K., Chuang I.L. (2021). Grand unification of quantum algorithms. *PRX Quantum*, 2(4), 040203.

**\[82\]** Costa P.C.S. et al. (2022). Optimal scaling quantum linear-systems solver via discrete adiabatic theorem. *PRX Quantum*, 3(4), 040303.

**\[83\]** Chakraborty S., Morolia A., Peduri A. (2023). Quantum regularized least squares. *Quantum*, 7, 988.

**\[84\]** Lin L., Tong Y. (2020). Near-optimal ground state preparation. *Quantum*, 4, 372.

**\[118\]** Harrow A.W., Hassidim A., Lloyd S. (2009). Quantum algorithm for linear systems of equations. *PRL*, 103(15), 150502.

**\[119\]** Kerenidis I., Prakash A. (2017). Quantum recommendation systems. *ITCS 2017*, 49.

**\[120\]** Rebentrost P., Mohseni M., Lloyd S. (2014). Quantum support vector machine for big data classification. *PRL*, 113(13), 130503.

**\[121\]** Lloyd S., Mohseni M., Rebentrost P. (2014). Quantum principal component analysis. *Nature Physics*, 10(9), 631–633.

**\[181\]** Morales M.E.S. et al. (2024). Quantum linear system solvers: A survey of algorithms and applications. arXiv:2411.02522.

---

### 第三章 相关（变分量子算法）

**\[32\]** Peruzzo A. et al. (2014). A variational eigenvalue solver on a photonic quantum processor. *Nature Communications*, 5(1), 4213.

**\[33\]** McClean J.R., Romero J., Babbush R., Aspuru-Guzik A. (2016). The theory of variational hybrid quantum-classical algorithms. *NJP*, 18(2), 023023.

**\[34\]** Cerezo M. et al. (2021). Variational quantum algorithms. *Nature Reviews Physics*, 3(9), 625–644.

**\[35\]** Cerezo M. et al. (2022). Challenges and opportunities in QML. *Nature Computational Science*, 2(9), 567–576.

**\[36\]** Du Y. et al. (2025). Quantum machine learning: A hands-on tutorial. arXiv:2502.01146.

**\[37\]** McClean J.R., Boixo S., Smelyanskiy V.N., Babbush R., Neven H. (2018). Barren plateaus in quantum neural network training landscapes. *Nature Communications*, 9(1), 4812.

**\[38\]** Cerezo M. et al. (2021). Cost function dependent barren plateaus in shallow parametrized quantum circuits. *Nature Communications*, 12(1), 1791.

**\[39\]** Wang S. et al. (2021). Noise-induced barren plateaus in variational quantum algorithms. *Nature Communications*, 12(1), 6961.

**\[40\]** Larocca M. et al. (2025). Barren plateaus in variational quantum computing. *Nature Reviews Physics*, 7, 174–189.

**\[41\]** Anschuetz E.R., Kiani B.T. (2022). Quantum variational algorithms are swamped with traps. *Nature Communications*, 13(1), 7760.

---

### 第四章 相关（量子核与去量子化）

**\[19\]** Gil-Fuster E. et al. (2024). On the relation between trainability and dequantization. arXiv:2406.07072.

**\[26\]** Liu Y., Arunachalam S., Temme K. (2021). A rigorous and robust quantum speed-up in supervised machine learning. *Nature Physics*, 17(9), 1013–1017.

**\[27\]** Gyurik C., Dunjko V. (2023). Exponential separations between classical and quantum learners. arXiv:2306.16028.

**\[29\]** Tang E. (2019). A quantum-inspired classical algorithm for recommendation systems. *Proc. STOC 2019*, 217–228.

**\[30\]** Tang E. (2021). Quantum PCA only achieves exponential speedup because of its state preparation assumptions. *PRL*, 127(6), 060503.

**\[31\]** Tang E. (2023). Quantum machine learning without any quantum. PhD Thesis, University of Washington.

**\[外\]** Larsen K.G. et al. (2024). An exponential separation between quantum and quantum-inspired classical algorithms for machine learning. arXiv:2411.02087.

---

### 第五章 相关（量子学习理论）

**\[20\]** Zhao H., Deng D.-L. (2025). Entanglement-induced provable and robust quantum learning advantages. *npj Quantum Information*, 11(1), 127.

**\[24\]** Anschuetz E.R., Gao X. (2026). Arbitrary polynomial separations in trainable QML. *Quantum*, 10, 1976.

**\[25\]** Zhang Z. et al. (2024). Quantum-classical separations in shallow-circuit-based learning. *Communications Physics*, 7(1), 290.

**\[73\]** Kretschmer W. et al. (2025). Demonstrating an unconditional separation between quantum and classical information resources. arXiv:2509.07255.

**\[109\]** Huang H.-Y. et al. (2022). Quantum advantage in learning from experiments. *Science*, 376(6598), 1182–1186.

**\[110\]** Chen S., Cotler J., Huang H.-Y., Li J. (2022). Exponential separations between learning with and without quantum memory. *FOCS 2021*, 574–585.

**\[113\]** Liu Z.-H. et al. (2025). Quantum learning advantage on a scalable photonic platform. *Science*, 389(6767), 1332–1335.

**\[135\]** Chen S., Cotler J., Huang H.-Y., Li J. (2023). The complexity of NISQ. *Nature Communications*, 14(1), 6001.

**\[137\]** Bravyi S., Gosset D., König R. (2018). Quantum advantage with shallow circuits. *Science*, 362(6412), 308–311.

**\[外\]** Lewis L., Gilboa D., McClean J.R. (2026). Quantum advantage for learning shallow neural networks with natural data distributions. *Nature Communications*, 17, 1341.

---

### 第六章 相关（经典影子）

**\[74\]** Huang H.-Y., Kueng R., Preskill J. (2020). Predicting many properties of a quantum system from very few measurements. *Nature Physics*, 16(10), 1050–1057.

**\[外\]** Jerbi S. et al. (2024). Shadows of quantum machine learning. *Nature Communications*, 15, 5676.

**\[外\]** Votto M. et al. (2026). MPO-based reconstruction of 96-qubit quantum states via classical shadows. *Physical Review Letters* (2026).

---

### 第七章 相关（流式算法与复杂度）

**\[60\]** Andoni A. et al. (2020). Streaming complexity of SVMs. arXiv:2007.03633.

**\[61\]** Mitliagkas I., Caramanis C., Jain P. (2013). Memory limited, streaming PCA. *NeurIPS* 26.

**\[68\]** Kallaugher J. (2022). A quantum advantage for a natural streaming problem. *FOCS 2021*, 897–908.

**\[69\]** Kallaugher J., Parekh O., Voronova N. (2024). Exponential quantum space advantage for approximating MaxDiCut. *Proc. STOC 2024*, 1805–1815.

**\[70\]** Kallaugher J., Parekh O., Voronova N. (2025). How to design a quantum streaming algorithm without knowing anything about quantum computing. *SOSA 2025*, 9–45.

**\[71\]** Gilboa D., Michaeli H., Soudry D., McClean J. (2024). Exponential quantum communication advantage in distributed inference and learning. *NeurIPS* 37, 30425–30473.

**\[72\]** Niroula P. et al. (2025). Realization of a quantum streaming algorithm on long-lived trapped-ion qubits. arXiv:2511.03689.

**\[95\]** Aaronson S., Ambainis A. (2015). Forrelation: A problem that optimally separates quantum from classical computing. *Proc. STOC 2015*, 307–316.

**\[96\]** Bansal N., Sinha M. (2021). k-forrelation optimally separates quantum and classical query complexity. *Proc. STOC 2021*, 1303–1316.

**\[158\]** Raz R. (2018). Fast learning requires good memory: A time-space lower bound for parity learning. *JACM*, 66(1), 1–18.

**\[163\]** Liu Q., Raz R., Zhan W. (2023). Memory-sample lower bounds for learning with classical-quantum hybrid memory. *Proc. STOC 2023*, 1097–1110.

---

### 第八章 相关（应用领域）

**\[21\]** Gao X., Zhang Z.-Y., Duan L.-M. (2018). A QML algorithm based on generative models. *Science Advances*, 4(12), eaat9004.

**\[22\]** Gao X., Anschuetz E.R., Wang S.-T., Cirac J.I., Lukin M.D. (2022). Enhancing generative models via quantum correlations. *PRX*, 12(2), 021037.

**\[23\]** Anschuetz E.R., Hu H.-Y., Huang J.-L., Gao X. (2023). Interpretable quantum advantage in neural sequence learning. *PRX Quantum*, 4(2), 020338.

**\[28\]** Huang H.-Y. et al. (2025). Generative quantum advantage for classical and quantum problems. arXiv:2509.09033.

**\[115\]** Allen R.R., Machado F., Chuang I.L., Huang H.-Y., Choi S. (2025). Quantum computing enhanced sensing. arXiv:2501.07625.

**\[122\]** Lloyd S., Weedbrook C. (2018). Quantum generative adversarial learning. *PRL*, 121(4), 040502.

**\[131\]** Babbush R. et al. (2018). Encoding electronic spectra in quantum circuits with linear T complexity. *PRX*, 8(4), 041015.

**\[132\]** Zhou S., Zhang M., Preskill J., Jiang L. (2018). Achieving the Heisenberg limit in quantum metrology using quantum error correction. *Nature Communications*, 9(1), 78.

---

## 第十章　量子随机游走、LCU 与量子蒙特卡洛

### 10.1　量子随机游走（Quantum Walk）

#### 经典随机游走回顾

经典随机游走（Random Walk）在 $N$ 节点图 $G = (V, E)$ 上的转移矩阵为 $P = D^{-1}A$（$A$ 是邻接矩阵，$D$ 是度对角矩阵）。从初始状态 $\vec{v}_0$ 经 $t$ 步演化到稳态分布需要 $O(1/\text{gap}(P))$ 步（gap 为转移矩阵谱间隙）。混合时间 $T_{\text{mix}} = O(\delta^{-1}\log(N/\varepsilon))$（$\delta = 1 - \lambda_2(P)$ 是谱间隙）。

经典随机游走的机器学习应用：
- **PageRank**：图节点重要性的幂法迭代；
- **谱聚类**（Spectral Clustering）：利用 Laplacian 特征向量分割图；
- **图嵌入**（DeepWalk/Node2Vec）：通过随机游走采样学习节点表示。

**经典瓶颈**：对 $N$ 节点图，计算前 $k$ 个特征值/特征向量的复杂度 $O(Nk/\delta^2)$（Lanczos 算法）——$N$ 很大时（如 Web 图，$N \sim 10^{10}$）不可接受。

#### 量子游走的两种形式

**离散量子游走（Discrete Quantum Walk）**

Szegedy（2004）量子化经典随机游走 $P$：

**状态空间**：边 Hilbert 空间 $\mathcal{H} = \mathbb{C}^{|V|} \otimes \mathbb{C}^{|V|}$，基向量 $|u, v\rangle$（$u$ 是当前节点，$v$ 是方向）。

**Szegedy 算符**：定义「扩散态」$|\psi_u\rangle = \sum_v \sqrt{P_{uv}}|u, v\rangle$，投影算符 $\Pi = \sum_u |\psi_u\rangle\langle\psi_u|$，Szegedy 算符：

$$W = (2\Pi - I)(2\Pi^{\text{swap}} - I)$$

其中 $\Pi^{\text{swap}} = \text{SWAP} \circ \Pi \circ \text{SWAP}$ 是「镜像投影」。

**谱关系（Szegedy 定理）**：若经典游走有本征值 $\lambda$，对应量子游走有本征相位 $e^{\pm i\arccos(\lambda)}$。特别地，经典混合时间 $T_{\text{mix}}^C \sim 1/\delta$ 对应量子游走二次加速：

$$T_{\text{mix}}^Q \sim \frac{1}{\sqrt{\delta}} = \sqrt{T_{\text{mix}}^C}$$

**连续时间量子游走（Continuous-Time Quantum Walk, CTQW）**

由图 Laplacian $L = D - A$ 生成的薛定谔方程：

$$i\frac{d|\psi(t)\rangle}{dt} = L|\psi(t)\rangle \implies |\psi(t)\rangle = e^{-iLt}|\psi(0)\rangle$$

本征态即 $L$ 的特征向量，本征能量即特征值 $\lambda_k$。量子态在时间 $t$ 后的概率分布为 $p_k(t) = |\langle k|\psi(t)\rangle|^2$，具有比经典更快的扩散速度（$\sigma_x \sim t$ vs. 经典 $\sigma_x \sim \sqrt{t}$）。

#### 量子游走在 ML 中的应用

**应用 1：量子化谱聚类（Quantum Spectral Clustering）**

传统谱聚类需要计算图 Laplacian 的前 $k$ 个特征向量。利用 QSVT：

1. 将图 Laplacian $L$ 构造为 block encoding $U_L$（$O(\text{deg}(G) \cdot \log N)$ 门）；
2. 用 QSVT 实现「低通滤波」多项式 $p_{\text{filter}}(x) = [x \leq k\text{-th eigenvalue}]$（近似符号函数），将图信号投影到前 $k$ 个特征子空间；
3. 利用经典影子协议提取特征向量的经典表示；
4. 在经典计算机上完成 $k$-means 聚类。

量子加速：特征向量计算从 $O(Nk/\delta^2)$（经典）降至 $O(\sqrt{N}\mathrm{poly}(k)/\delta)$（量子游走）。

**应用 2：量子 PageRank**

将 PageRank 迭代 $\vec{r}_{t+1} = \alpha P \vec{r}_t + (1-\alpha)\vec{e}$ 量子化：

1. 用量子游走一步算符 $W$ 替代经典转移矩阵 $P$；
2. 利用量子相位估计（QPE）定位接近 1 的本征值（对应 PageRank 向量）；
3. 量子游走的混合时间 $O(1/\sqrt{\delta})$ vs. 经典 $O(1/\delta)$，提供二次加速。

数值实验（Paparo & Martin-Delgado 2012）：在真实 Web 图子集上，量子 PageRank 产生与经典 PageRank 不同的节点排序——量子干涉改变了「重要性」的定义，可能揭示不同的图拓扑特征。

---

### 10.2　线性组合酉算符（LCU）技术

#### LCU 的基本思想

**动机**：QSVT 需要 block encoding，而 block encoding 有时难以直接构造。LCU 提供了一种通过「量子线性组合」实现任意矩阵函数的方法。

**设定**：目标矩阵 $A = \sum_{k=0}^{K-1} c_k U_k$（$c_k \geq 0$，$\sum_k c_k = 1$，$U_k$ 是酉算符）。

**LCU 协议**：

1. **准备步骤**：在辅助寄存器 $|\text{anc}\rangle$ 上准备状态 $|c\rangle = \sum_k \sqrt{c_k}|k\rangle$（调用「准备」酉算符 $\text{PREP}$）；

2. **选择步骤**：受控地施加 $U_k$（以辅助寄存器 $|k\rangle$ 控制目标寄存器）：

$$\text{SEL}: |k\rangle|\psi\rangle \to |k\rangle U_k|\psi\rangle$$

3. **投影步骤**：施加 $\text{PREP}^\dagger$，检查辅助寄存器是否回到 $|0\rangle$：

$$\langle 0| \text{PREP}^\dagger \cdot \text{SEL} \cdot \text{PREP}|0\rangle|\psi\rangle = \sum_k c_k U_k|\psi\rangle = A|\psi\rangle$$

**电路图**：

```
|0⟩ ─── PREP ─── (控制线) ─── PREP† ─── 测量（后选 |0⟩）
                    │
|ψ⟩ ─── ────── SEL ─────── ────────── A|ψ⟩/‖A|ψ⟩‖
```

后选成功概率 $\propto \|A|\psi\rangle\|^2$，可用振幅放大提升至 $\Omega(1)$（代价 $O(\lambda/\|A\|)$ 次放大，$\lambda = \sum_k |c_k|$）。

#### LCU 在哈密顿量模拟中的应用

**Pauli 展开**：任意 $n$ 量子比特哈密顿量 $H = \sum_\alpha h_\alpha P_\alpha$（$P_\alpha \in \{I,X,Y,Z\}^{\otimes n}$，$h_\alpha \in \mathbb{R}$）。

LCU 实现 $e^{-iHt}$（目标：哈密顿量模拟）：

**Dyson 展开截断**（Haah et al. 2021 最优算法）：

$$e^{-iHt} = \sum_{k=0}^K \frac{(-iHt)^k}{k!} + O\!\left(\frac{(e\lambda t)^{K+1}}{(K+1)!}\right)$$

截断到 $K = O(e\lambda t + \log(1/\varepsilon))$ 阶，误差 $\leq \varepsilon$。

每项 $(-iHt)^k$ 是 $k$ 个 Pauli 算符的乘积和，可以用 LCU 技术实现为一系列受控 Pauli 门。

**复杂度**：$T_{\text{LCU}} = O(\lambda t \log(\lambda t / \varepsilon) / \log\log(\lambda t / \varepsilon))$——与 Trotter 展开（误差 $\varepsilon$ 需要 $O(\lambda^2 t^2/\varepsilon)$ 步）相比，LCU 是**近最优**的哈密顿量模拟算法（$t, \varepsilon$ 依赖性几乎最优）。

#### LCU 与 Oracle Sketching 的联系

Zhao et al. 的 oracle sketching 实质上是将流式经典数据 $\{(x_t, f(x_t))\}$ 表示为如下形式的 LCU：

$$O_f \approx \prod_{t=1}^M e^{-i(\pi/M) f(x_t)|x_t\rangle\langle x_t|} = \text{LCU of single-mode phase gates}$$

正是利用了对角哈密顿量的 LCU 结构（对易性），才得到了精确的误差分析（Trotter 误差为零）。

---

### 10.3　量子蒙特卡洛（Quantum Monte Carlo）

#### 路径积分量子蒙特卡洛（PIMC）回顾

经典量子蒙特卡洛（QMC）通过 Feynman 路径积分将量子配分函数转化为经典统计力学问题：

$$Z = \mathrm{Tr}[e^{-\beta H}] = \int \mathcal{D}[\text{path}] \; e^{-S[\text{path}]}$$

然后用经典蒙特卡洛（Metropolis 算法）采样路径。

**符号问题（Sign Problem）**：对费米子系统（电子、夸克等），路径权重 $e^{-S}$ 不再是正数（因为费米子反对易），导致经典 QMC 在低温时方差指数增大，需要指数多样本——这是经典量子化学最重要的计算障碍。

#### 量子相位估计辅助的量子蒙特卡洛

**量子加速**（Babbush et al. 2019，Brassard et al. 2002）：

目标：估计哈密顿量 $H$ 在温度 $T = 1/\beta$ 下的热力学量 $\langle O \rangle_\beta = \mathrm{Tr}[Oe^{-\beta H}]/Z$。

**量子算法步骤**：

1. **初始态制备**：准备最大混合态（高温极限）$\rho_0 = I/2^n$，或通过 QSVT 制备热态的 purification：

$$|\Psi_\beta\rangle = \frac{1}{\sqrt{Z}}\sum_k e^{-\beta E_k/2} |E_k\rangle_{\text{system}} \otimes |E_k\rangle_{\text{ancilla}}$$

2. **热态 purification 的制备**（量子热化算法）：

```python
def quantum_thermal_state(H, beta, epsilon):
    n = num_qubits(H)
    ancilla = n_qubit_register()
    system = n_qubit_register()
    
    # 制备 Bell 对（最大纠缠态作为高温初始态）
    apply_Hadamard(ancilla)
    apply_CNOT(ancilla, system)
    
    # 量子绝热路径：从 β=0（均匀叠加）到目标 β
    for step in adiabatic_path(beta, steps=K):
        apply_LCU_Hamiltonian_simulation(H, step, system)
        apply_reflection(system, ancilla)
    
    return measure_thermal_average(system, O)
```

3. **估计期望值**：利用量子相位估计或经典影子提取 $\langle O \rangle_\beta$。

**量子优势**：避免了符号问题（量子计算机天然处理费米子，无需路径积分）；热态制备复杂度 $O(\sqrt{N/Z}/\varepsilon)$（振幅估计），vs. 经典 QMC 的指数代价（低温强关联费米子）。

#### 量子振幅估计（QAE）替代蒙特卡洛积分

**经典 MC 积分**：估计 $\mu = \mathbb{E}_{x \sim p}[f(x)] = \int f(x) p(x) dx$，需要 $O(1/\varepsilon^2)$ 个样本（中心极限定理）。

**量子振幅估计（Brassard et al. 2002）**：设量子电路 $\mathcal{A}$ 满足 $\mathcal{A}|0\rangle = \sqrt{\mu}|\psi_{\text{good}}\rangle + \sqrt{1-\mu}|\psi_{\text{bad}}\rangle$（$\mu$ 是成功概率），则可用量子相位估计以 $O(1/\varepsilon)$ 次 $\mathcal{A}$ 调用（精度 $\varepsilon$）——比经典的 $O(1/\varepsilon^2)$ 有**二次加速**。

**在机器学习中的应用**：
- **期望损失估计**：$L(\theta) = \mathbb{E}_{(\vec{x},y)}[\ell(f_\theta(\vec{x}), y)]$，量子加速训练集扫描；
- **积分核近似**：核函数 $k(\vec{x}, \vec{x}') = \int \phi(\vec{x}, \omega)\phi(\vec{x}', \omega)d\omega$，量子振幅估计加速核计算；
- **贝叶斯推断**：后验期望 $\mathbb{E}_{\theta|\mathcal{D}}[f(\theta)]$，量子 MCMC 加速采样。

---

## 第十一章　量子优化算法深度解析

### 11.1　QAOA 的理论极限与实践现状

#### QAOA 的参数优化景观

**Warm-start 初始化**：随机初始参数 $(\vec{\gamma}, \vec{\beta})$ 往往陷入 Barren Plateaus。已有改进策略：

1. **QAOA+ (Egger et al. 2021)**：从连续时间量子退火（绝热演化的离散化）获取初始参数：
$$\gamma_l^{(0)} = l \cdot T/p, \quad \beta_l^{(0)} = (1 - l/p) \cdot T/p$$
其中 $T$ 是绝热演化总时间；

2. **机器学习辅助**（Khairy et al. 2020）：用监督学习（神经网络/贝叶斯优化）预测给定图结构下的最优 QAOA 参数，实现从训练图到测试图的迁移。样本复杂度 $O(\text{poly}(p)/\varepsilon)$（$p$ 是 QAOA 层数）；

3. **图结构特化参数**（Brandão et al. 2018）：对 $d$-正则图，QAOA 参数仅依赖图的局部结构（$2p$ 邻域），可以通过分析少数代表性图高效推导，然后迁移到任意 $d$-正则图。

**QAOA 的近似比深度依赖（已知结果）**：

| 问题 | 深度 $p$ | 近似比 | 最优经典算法 |
|:---|:---:|:---:|:---:|
| MaxCut（3-正则） | 1 | $\geq 0.6924$ | GW：$\approx 0.878$ |
| MaxCut（3-正则） | 2 | $\geq 0.7559$ | GW：$\approx 0.878$ |
| MaxCut（任意图） | $\infty$ | $\to 1$（精确） | GW：$\approx 0.878$ |
| Max-3-XOR | 1 | $\geq 1/2 + O(1/n)$ | $1/2 + \Omega(1/\sqrt{n})$（经典） |
| Max-$k$-SAT | $O(\log n)$ | $\geq 1 - 1/k$ | 1（随机化经典） |

#### QAOA 的量子加速区间

**关键定理（Bravyi et al. 2020）**：QAOA（深度 $p < n/(4\log_2 n)$）**无法**超越某个固定的经典多项式时间算法在随机 MaxCut 实例上的平均性能。即在浅层电路下，QAOA 不具备超越经典的平均情况优势。

**反驳（条件性）**：上述定理仅适用于经典局部算法。对某些有非局部结构的问题（如量子近似 CSP），QAOA 有望在 $p = O(\log n)$ 时超越所有经典算法（基于非局域性论证，Hastings 2019）。

---

### 11.2　量子退火（Quantum Annealing）

#### 绝热量子计算（AQC）

**绝热定理**：设哈密顿量随时间缓慢变化 $H(s) = (1-s)H_0 + s H_P$（$s = t/T$ 从 0 变化到 1），初始态为 $H_0$ 的基态 $|g_0\rangle$，则最终态（$T \to \infty$）为 $H_P$ 的基态 $|g_P\rangle$。

**所需时间**：$T = O\!\left(\frac{\|dH/ds\|_{\max}}{\Delta_{\min}^2}\right)$，其中 $\Delta_{\min} = \min_{0 \leq s \leq 1}(\lambda_1(H(s)) - \lambda_0(H(s)))$ 是演化路径上的最小谱间隙。

**复杂性**：若 $\Delta_{\min} = \Omega(\text{poly}(1/n))$（多项式间隙），则 $T = O(\text{poly}(n))$，AQC 是高效的；若 $\Delta_{\min} = O(e^{-n})$（指数小间隙，如 NP-hard 问题常见），则 $T = O(e^n)$，AQC 不高效。

**D-Wave 量子退火机（实际设备）**：

D-Wave 2000Q/Advantage 实现了 $\sim 5000$ 个物理量子比特的横场伊辛（Transverse-field Ising）哈密顿量：

$$H(s) = -A(s)\sum_i \sigma_i^x + B(s)\!\left(\sum_i h_i \sigma_i^z + \sum_{ij} J_{ij} \sigma_i^z \sigma_j^z\right)$$

其中 $h_i$（偏置）和 $J_{ij}$（耦合）可编程，$A(s), B(s)$ 是预设的退火时间表。

**实验结果**：对特定设计的优化实例（如量子模拟退火基准），D-Wave 某些情形下比经典 SA（模拟退火）快 $O(10^8)$ 倍（King et al. 2019）；但在通用 TSP/MaxCut 等基准上，目前无法胜过专业经典求解器（Gurobi, CPLEX）。

#### 量子隧穿 vs. 热涨落

**量子退火的优势来源**（理论）：量子涨落（横场 $B\sigma^x$）允许量子隧穿（Quantum Tunneling），可以穿越经典方法（模拟退火需要翻越的）势垒：

- 经典：需要「爬过」高度 $\Delta E$ 的势垒，Boltzmann 因子 $e^{-\Delta E / k_BT}$（需要高温）；
- 量子：可以「穿越」宽度 $\Delta x$、高度 $V$ 的势垒，隧穿概率 $\sim e^{-\Delta x \sqrt{2mV}/\hbar}$（对窄高势垒更有效）。

**实验证据**：Boixo et al.（2016）在玻璃相优化问题上观察到量子纠错与量子隧穿的证据，D-Wave 行为与经典 SQA（随机量子退火模拟）一致，但比经典 SA 快。

---

### 11.3　量子整数规划与 QUBO 映射

#### QUBO 映射的完整方法

**二次无约束二值优化（QUBO）**：

$$\min_{\vec{x} \in \{0,1\}^n} \vec{x}^\top Q \vec{x} = \min_{\vec{x}} \sum_{ij} Q_{ij} x_i x_j$$

**约束处理（惩罚项方法）**：对含约束 $g_k(\vec{x}) = 0$ 的整数规划，添加惩罚项：

$$Q_{\text{aug}} = Q + \lambda \sum_k g_k(\vec{x})^2$$

选择足够大的 $\lambda$ 使约束满足时对应 QUBO 最小值。

**典型映射**：

| 约束 | 惩罚项 $\lambda \cdot g^2$ |
|:---|:---|
| $x_i + x_j = 1$ | $\lambda(x_i + x_j - 1)^2 = \lambda(1 - 2x_ix_j)$ |
| $\sum_i x_i = k$ | $\lambda(\sum_i x_i - k)^2$ |
| $x_i \leq x_j$（蕴含） | $\lambda x_i(1 - x_j)$ |

**应用案例：投资组合优化（Markowitz 模型）**：

$$\min_{\vec{x}} \vec{x}^\top \Sigma \vec{x} - \mu \vec{\mu}^\top \vec{x} \quad \text{s.t.} \; \sum_i x_i = k, \; x_i \in \{0, 1\}$$

（$\Sigma$ 是资产协方差矩阵，$\vec{\mu}$ 是预期收益，$k$ 是持仓数量，$x_i$ 是否持有资产 $i$）

QUBO 映射后可以提交给 D-Wave 或 QAOA 求解。对 $n = 50$ 资产，QUBO 矩阵大小 $50 \times 50$，QAOA 深度 $p = 10$ 的电路需约 $500 \times p = 5000$ 次参数更新（经典-量子混合优化）。

---

## 第十二章　量子误差缓解技术全景

### 12.1　噪声模型与误差来源

#### NISQ 设备的主要噪声类型

**1. 退相干（Decoherence）**

- **$T_1$（能量弛豫）时间**：$|1\rangle \to |0\rangle$ 的自发衰变；
- **$T_2$（相位弛豫）时间**：叠加态 $(|0\rangle + |1\rangle)/\sqrt{2}$ 的相位随机化；
- 典型值：超导量子比特 $T_1 \sim 100\mu s$，$T_2 \sim 50\mu s$；捕获离子 $T_1 \sim 10^3 s$，$T_2 \sim 1s$。

**2. 门误差（Gate Error）**

- 单量子比特门误差率 $\varepsilon_1 \sim 10^{-3}$（超导），$\sim 10^{-5}$（离子阱）；
- 双量子比特门（CNOT）误差率 $\varepsilon_2 \sim 10^{-2}$（超导），$\sim 10^{-3}$（离子阱）；
- 测量误差率 $\varepsilon_m \sim 10^{-2}$（超导），$\sim 10^{-3}$（离子阱）。

**3. 串扰（Crosstalk）**

对近邻量子比特同时操控时，一个量子比特的操作会影响另一个——这在超导量子比特中尤为严重（量子比特通过共振腔耦合）。

**4. 读取误差（Readout Error）**

$|0\rangle$ 被误读为 $|1\rangle$ 的概率（$\varepsilon_{0\to1}$）和 $|1\rangle$ 被误读为 $|0\rangle$ 的概率（$\varepsilon_{1\to0}$）。

**Pauli 噪声模型（常用近似）**：每个量子门后，以概率 $(1-p)$ 无误差，以概率 $p/3$ 各施加 $X$、$Y$、$Z$ 随机 Pauli 噪声：

$$\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

---

### 12.2　零噪声外推（ZNE）

#### 思想直觉

如果知道期望值 $\langle O\rangle(\lambda)$ 是噪声强度 $\lambda$ 的（光滑）函数，且 $\lambda = 0$ 对应理想（无噪声）值，则可以在 $\lambda > 0$ 时测量多个数据点，然后外推到 $\lambda = 0$。

#### Richardson 外推

**操作**：通过「噪声放大」实现在 $\lambda_1 < \lambda_2 < \lambda_3$ 处的测量（例如将每个门替换为 $GG^\dagger G$ 使噪声强度翻倍）。

**线性外推**（$k=2$ 点）：

$$\langle O\rangle^{\text{ZNE}} = \frac{\lambda_2 \langle O\rangle(\lambda_1) - \lambda_1 \langle O\rangle(\lambda_2)}{\lambda_2 - \lambda_1}$$

**多项式外推**（$k$ 点 Richardson 外推，$k$ 阶误差消除）：

$$\langle O\rangle^{\text{ZNE}} = \sum_{i=1}^k \gamma_i \langle O\rangle(\lambda_i)$$

其中 Richardson 系数 $\gamma_i = \prod_{j \neq i} \frac{\lambda_j}{\lambda_j - \lambda_i}$。

**噪声放大方法**：
1. **门折叠（Gate Folding）**：将单门 $G$ 替换为 $G(G^\dagger G)^m$（实现 $2m+1$ 倍噪声放大）；
2. **全局折叠**：将整个电路 $U$ 替换为 $U(U^\dagger U)^m$；
3. **脉冲拉伸**（Pulse Stretching）：在脉冲控制层面放大脉冲时长，实现连续噪声放大（比离散折叠更精细）。

**误差分析**：设真实期望值 $E_0 = \langle O\rangle(0)$，$k$ 阶外推误差为 $O(\lambda^k)$（相比不缓解的 $O(\lambda)$），代价是测量次数增加 $O(k^2)$ 倍（方差放大）。

---

### 12.3　概率误差消除（PEC）

#### 拟概率表示

**思想**：任意（有误差的）量子信道 $\mathcal{E}$ 可以表示为理想酉门的**拟概率**线性组合：

$$\mathcal{E} = \sum_i q_i \mathcal{U}_i \quad (\text{其中某些 } q_i < 0)$$

若 $\sum_i |q_i| = \gamma$（$\gamma > 1$），则通过采样实现理想信道：
- 以概率 $|q_i| / \gamma$ 执行 $\text{sign}(q_i) \cdot \mathcal{U}_i$（负号通过翻转测量结果符号处理）；
- 期望值为理想值，但方差增加 $\gamma^2$ 倍（需要 $\gamma^2$ 倍更多测量）。

**采样代价**：深度 $d$ 的电路，每层误差率 $\varepsilon$，$\gamma \approx (1 + c\varepsilon)^d \approx e^{c\varepsilon d}$，测量次数增加 $e^{2c\varepsilon d}$ 倍。对 $\varepsilon d = O(1)$（合理的电路规模），代价可接受；对 $\varepsilon d \gg 1$（深层有噪电路），PEC 的代价指数增大。

**最优拟概率分解**（Takagi 2021）：对 Pauli 噪声 $\mathcal{E}(\rho) = (1-\varepsilon)\rho + \varepsilon E\rho E$（$E$ 是 Pauli 算符），最优 $\gamma = 1 + 2\varepsilon$，即每个有噪门的取样代价约为 $1 + 2\varepsilon$。

---

### 12.4　对称性验证与后选择

**思想**：若理想电路的输出具有某种对称性（如总粒子数守恒、自旋守恒），有噪电路的输出可能违反该对称性。通过测量对称算符 $S$（满足 $[S, H] = 0$），后选择满足对称性的测量结果，可以滤除部分噪声。

**例子（量子化学 VQE）**：总电子数算符 $\hat{N} = \sum_i a_i^\dagger a_i$ 与分子哈密顿量对易。理想电路保持电子数，有噪电路可能产生错误的电子数。通过测量 $\hat{N}$ 并后选电子数正确的测量结果，过滤掉因噪声产生的「错误扇区」贡献。

**代价**：后选概率 $p_{\text{accept}} \in (0, 1)$，需要约 $1/p_{\text{accept}}$ 倍更多测量。对噪声率 $\varepsilon$ 的 $d$ 层电路，$p_{\text{accept}} \approx (1-\varepsilon)^d$，在合理深度下有效。

---

### 12.5　量子误差纠正（QEC）的基础

#### Stabilizer 码框架

**稳定子码（Stabilizer Code）**：用 $k$ 个逻辑比特编码 $n$ 个物理比特（$[[n, k, d]]$ 码，$d$ 是码距），由 $n-k$ 个对易的 Pauli 算符（稳定子生成元 $g_1, \ldots, g_{n-k}$）定义码空间：

$$\mathcal{C} = \{|\psi\rangle : g_i|\psi\rangle = +|\psi\rangle, \; \forall i\}$$

**Surface Code（表面码）**：目前最接近实用的量子纠错码，定义在二维方格网格上：

- $n$ 物理比特构成 $L \times L$ 格点（$n = L^2$），编码 $k = 1$ 个逻辑比特；
- 码距 $d = L$，可以纠正最多 $t = \lfloor(d-1)/2\rfloor$ 个任意单量子比特错误；
- 稳定子测量（综合征提取）：$Z$-型和 $X$-型 Plaquette 算符；
- **物理/逻辑比特比**：约 $L^2 : 1$（$d = 7$ 时约 $49 : 1$，$d = 31$ 时约 $961 : 1$）；
- **容错阈值**：物理错误率 $< 0.5\%$ 时，逻辑错误率随 $L$ 指数减小。

**T 门的高代价**：Clifford 门（H, S, CNOT）可以用稳定子码直接实现（横向门），但通用计算需要**非 Clifford 门**（如 T 门：$T = \mathrm{diag}(1, e^{i\pi/4})$）。T 门需要通过**门注入（Magic State Injection）** 实现：

1. 在工厂量子比特上制备「魔态」$|T\rangle = T|+\rangle = (|0\rangle + e^{i\pi/4}|1\rangle)/\sqrt{2}$；
2. 用魔态注入协议将 T 门应用到逻辑比特；
3. 魔态蒸馏（Magic State Distillation）：从低保真度物理魔态提纯出高保真度逻辑魔态，需要大量物理量子比特（约 15~1000:1）。

**Oracle Sketching 的 T 门代价**：每个多控相位门（$n = \log N$ 控制比特）分解为 $O(n^2)$ 个 T 门（Selinger 2013），总 T 门数 $M \times O(n^2) = O(N \log^2 N / \varepsilon)$。这是当前量子纠错资源估算的主要对象。

---

## 第十三章　量子自然语言处理（QNLP）

### 13.1　经典 NLP 背景与量子化动机

#### Transformer 架构的计算瓶颈

现代大语言模型（LLM）基于 Transformer 架构（Vaswani et al. 2017），核心是自注意力机制（Self-Attention）：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

其中 $Q, K, V \in \mathbb{R}^{N \times d}$（$N$ 是序列长度，$d$ 是模型维度）。

**计算瓶颈**：注意力矩阵 $QK^\top \in \mathbb{R}^{N \times N}$ 的计算需要 $O(N^2 d)$ 时间和 $O(N^2)$ 空间——对长文本（$N \sim 10^4 - 10^6$）是主要瓶颈。

**量子化动机**：若能用量子算法（如 Zhao et al. 的 oracle sketching）将注意力矩阵的「主要信息」压缩到 $O(\log N)$ 量子比特，则有望实现注意力计算的指数内存优势。

---

### 13.2　DisCoCat 框架：量子语义学

#### 分布式组合性分类语义（DisCoCat）

DisCoCat（Coecke, Sadrzadeh & Clark 2010）通过**紧凑闭范畴**（Compact Closed Category）将语法结构（Pregroup 语法）映射到语义计算（向量空间）。

**基本对应**：
- 名词（Noun）$n$：向量空间 $N$（典型地，词嵌入 $\vec{w} \in \mathbb{R}^d$）；
- 句子（Sentence）$s$：一维向量（标量，表示「句子的语义相似度」）；
- 动词（Verb）$n^r \cdot s \cdot n^l$：线性映射 $V: N \otimes N \to S$（双线性形式）。

**量子化**：将词嵌入向量空间替换为量子态 Hilbert 空间，使用量子电路实现语义组合。

**例子**：句子「Alice loves Bob」的语义计算：

```
       n      n·s·n⊤       n
Alice: |φ_A⟩  loves: T_loves  Bob: |φ_B⟩

Meaning = ⟨φ_A|T_loves|φ_B⟩ (内积/量子测量)
```

若 $|φ_A\rangle, |φ_B\rangle$ 是名词的量子态（在量子嵌入空间中），$T_{\text{loves}}$ 是动词对应的酉算符，则句子语义是量子内积——直接对应干涉型经典影子可提取的量！

#### Lambeq 工具箱与量子语言模型

**Lambeq（Cambridge Quantum Computing, 2021）**：第一个完整的 QNLP 工具箱，包含：
1. **解析器**：将自然语言句子转换为 DisCoCat 语义图（基于 CCG 语法）；
2. **量子电路生成器**：将 DisCoCat 图映射到参数化量子电路（PQC）；
3. **训练框架**：在 NISQ 设备上训练 PQC 参数，优化 NLP 任务（情感分析、机器翻译等）。

**量子语义电路的结构**（以情感分析为例）：

```
词嵌入:
Alice: ─── Ry(θ_A) ─── Rz(φ_A) ───●─── 
                                    │
loves: ─── Ry(θ_v) ─── ─────────── ⊕ ──●──
                                       │
Bob:   ─── Ry(θ_B) ─── Rz(φ_B) ───────⊕──
                                         │
句子语义测量: ─────────────────────────── 测量 Z
```

情感极性 $= \langle Z\rangle$（正 = 正面情感，负 = 负面情感）。

**训练**：参数偏移规则计算梯度，Adam 优化器最小化交叉熵损失。在 Amazon 评论情感分析数据集上，量子模型（$n = 7$ 比特）达到与经典 LSTM（$10^4$ 参数）相近的准确率（$\sim 87\%$），但参数数量仅 $\sim 100$ 个——显示出参数效率上的潜在优势。

---

### 13.3　量子注意力机制

#### 量子键值存储（Quantum Key-Value Store）

将经典注意力中的键矩阵 $K$ 和查询向量 $q$ 替换为量子态：

$$K \to \{|k_i\rangle\}_{i=1}^N \quad (\text{N 个量子键态}), \quad q \to |q\rangle (\text{量子查询态})$$

量子注意力权重：$\alpha_i = |\langle q | k_i \rangle|^2$（量子 Overlap，即键查询内积的模平方）。

**量子关联内存（QAM，Associative Memory）**：利用量子叠加并行计算所有 $\alpha_i$：

$$|q\rangle|0\rangle \xrightarrow{\text{HAM}} \sum_i \alpha_i |i\rangle$$

其中 HAM（Hopfield Associative Memory）算符一次性生成注意力权重的量子叠加，后续测量以概率 $\alpha_i$ 输出 $i$——等同于注意力采样（attention sampling）。

**Quantum Hopfield Network（Rebentrost et al. 2018）**：利用量子相位估计实现联想记忆检索，检索时间 $O(\log N)$（vs. 经典 Hopfield 的 $O(N)$），样本容量 $O(N)$（与经典相同）。但需要 QRAM 支持，去量子化风险同样存在。

---

### 13.4　量子词嵌入与语言模型

#### 量子词向量（Quantum Word Vectors）

**量子词嵌入思路**：将每个词 $w$ 映射到 $n$ 量子比特的密度矩阵 $\rho_w$（混合态允许表达词义的不确定性/多义性）：

$$\rho_w = \sum_k p_k^{(w)} |e_k^{(w)}\rangle\langle e_k^{(w)}|$$

其中 $\{|e_k^{(w)}\rangle\}$ 是词 $w$ 的「义项」量子态，$p_k^{(w)}$ 是各义项的概率。

**语义相似度**：量子保真度 $F(\rho_u, \rho_v) = \mathrm{Tr}[\sqrt{\sqrt{\rho_u}\rho_v\sqrt{\rho_u}}]^2$ 作为词语相似度度量。相比经典余弦相似度，量子保真度能捕捉词义分布的「重叠程度」而非仅是点估计的距离。

**训练方法（量子自然梯度）**：在量子信息几何（量子 Fisher 度量）下优化嵌入参数，对应 QFIM 的自然梯度更新——这在语义空间的曲率上比欧式梯度下降更自然。

#### 量子语言模型（QLM）

**自回归量子语言模型**：将语言模型 $p(w_t | w_1, \ldots, w_{t-1})$ 参数化为量子测量概率：

$$p(w_t | w_{1:t-1}) = \langle w_t | U(\vec{\theta})^\dagger \rho_{\text{context}} U(\vec{\theta}) | w_t \rangle$$

其中 $\rho_{\text{context}} = \rho_{w_1} \otimes \cdots \otimes \rho_{w_{t-1}}$ 是上下文量子态的张量积，$U(\vec{\theta})$ 是参数化「语境整合」酉算符。

**潜在优势（尚未被严格证明）**：量子纠缠允许上下文中的「远距离语义依赖」被更紧凑地表示（$O(n)$ 量子比特 vs. 经典 $O(N^2)$ 注意力矩阵），与 Zhao et al. 的空间优势框架在精神上相通。

---

## 第十四章　量子金融与量子强化学习

### 14.1　量子金融的核心应用场景

#### 量子蒙特卡洛加速衍生品定价

**背景**：金融衍生品（期权、期货等）的定价需要大量蒙特卡洛模拟。例如，欧式期权（Black-Scholes 模型）的价格：

$$V = e^{-rT} \mathbb{E}[\max(S_T - K, 0)]$$

其中 $S_T$ 是标的资产到期价格（服从对数正态分布），$K$ 是行权价，$r$ 是无风险利率，$T$ 是到期时间。

**经典 MC**：模拟 $M$ 条路径 $\{S_T^{(i)}\}$，估计 $\hat{V} = e^{-rT} \frac{1}{M}\sum_i \max(S_T^{(i)} - K, 0)$，误差 $\varepsilon \sim 1/\sqrt{M}$，需 $M = O(1/\varepsilon^2)$ 条路径。

**量子振幅估计（QAE）加速**：

1. 构造量子电路 $\mathcal{A}$ 使得 $\mathcal{A}|0\rangle = \sum_{s} \sqrt{p(s)}|s\rangle|\max(s-K, 0)\rangle$（同时编码所有路径和对应的支付函数）；
2. 用 QAE 估计 $\mathbb{E}[\max(S_T - K, 0)]$，精度 $\varepsilon$ 需要 $O(1/\varepsilon)$ 次量子电路执行——二次加速！

**实际挑战**：构造 $\mathcal{A}$ 需要对数正态分布的量子态制备（$O(\log(1/\varepsilon))$ 个量子比特表示连续分布）和量子算术（计算 $\max(s-K, 0)$）。最优实现（Stamatopoulos et al. 2020）的完整逻辑比特数约为 30-50 个——在 2027-2030 年的容错量子计算机上可行。

#### 投资组合优化

**完整量子流程（Hybrid Quantum-Classical Pipeline）**：

```python
def quantum_portfolio_optimization(returns, cov_matrix, k, lambda_risk):
    n = len(returns)  # 资产数量
    
    # 步骤 1：用 QUBO 映射（经典预处理）
    Q = lambda_risk * cov_matrix - np.diag(returns)
    penalty = large_constant * (sum_constraint_matrix)
    Q_aug = Q + penalty
    
    # 步骤 2：量子求解（QAOA 或 D-Wave）
    optimal_portfolio = QAOA_solver(Q_aug, p=10, shots=1000)
    
    # 步骤 3：经典后处理
    return optimal_portfolio
```

**基准对比**（Mugel et al. 2022，真实股票数据）：
- $n = 50$ 资产，$k = 10$ 持仓；
- 经典精确求解（整数规划）：最优解，但 $\sim 10$ 分钟；
- QAOA（$p = 10$，IBM 量子计算机）：次优解（距离最优 $\sim 2\%$），但理论上可并行化；
- D-Wave 2000Q：近似最优（$\sim 1\%$ 差距），$\sim 1$ 秒；
- 趋势：随着量子比特增加，量子方法有望在 $n > 1000$ 资产时超越经典求解器。

---

### 14.2　量子强化学习（QRL）

#### 强化学习基础

**MDP 框架**：马尔可夫决策过程 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$：
- $\mathcal{S}$：状态空间；
- $\mathcal{A}$：动作空间；
- $P(s'|s, a)$：转移概率；
- $R(s, a)$：即时奖励；
- $\gamma \in (0,1)$：折扣因子。

目标：学习策略 $\pi: \mathcal{S} \to \mathcal{A}$，最大化累积奖励 $G = \sum_{t=0}^\infty \gamma^t R(s_t, a_t)$。

**Q-learning（Deep Q-Network, DQN）**：学习动作价值函数 $Q^\pi(s, a)$，用神经网络近似 $Q$，通过 Bellman 方程迭代优化。

#### 量子强化学习的三种范式

**范式 1：VQC 作为策略函数（PQC Policy）**

将策略网络替换为参数化量子电路：

$$\pi_{\vec{\theta}}(a|s) = |\langle a | U(\vec{\theta}) | \phi(s)\rangle|^2$$

其中 $|\phi(s)\rangle$ 是状态 $s$ 的量子编码（角度编码），$U(\vec{\theta})$ 是可训练量子电路，$\{|a\rangle\}$ 是测量基。

训练：量子策略梯度（参数偏移规则），最大化期望累积奖励：

$$\nabla_{\vec{\theta}} J(\vec{\theta}) = \mathbb{E}_{\tau \sim \pi_{\vec{\theta}}}\left[\sum_t G_t \nabla_{\vec{\theta}} \log \pi_{\vec{\theta}}(a_t|s_t)\right]$$

**潜在优势**：$n$ 量子比特 PQC 参数数量 $\sim O(n)$，但表达力（VC 维）$\sim O(2^n)$——高参数效率（**表达效率**优势）。实验结果（Lockwood & Si 2021）：在 CartPole 环境中，6 量子比特 PQC（$\sim 20$ 参数）与 64 神经元经典网络（$\sim 10^4$ 参数）性能相当，参数效率高 $500\times$。

**范式 2：量子模拟环境**

对于量子系统作为环境（如量子硬件校准、量子纠错策略优化）：

$$\text{Environment: quantum circuit} \quad \text{Agent: classical/quantum RL}$$

**例子（量子纠错的 RL）**（Andreasson et al. 2019）：用 RL（Deep Q-network）学习表面码的最优解码策略，在随机 Pauli 噪声下达到接近最优解码器的性能（$\sim 99\%$ 阈值）。RL 解码器相比传统最小权重完美匹配（MWPM）解码器，在非对称噪声下更鲁棒。

**范式 3：量子探索加速**

RL 中的探索-利用权衡（Exploration-Exploitation Tradeoff）需要高效搜索状态空间。量子计算可以通过**Grover 搜索**加速找到最优动作：

$$Q^*(s, a^*) = \max_a Q^*(s, a) \quad \text{(量子最大值搜索)}$$

利用 Dürr-Høyer 量子最大化算法（1996），在 $|\mathcal{A}|$ 个动作中找到最大值 Q 只需 $O(\sqrt{|\mathcal{A}|})$ 次 Q 函数评估（vs. 经典 $O(|\mathcal{A}|)$）——这提供了与动作空间大小相关的二次加速。

---

### 14.3　量子迁移学习

#### 经典迁移学习回顾

**思想**：在大规模预训练模型（如 VGG、BERT）的特征表示上，针对具体下游任务进行微调，节省训练成本。

**典型流程**：
1. 用大型预训练网络（源任务）提取特征 $\vec{z} = f_{\text{pretrained}}(\vec{x})$；
2. 在小规模目标任务数据集上训练小型分类头 $g_\theta(\vec{z})$；
3. 可选：解冻部分预训练层联合微调。

#### 量子迁移学习框架

**Caro et al.（2022）的量子迁移学习定理**：

设量子电路 $U(\vec{\theta}) = V(\vec{\phi}) W$（$W$ 是预训练的「量子特征提取」部分，$V(\vec{\phi})$ 是待微调的目标任务头），则：

**泛化误差界**：

$$R[V(\vec{\phi}) W] \leq \hat{R}_N[V(\vec{\phi}) W] + O\!\left(\sqrt{\frac{d_{\text{eff}}(\vec{\phi}) + \log(1/\delta)}{N}}\right)$$

其中 $d_{\text{eff}}(\vec{\phi})$ 是**有效维度**（effective dimension），定义为 QFIM 的「有效秩」，满足 $d_{\text{eff}}(\vec{\phi}) \leq |\vec{\phi}|$（参数数量）。

**关键结论**：若预训练好的 $W$ 使得目标任务的 QFIM 仅有少数大特征值（低有效维度），则小目标数据集即可泛化——这是量子迁移学习的信息论保证。

**量子-经典混合迁移**：在实践中，经典预训练（如 ResNet）提取图像特征 $\vec{z}$，量子电路处理 $\vec{z}$（通过角度编码）并与经典分类器结合：

```
Image → [Classical CNN (frozen)] → z → [Quantum PQC (trainable)] → Measurement → Label
```

Hybrid 架构在小样本学习（Few-Shot Learning）任务上，量子 PQC 头相比等参数量经典全连接头有更好的泛化性（Mari et al. 2020 实验）——但优势来源仍有争议（是否来自量子性或仅是参数正则化效果）。

---

## 第十五章　等变量子网络与几何量子机器学习

### 15.1　等变量子神经网络（EQNN）

#### 对称性在机器学习中的作用

经典深度学习的一个核心进展是**将问题的物理对称性编码到网络结构中**：
- 卷积神经网络（CNN）：平移等变性（图像平移后输出相应平移）；
- 图神经网络（GNN）：节点置换不变性；
- 球面神经网络（Spherical CNN）：旋转等变性。

**等变性定义**：设群 $G$ 作用于输入 $\vec{x}$（通过表示 $\rho_{\text{in}}$），网络 $f$ 满足 $f(\rho_{\text{in}}(g)\vec{x}) = \rho_{\text{out}}(g)f(\vec{x})$（$\forall g \in G$），则称 $f$ 是 $G$-等变的。

**优势**：
1. 参数数量减少（共享对称相关的权重）；
2. 泛化性提高（网络天然满足测试集的对称约束）；
3. 避免 Barren Plateaus（有对称性约束的参数空间梯度不指数消失）。

#### 量子等变性的形式化

**量子 $G$-等变电路**（Meyer et al. 2023, Nguyen et al. 2022）：设群 $G$ 在量子 Hilbert 空间 $\mathcal{H}$ 上有酉表示 $\{U_g\}_{g \in G}$，参数化量子电路 $V(\vec{\theta})$ 满足：

$$[V(\vec{\theta}), U_g] = 0 \quad \forall g \in G$$

（即 $V(\vec{\theta})$ 在每个 $U_g$ 的对易子群中），则称 $V(\vec{\theta})$ 是 $G$-等变量子电路。

**等变参数空间的 Lie 代数刻画**：$G$-等变门构成 $G$ 的中心化 Lie 代数的指数映射，可以用 $G$ 的 Schur 引理系统性构造。

**重要结论（Schur-Weyl 对偶）**：$n$ 量子比特置换对称（$S_n$ 等变）量子电路的参数数量：

$$|\vec{\theta}|_{\text{eqv}} = O(p(n)) \quad (\text{其中 } p(n) \text{ 是整数 } n \text{ 的分拆数，次多项式增长})$$

远少于一般 VQA 的 $O(n \cdot L)$ 个参数，且无 Barren Plateaus（Schur-Weyl 对偶保证梯度下界多项式）。

---

### 15.2　数据重上传（Data Re-uploading）

#### 思想

经典神经网络通过多层非线性变换实现复杂函数近似：$f(\vec{x}) = \sigma(W_L \sigma(W_{L-1} \cdots \sigma(W_1\vec{x})))$。

量子电路的数据编码（$E(\vec{x})$）通常只进行一次（输入层），后续层不再接触数据。**数据重上传（Pérez-Salinas et al. 2020）**打破这一限制：在每个电路层都重新编码数据：

$$U(\vec{x}, \vec{\theta}) = \prod_{l=1}^L V_l(\vec{\theta}_l) E_l(\vec{x})$$

**为什么有效**：对于单量子比特电路，可以证明单量子比特量子电路通过数据重上传可以**通用近似任意连续函数**（类比经典神经网络的通用近似定理）——这赋予了浅层 PQC 与深层经典网络等价的表达能力。

**正式定理（Pérez-Salinas et al. 2020）**：对任意连续函数 $f: [-1,1]^d \to [-1,1]$ 和 $\varepsilon > 0$，存在深度为 $O(d/\varepsilon)$ 的数据重上传量子电路，使得期望值 $\langle Z\rangle$ 满足 $\sup_x|f(x) - \langle Z\rangle_x| \leq \varepsilon$。

**与 Fourier 分析的联系**（Schuld et al. 2021）：数据编码层 $E(\vec{x}) = e^{-ix\cdot\Omega}$（$\Omega$ 是编码频率矩阵），则量子电路的输出是输入 $\vec{x}$ 的 Fourier 级数：

$$f(\vec{x}) = \sum_{\vec{\omega} \in \Omega} c_{\vec{\omega}} e^{i\vec{\omega}\cdot\vec{x}}$$

Fourier 系数 $c_{\vec{\omega}}$ 由电路参数 $\vec{\theta}$ 决定，**可访问的频率** $\Omega$ 由编码层的特征值决定。数据重上传增加电路层数 = 增加可表达的 Fourier 频率数——直接对应经典神经网络加深 = 增加函数复杂度。

#### 实现示例（分类任务）

```python
def data_reuploading_circuit(x, theta, L=4):
    """
    x: 输入特征向量 (dim d)
    theta: 参数矩阵 (L x 3 x d)
    L: 电路层数
    """
    qc = QuantumCircuit(n_qubits)
    
    for l in range(L):
        # 数据编码层
        for i in range(d):
            qc.rx(x[i], i)  # 数据重上传
            qc.ry(theta[l, 0, i], i)  # 参数旋转
            qc.rz(theta[l, 1, i], i)
        
        # 纠缠层（环形 CNOT）
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(n_qubits - 1, 0)
    
    # 测量
    return qc.measure_all()
```

---

### 15.3　量子核对齐（Quantum Kernel Alignment）

#### 核对齐的思想

**经典核对齐（Kernel Alignment, KA）**：衡量核矩阵 $K$ 与理想核（标签矩阵 $yy^\top$）的相似度：

$$A(K, y) = \frac{\langle K, yy^\top \rangle_F}{\|K\|_F \|yy^\top\|_F} = \frac{\sum_{ij} K_{ij} y_i y_j}{\sqrt{\sum_{ij} K_{ij}^2} \cdot N}$$

最大核对齐意味着核矩阵中相同类别的样本内积大，不同类别的小——即理想分类核。

**量子核对齐训练**：在有标签数据上优化量子电路参数 $\vec{\theta}$，最大化量子核的核对齐：

$$\max_{\vec{\theta}} A(K_Q(\vec{\theta}), y) = \frac{\sum_{ij} |\langle\phi(\vec{x}_i; \vec{\theta})|\phi(\vec{x}_j; \vec{\theta})\rangle|^2 y_i y_j}{\|K_Q(\vec{\theta})\|_F \cdot N}$$

梯度通过参数偏移规则估计（每个 $K_{ij}$ 的梯度需 4 次量子电路执行，$N^2$ 个元素共需 $4N^2$ 次）。

**核对齐 vs. 普通 SVM 训练**：核对齐是直接优化核矩阵的「质量」，而非最小化 hinge 损失——适合在 NISQ 环境下（噪声高、每次电路执行代价高）高效利用标注数据。

**实验结果（Hubregtsen et al. 2022）**：在医学影像分类（乳腺癌识别）数据集上，量子核对齐训练后的 QSVM 相比无对齐的 QSVM 准确率提升 $5\%$-$10\%$（$n = 2$ 量子比特，$N = 100$ 训练样本）。

---

## 第十六章　量子硬件与工程实现路径

### 16.1　主流量子计算平台对比

| 平台 | 代表系统 | 量子比特数（2025） | 单门误差 | 双门误差 | T2 时间 | 连接性 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 超导（Transmon） | IBM Heron, Google Willow | $\sim 1000$ | $\sim 0.1\%$ | $\sim 0.3\%$ | $\sim 300\mu s$ | 近邻 2D 格点 |
| 捕获离子 | IonQ Forte, Quantinuum H2 | $\sim 32$ | $\sim 0.01\%$ | $\sim 0.1\%$ | $\sim 1s$ | 全连接 |
| 光量子（MBQC） | PsiQuantum | $\sim 100$ | $\sim 0.5\%$ | 天然 | $\sim \mu s$ | 光子链路 |
| 中性原子 | QuEra Aquila, Pasqua | $\sim 256$ | $\sim 0.5\%$ | $\sim 0.5\%$ | $\sim 10s$ | 可重构 |
| 拓扑量子比特 | Microsoft InAsSb | $< 10$（测试中） | $< 0.01\%$（理论） | $< 0.01\%$（理论） | 很长（理论） | TBD |

#### 超导量子比特：最成熟的 NISQ 平台

**Transmon 量子比特**：超导 Josephson 结与电容构成的非线性谐振子，调制频率 $\sim 4$-$7$ GHz，非谐性 $\sim 200$ MHz（允许选择性激励 $|0\rangle \leftrightarrow |1\rangle$ 而不激励 $|1\rangle \leftrightarrow |2\rangle$）。

**门操作**：
- 单量子比特门：微波脉冲（$\sim 20$-$50$ ns），任意 SU(2) 旋转；
- 双量子比特门（CNOT/CZ）：利用可调耦合器（Tunable Coupler）调制量子比特间的交换相互作用，$\sim 100$-$200$ ns；
- 测量：通过读取腔（Readout Resonator）的色散相移，$\sim 100$-$500$ ns。

**规模化挑战（制造、控制、纠错）**：
1. 制造一致性：不同量子比特频率/连接的随机涨落（制造噪声），需要个别校准；
2. 控制电子：每个量子比特需要 1-2 条微波控制线，1000 量子比特需要 ~2000 条线（低温 $\sim 20$ mK 与室温之间的热负荷）；
3. 频率碰撞：固定频率 Transmon 的频率分布有限，$\sim 1000$ 量子比特时频率冲突概率增大。

#### 捕获离子：最精确的量子系统

**离子阱量子比特**：$^{171}\text{Yb}^+$ 或 $^{40}\text{Ca}^+$ 离子中超精细或光学跃迁作为量子比特，Paul 阱（射频四极场）囚禁离子，激光或微波实现量子门。

**全连接拓扑**：所有离子共享同一个阱，通过集体声子模式（共同振动）实现任意对离子间的双量子比特门（无需路由）。

**门操作细节**（Mølmer-Sørensen 门）：
$$U_{MS}(\vec{\theta}) = \exp\!\left(-i\theta \sum_{i<j} (\vec{X}_i \cdot \vec{X}_j + \vec{Y}_i \cdot \vec{Y}_j)\right)$$
激光拉比耦合 $\Omega$，演化时间 $t = \pi/(2\eta\Omega)$（$\eta$ 是 Lamb-Dicke 参数），实现最大纠缠 $\theta = \pi/4$。

**Oracle Sketching 在离子阱上的可行性**：
- 多控相位门：在全连接离子阱上，无需路由开销，$O(\log N)$ 控制比特的多控门可用 $O(\log N)$ 个 MS 门实现（每个 MS 门纠缠所有 $\log N$ 个比特）；
- 对比超导：超导需要 $O(\log^2 N)$ 个双比特门（因路由开销）；
- 捕获离子门时间（$\sim 100\mu s$）比超导（$\sim 200$ ns）慢约 500 倍，但错误率低约 10 倍，整体容错资源与超导相当。

#### 中性原子：可扩展的新星

**Rydberg 原子量子比特**：$^{87}$Rb 或 $^{133}$Cs 原子，超精细基态作为量子比特，激光阵列（AOD/SLM）捕获，二维可重构布局。

**里德伯纠缠**：激发到里德伯态 $|r\rangle$（$n \sim 60$），两原子间里德伯阻塞（Rydberg Blockade）：

$$V_{dd} = \frac{C_6}{R^6} \gg \Omega \implies \text{不能同时激发两个相邻原子}$$

利用里德伯阻塞实现 CZ 门：$|11\rangle \to -|11\rangle$，$|00\rangle, |01\rangle, |10\rangle$ 不变，门时间 $\sim 1\mu s$，误差 $\sim 0.3\%$。

**动态重构**：AOD 可以实时移动原子位置，实现任意拓扑的连接——这对 QAOA（图结构变化的优化）和量子纠错（动态注入新鲜量子比特）特别有利。

**QuEra Aquila（2022）**：256 量子比特 Rydberg 原子平台，已演示 $100^+$ 量子比特纠缠态，弗拉斯特（Frustration）相变，以及组合优化问题（独立集，MIS）的近似求解。

---

### 16.2　量子编译器与量子电路优化

#### 量子编译的挑战

**物理约束**：实际量子计算机仅支持有限门集（如 IBM 的 $\{CNOT, R_Z, \sqrt{X}\}$，IonQ 的 $\{MS, R_{xx}, R_z\}$），且双量子比特门仅在拓扑相邻的量子比特之间可用（超导）。

**编译目标**：将逻辑量子电路转换为满足物理约束的物理电路，同时：
1. 最小化双量子比特门数量（主要误差来源）；
2. 最小化电路深度（减少退相干损失）；
3. 满足连接性约束（路由：通过 SWAP 门移动量子比特）。

**编译复杂性**：最优 SWAP 路由是 NP-hard（Siraichi et al. 2018），但有多种启发式算法：
- **SABRE**（Li et al. 2019）：基于启发式搜索的双向路由算法，IBM Qiskit 默认；
- **TKET**（Cambridge Quantum Computing）：利用 ZX-Calculus 化简量子电路，减少 T 门数量（对容错量子计算关键）；
- **机器学习辅助编译**（RL 编译器）：用强化学习学习路由策略，超越 SABRE 约 $10\%$-$20\%$ 的 SWAP 开销（He et al. 2022）。

#### T 门优化（容错量子计算的关键）

在容错设置下，Clifford 门（$H, S, CNOT$）可以通过横向门高效实现，而 T 门需要代价高昂的魔态蒸馏。**T 门计数**是评估容错电路资源的首要指标。

**T 门减少技术**：

1. **旋转合并**（Rotation Merging）：将连续的旋转门 $R_z(\theta_1) R_z(\theta_2) = R_z(\theta_1 + \theta_2)$ 合并，减少门数；

2. **相位多项式优化**（Phase Polynomial Optimization, Amy et al. 2014）：对 Clifford+T 电路的 T 门数量，利用整数线性规划找到最优 T-count 分解；对 Toffoli 门（$3$ 控制位 CNOT）：朴素分解需 $7$ 个 T 门，最优分解仅需 $4$ 个（Selinger 2013）；

3. **ZX-Calculus 化简**：利用量子电路的图论表示（ZX-图），用重写规则系统性化简，可将 T 门数减少 $30\%$-$50\%$（Kissinger & van de Wetering 2020）。

**Oracle Sketching 的 T 门优化**：每个「多控相位门」$\exp(-i\theta |x_t\rangle\langle x_t|)$ 可以通过 Gray 码分解减少 CNOT 数，再结合 T 门优化，预计将总 T 门数从 $O(N \log^2 N)$ 优化到 $O(N \log N \cdot \log\log N)$（结合 Selinger 最优 Toffoli 分解）。

---

### 16.3　量子-经典混合计算架构

#### 混合架构的层次

**Layer 1（量子处理单元 QPU）**：执行量子电路，提供量子测量结果。
**Layer 2（经典控制处理器）**：实时控制 QPU（脉冲序列生成），处理测量结果，计算下一步参数（$\mu s$-$ms$ 延迟）。
**Layer 3（经典 HPC 节点）**：运行优化算法（梯度下降、BFGS、贝叶斯优化），处理大规模数据预处理/后处理（$ms$-$s$ 延迟）。
**Layer 4（数据存储与预处理）**：数据库、流式数据管道、特征工程（$s$-$min$ 延迟）。

**Oracle Sketching 的架构对应**：

```
┌─────────────────────────────────────────────────────┐
│  Layer 4: 流式数据输入 {(x_t, f(x_t))}               │
│           ↓ 每次到达一个样本                          │
│  Layer 3: 计算多控相位门参数 θ_t = π·f(x_t)/M        │
│           ↓ 编译为物理脉冲序列                        │
│  Layer 2: 实时更新量子电路 V_t = V_{t-1} · e^{-iθ_t|x_t⟩⟨x_t|}│
│           ↓ M 步完成 oracle 草图                      │
│  Layer 1: QPU 执行干涉型经典影子测量（Hadamard 测试）  │
│           ↓ 返回 b = 0 或 1 的测量结果               │
│  Layer 3: 统计分析 → 构建经典预测模型               │
└─────────────────────────────────────────────────────┘
```

**延迟瓶颈**：Layer 3→Layer 2 的「流式数据 → 量子电路参数」的实时编译需要 $\ll \tau$（样本刷新时间）完成，否则流式假设被违反。对 IMDb/scRNA-seq，$\tau \sim O(1)$ 秒，这对当前控制硬件（$\mu s$ 级延迟）是完全可行的。

---

*本综述基于 2026 年 4 月前公开发表的文献，持续更新。*

*版本：v3.0（全面扩展版），2026 年 4 月。*
