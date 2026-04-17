# 第二章后半节修订（ReaxFF 参数优化方法 与 MLP 残差能量修正方法）

> 本文件对原稿《毕业论文-孙-第二章-修订.md》中"ReaxFF 参数优化方法"与"MLP 残差能量修正方法"两节进行了公式核对与扩充。式编号与原稿保持一致 (式 2-25 ~ 2-48)。修订要点详见正文末附"修订一览"。

---

## ReaxFF 参数优化方法

### 优化问题与总体思路

ReaxFF 参数化的核心目标是利用一组参考数据（DFT 计算所得的相对能量、原子受力、键长键角、原子电荷及形成能等）约束势能面 $E_{\text{ReaxFF}}(\mathbf R;\boldsymbol\theta)$ 的形状，使力场能够同时合理描述稳定结构、过渡态及反应路径。对于本文涉及的 Pd/Zn/C/H/O 体系，待优化参数 $\boldsymbol\theta$ 主要包括三类：

- **单原子参数**：电负性 $\chi_i$、硬度 $\eta_i$、价电子数 $\mathrm{Val}_i$、孤对电子最优数 $n_{i,\text{opt}}^{lp}$、过/欠配位修正参数 $p_{ovun}$ 等；
- **双原子键参数**：平衡键长 $r_{0,ij}^{\sigma/\pi/\pi\pi}$、键解离能 $D_e^{\sigma/\pi/\pi\pi}$、键级形状参数 $p_{bo,1\!-\!6}$、Morse vdW 参数 $D_{ij},\alpha_{ij},r_{vdW}$ 等；
- **多体参数**：三体角参数 $\theta_{0,0},k_a,k_b,p_{val}$ 与四体扭转参数 $V_1,V_2,V_3,p_{tor}$ 等。

由于不同参数之间存在显著耦合，独立参数总数通常在百维量级，因此 ReaxFF 参数优化本质上是一个**高维、强耦合、多峰的非线性最优化问题**，并且对数值稳定性要求很高（键级公式中分母可能接近零、指数项可能溢出）。

为消除 DFT 绝对能量基准差异的影响，本文均采用**相对能量**形式：在每一类构型集合 $\mathcal{S}$ 内选取每原子能量最接近平均值的构型作为参考构型 $i_{\text{ref}}$，其余构型 $i$ 的训练目标为

$$
\Delta E_i = \frac{E_i - E_{i_{\text{ref}}}}{N_i}\bigl(\text{kcal/mol·atom}\bigr),
$$

DFT 与 ReaxFF 端均使用同一参考构型，从而消除整体能量平移。

参数优化通过最小化下式定义的加权误差泛函实现：

$$
f(\mathbf{p}) = \sum_i \left[\frac{x_{i,\text{QM}} - x_{i,\text{ReaxFF}}(\mathbf{p})}{\sigma_i}\right]^2
\tag{2-25}
$$

式 (2-25) 中 $x_i$ 可代表能量、力、键长、电荷等任意可测物理量，$\sigma_i$ 为相应数据的归一化标度。$\sigma_i$ 的两重作用：(i) 把不同量纲的物理量统一无量纲化，避免数值失衡；(ii) 通过权重突出对关键物理量（如平衡构型能量、吸附能、反应中间态能量）的约束，提高力场在目标化学环境中的可靠性。

权重 $\sigma_i$ 的设定反映了不同训练数据的相对可信度与物理重要性：DFT 误差较小且对目标反应行为影响显著的数据应赋予更高权重，而高能、远离平衡区域的构型则可适当降低权重，以避免少量异常点主导整体优化方向。

历史上，ReaxFF 参数优化长期主要依赖两类无导数方法：单参数顺序搜索 (SPS) 算法[40]和 Metropolis 蒙特卡洛-模拟退火 (MMC-SA) 算法[41-42]。前者通过逐参数局部更新降低误差，方法直观但难以处理强耦合；后者借助随机扰动和退火接受准则增强全局搜索能力，但收敛速度慢、计算代价高。对本文涉及的 Pd-Zn-C-H-O 多组分体系，这两类方法只能提供初值，难以独立完成最终参数化，因此有必要引入显式利用梯度信息的优化框架。

### JAX-ReaxFF 框架与自动微分

JAX-ReaxFF[43] 是本文全部参数优化工作的统一计算框架。该框架以 JAX[44] 为基础，将 ReaxFF 的能量计算流程——包括键级 $BO_{ij}$、键级修正 $BO'_{ij}$、各能量组分（共价键、孤对、过欠配位、价角、扭转、氢键、vdW、Coul）、电荷平衡 (EEM) 求解及总能量汇总——统一实现为**可微计算图**，从而使损失函数 $\mathcal{L}(\boldsymbol\theta)$ 对参数 $\boldsymbol\theta$ 的精确梯度可由反向模式自动微分 (reverse-mode automatic differentiation, RMAD) 一次反向传播获得：

$$
\nabla_{\boldsymbol\theta}\mathcal{L}(\boldsymbol\theta)
=\frac{\partial \mathcal{L}}{\partial E_{\text{ReaxFF}}}\cdot
\Bigl(\frac{\partial E_{\text{ReaxFF}}}{\partial \boldsymbol\theta}\Bigr)
+ \frac{\partial \mathcal{L}}{\partial \mathbf q^*}\cdot
\frac{\partial \mathbf q^*}{\partial \boldsymbol\theta},
$$

其中 $\mathbf q^*(\boldsymbol\theta)$ 为 EEM 隐式给出的平衡电荷向量，其对 $\boldsymbol\theta$ 的依赖通过对线性系统求解过程的可微反传（隐式微分）完成。对于 ReaxFF 这类百维至千维参数体系，反向传播的代价仅约为前向计算代价的 2–3 倍，因此较传统逐参数扰动法节省 $O(N)$ 倍计算量。同时，JAX 的 JIT 编译与 `vmap` 向量化能力可对一批训练构型并行评估，使梯度驱动优化与后续残差模型训练可在统一的数值实现中稳定开展。

### 优化目标与损失函数

参数优化目标是最小化训练集上的加权能量误差函数：

$$
\mathcal{L}(\boldsymbol\theta) = \frac{1}{N}\sum_{i=1}^{N} w_i\left[\frac{E_{i,\text{ReaxFF}}(\boldsymbol\theta) - E_{i,\text{DFT}}}{\sigma_i}\right]^{2}
\tag{2-26}
$$

其中 $N$ 为训练集数据点总数；$E_{i,\text{ReaxFF}}(\boldsymbol\theta)$ 为参数 $\boldsymbol\theta$ 下 ReaxFF 对第 $i$ 个构型的预测能量；$E_{i,\text{DFT}}$ 为 DFT 参考能量；$\sigma_i$ 为标度因子，能量类数据通常取 $0.1\!\sim\!1.0~\text{kcal/mol}$；$w_i$ 为附加数据类型权重，可对状态方程 (EOS)、形成能、吸附能等差异化设置。

### L-BFGS 优化器

L-BFGS[45] 是 JAX-ReaxFF 框架下可直接调用的经典拟牛顿优化器。其基本思想是**不显式构造 Hessian 矩阵**，而仅利用最近 $m$ 步参数增量与梯度增量近似目标函数的局部二阶曲率信息，因此特别适合 ReaxFF 这类高维参数优化问题。

记第 $k$ 步参数为 $\boldsymbol\theta_k$、梯度为 $\mathbf g_k=\nabla_{\boldsymbol\theta}\mathcal{L}(\boldsymbol\theta_k)$，定义

$$
\mathbf s_k = \boldsymbol\theta_{k+1}-\boldsymbol\theta_k,\quad
\mathbf y_k = \mathbf g_{k+1}-\mathbf g_k,\quad
\rho_k = \frac{1}{\mathbf y_k^{\top}\mathbf s_k},
$$

则对**逆 Hessian 近似** $H_k\approx[\nabla^2\mathcal L(\boldsymbol\theta_k)]^{-1}$ 的 BFGS 递推更新为：

$$
H_{k+1} = \left(I - \rho_k\,\mathbf{s}_k \mathbf{y}_k^{\top}\right)H_k\left(I - \rho_k\,\mathbf{y}_k \mathbf{s}_k^{\top}\right) + \rho_k\,\mathbf{s}_k \mathbf{s}_k^{\top}
\tag{2-27}
$$

对 $n$ 维参数空间，显式存储 $H_k$ 需 $O(n^{2})$ 内存，难以承受。L-BFGS 仅保留最近 $m$ 对 $(\mathbf s_k,\mathbf y_k)$（典型 $m\!=\!5\!\sim\!20$），并通过**双循环递推 (two-loop recursion)** 直接计算搜索方向 $\mathbf p_k = -H_k\mathbf g_k$，将每步内存与计算复杂度降至 $O(mn)$。随后结合 Wolfe 条件线搜索确定步长，完成参数更新

$$
\boldsymbol\theta_{k+1} = \boldsymbol\theta_k + \alpha_k \mathbf p_k.
$$

对局部近似二次的损失面，L-BFGS 通常具有较强的后期收敛能力。本文第 3、4 章所使用的参数集即基于 L-BFGS 完成；但其对初始点和数值稳定性较为敏感，这也是后续引入更稳健的一阶优化器的动机。

### Adam 优化器

Adam[46] 是本文在 JAX-ReaxFF 框架中重点引入的一阶自适应优化器。与依赖局部曲率近似的 L-BFGS 不同，Adam 通过同时维护梯度的一阶矩 (动量) 和二阶矩 (能量) 估计，为每个参数自适应分配更新步长，因此在优化初期更适合处理梯度尺度差异显著、噪声较大且数值稳定性要求较高的 ReaxFF 参数空间。其逐步更新规则如下：

$$
\mathbf g_t = \nabla_{\boldsymbol\theta}\mathcal{L}(\boldsymbol\theta_{t-1})
\tag{2-28a}
$$

$$
\mathbf m_t = \beta_1 \mathbf m_{t-1} + (1-\beta_1)\mathbf g_t
\tag{2-28b}
$$

$$
\mathbf v_t = \beta_2 \mathbf v_{t-1} + (1-\beta_2)\,\mathbf g_t\odot \mathbf g_t
\tag{2-28c}
$$

$$
\hat{\mathbf m}_t = \frac{\mathbf m_t}{1-\beta_1^{\,t}};\qquad
\hat{\mathbf v}_t = \frac{\mathbf v_t}{1-\beta_2^{\,t}}
\tag{2-28d}
$$

$$
\boldsymbol\theta_t = \boldsymbol\theta_{t-1} - \alpha\,\frac{\hat{\mathbf m}_t}{\sqrt{\hat{\mathbf v}_t} + \varepsilon}
\tag{2-28e}
$$

其中 $\odot$ 表示逐元素积；$\alpha$ 为全局学习率（典型 $10^{-3}\!\sim\!10^{-4}$）；$\beta_1=0.9,\,\beta_2=0.999$ 分别控制一阶/二阶矩的指数滑动平均；$\varepsilon=10^{-8}$ 用于防止除零。式 (2-28d) 的偏差修正用于抵消 $\mathbf m_0=\mathbf v_0=\mathbf 0$ 初始化在前若干步引入的零偏置。

直观上，$\hat{\mathbf m}_t$ 给出梯度的低通滤波 (动量)，$\sqrt{\hat{\mathbf v}_t}$ 估计每个参数梯度的近期均方根 (RMS)，因此 Adam 在每个参数维度上的有效学习率约为 $\alpha/\mathrm{RMS}(g)$，能够自动放大梯度小、缩小梯度大的参数维度，从而减轻人工分组调参的负担。

本文在 JAX-ReaxFF 框架内实现 Adam，并在此基础上进一步叠加梯度裁剪、学习率调度、参数边界约束与早停等机制，形成后续章节所采用的稳定优化流程。

### 梯度裁剪

在 ReaxFF 参数优化初期，某些参数可能位于势能面的高曲率区域，导致梯度幅值异常增大。若直接利用此类梯度更新参数，可能触发 ReaxFF 内部的数值不稳定（如键级修正中分母趋零、指数项溢出），从而产生 NaN 并中断训练。为此，本文采用**全局范数裁剪 (Global Norm Clipping)** 策略：

$$
\|\mathbf{g}\|_2 = \sqrt{\sum_i g_i^{\,2}}
\tag{2-29}
$$

$$
\mathbf{g}_{\text{clipped}} = \mathbf{g}\cdot\min\!\left(1,\;\frac{\text{clip\_norm}}{\|\mathbf{g}\|_2}\right)
\tag{2-30}
$$

式 (2-30) 在 $\|\mathbf g\|_2\le \text{clip\_norm}$ 时保持梯度不变，在超过阈值时按比例缩放至范数恰为 $\text{clip\_norm}$。该操作只改变梯度幅值而不改变其方向，因此既能保持下降方向的有效性，又能将更新步长限制在安全范围内。在具体实现中，本文采用"先全局范数裁剪、再 Adam 更新"的串联流程，以兼顾数值稳定性与收敛效率。

### 学习率调度

固定学习率在接近最优解时易产生振荡，而过小的学习率又会导致前期收敛缓慢。为兼顾优化初期的稳定性与后期的收敛效率，本文引入**预热-余弦退火 (warmup + cosine annealing)** 调度策略：

$$
\alpha(t) = \alpha_{\max}\,\frac{t}{T_{\text{warm}}},\quad 0 \le t \le T_{\text{warm}}
\tag{2-31}
$$

$$
\alpha(t) = \alpha_{\min} + \frac{\alpha_{\max}-\alpha_{\min}}{2}\left[1 + \cos\!\left(\frac{\pi(t-T_{\text{warm}})}{T-T_{\text{warm}}}\right)\right],\quad T_{\text{warm}} < t \le T
\tag{2-32}
$$

预热阶段，学习率由 0 线性增加至 $\alpha_{\max}$，避免 Adam 初期偏差修正未稳定时大梯度与大学习率叠加引起的震荡；退火阶段，学习率再由 $\alpha_{\max}$ 平滑衰减至 $\alpha_{\min}$，在后期通过逐步减小步长实现更稳定的收敛。本文典型设置为 $T_{\text{warm}}=0.05T$、$\alpha_{\max}=10^{-3}$、$\alpha_{\min}=10^{-5}$。此外，本文也保留指数衰减调度 $\alpha(t)=\alpha_0\gamma^t\,(\gamma\in(0,1))$，以适应快速粗优化场景。

### 参数边界约束与早停策略

**参数边界约束**：在每一步参数更新后，本文均将参数限制在物理合理边界 $[\boldsymbol\theta_{\min},\boldsymbol\theta_{\max}]$ 内，其更新形式为：
$$
\boldsymbol\theta_t = \mathrm{clip}\bigl(\boldsymbol\theta_t,\,\boldsymbol\theta_{\min},\,\boldsymbol\theta_{\max}\bigr),
\quad
[\mathrm{clip}(\boldsymbol\theta,\boldsymbol\theta_{\min},\boldsymbol\theta_{\max})]_i = \max\!\bigl(\theta_{\min,i},\,\min(\theta_i,\theta_{\max,i})\bigr)
\tag{2-33}
$$

边界由 ReaxFF 物理意义给出，例如键解离能 $D_e>0$、平衡键长 $r_0>0$、孤对电子参数取经验合理区间等。

**早停策略**：将训练集按 9:1 划分为训练子集 $\mathcal D_{\text{tr}}$ 和验证子集 $\mathcal D_{\text{val}}$；每隔 $N_{\text{eval}}$ 步（如 50 步）评估一次验证集误差 $\mathcal L_{\text{val}}$；若连续 $K$ 个评估周期内 $\mathcal L_{\text{val}}$ 的相对改善低于预设容差 $\mathrm{tol}$（如 $10^{-5}$），则停止优化并返回历史最优 $\boldsymbol\theta^*=\arg\min_t\mathcal L_{\text{val}}(\boldsymbol\theta_t)$。该策略可避免模型对特定构型过拟合，并提高力场在小样本训练条件下的泛化能力。

---

## MLP 残差能量修正方法

### 总体框架

在完成基于 JAX-ReaxFF 框架的 ReaxFF 参数优化后，本文进一步在同一自动微分计算图基础上引入 MLP 残差能量修正：

$$
E_{\text{total}}(\mathbf{R}) = E_{\text{ReaxFF}}\!\bigl(\mathbf{R};\boldsymbol\theta^{*}\bigr) + E_{\text{ML}}\!\bigl(\mathbf{R};\boldsymbol\phi\bigr)
\tag{2-34}
$$

$$
\mathbf{F}_j^{\text{total}}(\mathbf{R})
= -\nabla_{\mathbf R_j} E_{\text{total}}(\mathbf R)
= \mathbf F_j^{\text{ReaxFF}} + \mathbf F_j^{\text{ML}},\quad
\mathbf F_j^{\text{ML}} = -\nabla_{\mathbf R_j} E_{\text{ML}}(\mathbf R;\boldsymbol\phi)
\tag{2-35}
$$

其中 $\boldsymbol\theta^{*}$ 表示已优化完成并冻结的 ReaxFF 参数，$\boldsymbol\phi$ 表示 MLP 的可训练参数。

该残差框架并非以纯机器学习势直接替代 ReaxFF，而是令 MLP 仅学习 DFT 势能面与 ReaxFF 基线势能面之间的**系统偏差** $\Delta E^{\text{DFT-ReaxFF}}=E_{\text{DFT}}-E_{\text{ReaxFF}}$。这样既保留了 ReaxFF 对反应性拓扑变化的物理先验，又借助数据驱动模型提高了局域能量与力的定量精度。同时，由式 (2-35)，力由总能量对原子坐标的负梯度严格给出，因此 $E_{\text{ML}}$ 对应的力可通过自动微分一致获得，从而保证能量与力之间的解析一致性[43-44]。

### SE(3) 对称性保证

为保证残差模型具有明确的物理意义，$E_{\text{ML}}$ 必须满足与原始势函数相同的 **SE(3) 不变性**，即对任意平移向量 $\mathbf t\in\mathbb R^3$ 与旋转矩阵 $\mathbf Q\in SO(3)$，体系总能量满足

$$
E_{\text{ML}}(\mathbf Q\mathbf R+\mathbf t)=E_{\text{ML}}(\mathbf R),\quad
\mathbf F_j^{\text{ML}}(\mathbf Q\mathbf R)=\mathbf Q\,\mathbf F_j^{\text{ML}}(\mathbf R).
$$

在本文框架中，这一要求通过两层机制保证：
1. **不变特征构造**：MLP 的输入特征 $\mathbf f_i$ 全部由原子间距 $r_{ij}$、键级 $BO'_{ij}$、键角 $\theta_{ijk}$ 及其统计量 (求和、最大、方差) 构造，这些均为 SE(3) 的零阶张量 (标量)，对平移和旋转天然不变；
2. **能量梯度给出力**：力作为能量对坐标的负梯度，等变性可由能量表达式自动继承。

由此，所构建的 MLP 残差模型不会破坏体系原有的基本对称性[30]。

### 逐原子特征描述符

残差网络采用逐原子标量特征 $\mathbf f_i\in\mathbb R^d$ 作为输入，全部由 ReaxFF 计算流程中的可微中间量提取，因此与 ReaxFF 力学模型保持一致。本文使用两类特征集：

**基础特征 (12 维)**，主要表征局域配位环境：

$$
\mathbf f_i^{(\text{base})} = \bigl[\,
\underbrace{\Delta_i,\,\Delta_i^{lp},\,\Delta_i^{ovun}}_{\text{配位/孤对/过欠配位}},\;
\underbrace{q_i,\,n_i^{lp}}_{\text{电荷/孤对数}},\;
\underbrace{\textstyle\sum_j BO'_{ij},\,\sum_j BO_{ij}^{\sigma},\,\sum_j BO_{ij}^{\pi},\,\sum_j BO_{ij}^{\pi\pi}}_{\text{总/分量键级求和}},\;
\underbrace{\max_j BO'_{ij},\,\overline{BO'_{ij}},\,\sigma(BO'_{ij})}_{\text{近邻键级统计量}}
\,\bigr],
$$

**扩展特征 (43 维)**，在基础特征之上进一步加入：

- 键角统计：$\sum_{jk}\theta_{ijk},\;\overline{\theta_{ijk}},\;\sigma(\theta_{ijk})$；
- 键级波动统计：$\mathrm{Var}_j(BO'_{ij}),\;\sum_j(BO'_{ij}-\overline{BO'})^{2}$；
- 电荷-键级耦合：$q_i\cdot\sum_j BO'_{ij},\;q_i^{\,2}\cdot\Delta_i,\;q_i\cdot n_i^{lp}$ 等非线性交叉项。

基础特征反映局域配位环境，扩展特征进一步增强对界面极化效应和局域非线性相互作用的表征能力，因此更适用于 Pd-Zn-C-H-O 多组分体系的残差修正。

### 特征维度自注意力机制 (Feature-Wise Self-Attention)

不同化学环境下，各特征维度对残差能量的贡献并不相同。为提高模型对局域环境变化的自适应能力，本文在特征输入层引入**特征维度自注意力机制**[47]。该机制并不在不同原子之间传递信息，而仅在同一原子的特征向量内部对各维度进行重加权，因此既保留了逐原子能量分解结构，又不会破坏 SE(3) 不变性，可视为通道注意力 (channel attention) 的简化形式。

对第 $i$ 个原子，记其输入特征向量为 $\mathbf{f}_i\in\mathbb R^{d}$，自注意力层首先通过线性映射生成查询、键、值向量：

$$
\mathbf{Q}_i = \mathbf{f}_i W_Q;\quad \mathbf{K}_i = \mathbf{f}_i W_K;\quad \mathbf{V}_i = \mathbf{f}_i W_V,\quad W_Q,W_K,W_V\in\mathbb R^{d\times d}
\tag{2-36}
$$

其中 $W_Q,W_K,W_V$ 均为可训练投影矩阵。由于 $\mathbf Q_i,\mathbf K_i,\mathbf V_i$ 均来自同一原子的局域特征，因此该注意力机制本质上是在**同一原子的不同特征维度之间**建立相关性，而非执行跨原子的信息交换。

随后通过逐元素相关性计算特征维度得分，并经 softmax 归一化得到注意力系数：

$$
\mathrm{scores}_i = \frac{1}{\sqrt{d}}\,\mathbf{Q}_i \odot \mathbf{K}_i\;\in\mathbb R^{d}
\tag{2-37}
$$

$$
\alpha_{i,k} = \frac{\exp(\mathrm{scores}_{i,k})}{\sum_{l=1}^{d}\exp(\mathrm{scores}_{i,l})},\quad k=1,\dots,d
\tag{2-38}
$$

式中 $\odot$ 为 Hadamard (逐元素) 积，$1/\sqrt{d}$ 温度因子用于防止 score 量级过大导致 softmax 饱和[47]。$\boldsymbol\alpha_i\in\mathbb R^{d}$ 为针对特征维度的权重向量，其分量 $\alpha_{i,k}$ 表示第 $k$ 个特征在当前局域化学环境中的相对重要性。与标准 Transformer 中跨 token 的注意力 ($\mathbf Q\mathbf K^{\top}\!\!\in\mathbb R^{n\times n}$) 不同，本文的注意力机制仅针对单原子特征维度进行加权 ($\mathrm{scores}\in\mathbb R^{d}$)，因此实现更为轻量，也更符合逐原子残差建模的结构假设。

加权后的特征向量经输出投影 $W_O\in\mathbb R^{d\times d}$ 与残差连接得到：

$$
\mathbf{f}'_i = \boldsymbol\alpha_i \odot (\mathbf{V}_i W_O) + \mathbf{f}_i
\tag{2-39}
$$

残差连接保证原始特征能够直接传递至输出层，从而缓解深层网络训练中的梯度衰减问题，并使网络能够从近似恒等映射出发进行稳定优化。对本文采用的 43 维特征而言，该注意力层参数量仅为 $4d^{2}\!\approx\!7.4\times10^{3}$，不会显著增加整体模型复杂度。

### ResNet-MLP 层

经注意力加权后的特征 $\mathbf f'_i$ 进一步送入 ResNet-MLP 网络[48]，用于计算第 $i$ 个原子的残差能量贡献 $E_i$。为保证网络在较深层数下仍具有稳定训练能力，本文在隐藏层中引入残差连接。设第 $l$ 层隐藏向量为 $\mathbf h^{(l)}\in\mathbb R^{d_h}$，权重矩阵 $W^{(l)}\in\mathbb R^{d_h\times d_h}$、偏置 $\mathbf b^{(l)}\in\mathbb R^{d_h}$，则有：

$$
\mathbf h^{(l+1)} = \mathrm{Swish}\!\left(\mathbf h^{(l)} W^{(l)} + \mathbf b^{(l)}\right) + \mathbf h^{(l)}\quad(\text{含残差})
\tag{2-40}
$$

$$
\mathbf h^{(l+1)} = \mathrm{Swish}\!\left(\mathbf h^{(l)} W^{(l)} + \mathbf b^{(l)}\right)\quad(\text{无残差})
\tag{2-41}
$$

输入层与输出层不含残差，中间隐藏层均添加残差连接。本文采用 $L=4$ 层、$d_h=64$ 的网络规模。最终原子能量与总残差能量分别由：

$$
E_i = \mathbf h_i^{(L)} W_{\text{out}} + b_{\text{out}},\quad W_{\text{out}}\in\mathbb R^{d_h\times 1}
\tag{2-42}
$$

$$
E_{\text{ML}}(\mathbf{R}) = \sum_{i\in\mathcal A} m_i\,E_i
\tag{2-43}
$$

给出。其中 $\mathrm{Swish}(x)=x\,\sigma(\beta x)=x/(1+e^{-\beta x})$（典型 $\beta=1$）[49]，具有平滑且处处可导的特点，更适合满足力相关任务对导数连续性的要求。残差连接使网络的学习目标由完整非线性映射转化为增量修正，与 MLP 仅学习 ReaxFF 残差的总体思路相一致。所有原子残差能量按式 (2-43) 求和后得到 $E_{\text{ML}}$，因此总模型仍保持清晰的逐原子可加结构；式中 $m_i\in\{0,1\}$ 为下节定义的原子类型门控掩码。

### 原子类型门控 (Atom-Type Gating)

在多组分体系中，并非所有原子类型都需要同等程度的机器学习修正。为避免对已较为可靠的子体系引入不必要扰动，本文引入**原子类型门控机制**：仅允许指定原子集合 $\mathcal A$ 上的能量残差项参与求和，其余原子完全由 ReaxFF 描述。形式上，定义掩码

$$
m_i =
\begin{cases}
1,& \text{atom-type}(i)\in\mathcal A_{\text{learn}}\\
0,& \text{otherwise}
\end{cases},
$$

则残差能量表达式为：

$$
E_{\text{ML}}(\mathbf{R}) = \sum_i m_i \cdot e_{\text{ML}}\!\left(\mathbf{f}_i(\mathbf{R});\boldsymbol\phi\right)
\tag{2-44}
$$

以 Pd-Zn-C-H-O 体系为例，若 Pd-Zn-O 子体系的 ReaxFF 参数已较为可靠，则可令 $\mathcal A_{\text{learn}}=\{\mathrm C,\mathrm H\}$，使 MLP 主要修正 C/H 相关局域环境。门控机制的意义在于：(i) 保持已验证子体系的稳定性；(ii) 将模型自由度集中用于描述新增元素及界面相互作用；(iii) 减少有效参数数目、降低过拟合风险。需注意，尽管某些原子本身不直接贡献 ML 能量项，但其坐标仍可能通过邻域特征 $\mathbf f_i$ 进入其他原子的输入，因此整体力 $\mathbf F_j^{\text{ML}}=-\nabla_{\mathbf R_j}E_{\text{ML}}$ 对包括非学习原子在内的所有原子均可产生一致且物理合理的修正。

### 受保护坐标正则化 (Protected-Coordinate Regularization)

仅依赖原子类型门控时，MLP 仍可能沿某些特定键长或键角方向产生不期望的力修正。为进一步约束模型行为，本文引入**受保护坐标正则化**，对关键内部坐标方向上的 ML 力投影施加惩罚，使残差模型在这些方向上尽量保持与原始 ReaxFF 一致。

**受保护键长 (e.g. Pd-Zn)**：设两原子位置 $\mathbf R_{\text{Pd}},\mathbf R_{\text{Zn}}$，键长 $r_{\text{Pd-Zn}}=\|\mathbf R_{\text{Pd}}-\mathbf R_{\text{Zn}}\|$，单位向量 $\hat{\mathbf r}=(\mathbf R_{\text{Pd}}-\mathbf R_{\text{Zn}})/r_{\text{Pd-Zn}}$。考虑两原子等量反向位移所对应的纯键拉伸方向，定义 ML 力沿键方向的广义投影力：

$$
q_{\text{Pd-Zn}} = \tfrac{1}{2}\Bigl(\mathbf{F}^{ML}_{\text{Pd}} - \mathbf{F}^{ML}_{\text{Zn}}\Bigr)\cdot\hat{\mathbf{r}}
\tag{2-45}
$$

**受保护三体角 (e.g. Pd-O-Zn)**：设角度 $\theta=\angle(\mathbf R_{\text{Pd}},\mathbf R_{\text{O}},\mathbf R_{\text{Zn}})$，对应梯度向量 $\nabla\theta=\bigl(\partial\theta/\partial\mathbf R_j\bigr)_{j\in\{Pd,O,Zn\}}$。ML 力沿角度变化方向的归一化投影定义为：

$$
q_{\text{Pd-O-Zn}}
= \frac{\sum_{j\in\{Pd,O,Zn\}} \mathbf{F}^{ML}_j \cdot \dfrac{\partial\theta}{\partial\mathbf{R}_j}}
       {\sqrt{\sum_{j} \bigl\|\partial\theta/\partial\mathbf{R}_j\bigr\|^{2}}}
\tag{2-46}
$$

分母的归一化使 $q_{\text{Pd-O-Zn}}$ 具有 [力] = $\text{kcal}/(\text{mol}\cdot\text{Å})$ 的良好量纲，便于与 (2-45) 在同一损失尺度下处理。$\partial\theta/\partial\mathbf R_j$ 由角度公式 $\cos\theta=\hat{\mathbf u}\cdot\hat{\mathbf v}$ ($\hat{\mathbf u}=(\mathbf R_{\text{Pd}}-\mathbf R_{\text{O}})/r_{PO},\,\hat{\mathbf v}=(\mathbf R_{\text{Zn}}-\mathbf R_{\text{O}})/r_{ZO}$) 解析给出。

**总损失**：通过在总损失函数中加入正则项可控制 MLP 在受保护坐标方向上的修正幅度：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_E + w_F\,\mathcal{L}_F + \lambda_{\text{prot}}\,\mathcal{L}_{\text{prot}}
\tag{2-47}
$$

$$
\mathcal{L}_{\text{prot}} = \mathbb{E}_{(R,\,\text{Pd-Zn})}\!\bigl[q_{\text{Pd-Zn}}^{2}\bigr]
+ \mathbb{E}_{(R,\,\text{Pd-O-Zn})}\!\bigl[q_{\text{Pd-O-Zn}}^{2}\bigr]
\tag{2-48}
$$

其中 $\mathbb E$ 表示对训练批次内所有满足该几何模式的键/角实例取平均。正则化强度 $\lambda_{\text{prot}}$ 越大，模型在这些方向上的自由修正越受抑制；$\lambda_{\text{prot}}\to\infty$ 对应在受保护坐标方向上完全保留原始 ReaxFF 行为。该策略与原子类型门控相互配合，使残差模型既具备足够的表达能力，又能维持对关键化学坐标的物理约束。

### 损失函数与训练策略

MLP 残差模型的训练采用**能量与力联合优化**策略。对一批训练构型 $\{\mathbf R^{(b)}\}_{b=1}^{B}$，每个构型含 $N_b$ 个原子，其能量误差与力误差损失分别为：

$$
\mathcal{L}_E
= \frac{1}{B}\sum_{b=1}^{B}
\frac{1}{N_b^{\,2}}\bigl(E_{\text{total}}^{(b)} - E_{\text{DFT}}^{(b)}\bigr)^{2},
$$

$$
\mathcal{L}_F
= \frac{1}{B}\sum_{b=1}^{B}\frac{1}{3 N_b}\sum_{j=1}^{N_b}
\bigl\|\mathbf F_j^{\text{total},(b)} - \mathbf F_j^{\text{DFT},(b)}\bigr\|^{2}.
$$

能量误差除以 $N_b^{\,2}$ 等价于"每原子能量 MSE"，可减弱体系规模差异带来的偏置；力误差对所有原子分量取平均，以直接约束动力学相关量。总损失 (2-47) 中典型权重设置为 $w_F\!\sim\!10\!-\!100~\text{Å}^{2}$ (与能量同量纲)、$\lambda_{\text{prot}}\!\sim\!1\!-\!10$。

**训练流程**进一步结合：(i) **批次中心化参考能量**——在批次内对 $E_{\text{DFT}}-E_{\text{ReaxFF}}$ 减去其批均值，使 MLP 仅学习相对偏差，避免吸收基线整体平移；(ii) **聚类批次采样**——按构型类型 (晶相、表面、吸附、过渡态) 在每个批次中均匀采样，防止样本不平衡；(iii) **两阶段训练策略**——先完成 ReaxFF 参数优化得到 $\boldsymbol\theta^{*}$，再冻结 ReaxFF 并训练 MLP 残差项 $\boldsymbol\phi$，从而保留物理模型的先验约束，又能充分发挥数据驱动修正的灵活性[43]。

---

## 修订一览

| 编号 | 原稿问题 | 修订内容 |
| :--- | :--- | :--- |
| (2-25) | 仅给出形式，未明确相对能量基准 | 补充批次内"减去最接近均值构型"的相对能量构造说明 |
| (2-26) | 形式正确 | 维持 |
| (2-27) | 缺 $s_k,y_k,\rho_k$ 定义；未说明 L-BFGS 与 BFGS 的存储差异 | 补全定义、说明二循环递推与 $O(mn)$ 复杂度 |
| (2-28) | $g_t^2$ 表述模糊 | 拆为 (2-28a)–(2-28e) 并明确 $\mathbf g_t\odot\mathbf g_t$ 为逐元素积 |
| (2-30) | 条件式写法不规范 | 改写为 $\mathbf g_{\text{clipped}}=\mathbf g\cdot\min(1,\,\text{clip\_norm}/\|\mathbf g\|_2)$ 统一形式 |
| (2-31)(2-32) | 形式正确 | 维持，补默认数值 ($T_{\text{warm}}=0.05T,\,\alpha_{\max}=10^{-3},\,\alpha_{\min}=10^{-5}$) |
| (2-33) | 仅给出 $\mathrm{clip}$ 符号 | 补出逐分量定义 $\max(\theta_{\min},\min(\theta,\theta_{\max}))$ |
| (2-34)(2-35) | 力的来源未显式写出 | 显式写出 $\mathbf F_j^{\text{ML}}=-\nabla_{\mathbf R_j}E_{\text{ML}}$ |
| 描述符 | 原稿仅文字描述 12/43 维特征 | 增补显式特征向量定义 (基础 12 维与扩展项分块) |
| (2-37) | $Q_i\odot K_i$ 缺尺度因子 | 增加 $1/\sqrt d$ 温度缩放，并明确输出维度为 $\mathbb R^{d}$ |
| (2-38) | 求和指标与外部下标都用 $k$，记号冲突 | 改为 $\alpha_{i,k}=\exp(\text{scores}_{i,k})/\sum_l\exp(\text{scores}_{i,l})$ |
| (2-39) | 形式正确 | 维持，补 $W_O$ 维度说明 |
| (2-40)(2-41) | 形式正确 | 标注"含/无残差"，补 $L=4,d_h=64$ 默认网络规模 |
| (2-42)(2-43) | $E_{\text{ML}}$ 求和未与门控式 (2-44) 一致 | 把求和写成 $\sum_i m_i E_i$ 与 (2-44) 一致 |
| (2-44) | 形式正确 | 增补 $m_i\in\{0,1\}$ 显式定义 |
| (2-45) | 缺 $1/2$ 系数（动量守恒下纯键拉伸） | 补为 $\tfrac{1}{2}(\mathbf F_{Pd}^{ML}-\mathbf F_{Zn}^{ML})\cdot\hat{\mathbf r}$ |
| (2-46) | 投影未归一化，量纲不规整 | 除以 $\sqrt{\sum_j\|\partial\theta/\partial\mathbf R_j\|^{2}}$ 得到力量纲投影 |
| (2-48) | $\mathbb E$ 无明确含义 | 标明对训练批次内符合该几何模式的键/角实例取平均 |
| 损失函数节 | $\mathcal L_E,\mathcal L_F$ 未给出显式表达 | 补出每原子归一化的 MSE 形式 |
