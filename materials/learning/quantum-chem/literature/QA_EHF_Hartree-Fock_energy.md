# 专题问答：$E_{\mathrm{HF}}$（Hartree–Fock 能量）是怎么求的？

**类型**：电子结构基础  
**总索引**：[学习问答记录.md](./学习问答记录.md) **条目 31**

## 使用说明

- 公式：行内 `$...$`；块公式 `$$` 独占行，**块末 `$$` 后空一行**。

---

## 问答日志

### 条目 1

**标签：** `#Hartree-Fock` `#HF` `#SCF` `#PySCF`

**日期：** 2026-04-08

**问题：**

**$E_{\mathrm{HF}}$**（也常写作 **EHF**、**HF energy**）是怎么求出来的？

**回答：**

---

### （1）符号指什么

在量子化学里，**$E_{\mathrm{HF}}$** 一般指 **Hartree–Fock（HF）近似下的体系总能量**（多数程序里 **已含核–核排斥** $E_{\mathrm{nn}}$，即与实验对比时常说的「Born–Oppenheimer 总能量」的 HF 级别；个别文献把「纯电子能」单独写出，读文时注意定义）。

它对应 **单个 Slater 行列式**（闭壳层常取 **限制 HF，RHF**）波函数 $\lvert\Phi_{\mathrm{HF}}\rangle$ 下的 **期望值**：

$$
E_{\mathrm{HF}} \;=\; \frac{\langle\Phi_{\mathrm{HF}}\vert \hat{H}\vert\Phi_{\mathrm{HF}}\rangle}{\langle\Phi_{\mathrm{HF}}\vert\Phi_{\mathrm{HF}}\rangle},
$$

其中 $\hat{H}$ 是 **非相对论、Born–Oppenheimer** 下的电子哈密顿量（动能 + 核吸引 + 电子排斥）**加上** $E_{\mathrm{nn}}$。HF 用 **平均场** 把多电子问题变成 **有效单粒子** 问题。

---

### （1.5）波函数图像：「正常」多行列式 vs HF 只取一个行列式

在 **固定的单粒子基**（有限个自旋轨道）下，**同一电子数与对称性** 下所有物理上允许的反对称波函数，都可以写成 **Slater 行列式的线性组合**：

$$
\lvert\Psi\rangle = \sum_I c_I \lvert D_I\rangle.
$$

**全组态相互作用（FCI）** 在数学上就是（在该基与对称性约束下）**取遍** 所有行列式 $\lvert D_I\rangle$ 并求 $\{c_I\}$，使能量最低；此时 $\lvert\Psi\rangle$ 是该有限基下的 **精确多电子解**（仍非「物理上绝对精确」，因基组有限）。

**Hartree–Fock** 则 **不** 对全部 $\{c_I\}$ 优化，而是把试探函数 **限制为恰好一个** Slater 行列式 $\lvert\Phi_{\mathrm{HF}}\rangle$，并 **只** 优化其中的 **分子轨道**（在单粒子基里怎么转），使得在该 **单行列式族** 内能量最低。因此：

- 你的理解 **对**：**精确（FCI）波函数** 一般是 **多行列式叠加**；**HF** 是 **单一行列式** 近似。  
- **特例**：若基态在某种基下 **几乎只有一个行列式占主导**（单参考好），HF 常是良好起点；**强关联 / 多参考** 时许多 $\lvert D_I\rangle$ 权重相当，**单靠 HF 一个行列式** 形状会 **明显偏离** 真实波函数，$E_{\mathrm{HF}}$ 与 FCI 的差即 **相关能** 的主要部分。

条目 31 正文侧重 **$E_{\mathrm{HF}}$ 怎么算**；波函数上 **「单行列式」** 正是 HF 与 CI/CC 的 **根本区别**。

---

### （2）能量公式（与积分、密度矩阵）

在 **原子轨道（AO）基** 下，引入 **密度矩阵** $\mathbf{P}$ 与 **Fock 矩阵** $\mathbf{F}$，闭壳层 $n$ 电子（$n/2$ 个占据空间轨道）时常写成：

$$
E_{\mathrm{HF}} \;=\; \frac{1}{2}\,\mathrm{Tr}\bigl[\mathbf{P}(\mathbf{h}+\mathbf{F})\bigr] \;+\; E_{\mathrm{nn}},
$$

其中 $\mathbf{h}$ 为 **单电子**（动能 + 核吸引）矩阵；$\mathbf{F}=\mathbf{h}+\mathbf{G}(\mathbf{P})$，$\mathbf{G}$ 汇集 **Coulomb – Exchange**（由双电子积分与 $\mathbf{P}$ 构造）。  
（开壳层 ROHF/UHF 形式类似，但 $\mathbf{P}$、$\mathbf{F}$ 分 $\alpha/\beta$ 或需用 ROHF 特有公式。）

**要点**：$E_{\mathrm{HF}}$ **不是** 单纯「轨道能量之和」；用轨道能 $\varepsilon_i$ 表示时需加 **双电子修正项**（教材中的「轨道能量之和减重复计数」形式）。

---

### （3）数值上怎么「求」：自洽场（SCF）迭代

HF 方程 $\mathbf{F}\mathbf{C}=\mathbf{S}\mathbf{C}\boldsymbol{\varepsilon}$ 里，$\mathbf{F}$ **依赖于** 系数矩阵 $\mathbf{C}$（通过 $\mathbf{P}$），故必须 **迭代**：

1. **初猜** $\mathbf{P}$（如核哈密顿量本征向量、叠加原子密度等）。  
2. 用当前 $\mathbf{P}$ 造 **$\mathbf{G}$** 与 **$\mathbf{F}$**。  
3. **广义本征问题** 解出 $\mathbf{C}$、占据轨与 $\boldsymbol{\varepsilon}$。  
4. 由 $\mathbf{C}$ 更新 $\mathbf{P}$；若 $\mathbf{P}$（或 $E_{\mathrm{HF}}$、$\mathbf{F}$）变化小于阈值则 **收敛**，否则回到步骤 2。

收敛后，用上面的 **能量泛函** 计算 **$E_{\mathrm{HF}}$**（与 $\langle\Phi\vert\hat{H}\vert\Phi\rangle$ 一致）。

---

### （4）与变分原理的关系

HF 在 **单行列式族** 内 **变分最优**：$E_{\mathrm{HF}} \ge E_0$（真非相对论基态能量，同基组内也 $\ge$ 该基组下 FCI）。  
**相关能** 大致指 $E_{\mathrm{exact}} - E_{\mathrm{HF}}$，由后 HF 方法（MP2、CC、CI 等）拾回。

---

### （5）在本仓库工程里对应什么

`dft_qc_pipeline` 等路径里的 **`e_hf` / `energy_hf`**：对 PySCF 即 **`scf.RHF`（或 ROHF/UHF）`mf.kernel()` 收敛后给出的总能量**（含 $E_{\mathrm{nn}}$），与上文的 $E_{\mathrm{HF}}$ 一致。

---

*条目同步：[学习问答记录.md](./学习问答记录.md) **条目 31**。*
