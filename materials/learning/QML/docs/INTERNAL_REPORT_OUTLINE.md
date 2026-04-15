# 内部技术报告大纲（量子 × 分子模拟 / 材料 / QML）

用于阶段 2–3 对外/对内对齐；定长 **8–12 页** 时按章节压缩。个人长期目标与文献/复现节奏见 [`DOMAIN_EXPERT_ROADMAP.md`](DOMAIN_EXPERT_ROADMAP.md)。

## 1. 摘要

- 业务问题（例如：吸附能、包覆速率、力场残差）
- 方法一句话（经典基线 + 量子模块 / 纯经典对照）
- 主要数值结论（MAE/RMSE，**含单位**）与数据规模

## 2. 问题与数据

### 2.1 任务定义

- 输入（结构文件 / 轨迹 / 手工特征）
- 输出（回归 / 分类标量）
- 标签定义（如 \(E_\mathrm{ads}\) 公式、参考计算方法）

### 2.2 数据集

- 样本量、特征维度、时间窗长度（若适用）
- **划分策略**：随机 / **group split**（材料、表面、吸附物分组）
- 数据泄漏检查清单（同一体系不能同时出现在 train 与 test）

## 3. 基线方法

- Ridge / RBF-KRR / GBDT / GNN（按任务选）
- 超参数与 **交叉验证** 设置（折数、seed）
- 预期：**量子方法不应弱于强经典基线太多**（否则先检查特征与泄漏）

## 4. 量子方法

- 电子结构 / VQE / 映射等**理论书面课**（可选附录引用）：[`qc_learn/README.md`](qc_learn/README.md)
- 量子核 vs VQC（见 [`QML_KERNEL_VS_VQC.md`](QML_KERNEL_VS_VQC.md)）
- **分子力场 / 等变 PES**：若使用自研 **QML-FF**（`/Users/shl/nvidia/QML-FF`），见 [`QMLFF_INTEGRATION.md`](QMLFF_INTEGRATION.md)：对称性约定、Wigner/Schur/QMP、JAX 力与 `config` 版本冻结
- 电路：qubit 数、层数、编码、测量
- 模拟器 / 硬件：名称与版本

## 5. 结果

- 主表：各模型 **MAE / RMSE / R²**（或分类指标）
- **Parity plot**（true vs pred）
- 外推折（若使用 group split）：单独报告

## 6. 失败案例与局限

- 过拟合、方差大、某折崩溃
- 噪声、样本量不足、几何离群
- **不夸大量子优势**：写明 NISQ 限制

## 7. 复现性

- Git commit、随机种子、`requirements.txt` 或 conda env 导出
- 关键命令（见 [`PHASE2_PIPELINES.md`](PHASE2_PIPELINES.md)）

## 8. 附录

- 符号表（与 [`EXTERNAL_ANCHORS.md`](EXTERNAL_ANCHORS.md) 一致）
- 补充图表
