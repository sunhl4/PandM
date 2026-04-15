# 量子核（QKR）与变分量子电路（VQC）：对比表（工程绑定）

本表与 **[`docs/topics/chemistry_qml_application_notes.md`](topics/chemistry_qml_application_notes.md)**（Part A 吸附能基线）的叙事一致，并指向进一步阅读。

## 何时考虑哪一种

| 维度 | 量子核 / 量子特征映射 + 经典回归（如 KRR） | VQC / QNN 端到端训练 |
|------|---------------------------------------------|------------------------|
| **训练稳定性** | 通常更稳（量子部分常固定，训练在核权重上） | 更易遇贫瘠高原、对初始化/深度敏感 |
| **数据规模** | 核矩阵 \(O(N^2)\) 随样本数放大 | 单次推断可能更快，但训练需多次电路 |
| **归纳偏置** | 由特征映射与核决定 | 由 ansatz、编码、测量共同决定 |
| **与经典 ML 对比** | 常对标 RBF 核、GPR（见吸附笔记 §2–3） | 常对标 MLP / GNN 头 |
| **典型用途** | 小样本回归、需可解释核矩阵时 | 需端到端学习量子参数时 |

## 与仓库文档的对应关系

| 主题 | 仓库位置 |
|------|----------|
| 贫瘠高原与可训练性 | [`qml_training_landscape_compendium.md`](notes/qml_training_landscape_compendium.md) Part 1 |
| QNTK / 训练动力学（理论） | 同上 Part 2；力场综述 [`molecular_ff_survey_enhanced.md`](topics/molecular_ff_survey_enhanced.md) §11–12 |
| 吸附能任务基线与 ⟨Z⟩/⟨X⟩/⟨Y⟩ 经验 | [`chemistry_qml_application_notes.md`](topics/chemistry_qml_application_notes.md) Part A |
| 代码演示 | [`week3_quantum_kernel_ridge_demo.py`](../week3_quantum_kernel_ridge_demo.py)、[`week4_vqc_regression_demo.py`](../week4_vqc_regression_demo.py) |

## 实践清单（阶段 1 出口）

1. 跑通 `week3` 与 `week4`（见下方命令）。
2. 在 **同一合成数据与同一随机种子** 下记录 **MAE/RMSE 与墙钟时间**。
3. 对业务方一句话：**“先强经典基线（Ridge/RBF-KRR），量子模块作为特征/核的消融”**。

```bash
cd /path/to/QML
# 全量演示（较慢，量子核矩阵 O(N²)）
python week3_quantum_kernel_ridge_demo.py
python week4_vqc_regression_demo.py

# 快速验证（48 样本、3 折 CV；与 CI/阶段检查兼容）
QML_SMOKE=1 python week3_quantum_kernel_ridge_demo.py
QML_SMOKE=1 python week4_vqc_regression_demo.py
```

**阶段 1 阅读（仓库内）**：通读 [`qml_training_landscape_compendium.md`](notes/qml_training_landscape_compendium.md) Part 1；力场综述 [`molecular_ff_survey_enhanced.md`](topics/molecular_ff_survey_enhanced.md) 第 10–12 节（QELM/QRC/QNTK）。引用原文前请用 Zotero/arXiv 核对作者与年份。

---

## 备忘：量子核与投影（原 `Future-notes.md`）

### 自适应投影（可能方向）

1. 学习最优投影态：通过训练选择信息量最大的投影。  
2. 动态投影：按数据自适应调整。  
3. 多投影融合：同时测量多个投影。

### 当前实践

全零态投影在简单性与性能之间仍是常用折中。

*（摘录合并自根目录 `Future-notes.md`，2026-03-29）*
