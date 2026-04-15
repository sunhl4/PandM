# 量子计算化学与分子模拟 · 长期工程（QML 为辅）

本仓库面向 **长期维护**：以 **量子计算在计算化学与分子模拟** 为主线，以 **量子计算与机器学习的结合** 为副线，支撑文献阅读、系统学习、**复现文献算法**与**实现新想法**。

- **唯一理论 + 推导主稿**：[docs/theory_and_derivations.md](docs/theory_and_derivations.md)（`python3 tools/build_theory_master.py` 更新）  
- **领域问答与备忘**：[docs/domain_qa.md](docs/domain_qa.md)  
- **总纲与维护节奏**：[docs/DOMAIN_EXPERT_ROADMAP.md](docs/DOMAIN_EXPERT_ROADMAP.md)  
- **文档索引（压缩）**：[docs/README.md](docs/README.md) · **全路径清单**：[docs/PROJECT_FILE_MANIFEST.md](docs/PROJECT_FILE_MANIFEST.md)  

**代码入口**：`quantum_chemistry/`（电子结构/VQE/映射）、根目录 `week*.py`（QML 演示）、`gnn_adsorption/` / `zno_pd_qml/`（分子任务管线）、`benchmarks/`（可复现小基准）。

**文献与合并专题**：[`docs/literature/README.md`](docs/literature/README.md)（综述 2007–2025、QAOA、化材精读模板）；理论/应用合并稿见 [`docs/README.md`](docs/README.md) 节 C′。

**量子化学理论统一枢纽**（`quantum_chemistry/docs` 与 `docs/qc_learn` 对齐 + 合订本）：[`docs/unified_chemistry_theory/README.md`](docs/unified_chemistry_theory/README.md)。

**文档合并覆盖说明**（是否还有相近未合并）：[`docs/MERGE_COVERAGE_AUDIT.md`](docs/MERGE_COVERAGE_AUDIT.md)。

**环境**：`requirements.txt`（主）、`requirements_gnn.txt`（GNN）、`quantum_chemistry/requirements.txt`（量子化学）。
