# QML 工程文件清单（全量索引）

> 生成目的：单页总览仓库内**每一个文件**的路径与用途；2026-03-30 合并文献与专题稿后更新。  
> 若增删文件，请同步更新本页与 [README.md](README.md)。

## 1. 学习计划与工程文档（`docs/`）

| 路径 | 说明 |
|------|------|
| [theory_and_derivations.md](theory_and_derivations.md) | **唯一理论+推导主稿**（`tools/build_theory_master.py` 生成） |
| [domain_qa.md](domain_qa.md) | **领域问答与备忘**（长期手写维护） |
| [DOMAIN_EXPERT_ROADMAP.md](DOMAIN_EXPERT_ROADMAP.md) | **领域专家路线总纲**：主线/副线、维护循环、复现与创新工作流 |
| [README.md](README.md) | 文档索引（压缩：两份核心 + 源表） |
| [MERGE_COVERAGE_AUDIT.md](MERGE_COVERAGE_AUDIT.md) | **合并与重叠审计**（哪些已合并、哪些刻意保留多份） |
| [PROJECT_FILE_MANIFEST.md](PROJECT_FILE_MANIFEST.md) | 本清单 |
| [literature_notes/README.md](literature_notes/README.md) | 精读笔记目录说明（配合周文献节奏） |
| [PHASE0_H2_BASELINE.md](PHASE0_H2_BASELINE.md) | H₂ VQE 基线、作图、参考能量 |
| [PHASE2_PIPELINES.md](PHASE2_PIPELINES.md) | zno_pd_qml / gnn_adsorption 烟测 |
| [QML_KERNEL_VS_VQC.md](QML_KERNEL_VS_VQC.md) | 量子核 vs VQC；含量子核投影相关备忘 |
| [QMLFF_INTEGRATION.md](QMLFF_INTEGRATION.md) | 与 `QML-FF` 仓库对齐说明 |
| [INTERNAL_REPORT_OUTLINE.md](INTERNAL_REPORT_OUTLINE.md) | 内部技术报告大纲 |
| [WEEKLY_LITERATURE_WORKFLOW.md](WEEKLY_LITERATURE_WORKFLOW.md) | 每周文献节奏 |
| [EXTERNAL_ANCHORS.md](EXTERNAL_ANCHORS.md) | 教材与栈文档锚点 |
| [SHAREABLE_TECH_NOTE_TEMPLATE.md](SHAREABLE_TECH_NOTE_TEMPLATE.md) | 对外短讯模板 |
| [figures/h2_pes.png](figures/h2_pes.png) | H₂ 势能面图（**canonical**） |
| [figures/h2_vqe_convergence.png](figures/h2_vqe_convergence.png) | H₂ VQE 收敛图（**canonical**） |
| [qml_tutorial_zh_consolidated.md](qml_tutorial_zh_consolidated.md) | QML Tutorial（qml-tutorial.github.io）中文版综合长文 |
| [QML_GUIDE_INDEX.md](QML_GUIDE_INDEX.md) | `docs/guides/`「量子机器学习完整指南」多版本说明 |
| [group_meeting_reaxff_qml.md](group_meeting_reaxff_qml.md) | 组会：ReaxFF–DFT 残差 + QML（合并稿） |
| [literature/README.md](literature/README.md) | 文献子目录总览（综述、QAOA、化材精读） |
| [notes/qml_training_landscape_compendium.md](notes/qml_training_landscape_compendium.md) | QML 理论五合一合并稿 |
| [topics/chemistry_qml_application_notes.md](topics/chemistry_qml_application_notes.md) | 化学应用笔记三合一合并稿 |

### 1.05 `docs/unified_chemistry_theory/`（理论统一枢纽，2026-03-30）

| 路径 | 说明 |
|------|------|
| [unified_chemistry_theory/README.md](unified_chemistry_theory/README.md) | 总索引：`quantum_chemistry/docs` 与 `qc_learn` 对齐的阅读路线 |
| [unified_chemistry_theory/vol_package_01_05_theory.md](unified_chemistry_theory/vol_package_01_05_theory.md) | 包理论 01–05 合订本（脚本生成） |
| [unified_chemistry_theory/vol_qc_learn_weeks_3_4_and_project.md](unified_chemistry_theory/vol_qc_learn_weeks_3_4_and_project.md) | QC-learn 第 3–4 周 + 项目思路合订本（脚本生成） |

### 1.1 `docs/qc_learn/`（原 QC-learn，2026-03-29 并入）

| 路径 | 说明 |
|------|------|
| [qc_learn/README.md](qc_learn/README.md) | 4 周量子化学 + ML 书面课总览与进度 |
| [qc_learn/quantum_chemistry_foundations.md](qc_learn/quantum_chemistry_foundations.md) | 第 1–2 周合并稿：TISE、HF、后 HF、DFT、基组、误差 |
| [qc_learn/week3_classical_ml.md](qc_learn/week3_classical_ml.md) | 第 3 周：NNQS、VMC+ML 等 |
| [qc_learn/week4_quantum_ml.md](qc_learn/week4_quantum_ml.md) | 第 4 周：映射、VQE、QPE、ADAPT-VQE、容错展望 |
| [qc_learn/week4_quantum_ml.pdf](qc_learn/week4_quantum_ml.pdf) | 第 4 周 Markdown 的 PDF 导出 |
| [qc_learn/final_project_ideas.md](qc_learn/final_project_ideas.md) | 项目与创新方向框架 |
| [literature/references/README.md](literature/references/README.md) | 教材与主题论文列表（原 `qc_learn/references/`） |
| [literature/references/textbooks.md](literature/references/textbooks.md) | 教材书目 |
| [literature/references/ml_quantum_chemistry.md](literature/references/ml_quantum_chemistry.md) | ML×量子化学论文列表 |
| [literature/references/quantum_ml.md](literature/references/quantum_ml.md) | 量子 ML / VQE 论文列表 |
| [qc_learn/references/README.md](qc_learn/references/README.md) | 占位：指向 `literature/references/` |

## 2. 仓库根目录：入口与周次脚本

| 路径 | 说明 |
|------|------|
| `README.md` | 仓库入口，指向 `docs/` 与清单 |
| `requirements.txt` | 主环境依赖 |
| `requirements_gnn.txt` | GNN / Phase2 额外依赖 |
| `week1_comprehensive_demo.py` | Week 1 综合演示 |
| `week2_encoding_compare.py` | Week 2 编码对比 |
| `week2_quantum_neural_network.py` | Week 2 QNN |
| `week3_quantum_kernel_ridge_demo.py` | Week 3 量子核岭回归 |
| `week4_vqc_regression_demo.py` | Week 4 VQC 回归 |
| `week5_advanced_features.py` | Week 5 进阶特性 |
| `week3_parity_quantum_krr.png` / `week3_parity_rbf_krr.png` | Week3 输出图 |

## 3. 合并后的专题文档（`docs/` 子目录）

原根目录多篇独立 `.md` 已合并或迁入下表；**请勿在根目录恢复同名旧文件**。

| 路径 | 说明 |
|------|------|
| [literature/README.md](literature/README.md) | 文献子目录索引 |
| [literature/qml_literature_complete_2007_2025.md](literature/qml_literature_complete_2007_2025.md) | QML 文献综述 2007–2025（含逐年条目；**已替代**根目录按年拆分文件） |
| [literature/qaoa_literature_and_notes.md](literature/qaoa_literature_and_notes.md) | QAOA 主题（原 `QAOA.md`） |
| [literature/chemistry_materials_reading.md](literature/chemistry_materials_reading.md) | 化材精读模板 + 2025 优先清单（原模板与 `.txt` 合并） |
| [notes/qml_training_landscape_compendium.md](notes/qml_training_landscape_compendium.md) | 贫瘠高原、QNTK、非线性、旋转/纠缠、QRC **五合一** |
| [topics/chemistry_qml_application_notes.md](topics/chemistry_qml_application_notes.md) | 吸附、力场论文列表、单篇示例 **三合一** |
| [topics/molecular_ff_survey_base.md](topics/molecular_ff_survey_base.md) | 分子力场 QML 综述（基础全文） |
| [topics/molecular_ff_survey_enhanced.md](topics/molecular_ff_survey_enhanced.md) | 同上增强版（第 10 章起增量） |
| [guides/quantum_ml_complete_guide.md](guides/quantum_ml_complete_guide.md) | 量子机器学习完整指南（主修订版） |
| [guides/quantum_ml_complete_guide_3900lines_version.md](guides/quantum_ml_complete_guide_3900lines_version.md) | 完整指南加长/历史稿 |

**再生合并稿**（若需从备份还原后重跑）：`tools/build_qml_theory_compendium.py`、`tools/build_chemistry_application_notes.py`（依赖已删除的根目录源文件时需先恢复源）。

## 4. 根目录：其他资产

| 路径 | 说明 |
|------|------|
| `QML_Group_Meeting_PPT.pptx` | 组会 PPT 二进制 |
| `fig/week0_*.png` | Week0 特征/目标可视化图（5 张） |

## 5. `quantum_chemistry/`（量子化学子工程）

| 路径 | 说明 |
|------|------|
| `README.md` | 子项目说明 |
| `requirements.txt` | 子环境依赖 |
| `__init__.py` | 包初始化 |
| `core/*.py` | 分子积分、费米子算符、qubit 映射 |
| `ansatz/*.py` | UCCSD、HEA、自适应 ansatz |
| `vqe/*.py` | VQE 求解器、优化器、测量 |
| `utils/*.py` | 分析、可视化 |
| `docs/*.md` | 二次量子化、映射、VQE、ansatz、激发态理论 |
| `tutorials/*.py`, `01_complete_h2_vqe_explanation.md` | H₂ VQE 与多框架教程 |
| `tutorials/plot_h2_baseline_figures.py` | 写入 `docs/figures/` 的基线图 |
| `.vscode/settings.json` | 编辑器设置 |

## 6. `gnn_adsorption/` 与 `zno_pd_qml/`

| 路径 | 说明 |
|------|------|
| `gnn_adsorption/README.md` | 吸附 GNN 说明 |
| `gnn_adsorption/*.py` | 数据集、模型、量子 KRR 嵌入、训练脚本 |
| `zno_pd_qml/README.md` | ZnO/Pd QML 管线说明 |
| `zno_pd_qml/*.py` | 几何、特征、标签、LAMMPS dump、数据集、训练 |

## 7. `benchmarks/`、`fixtures/`、`tools/`

| 路径 | 说明 |
|------|------|
| `benchmarks/reproducible_h2_vqe_benchmark.py` | 可复现 H₂ VQE benchmark |
| `benchmarks/h2_vqe_benchmark_result.json` | benchmark 结果 JSON |
| `fixtures/phase2_manifest.json` | Phase2 fixture 清单 |
| `fixtures/gnn_smoke/*` | GNN 烟测 CSV + CONTCAR |
| `fixtures/zno_smoke/dataset.npz` | ZnO 烟测数据 |
| `tools/arxiv_qchem_digest.py` | arXiv 抓取 |
| `tools/generate_phase2_smoke_fixtures.py` | 生成 Phase2 fixtures |
| `tools/validate_gnn_smoke_fixtures.py` | 校验 GNN fixtures |
| `tools/zotero_resolve_arxiv_search_links.py` | 解析 arXiv 搜索链接 → Zotero 导入用文本 |
| `tools/build_qml_theory_compendium.py` | 从根目录源文件生成 `notes/qml_training_landscape_compendium.md`（源已删则仅作存档） |
| `tools/build_chemistry_application_notes.py` | 从根目录源文件生成 `topics/chemistry_qml_application_notes.md`（源已删则仅作存档） |
| `tools/build_unified_package_theory.py` | 生成 `unified_chemistry_theory/vol_package_01_05_theory.md` |
| `tools/build_unified_qc_learn_weeks.py` | 生成 `unified_chemistry_theory/vol_qc_learn_weeks_3_4_and_project.md` |
| `tools/build_theory_master.py` | 生成 **`theory_and_derivations.md`**（唯一理论主稿） |

## 8. Zotero 解析输出（**唯一保留：`zotero_import_full_v2/`**）

| 路径 | 说明 |
|------|------|
| `zotero_import_full_v2/abs_urls_all.txt` | 摘要页 URL |
| `zotero_import_full_v2/identifiers_all.txt` | arXiv ID 列表 |
| `zotero_import_full_v2/cache.json` | 解析缓存 |
| `zotero_import_full_v2/report.csv` / `unresolved.csv` | 报告与未解析项 |
| `zotero_import_full_v2/per_file/*.txt` | 按源 Markdown 分列的 ID |

已删除的重复目录（内容已被 v2 或试验目录替代）：`zotero_import/`、`zotero_import_full/`、`zotero_import_test2/`、`zotero_import_test3/`、`zotero_import_test4/`。

## 9. IDE 与缓存（建议不纳入版本控制）

| 路径 | 说明 |
|------|------|
| `.vscode/*` | VS Code / Cursor 配置 |
| `.idea/*` | JetBrains 工程文件 |
| `.DS_Store` | macOS 文件夹元数据 |
| `__pycache__/*.pyc` | Python 字节码（各包下）；根目录孤立的 `.pyc` 已删 |

## 10. 冗余已清理项（备忘）

- 根目录 `h2_pes.png`、`h2_vqe_convergence.png` 与 `docs/figures/` 下文件**内容相同**，已删根目录副本，请以 `docs/figures/` 为准。
- `quantum_ml_complete_guide copy.md`：较旧分支版本，已由 `quantum_ml_complete_guide.md` 覆盖，已删除。
- `Future-notes.md`：已并入 `docs/QML_KERNEL_VS_VQC.md` 文末「备忘：量子核与投影」。
- `QML_Group_Meeting_PPT.md` 与 `group_meeting_ppt_qml_reaxff_residual.md`：已合并为 `docs/group_meeting_reaxff_qml.md`。
- `A.md`：已迁至 `docs/qml_tutorial_zh_consolidated.md`。
- 重复 Zotero 输出目录：仅保留 `zotero_import_full_v2/`；`zotero_import`、`zotero_import_full`、`zotero_import_test2`–`test4` 已删除。
- **2026-03-30 文献与笔记合并**：根目录按年 QML 文献综述、`QAOA.md`、多篇 QML 理论笔记、吸附/力场/单篇示例、化材模板与 txt、力场综述两篇，已迁入 `docs/literature/`、`docs/notes/`、`docs/topics/`、`docs/guides/`（见 §3）。
- **2026-03-30 量子化学理论枢纽**：新增 `docs/unified_chemistry_theory/`（总索引 + `vol_package_01_05`、`vol_qc_learn_weeks_3_4` 合订本；`tools/build_unified_*.py` 可重生成）。
- **2026-03-30 参考文献去重**：`docs/qc_learn/references/*.md`（教材与三列表）迁入 `docs/literature/references/`；`qc_learn/references/README.md` 仅作占位跳转。

## 11. 文件计数（约）

最近一次整理后：`find … -type f` 约为 **158**（含 `.idea` / `__pycache__` / `.DS_Store`）；以本机实时 `find` 为准。
