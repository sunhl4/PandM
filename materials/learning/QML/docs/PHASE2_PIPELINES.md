# Phase 2：应用管线命令（烟测 fixtures）

**等变量子力场（主工程）**：完整 DFT/JAX 力场训练在 **`/Users/shl/nvidia/QML-FF`**（包 `qmlff`），与阶段映射见 [`QMLFF_INTEGRATION.md`](QMLFF_INTEGRATION.md)。下列为本仓库内的轻量烟测，用于流程与报告骨架。

## 生成 fixtures

```bash
cd /path/to/QML
python tools/generate_phase2_smoke_fixtures.py
python tools/validate_gnn_smoke_fixtures.py   # ASE 校验 CONTCAR，无需 PyG
```

产物：`fixtures/gnn_smoke/adsorption.csv`、`fixtures/zno_smoke/dataset.npz`（见 `fixtures/phase2_manifest.json`）。

## A. `zno_pd_qml`（轨迹特征 → Ridge / RBF-KRR / Quantum-KRR）

已在烟测数据上可跑通（12 样本，8 维特征，`y_cover_rate`）：

```bash
python -m zno_pd_qml.train_qml \
  --dataset_npz fixtures/zno_smoke/dataset.npz \
  --task regression \
  --target y_cover_rate \
  --group_key o_model \
  --split_mode groupkfold \
  --cv_folds 3 \
  --n_qubits 4 \
  --pca_components 4 \
  --out_dir outputs/zno_smoke_run \
  --save_parity \
  --n_workers 1
```

真实 LAMMPS 轨迹请仍按 [`zno_pd_qml/README.md`](../zno_pd_qml/README.md) 生成 `dataset.npz`。

## B. `gnn_adsorption`（CONTCAR + CSV → GNN）

依赖 **PyTorch Geometric**（与 PyTorch 版本绑定，需按 [官方说明](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) 安装）。

```bash
python -m gnn_adsorption.train_gnn_regressor \
  --csv_path fixtures/gnn_smoke/adsorption.csv \
  --root_dir . \
  --path_col path \
  --target_col y \
  --group_col group \
  --epochs 30 \
  --batch_size 2 \
  --out_dir outputs/gnn_smoke_run
```

训练完成后可做 embedding + 量子核头（见 [`gnn_adsorption/README.md`](../gnn_adsorption/README.md) 第 4 节）。

## C. 与内部报告的关系

将上述命令、指标与图表填入 [`INTERNAL_REPORT_OUTLINE.md`](INTERNAL_REPORT_OUTLINE.md)。
