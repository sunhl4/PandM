# GNN Adsorption Energy Regression (CONTCAR + CSV)

This folder provides a **minimal, practical** pipeline for:

- Loading **DFT structure files** (VASP `CONTCAR` / `POSCAR`) via ASE
- Building a **neighbor graph** with **PBC-aware** edges (cutoff radius)
- Training a small **message-passing GNN** regressor on adsorption energy
- (Optional) Exporting GNN embeddings and running **Quantum Kernel Ridge** as a head for ablations

## 1) CSV schema (required)

Your CSV must contain at least:

- **`path`**: path to `CONTCAR` (absolute or relative to `--root_dir`)
- **`y`**: adsorption energy label (float)

Optional columns (recommended):

- **`group`**: group id for leakage-safe splitting (e.g., surface id / adsorbate id)

Example:

```text
path,y,group
data/0001/CONTCAR,-1.234,Pt111_CO
data/0002/CONTCAR,-0.876,Pt111_CO
data/0101/CONTCAR,-1.455,Ni111_O
```

## 2) Install notes (deps)

This code uses:

- `ase` (read CONTCAR + PBC neighbor list)
- `torch`
- `torch_geometric` (PyG)

`torch_geometric` installation depends on your OS / PyTorch version. Follow the official PyG install guide.

## 3) Train a GNN regressor

```bash
python -m gnn_adsorption.train_gnn_regressor \
  --csv_path /path/to/adsorption.csv \
  --root_dir /path/to/project \
  --path_col path \
  --target_col y \
  --cutoff 6.0 \
  --epochs 200 \
  --batch_size 16 \
  --lr 1e-3 \
  --seed 42 \
  --out_dir outputs/gnn_run_01
```

Outputs:

- `outputs/gnn_run_01/best.pt` (model checkpoint)
- `outputs/gnn_run_01/preds.csv` (true vs pred for train/val/test)
- `outputs/gnn_run_01/config.json`

## 4) Optional: embeddings → (Quantum) Kernel Ridge head

This script:

1) Loads a trained GNN checkpoint
2) Exports per-structure embeddings
3) Compares Ridge / RBF-KRR / Quantum-KRR on the embeddings

```bash
python -m gnn_adsorption.embedding_quantum_krr \
  --csv_path /path/to/adsorption.csv \
  --root_dir /path/to/project \
  --ckpt outputs/gnn_run_01/best.pt \
  --out_dir outputs/gnn_run_01/embedding_krr \
  --n_qubits 4
```

## 5) Common pitfalls

- **Unit consistency**: make sure all `y` are in the same unit (eV).
- **Data leakage**: random splitting can be overly optimistic. Prefer `group` split when possible.
- **Cutoff**: too small misses interactions; too large increases compute and noise.
- **Vacuum slabs**: for slab models, PBC often only applies in x/y; ASE will handle `atoms.pbc`.


