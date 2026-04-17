from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from torch_geometric.loader import DataLoader
except Exception as e:  # pragma: no cover
    raise ImportError("gnn_adsorption requires torch_geometric. Install PyG first.") from e

from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import pennylane as qml

from gnn_adsorption.dataset_pyg import ContcarCSVDataset
from gnn_adsorption.models import ModelConfig, PBCMPNNRegressor


def make_quantum_kernel(n_qubits: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x1: np.ndarray, x2: np.ndarray):
        qml.AngleEmbedding(x1, wires=range(n_qubits), rotation="Y")
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation="Y")
        return qml.expval(qml.Projector([0] * n_qubits, wires=range(n_qubits)))

    def kernel(x1: np.ndarray, x2: np.ndarray) -> float:
        return float(kernel_circuit(x1, x2))

    return kernel


def compute_kernel_matrix(X1: np.ndarray, X2: np.ndarray, kernel_fn) -> np.ndarray:
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2), dtype=float)
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_fn(X1[i], X2[j])
    return K


def fit_angle_mapper(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_train = np.asarray(X_train, dtype=float)
    mn = X_train.min(axis=0)
    mx = X_train.max(axis=0)
    return mn, mx


def to_angle_range_with(mn: np.ndarray, mx: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    denom = np.where((mx - mn) < 1e-12, 1.0, (mx - mn))
    out = (X - mn) / denom * np.pi
    out = np.where((mx - mn) < 1e-12, 0.0, out)
    return out


@dataclass
class Metrics:
    rmse: float
    mae: float
    r2: float


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return Metrics(
        rmse=float(math.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
    )


@torch.no_grad()
def export_embeddings(
    model: PBCMPNNRegressor,
    dataset: ContcarCSVDataset,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    subset = [dataset[int(i)] for i in indices]
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()

    embs: List[np.ndarray] = []
    ys: List[float] = []
    paths: List[str] = []

    for batch in loader:
        batch = batch.to(device)
        g = model.encode(
            z=batch.z,
            edge_index=batch.edge_index,
            edge_dist=batch.edge_dist,
            batch=batch.batch,
        )  # (B, emb_dim)
        embs.append(g.detach().cpu().numpy())
        ys.extend(batch.y.view(-1).detach().cpu().numpy().tolist())

        p = list(batch.path) if isinstance(batch.path, (list, tuple)) else batch.path
        if isinstance(p, str):
            p = [p]
        paths.extend([str(x) for x in p])

    E = np.concatenate(embs, axis=0)
    y = np.asarray(ys, dtype=float)
    return E, y, paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--root_dir", type=str, default=None)
    ap.add_argument("--path_col", type=str, default="path")
    ap.add_argument("--target_col", type=str, default="y")
    ap.add_argument("--group_col", type=str, default=None)
    ap.add_argument("--cutoff", type=float, default=6.0)

    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--n_qubits", type=int, default=4, help="PCA dims == qubits for quantum kernel")
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ContcarCSVDataset(
        csv_path=args.csv_path,
        root_dir=args.root_dir,
        path_col=args.path_col,
        target_col=args.target_col,
        group_col=args.group_col,
        cutoff=args.cutoff,
        cache_structures=False,
    )

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ModelConfig(**ckpt["config"])
    model = PBCMPNNRegressor(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    split = ckpt.get("split", None)
    if split is None:
        # fallback: use all data
        idx_all = list(range(len(ds)))
        split = {"train": idx_all, "val": [], "test": []}

    tr_idx = [int(i) for i in split.get("train", [])]
    te_idx = [int(i) for i in split.get("test", [])]
    if len(te_idx) == 0:
        # If checkpoint has no test split, just do CV on all.
        te_idx = tr_idx

    E_tr, y_tr, paths_tr = export_embeddings(
        model=model, dataset=ds, indices=tr_idx, device=device, batch_size=args.batch_size, num_workers=args.num_workers
    )
    E_te, y_te, paths_te = export_embeddings(
        model=model, dataset=ds, indices=te_idx, device=device, batch_size=args.batch_size, num_workers=args.num_workers
    )

    np.savez(out_dir / "embeddings.npz", E_tr=E_tr, y_tr=y_tr, paths_tr=np.array(paths_tr), E_te=E_te, y_te=y_te, paths_te=np.array(paths_te))

    # Compare heads on embeddings via CV (on train split)
    scaler = StandardScaler()
    E_tr_s = scaler.fit_transform(E_tr)

    # compress to n_qubits dims for quantum kernel (and keep same for classical baselines for fairness)
    d_q = int(args.n_qubits)
    if E_tr_s.shape[1] < d_q:
        raise ValueError(f"Embedding dim={E_tr_s.shape[1]} < n_qubits={d_q}. Reduce n_qubits or increase emb_dim.")
    pca = PCA(n_components=d_q, random_state=args.seed)
    X = pca.fit_transform(E_tr_s)

    # Baselines
    ridge = Ridge(alpha=1e-2)
    rbfkrr = KernelRidge(alpha=1e-2, kernel="rbf", gamma=1.0 / d_q)
    qkrr = KernelRidge(alpha=1e-2, kernel="precomputed")
    qkernel = make_quantum_kernel(n_qubits=d_q)

    kf = KFold(n_splits=int(args.cv_folds), shuffle=True, random_state=args.seed)

    rows: List[Dict[str, object]] = []
    for fold, (tr, te) in enumerate(kf.split(X), start=1):
        X_tr, X_va = X[tr], X[te]
        y1, y2 = y_tr[tr], y_tr[te]

        # Ridge
        ridge.fit(X_tr, y1)
        y_hat_r = ridge.predict(X_va)
        mr = _metrics(y2, y_hat_r)

        # RBF-KRR
        rbfkrr.fit(X_tr, y1)
        y_hat_k = rbfkrr.predict(X_va)
        mk = _metrics(y2, y_hat_k)

        # Quantum KRR
        mn, mx = fit_angle_mapper(X_tr)
        X_tr_q = to_angle_range_with(mn, mx, X_tr)
        X_va_q = to_angle_range_with(mn, mx, X_va)
        K_tr = compute_kernel_matrix(X_tr_q, X_tr_q, qkernel)
        K_va = compute_kernel_matrix(X_va_q, X_tr_q, qkernel)
        qkrr.fit(K_tr, y1)
        y_hat_q = qkrr.predict(K_va)
        mq = _metrics(y2, y_hat_q)

        rows.extend(
            [
                {"fold": fold, "model": "Ridge", "rmse": mr.rmse, "mae": mr.mae, "r2": mr.r2},
                {"fold": fold, "model": "RBF-KRR", "rmse": mk.rmse, "mae": mk.mae, "r2": mk.r2},
                {"fold": fold, "model": "Quantum-KRR", "rmse": mq.rmse, "mae": mq.mae, "r2": mq.r2},
            ]
        )

    # Save CV table
    with open(out_dir / "cv_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "model", "rmse", "mae", "r2"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Simple summary
    def summarize(model_name: str) -> Dict[str, float]:
        arr = np.array([r["mae"] for r in rows if r["model"] == model_name], dtype=float)
        arr2 = np.array([r["rmse"] for r in rows if r["model"] == model_name], dtype=float)
        arr3 = np.array([r["r2"] for r in rows if r["model"] == model_name], dtype=float)
        return {
            "mae_mean": float(arr.mean()),
            "mae_std": float(arr.std()),
            "rmse_mean": float(arr2.mean()),
            "rmse_std": float(arr2.std()),
            "r2_mean": float(arr3.mean()),
            "r2_std": float(arr3.std()),
        }

    summary = {
        "Ridge": summarize("Ridge"),
        "RBF-KRR": summarize("RBF-KRR"),
        "Quantum-KRR": summarize("Quantum-KRR"),
        "n_qubits": d_q,
        "pca_components": d_q,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", out_dir / "embeddings.npz")
    print("Saved:", out_dir / "cv_metrics.csv")
    print("Saved:", out_dir / "summary.json")
    print("CV summary:", summary)


if __name__ == "__main__":
    main()


