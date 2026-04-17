from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from torch_geometric.loader import DataLoader
except Exception as e:  # pragma: no cover
    raise ImportError("gnn_adsorption requires torch_geometric. Install PyG first.") from e

from sklearn.metrics import mean_absolute_error, mean_squared_error

from gnn_adsorption.dataset_pyg import ContcarCSVDataset
from gnn_adsorption.models import ModelConfig, PBCMPNNRegressor


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices_random(n: int, seed: int, train_frac: float, val_frac: float) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    tr = idx[:n_train]
    va = idx[n_train : n_train + n_val]
    te = idx[n_train + n_val :]
    return {"train": tr, "val": va, "test": te}


def split_indices_group(
    groups: Sequence[Optional[str]],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> Dict[str, np.ndarray]:
    """
    Group-wise split to reduce leakage: samples with same group id stay in one split.
    Any None groups are treated as unique groups.
    """
    n = len(groups)
    g_ids: List[str] = []
    for i, g in enumerate(groups):
        g_ids.append(str(g) if g is not None else f"__none__{i}")

    uniq = np.array(sorted(set(g_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n_train = int(round(len(uniq) * train_frac))
    n_val = int(round(len(uniq) * val_frac))
    g_train = set(uniq[:n_train])
    g_val = set(uniq[n_train : n_train + n_val])
    g_test = set(uniq[n_train + n_val :])

    tr, va, te = [], [], []
    for i, g in enumerate(g_ids):
        if g in g_train:
            tr.append(i)
        elif g in g_val:
            va.append(i)
        else:
            te.append(i)

    return {"train": np.asarray(tr), "val": np.asarray(va), "test": np.asarray(te)}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, y_mean: float, y_std: float) -> Dict[str, float]:
    model.eval()
    ys: List[float] = []
    yhs: List[float] = []
    for batch in loader:
        batch = batch.to(device)
        pred_norm = model(batch)  # normalized space
        pred = pred_norm * y_std + y_mean
        y_true = batch.y.view(-1).float()
        ys.extend(y_true.detach().cpu().numpy().tolist())
        yhs.extend(pred.detach().cpu().numpy().tolist())

    ys_np = np.asarray(ys, dtype=float)
    yhs_np = np.asarray(yhs, dtype=float)
    return {
        "mae": float(mean_absolute_error(ys_np, yhs_np)),
        "rmse": float(math.sqrt(mean_squared_error(ys_np, yhs_np))),
    }


@torch.no_grad()
def predict_rows(model: nn.Module, loader: DataLoader, device: torch.device, y_mean: float, y_std: float) -> List[Tuple[str, float, float]]:
    model.eval()
    out: List[Tuple[str, float, float]] = []
    for batch in loader:
        batch = batch.to(device)
        pred_norm = model(batch)
        pred = pred_norm * y_std + y_mean
        y_true = batch.y.view(-1).float()
        paths = list(batch.path) if isinstance(batch.path, (list, tuple)) else batch.path

        # When collated, custom attrs can become a list of strings.
        if isinstance(paths, str):
            paths = [paths]

        for p, yt, yp in zip(paths, y_true.detach().cpu().numpy(), pred.detach().cpu().numpy()):
            out.append((str(p), float(yt), float(yp)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--root_dir", type=str, default=None)
    ap.add_argument("--path_col", type=str, default="path")
    ap.add_argument("--target_col", type=str, default="y")
    ap.add_argument("--group_col", type=str, default=None)

    ap.add_argument("--cutoff", type=float, default=6.0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_rbf", type=int, default=64)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--use_group_split", action="store_true")

    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--cache_structures", action="store_true")
    ap.add_argument("--out_dir", type=str, required=True)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ContcarCSVDataset(
        csv_path=args.csv_path,
        root_dir=args.root_dir,
        path_col=args.path_col,
        target_col=args.target_col,
        group_col=args.group_col,
        cutoff=args.cutoff,
        cache_structures=args.cache_structures,
    )

    if args.use_group_split:
        split = split_indices_group(ds.groups(), seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac)
    else:
        split = split_indices_random(len(ds), seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac)

    # Compute target normalization from train only
    y_train = np.array([float(ds[i].y.item()) for i in split["train"]], dtype=float)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() if y_train.std() > 1e-12 else 1.0)

    def make_loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        subset = [ds[int(i)] for i in indices.tolist()]
        return DataLoader(subset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)

    train_loader = make_loader(split["train"], shuffle=True)
    val_loader = make_loader(split["val"], shuffle=False)
    test_loader = make_loader(split["test"], shuffle=False)

    cfg = ModelConfig(
        cutoff=float(args.cutoff),
        hidden_dim=int(args.hidden_dim),
        n_layers=int(args.n_layers),
        n_rbf=int(args.n_rbf),
        emb_dim=int(args.emb_dim),
    )
    model = PBCMPNNRegressor(cfg).to(device)

    # Train in normalized y-space for stability
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            y = batch.y.view(-1).float()
            y_norm = (y - y_mean) / y_std

            pred_norm = model(batch)
            loss = loss_fn(pred_norm, y_norm)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        val_metrics = evaluate(model, val_loader, device=device, y_mean=y_mean, y_std=y_std)
        if val_metrics["mae"] < best_val:
            best_val = val_metrics["mae"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "split": {k: v.tolist() for k, v in split.items()},
                    "args": vars(args),
                },
                best_path,
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            tr_loss = float(np.mean(losses)) if losses else float("nan")
            msg = (
                f"[epoch {epoch:04d}/{args.epochs}] "
                f"train_mse_norm={tr_loss:.5f} | val_mae={val_metrics['mae']:.4f} | val_rmse={val_metrics['rmse']:.4f}"
            )
            print(msg)

    # Load best + evaluate
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    y_mean = float(ckpt["y_mean"])
    y_std = float(ckpt["y_std"])

    val_metrics = evaluate(model, val_loader, device=device, y_mean=y_mean, y_std=y_std)
    test_metrics = evaluate(model, test_loader, device=device, y_mean=y_mean, y_std=y_std)
    print(f"[best] val_mae={val_metrics['mae']:.4f} val_rmse={val_metrics['rmse']:.4f}")
    print(f"[best] test_mae={test_metrics['mae']:.4f} test_rmse={test_metrics['rmse']:.4f}")

    # Save predictions (train/val/test)
    preds_path = out_dir / "preds.csv"
    import csv as _csv

    with open(preds_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["split", "path", "y_true", "y_pred"])
        for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            for p, yt, yp in predict_rows(model, loader, device=device, y_mean=y_mean, y_std=y_std):
                w.writerow([split_name, p, yt, yp])

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                "model_config": asdict(cfg),
                "y_mean": y_mean,
                "y_std": y_std,
                "split": ckpt["split"],
                "args": vars(args),
                "metrics": {"val": val_metrics, "test": test_metrics},
                "device": str(device),
            },
            f,
            indent=2,
        )

    print(f"Saved: {best_path}")
    print(f"Saved: {preds_path}")


if __name__ == "__main__":
    main()


