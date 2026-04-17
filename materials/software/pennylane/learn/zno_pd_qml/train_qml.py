from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml


def _load_npz(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


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


def compute_kernel_matrix(
    X1: np.ndarray,
    X2: np.ndarray,
    kernel_fn,
    *,
    n_workers: int = 1,
) -> np.ndarray:
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    n1, n2 = X1.shape[0], X2.shape[0]
    if int(n_workers) <= 1:
        K = np.zeros((n1, n2), dtype=float)
        for i in range(n1):
            for j in range(n2):
                K[i, j] = kernel_fn(X1[i], X2[j])
        return K

    # Parallel version (joblib)
    from joblib import Parallel, delayed
    import itertools

    results = Parallel(n_jobs=int(n_workers), verbose=0)(
        delayed(kernel_fn)(X1[i], X2[j]) for i, j in itertools.product(range(n1), range(n2))
    )
    return np.asarray(results, dtype=float).reshape(n1, n2)


def _save_parity_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_png: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=25, alpha=0.8, edgecolor="black", linewidth=0.3)
    plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def _select_target(y: np.ndarray, y_keys: Sequence[str], target: str) -> np.ndarray:
    y_keys = [str(k) for k in y_keys]
    if target not in y_keys:
        raise ValueError(f"Unknown target '{target}'. Available: {y_keys}")
    j = y_keys.index(target)
    return y[:, j]


def _group_from_meta(meta: Sequence[Dict[str, Any]], group_key: str) -> np.ndarray:
    if group_key in ("none", "", "null"):
        return np.array(["all"] * len(meta), dtype=object)
    if group_key == "o_model":
        return np.array([str(m.get("o_model", "unknown")) for m in meta], dtype=object)
    if group_key == "T_K":
        return np.array([str(int(m.get("T_K", -1))) for m in meta], dtype=object)
    if group_key == "dPd_A":
        return np.array([str(m.get("dPd_A", "nan")) for m in meta], dtype=object)
    if group_key == "o_frac":
        return np.array([str(m.get("o_frac", "nan")) for m in meta], dtype=object)
    if "+" in group_key:
        keys = [k.strip() for k in group_key.split("+") if k.strip()]
        return np.array(["|".join([str(m.get(k, "NA")) for k in keys]) for m in meta], dtype=object)
    # fallback: direct meta key
    return np.array([str(m.get(group_key, "NA")) for m in meta], dtype=object)


@dataclass(frozen=True)
class RegMetrics:
    rmse: float
    mae: float
    r2: float


@dataclass(frozen=True)
class ClsMetrics:
    acc: float
    f1: float
    roc_auc: Optional[float]


def main() -> None:
    # Delay sklearn imports so that:
    # - `python -m zno_pd_qml.train_qml --help` works even if sklearn/scipy is broken
    # - we can surface a clearer error message for common SciPy binary mismatch issues
    try:
        from sklearn.decomposition import PCA
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.linear_model import Ridge
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            roc_auc_score,
        )
        from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Failed to import scikit-learn dependencies. This is often caused by a SciPy binary mismatch.\n"
            "Fix suggestions (pick one):\n"
            "  - In a clean env: `pip install -U numpy scipy scikit-learn`\n"
            "  - Or conda-forge: `conda install -c conda-forge numpy scipy scikit-learn`\n"
            "Then re-run this command."
        ) from e

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", type=str, required=True)
    ap.add_argument("--task", type=str, required=True, choices=["regression", "classification"])
    ap.add_argument("--target", type=str, required=True, help="One of y_keys inside dataset.npz")

    ap.add_argument("--group_key", type=str, default="o_model", help="o_model/T_K/dPd_A/o_frac/none or combo like o_model+T_K")
    ap.add_argument("--split_mode", type=str, default="groupkfold", choices=["groupkfold", "logo", "kfold"])
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--n_qubits", type=int, default=4)
    ap.add_argument("--pca_components", type=int, default=None, help="Default: n_qubits")
    ap.add_argument("--alpha", type=float, default=1e-2)
    ap.add_argument("--rbf_gamma", type=float, default=None, help="Default: 1/n_features_after_pca")
    ap.add_argument("--n_workers", type=int, default=1, help="Parallelism for quantum kernel matrix computation")

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--save_parity", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_npz(args.dataset_npz)
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    X_keys = [str(k) for k in data["X_keys"].tolist()]
    y_keys = [str(k) for k in data["y_keys"].tolist()]
    meta = data.get("meta", None)
    if meta is None:
        raise ValueError("dataset.npz missing 'meta' (expected from zno_pd_qml.build_dataset).")
    meta_list = [dict(m) for m in meta.tolist()]

    y_vec = _select_target(y, y_keys, target=str(args.target))

    groups = _group_from_meta(meta_list, group_key=str(args.group_key))

    # Splitter
    split_mode = str(args.split_mode)
    if split_mode == "kfold":
        splitter = KFold(n_splits=int(args.cv_folds), shuffle=True, random_state=int(args.seed))
        split_iter = splitter.split(X)
    elif split_mode == "logo":
        splitter = LeaveOneGroupOut()
        split_iter = splitter.split(X, y_vec, groups)
    else:
        splitter = GroupKFold(n_splits=int(args.cv_folds))
        split_iter = splitter.split(X, y_vec, groups)

    n_qubits = int(args.n_qubits)
    pca_components = int(args.pca_components) if args.pca_components is not None else n_qubits
    qkernel = make_quantum_kernel(n_qubits=n_qubits)

    rows: List[Dict[str, Any]] = []
    last_parity: Optional[Tuple[np.ndarray, np.ndarray, str]] = None

    fold = 0
    for tr_idx, te_idx in split_iter:
        fold += 1
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y_vec[tr_idx], y_vec[te_idx]

        # Scale -> PCA (fit on train only)
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        if X_tr_s.shape[1] < pca_components:
            raise ValueError(f"X dim={X_tr_s.shape[1]} < pca_components={pca_components}")
        pca = PCA(n_components=int(pca_components), random_state=int(args.seed)).fit(X_tr_s)
        Z_tr = pca.transform(X_tr_s)
        Z_te = pca.transform(X_te_s)

        # For quantum kernel, we use first n_qubits dims (or require equal)
        if Z_tr.shape[1] < n_qubits:
            raise ValueError(f"PCA dim={Z_tr.shape[1]} < n_qubits={n_qubits}")
        Z_tr_q = Z_tr[:, :n_qubits]
        Z_te_q = Z_te[:, :n_qubits]

        # Baseline features for classical models: use the same reduced Z for fairness
        Z_tr_c = Z_tr_q
        Z_te_c = Z_te_q
        gamma = float(args.rbf_gamma) if args.rbf_gamma is not None else 1.0 / float(Z_tr_c.shape[1])

        if str(args.task) == "regression":
            ridge = Ridge(alpha=float(args.alpha))
            rbfkrr = KernelRidge(alpha=float(args.alpha), kernel="rbf", gamma=gamma)
            qkrr = KernelRidge(alpha=float(args.alpha), kernel="precomputed")

            ridge.fit(Z_tr_c, y_tr)
            y_hat_r = ridge.predict(Z_te_c)
            mr = RegMetrics(
                rmse=float(math.sqrt(mean_squared_error(y_te, y_hat_r))),
                mae=float(mean_absolute_error(y_te, y_hat_r)),
                r2=float(r2_score(y_te, y_hat_r)),
            )
            rows.append({"fold": fold, "model": "Ridge", **mr.__dict__})

            rbfkrr.fit(Z_tr_c, y_tr)
            y_hat_k = rbfkrr.predict(Z_te_c)
            mk = RegMetrics(
                rmse=float(math.sqrt(mean_squared_error(y_te, y_hat_k))),
                mae=float(mean_absolute_error(y_te, y_hat_k)),
                r2=float(r2_score(y_te, y_hat_k)),
            )
            rows.append({"fold": fold, "model": "RBF-KRR", **mk.__dict__})

            mn, mx = fit_angle_mapper(Z_tr_q)
            Z_tr_ang = to_angle_range_with(mn, mx, Z_tr_q)
            Z_te_ang = to_angle_range_with(mn, mx, Z_te_q)
            K_tr = compute_kernel_matrix(Z_tr_ang, Z_tr_ang, qkernel, n_workers=int(args.n_workers))
            K_te = compute_kernel_matrix(Z_te_ang, Z_tr_ang, qkernel, n_workers=int(args.n_workers))
            qkrr.fit(K_tr, y_tr)
            y_hat_q = qkrr.predict(K_te)
            mq = RegMetrics(
                rmse=float(math.sqrt(mean_squared_error(y_te, y_hat_q))),
                mae=float(mean_absolute_error(y_te, y_hat_q)),
                r2=float(r2_score(y_te, y_hat_q)),
            )
            rows.append({"fold": fold, "model": "Quantum-KRR", **mq.__dict__})

            last_parity = (np.asarray(y_te, dtype=float), np.asarray(y_hat_q, dtype=float), f"fold{fold}")

        else:
            # classification
            y_tr_i = np.asarray(y_tr, dtype=int)
            y_te_i = np.asarray(y_te, dtype=int)
            svc_rbf = SVC(C=1.0, kernel="rbf", gamma=gamma, probability=True)
            svc_q = SVC(C=1.0, kernel="precomputed", probability=True)

            svc_rbf.fit(Z_tr_c, y_tr_i)
            y_hat = svc_rbf.predict(Z_te_c)
            y_prob = svc_rbf.predict_proba(Z_te_c)[:, 1] if len(np.unique(y_tr_i)) == 2 else None
            auc = float(roc_auc_score(y_te_i, y_prob)) if (y_prob is not None and len(np.unique(y_te_i)) == 2) else None
            m1 = ClsMetrics(
                acc=float(accuracy_score(y_te_i, y_hat)),
                f1=float(f1_score(y_te_i, y_hat, zero_division=0)),
                roc_auc=auc,
            )
            rows.append({"fold": fold, "model": "RBF-SVC", **m1.__dict__})

            mn, mx = fit_angle_mapper(Z_tr_q)
            Z_tr_ang = to_angle_range_with(mn, mx, Z_tr_q)
            Z_te_ang = to_angle_range_with(mn, mx, Z_te_q)
            K_tr = compute_kernel_matrix(Z_tr_ang, Z_tr_ang, qkernel, n_workers=int(args.n_workers))
            K_te = compute_kernel_matrix(Z_te_ang, Z_tr_ang, qkernel, n_workers=int(args.n_workers))
            svc_q.fit(K_tr, y_tr_i)
            y_hat_q = svc_q.predict(K_te)
            y_prob_q = svc_q.predict_proba(K_te)[:, 1] if len(np.unique(y_tr_i)) == 2 else None
            auc_q = float(roc_auc_score(y_te_i, y_prob_q)) if (y_prob_q is not None and len(np.unique(y_te_i)) == 2) else None
            m2 = ClsMetrics(
                acc=float(accuracy_score(y_te_i, y_hat_q)),
                f1=float(f1_score(y_te_i, y_hat_q, zero_division=0)),
                roc_auc=auc_q,
            )
            rows.append({"fold": fold, "model": "Quantum-SVC", **m2.__dict__})

    # Save metrics
    metrics_csv = out_dir / "cv_metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Summary
    def _summarize(model: str, key: str) -> Dict[str, float]:
        arr = np.array([r[key] for r in rows if r["model"] == model and r.get(key) is not None], dtype=float)
        if arr.size == 0:
            return {}
        return {"mean": float(arr.mean()), "std": float(arr.std())}

    models = sorted({r["model"] for r in rows})
    summary: Dict[str, Any] = {
        "task": str(args.task),
        "target": str(args.target),
        "group_key": str(args.group_key),
        "split_mode": str(args.split_mode),
        "n_qubits": n_qubits,
        "pca_components": pca_components,
        "models": {},
        "X_dim": int(X.shape[1]),
        "X_keys": X_keys,
        "y_keys": y_keys,
    }
    if str(args.task) == "regression":
        for m in models:
            summary["models"][m] = {
                "rmse": _summarize(m, "rmse"),
                "mae": _summarize(m, "mae"),
                "r2": _summarize(m, "r2"),
            }
    else:
        for m in models:
            summary["models"][m] = {
                "acc": _summarize(m, "acc"),
                "f1": _summarize(m, "f1"),
                "roc_auc": _summarize(m, "roc_auc"),
            }

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if str(args.task) == "regression" and bool(args.save_parity) and last_parity is not None:
        y_te, y_hat, tag = last_parity
        _save_parity_plot(y_te, y_hat, title=f"Quantum-KRR parity ({tag})", out_png=str(out_dir / "parity_quantum_krr.png"))

    print("Saved:", metrics_csv)
    print("Saved:", summary_json)


if __name__ == "__main__":
    main()


