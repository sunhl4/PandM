from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from torch_geometric.data import Data
except Exception as e:  # pragma: no cover
    raise ImportError(
        "gnn_adsorption requires torch_geometric. Install PyG first, then retry."
    ) from e


@dataclass(frozen=True)
class Row:
    path: str
    y: float
    group: Optional[str] = None


def _read_csv_rows(
    csv_path: str,
    path_col: str,
    target_col: str,
    group_col: Optional[str],
) -> List[Row]:
    rows: List[Row] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        for r in reader:
            p = r.get(path_col)
            if p is None or str(p).strip() == "":
                raise ValueError(f"Missing '{path_col}' in a row of {csv_path}")
            y_raw = r.get(target_col)
            if y_raw is None or str(y_raw).strip() == "":
                raise ValueError(f"Missing '{target_col}' in a row of {csv_path}")
            g = r.get(group_col) if group_col else None
            rows.append(Row(path=str(p), y=float(y_raw), group=(str(g) if g is not None else None)))
    if len(rows) == 0:
        raise ValueError(f"No rows loaded from: {csv_path}")
    return rows


def _resolve_path(p: str, root_dir: Optional[str]) -> str:
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    if root_dir is None:
        return str(pp)
    return str(Path(root_dir) / pp)


def build_pbc_radius_graph(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    cutoff: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build neighbor edges with ASE's PBC-aware neighbor list.

    Returns:
      edge_index: (2, E) int64 (directed, includes both directions)
      edge_vec:   (E, 3) float32, vector from i -> j (with PBC image shift)
      edge_dist:  (E,) float32, norm of edge_vec
    """
    try:
        from ase import Atoms
        from ase.neighborlist import neighbor_list
    except Exception as e:  # pragma: no cover
        raise ImportError("ASE is required to build PBC neighbor graphs. Install 'ase'.") from e

    atoms = Atoms(positions=positions, cell=cell, pbc=pbc)

    # i -> j with integer image shifts S (so j position is pos[j] + S @ cell)
    i, j, S = neighbor_list("ijS", atoms, cutoff)

    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    S = np.asarray(S, dtype=np.int64)
    cell = np.asarray(cell, dtype=np.float64)

    # Vector from i to the periodic image of j
    vec_ij = positions[j] + (S @ cell) - positions[i]
    dist_ij = np.linalg.norm(vec_ij, axis=1)

    # Make it explicitly undirected by adding reverse edges
    edge_index = np.stack([np.concatenate([i, j]), np.concatenate([j, i])], axis=0)
    edge_vec = np.concatenate([vec_ij, -vec_ij], axis=0).astype(np.float32)
    edge_dist = np.concatenate([dist_ij, dist_ij], axis=0).astype(np.float32)

    return edge_index, edge_vec, edge_dist


class ContcarCSVDataset(torch.utils.data.Dataset):
    """
    CSV-backed dataset:
      - reads VASP CONTCAR/POSCAR via ASE
      - builds PBC-aware radius graph edges (cutoff)
      - returns torch_geometric.data.Data
    """

    def __init__(
        self,
        csv_path: str,
        root_dir: Optional[str] = None,
        path_col: str = "path",
        target_col: str = "y",
        group_col: Optional[str] = None,
        cutoff: float = 6.0,
        cache_structures: bool = False,
    ) -> None:
        self.csv_path = str(csv_path)
        self.root_dir = str(root_dir) if root_dir is not None else None
        self.path_col = path_col
        self.target_col = target_col
        self.group_col = group_col
        self.cutoff = float(cutoff)
        self.rows = _read_csv_rows(self.csv_path, path_col, target_col, group_col)

        self._cache_structures = bool(cache_structures)
        self._cache: Dict[int, Data] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def groups(self) -> List[Optional[str]]:
        return [r.group for r in self.rows]

    def _read_atoms(self, path: str):
        try:
            from ase.io import read
        except Exception as e:  # pragma: no cover
            raise ImportError("ASE is required to read CONTCAR. Install 'ase'.") from e

        # ASE detects format from file content for VASP POSCAR-like files,
        # but we explicitly set it for clarity.
        return read(path, format="vasp")

    def __getitem__(self, idx: int) -> Data:
        if self._cache_structures and idx in self._cache:
            return self._cache[idx]

        row = self.rows[idx]
        abs_path = _resolve_path(row.path, self.root_dir)

        atoms = self._read_atoms(abs_path)
        Z = np.asarray(atoms.numbers, dtype=np.int64)
        pos = np.asarray(atoms.positions, dtype=np.float32)
        cell = np.asarray(atoms.cell.array, dtype=np.float32)
        pbc = np.asarray(atoms.pbc, dtype=bool)

        edge_index, edge_vec, edge_dist = build_pbc_radius_graph(
            positions=pos.astype(np.float64),
            cell=cell.astype(np.float64),
            pbc=pbc,
            cutoff=self.cutoff,
        )

        data = Data(
            z=torch.from_numpy(Z).long(),
            pos=torch.from_numpy(pos).float(),
            edge_index=torch.from_numpy(edge_index).long(),
            edge_vec=torch.from_numpy(edge_vec).float(),
            edge_dist=torch.from_numpy(edge_dist).float(),
            y=torch.tensor([row.y], dtype=torch.float32),
        )
        # Keep metadata for debugging / saving predictions
        data.path = abs_path
        if row.group is not None:
            data.group = row.group

        if self._cache_structures:
            self._cache[idx] = data
        return data


