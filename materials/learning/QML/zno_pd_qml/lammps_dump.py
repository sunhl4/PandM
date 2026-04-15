from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Box:
    """Simulation box bounds and periodicity flags (LAMMPS dump BOX BOUNDS line)."""

    xlo: float
    xhi: float
    ylo: float
    yhi: float
    zlo: float
    zhi: float
    periodic: Tuple[bool, bool, bool]

    @property
    def lengths(self) -> np.ndarray:
        return np.array([self.xhi - self.xlo, self.yhi - self.ylo, self.zhi - self.zlo], dtype=float)


@dataclass
class Frame:
    timestep: int
    box: Box
    ids: np.ndarray  # (N,)
    types: np.ndarray  # (N,)
    pos: np.ndarray  # (N,3) in Angstrom


def _parse_box_bounds(header_line: str) -> Tuple[bool, bool, bool]:
    """
    Example:
      ITEM: BOX BOUNDS pp pp ff
    """
    parts = header_line.strip().split()
    if len(parts) < 6:
        # fallback: assume periodic in x,y; non-periodic in z (common for slab)
        return (True, True, False)
    flags = parts[-3:]
    p = []
    for f in flags:
        p.append(f.lower().startswith("p"))
    return (bool(p[0]), bool(p[1]), bool(p[2]))


def _read_nonempty_line(f) -> str:
    line = f.readline()
    if not line:
        raise EOFError
    return line


def iter_lammps_dump_frames(
    dump_path: str | Path,
    *,
    stride_frames: int = 1,
    max_frames: Optional[int] = None,
    required_cols: Sequence[str] = ("id", "type", "x", "y", "z"),
) -> Iterator[Frame]:
    """
    Stream LAMMPS dump custom frames.

    Supports ATOMS columns:
      - (x,y,z) or (xs,ys,zs) scaled coords.

    Notes:
      - This is intentionally dependency-free (no MDAnalysis).
      - Distances should be computed with your own min-image if periodic.
    """
    dump_path = Path(dump_path)
    stride_frames = max(1, int(stride_frames))
    emitted = 0
    seen = 0

    with dump_path.open("r") as f:
        while True:
            try:
                line = _read_nonempty_line(f)
            except EOFError:
                return

            if not line.startswith("ITEM: TIMESTEP"):
                # tolerate leading junk / blank lines
                continue

            timestep = int(_read_nonempty_line(f).strip())

            # NUMBER OF ATOMS
            _ = _read_nonempty_line(f)  # ITEM: NUMBER OF ATOMS
            n_atoms = int(_read_nonempty_line(f).strip())

            # BOX BOUNDS
            box_hdr = _read_nonempty_line(f).strip()
            if not box_hdr.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Unexpected dump format near BOX BOUNDS: {box_hdr}")
            periodic = _parse_box_bounds(box_hdr)

            xlo, xhi = map(float, _read_nonempty_line(f).split()[:2])
            ylo, yhi = map(float, _read_nonempty_line(f).split()[:2])
            zlo, zhi = map(float, _read_nonempty_line(f).split()[:2])
            box = Box(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, zlo=zlo, zhi=zhi, periodic=periodic)

            # ATOMS
            atoms_hdr = _read_nonempty_line(f).strip()
            if not atoms_hdr.startswith("ITEM: ATOMS"):
                raise ValueError(f"Unexpected dump format near ATOMS header: {atoms_hdr}")
            cols = atoms_hdr.split()[2:]
            col_to_i = {c: i for i, c in enumerate(cols)}

            missing = [c for c in required_cols if c not in col_to_i]
            # Allow scaled coords alternative
            if missing:
                alt = {"x": "xs", "y": "ys", "z": "zs"}
                missing2 = [c for c in missing if alt.get(c, c) not in col_to_i]
                if missing2:
                    raise ValueError(
                        f"Dump missing required columns {missing2}. Found columns={cols}. "
                        f"Add them to your dump custom line."
                    )

            raw = np.zeros((n_atoms, len(cols)), dtype=float)
            for i in range(n_atoms):
                parts = _read_nonempty_line(f).split()
                # Some dumps may have extra whitespace; be defensive
                if len(parts) < len(cols):
                    raise ValueError(f"Bad ATOMS line (len={len(parts)} < {len(cols)}): {parts}")
                raw[i] = np.asarray(parts[: len(cols)], dtype=float)

            seen += 1
            if (seen - 1) % stride_frames != 0:
                continue
            if max_frames is not None and emitted >= int(max_frames):
                return

            ids = raw[:, col_to_i["id"]].astype(np.int64)
            types = raw[:, col_to_i["type"]].astype(np.int64)

            def _get_xyz() -> np.ndarray:
                if all(c in col_to_i for c in ("x", "y", "z")):
                    return raw[:, [col_to_i["x"], col_to_i["y"], col_to_i["z"]]]
                # scaled
                xs = raw[:, col_to_i.get("xs", col_to_i.get("x"))]
                ys = raw[:, col_to_i.get("ys", col_to_i.get("y"))]
                zs = raw[:, col_to_i.get("zs", col_to_i.get("z"))]
                L = box.lengths
                x = box.xlo + xs * L[0]
                y = box.ylo + ys * L[1]
                z = box.zlo + zs * L[2]
                return np.stack([x, y, z], axis=1)

            pos = _get_xyz().astype(float)

            # Sort by id for stable indexing (optional, but helps windows/diffs)
            order = np.argsort(ids)
            frame = Frame(
                timestep=timestep,
                box=box,
                ids=ids[order],
                types=types[order],
                pos=pos[order],
            )
            emitted += 1
            yield frame


def type_mask(types: np.ndarray, type_ids: Sequence[int]) -> np.ndarray:
    """Return boolean mask for LAMMPS integer atom types."""
    t = np.asarray(types, dtype=np.int64)
    ids = np.asarray(list(type_ids), dtype=np.int64)
    if ids.size == 0:
        return np.zeros_like(t, dtype=bool)
    return np.isin(t, ids)


def load_type_map(type_map: Dict[str, Sequence[int]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in type_map.items():
        out[k] = np.asarray(list(v), dtype=np.int64)
    return out


