from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import MessagePassing, global_mean_pool
except Exception as e:  # pragma: no cover
    raise ImportError(
        "gnn_adsorption requires torch_geometric. Install PyG first, then retry."
    ) from e


def cosine_cutoff(dist: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Smooth cutoff envelope in [0, cutoff].
    Returns 0 for dist >= cutoff.
    """
    x = dist / cutoff
    out = 0.5 * (torch.cos(math.pi * x).clamp(min=-1.0, max=1.0) + 1.0)
    return out * (dist < cutoff).to(out.dtype)


class RBFEmbedding(nn.Module):
    def __init__(self, n_rbf: int, cutoff: float) -> None:
        super().__init__()
        self.n_rbf = int(n_rbf)
        self.cutoff = float(cutoff)

        centers = torch.linspace(0.0, self.cutoff, self.n_rbf)
        # (n_rbf,) buffers so they move with device but aren't trainable
        self.register_buffer("centers", centers)
        # width chosen so adjacent centers overlap reasonably
        gamma = 1.0 / (centers[1] - centers[0]).clamp(min=1e-6) ** 2
        self.register_buffer("gamma", gamma)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: (E,)
        d = dist.unsqueeze(-1)  # (E,1)
        return torch.exp(-self.gamma * (d - self.centers) ** 2)  # (E, n_rbf)


class CFConv(MessagePassing):
    """
    A compact continuous-filter convolution (SchNet-style) that uses:
      - edge_dist -> RBF -> MLP filter
      - message: x_j * filter
    """

    def __init__(self, hidden_dim: int, n_rbf: int, cutoff: float) -> None:
        super().__init__(aggr="add")
        self.hidden_dim = int(hidden_dim)
        self.cutoff = float(cutoff)

        self.rbf = RBFEmbedding(n_rbf=n_rbf, cutoff=cutoff)
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_dist: torch.Tensor) -> torch.Tensor:
        # x: (N, H), edge_index: (2, E), edge_dist: (E,)
        rbf = self.rbf(edge_dist)  # (E, n_rbf)
        W = self.filter_net(rbf)  # (E, H)
        W = W * cosine_cutoff(edge_dist, self.cutoff).unsqueeze(-1)  # (E,H)
        out = self.propagate(edge_index=edge_index, x=x, W=W)  # (N,H)
        out = self.update_net(out)
        return x + out  # residual

    def message(self, x_j: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return x_j * W


@dataclass(frozen=True)
class ModelConfig:
    cutoff: float = 6.0
    hidden_dim: int = 128
    n_layers: int = 4
    n_rbf: int = 64
    emb_dim: int = 128  # graph embedding dim (after pooling)


class PBCMPNNRegressor(nn.Module):
    """
    PBC-aware message passing regressor:
      - atom embedding (Z -> hidden)
      - several CFConv layers using precomputed PBC edges
      - graph pooling -> embedding
      - MLP head -> scalar
    """

    def __init__(self, config: ModelConfig, max_atomic_num: int = 118) -> None:
        super().__init__()
        self.config = config

        self.atom_emb = nn.Embedding(max_atomic_num + 1, config.hidden_dim)
        self.convs = nn.ModuleList(
            [CFConv(config.hidden_dim, config.n_rbf, config.cutoff) for _ in range(config.n_layers)]
        )
        self.to_graph_emb = nn.Sequential(
            nn.Linear(config.hidden_dim, config.emb_dim),
            nn.SiLU(),
            nn.Linear(config.emb_dim, config.emb_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(config.emb_dim, config.emb_dim // 2),
            nn.SiLU(),
            nn.Linear(config.emb_dim // 2, 1),
        )

    def encode(self, z: torch.Tensor, edge_index: torch.Tensor, edge_dist: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.atom_emb(z)  # (N,H)
        for conv in self.convs:
            x = conv(x=x, edge_index=edge_index, edge_dist=edge_dist)
        g = global_mean_pool(x, batch=batch)  # (B,H)
        g = self.to_graph_emb(g)  # (B, emb_dim)
        return g

    def forward(self, data) -> torch.Tensor:
        g = self.encode(
            z=data.z,
            edge_index=data.edge_index,
            edge_dist=data.edge_dist,
            batch=data.batch,
        )
        return self.head(g).squeeze(-1)  # (B,)


