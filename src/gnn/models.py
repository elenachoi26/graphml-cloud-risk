"""
The four models in the ablation, from no graph to full graph.

    TabularBaseline (Ridge)  node features only, linear
    MLPBaseline              node features only, non-linear
    BasicGNN (GCN)           + graph structure, edge weight as normalisation
    HomoGAT                  + learned attention and edge features

The middle tier is the one that makes the result interpretable. If only Ridge
and the GCN were compared, a large gap could just mean "the relationship is
non-linear". The MLP has exactly the same eight input columns as the GCN and no
graph, so any remaining gap is attributable to structure and nothing else.

All four share the same feature set (`config.GNN_FEATURES`) and the same
two-layer-then-head shape, so the comparison isn't confounded by capacity.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv


def _metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    """
    Spearman first, because rank is what underwriting consumes.

    Nobody acts on "this firm scored 0.63". They act on "these six firms are the
    most exposed ones in the book". MAE and RMSE are reported alongside to show
    the scores are also calibrated, not just correctly ordered.
    """
    rho, _ = spearmanr(true, pred)
    return {
        "spearman": rho,
        "mae": mean_absolute_error(true, pred),
        "rmse": float(np.sqrt(np.mean((true - pred) ** 2))),
    }


# ── Tier 1: no graph ──────────────────────────────────────────────────────────

class TabularBaseline:
    """Ridge regression on node features — the performance floor."""

    def __init__(self, alpha: float = 1.0):
        self.model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])

    def fit(self, X: np.ndarray, y: np.ndarray):
        mask = ~np.isnan(y)
        self.model.fit(X[mask], y[mask])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        mask = ~np.isnan(y)
        return _metrics(y[mask], self.predict(X[mask]))


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x): return self.net(x).squeeze(-1)


class MLPBaseline:
    """Non-linear control: same features as the GNN, still no graph."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, lr: float = 1e-3,
                 epochs: int = 200, dropout: float = 0.2):
        self.model = MLPRegressor(in_dim, hidden_dim, dropout)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.epochs = epochs
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        mask = ~np.isnan(y)
        x_t = torch.tensor(self.scaler.fit_transform(X[mask]).astype(np.float32))
        y_t = torch.tensor(y[mask], dtype=torch.float)
        self.model.train()
        for _ in range(self.epochs):
            self.optim.zero_grad()
            self.loss_fn(self.model(x_t), y_t).backward()
            self.optim.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(self.scaler.transform(X).astype(np.float32))).numpy()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        mask = ~np.isnan(y)
        return _metrics(y[mask], self.predict(X[mask]))


# ── Tier 2: graph structure ───────────────────────────────────────────────────

class BasicGNN(nn.Module):
    """
    Two-layer GCN — the model shipped in the final pipeline.

    Two layers, not more, because the risk decomposition itself only reaches two
    hops (direct and cascading exposure): a deeper stack would let information
    travel further than the underlying formulation claims it does.

    edge_weight enters through degree normalisation only, never as a feature.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 4)
        self.head = nn.Sequential(nn.Linear(hidden_dim // 4, 32), nn.ReLU(), nn.Linear(32, 1))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_weight=edge_weight))
        return self.head(x).squeeze(-1)

    def forward_data(self, data: Data):
        ew = data.edge_attr[:, 0] if data.edge_attr is not None else None
        return self.forward(data.x, data.edge_index, edge_weight=ew)


# ── Tier 3: attention + edge features ─────────────────────────────────────────

class HomoGAT(nn.Module):
    """
    Two-layer GAT using the full edge feature vector as attention context.

    Explored as the richer alternative. It did not beat the GCN by enough to
    justify the extra opacity — and in an insurance setting the model has to be
    explainable to an underwriter and a regulator, so the simpler model wins ties.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_heads: int = 4,
                 dropout: float = 0.2, edge_dim: int = 5):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // num_heads, heads=num_heads,
                             dropout=dropout, edge_dim=edge_dim, concat=True)
        self.conv2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                             dropout=dropout, edge_dim=edge_dim, concat=False)
        self.head = nn.Sequential(nn.Linear(hidden_dim // num_heads, 32), nn.ReLU(),
                                  nn.Linear(32, 1))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_attr=edge_attr))
        return self.head(x).squeeze(-1)

    def forward_data(self, data: Data):
        return self.forward(data.x, data.edge_index, data.edge_attr)


def evaluate_predictions(true: np.ndarray, pred: np.ndarray) -> dict:
    """Shared metric helper for the graph models."""
    return _metrics(true, pred)
