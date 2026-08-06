"""
Turn a scored year-snapshot into a PyTorch Geometric `Data` object.

One homogeneous graph per year, firms and providers in the same node space.
Splitting them into a bipartite firm→provider structure would have been the
obvious modelling choice and is the wrong one here: the interesting paths are
firm→firm (a cloud provider is itself a customer of a chip vendor) and
provider→provider. A single node space lets message passing cross those
boundaries, which is exactly where cascading exposure lives.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from .data_loader import RiskDataLoader


class HomoGraphBuilder:
    def __init__(self, loader: RiskDataLoader):
        self.loader = loader

    def build(self, year: int) -> Data:
        X_df, E_df, Y = self.loader.load_year(year)
        X_df = X_df.sort_values("node_id").reset_index(drop=True)

        # node_id is global across all years so a firm keeps its identity between
        # snapshots, but PyG requires edge_index within [0, N) of *this* graph —
        # hence the remap. Skipping this silently drops edges or indexes out of
        # bounds, and it is the single easiest thing to get wrong here.
        global_to_local = {int(nid): i for i, nid in enumerate(X_df["node_id"])}
        E_df = E_df.copy()
        E_df["source_node_id"] = E_df["source_node_id"].map(global_to_local)
        E_df["target_node_id"] = E_df["target_node_id"].map(global_to_local)
        E_df = E_df.dropna(subset=["source_node_id", "target_node_id"])
        E_df["source_node_id"] = E_df["source_node_id"].astype(int)
        E_df["target_node_id"] = E_df["target_node_id"].astype(int)

        x = torch.tensor(self.loader.get_feature_matrix(X_df), dtype=torch.float)
        edge_index, edge_attr = self.loader.get_edge_index_and_attr(E_df)

        # Unlabelled nodes stay NaN and are masked out of the loss rather than
        # being treated as zero-risk.
        if Y is not None:
            row_of = {fid: i for i, fid in enumerate(X_df["canonical_firm_id"])}
            y_vals = np.full(len(X_df), np.nan, dtype=np.float32)
            for firm_id, score in Y.items():
                if firm_id in row_of:
                    y_vals[row_of[firm_id]] = score
            y = torch.tensor(y_vals, dtype=torch.float)
        else:
            y = None

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.node_type = torch.tensor(X_df["node_type_enc"].values, dtype=torch.long)
        data.canonical_firm_id = X_df["canonical_firm_id"].values
        data.canonical_name = X_df["canonical_name"].values
        data.year = year
        return data
