"""
Assemble the scored graph into model-ready frames.

Feature design is the load-bearing decision here. Node features are the six
10-K intrinsic risks + the incident signal + the node type — eight columns, and
nothing else. Every graph-derived quantity (pagerank, indegree centrality,
node_vulnerability, and the exposure components themselves) is deliberately
withheld.

Without that discipline the comparison in `cross_val.py` would be meaningless:
a tabular baseline handed `pagerank` as a column is not graph-free, and a GNN
handed `direct_exposure` is just copying its own label back out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config

NODE_FEAT_COLS = config.GNN_FEATURES

EDGE_FEAT_COLS = [
    "edge_weight",
    "contract_scale_score_norm",
    "risk_language_score_norm",
    "no_alternative_flag",
    "investment_flag",
]

LABEL_COL = "composite_risk"


class RiskDataLoader:
    """Reads the stage-2 artifacts out of `outputs/`."""

    def __init__(self, output_dir=None):
        self.out = config.OUT_DIR if output_dir is None else output_dir

    def load_node_features(self, year: int) -> pd.DataFrame:
        comp = pd.read_csv(self.out / f"composite_risk_{year}.csv")
        vuln = pd.read_csv(self.out / f"node_vulnerability_{year}.csv")

        # Pull only non-leaky columns across from the vulnerability table.
        keep = ["canonical_firm_id"] + config.INTRINSIC_FEATURES + ["incident_signal"]
        df = comp.merge(vuln[keep], on="canonical_firm_id", how="left", suffixes=("", "_v"))

        df["node_type_enc"] = (df["node_type"].map(config.NODE_TYPE_MAP)
                               .fillna(config.NODE_TYPE_MAP["target_entity"]).astype(int))

        # Target entities have no filing of their own — intrinsic features are
        # genuinely unobserved, not zero-risk. Zero is the neutral encoding; the
        # model distinguishes them via node_type_enc.
        is_target = df["node_type"] == "target_entity"
        df.loc[is_target, config.INTRINSIC_FEATURES] = (
            df.loc[is_target, config.INTRINSIC_FEATURES].fillna(0.0)
        )
        return df

    def load_edges(self, year: int) -> pd.DataFrame:
        edges = pd.read_csv(self.out / f"edges_weighted_{year}.csv")
        nodes = pd.read_csv(self.out / f"nodes_{year}.csv")

        id_map = nodes.set_index("canonical_firm_id")["node_id"].to_dict()
        edges["source_node_id"] = edges["risk_source_id"].map(id_map)
        edges["target_node_id"] = edges["risk_target_id"].map(id_map)

        edges = edges.dropna(subset=["source_node_id", "target_node_id"]).copy()
        edges["source_node_id"] = edges["source_node_id"].astype(int)
        edges["target_node_id"] = edges["target_node_id"].astype(int)
        return edges

    def load_year(self, year: int):
        """
        Returns (X_df, E_df, Y).

        Y is None for the prediction year: the rule-based CompositeRisk is never
        computed for 2025, which is what makes it a genuine held-out graph.
        """
        X_df = self.load_node_features(year)
        E_df = self.load_edges(year)
        Y = X_df.set_index("canonical_firm_id")[LABEL_COL] if year in config.TRAIN_YEARS else None
        return X_df, E_df, Y

    def load_weak_labels(self) -> pd.DataFrame:
        return pd.read_csv(self.out / "weak_labels_2022_2024.csv")

    def get_feature_matrix(self, X_df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in NODE_FEAT_COLS if c in X_df.columns]
        return X_df[cols].fillna(0.0).values.astype(np.float32)

    def get_edge_index_and_attr(self, E_df: pd.DataFrame):
        import torch

        edge_index = torch.tensor(
            np.stack([E_df["source_node_id"].values, E_df["target_node_id"].values], axis=0),
            dtype=torch.long,
        )
        cols = [c for c in EDGE_FEAT_COLS if c in E_df.columns]
        edge_attr = torch.tensor(
            E_df[cols].fillna(0.0).values.astype(np.float32), dtype=torch.float
        )
        return edge_index, edge_attr
