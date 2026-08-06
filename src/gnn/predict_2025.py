"""
Score the 2025 dependency graph with the model trained on 2022–2024.

2025 is the hold-out year. The rule-based CompositeRisk *is* computed for it by
step 6, but it is never used as a target or an input here — the model infers
systemic exposure from the 2025 topology alone, using patterns learned from
earlier snapshots, including for nodes that first appear in 2025 and have no
history at all. Keeping the 2025 rule score on disk is what makes the
rule-vs-model divergence analysis in `interpretation_report.py` possible.

Results are split three ways because they answer different questions:

    all nodes       where systemic exposure sits in the ecosystem overall
    source firms    which insurable firms to review first (the underwriting list)
    target entities which providers accumulate exposure (the CRO's list)

    python -m src.gnn.predict_2025
"""

from __future__ import annotations

import pandas as pd
import torch

from .. import config
from .data_loader import RiskDataLoader
from .graph_builder import HomoGraphBuilder
from .models import BasicGNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_final_gcn() -> BasicGNN:
    path = config.MODEL_DIR / "final_gcn_model.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m src.gnn.train_final` first."
        )
    ckpt = torch.load(path, map_location=DEVICE)
    model = BasicGNN(in_dim=ckpt["in_dim"], hidden_dim=ckpt["hidden_dim"], dropout=0.2).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("=" * 70)
    print(f"  loaded {ckpt.get('model_name')} — in_dim={ckpt['in_dim']}, "
          f"hidden={ckpt['hidden_dim']}, trained on {ckpt.get('train_years')}")
    print("=" * 70)
    return model


def run(year: int = config.PREDICT_YEAR) -> pd.DataFrame:
    model = load_final_gcn()
    loader = RiskDataLoader()
    data = HomoGraphBuilder(loader).build(year).to(DEVICE)

    with torch.no_grad():
        scores = model.forward_data(data).cpu().numpy()

    nodes = pd.read_csv(config.OUT_DIR / f"nodes_{year}.csv")
    pred = pd.DataFrame({"canonical_firm_id": data.canonical_firm_id, "gcn_score": scores})
    pred = nodes.merge(pred, on="canonical_firm_id", how="right")

    pred["gcn_rank_all"] = pred["gcn_score"].rank(ascending=False, method="min").astype(int)
    pred["gcn_rank_within_type"] = (pred.groupby("node_type")["gcn_score"]
                                    .rank(ascending=False, method="min").astype(int))
    pred = pred.sort_values("gcn_score", ascending=False)

    config.ensure_dirs()
    pred.to_csv(config.PRED_DIR / f"prediction_{year}_gcn_all_nodes.csv", index=False)
    for ntype, suffix in (("source_firm", "source_firms"), ("target_entity", "target_entities")):
        subset = pred[pred["node_type"] == ntype].sort_values("gcn_score", ascending=False)
        subset.to_csv(config.PRED_DIR / f"prediction_{year}_gcn_{suffix}.csv", index=False)
        print(f"\n  Top {ntype}s:")
        for _, r in subset.head(6).iterrows():
            print(f"    {r['gcn_rank_within_type']:>3}. {r['canonical_name']:<28} {r['gcn_score']:.4f}")

    print(f"\n  → {config.PRED_DIR} ({len(pred)} nodes scored for {year})")
    return pred


if __name__ == "__main__":
    run()
