"""
Step 6 — CompositeRisk, and the weak labels the GNN trains on.

    CompositeRisk = 0.50·Direct + 0.20·Cascading + 0.20·Concentration + 0.10·NonSub

Each component is percentile-rank normalised *within its year* first, because the
four are on incompatible raw scales and the graph grows year over year. Ranking
makes the blend meaningful and keeps scores comparable across snapshots.

Why weak supervision at all: the ideal label is "which insured firms lost how
much when provider X went down", and no such dataset exists publicly. So the
rule-based CompositeRisk becomes a proxy label over 2022–2024, and the GNN learns
the *structural pattern* that produces it — which then transfers to the 2025
graph. The 2025 score is still computed here, but it is never fed to the model as
a target or a feature; it is kept only so the rule and the model can be compared
on the hold-out year.

The honest limitation: the model can only be as right as the rule. What it adds
is generalisation to a graph the rule never saw, and the ablation in
`docs/04-model.md` is what shows the structure carries real signal.
"""

from __future__ import annotations

import pandas as pd

from . import config
from .utils import banner, percentile_rank_normalize, print_dist, save_csv

VULN_COLS = ["canonical_firm_id", "year", "intrinsic_risk", "incident_signal",
             "network_criticality", "node_vulnerability"]


def run(ctx: dict, dry_run: bool = False) -> dict:
    banner("STEP 6 — Composite Risk",
           " | ".join(f"{k.split('_')[0]}={v}" for k, v in config.COMPOSITE_WEIGHTS.items()))

    train_frames = []
    year_composite = {}

    for yr in config.YEARS:
        comp_df = ctx["year_components"][yr].copy()
        comp_df = comp_df.merge(
            ctx["year_vulnerability"][yr][VULN_COLS],
            on=["canonical_firm_id", "year"], how="left",
        )

        for col in config.COMPOSITE_WEIGHTS:
            comp_df[f"{col}_norm"] = percentile_rank_normalize(comp_df[col])

        comp_df["composite_risk"] = sum(
            w * comp_df[f"{col}_norm"] for col, w in config.COMPOSITE_WEIGHTS.items()
        )

        save_csv(comp_df, config.WRITE_DIR / f"composite_risk_{yr}.csv", dry_run)
        year_composite[yr] = comp_df
        if yr in config.TRAIN_YEARS:
            train_frames.append(comp_df)

        print_dist("composite_risk", yr, comp_df["composite_risk"],
                   name_map=comp_df["canonical_name"].to_dict())

    if train_frames:
        weak = pd.concat(train_frames, ignore_index=True)
        save_csv(weak, config.WRITE_DIR / "weak_labels_2022_2024.csv", dry_run)
        print(f"\n  weak_labels_2022_2024: {len(weak)} node-year rows "
              f"({weak['canonical_firm_id'].nunique()} unique firms)")

    ctx["year_composite"] = year_composite
    return ctx
