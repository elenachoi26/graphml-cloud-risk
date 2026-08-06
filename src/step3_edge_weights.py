"""
Step 3 — Edge weights: how much risk actually flows down a disclosed relationship.

    edge_weight = max(FLOOR, 0.40·CSS + 0.30·RLS + 0.20·NoAlt + 0.10·Inv)

A 10-K mentions "we use AWS" and "we have a multi-year, multi-billion dollar
commitment to AWS with no comparable alternative" in the same flat text. The
weight is what separates them:

  CSS   contract scale — disclosed contract or investment size
  RLS   risk language  — how alarmed the filing's own wording is
  NoAlt "sole source" / "no comparable alternative" language
  Inv   equity stake or long-term strategic partnership

FLOOR exists because a firm bothering to name a counterparty in a 10-K is itself
evidence of dependence — no disclosed edge is weightless.
"""

from __future__ import annotations

import pandas as pd

from . import config
from .utils import banner, minmax_norm, master_lookup, print_dist, read_output, save_csv


def run(ctx: dict, dry_run: bool = False) -> dict:
    banner("STEP 3 — Edge Weights",
           f"CSS={config.EW_CSS} | RLS={config.EW_RLS} | "
           f"NoAlt={config.EW_NOALT} | Inv={config.EW_INV} | floor={config.EW_FLOOR}")

    master = ctx["master"]
    names = master_lookup(master)["canonical_name"]

    year_edges_weighted = {}
    for yr in config.YEARS:
        e = ctx["year_edges"][yr].copy() if "year_edges" in ctx \
            else read_output(f"edges_{yr}_filtered.csv")

        e["contract_scale_score"] = e["contract_scale_score"].fillna(0.0)
        e["risk_language_score"] = e["risk_language_score"].fillna(0.0)
        e["no_alternative_mentioned"] = e["no_alternative_mentioned"].fillna("N")
        e["investment_mentioned"] = e["investment_mentioned"].fillna("N")

        e["contract_scale_score_norm"] = minmax_norm(e["contract_scale_score"])
        e["risk_language_score_norm"] = minmax_norm(e["risk_language_score"])
        e["no_alternative_flag"] = (e["no_alternative_mentioned"] == "Y").astype(float)
        e["investment_flag"] = (e["investment_mentioned"] == "Y").astype(float)

        e["edge_weight"] = (
            config.EW_CSS * e["contract_scale_score_norm"]
            + config.EW_RLS * e["risk_language_score_norm"]
            + config.EW_NOALT * e["no_alternative_flag"]
            + config.EW_INV * e["investment_flag"]
        ).clip(lower=config.EW_FLOOR)

        save_csv(e, config.WRITE_DIR / f"edges_weighted_{yr}.csv", dry_run)
        year_edges_weighted[yr] = e

        edge_names = pd.DataFrame({
            "src_name": e["risk_source_id"].map(names).fillna(e["risk_source_id"]).values,
            "tgt_name": e["risk_target_id"].map(names).fillna(e["risk_target_id"]).values,
        }, index=e.index)
        print_dist("edge_weight", yr, e["edge_weight"], is_edge=True, edge_names=edge_names)

    ctx["year_edges_weighted"] = year_edges_weighted
    return ctx
