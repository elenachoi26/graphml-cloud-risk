"""
Step 2 — Per-year graph snapshots and node typing.

One snapshot per fiscal year, because dependency disclosure changes: a vendor
named in the 2023 10-K may be absent in 2024. Node ids are assigned once across
all years so a firm keeps the same index in every snapshot, which is what lets
the 2022–2024 model be applied to the 2025 graph.

Two node types:
  source_firm   — a firm whose own 10-K was parsed (has intrinsic features)
  target_entity — an entity only ever mentioned by someone else's 10-K
"""

from __future__ import annotations

import pandas as pd

from . import config
from .utils import attr_getter, banner, save_csv


def run(ctx: dict, dry_run: bool = False) -> dict:
    banner("STEP 2 — Year-specific Graph Snapshots")

    edges = ctx["edges"]
    master = ctx["master"]
    source_firm_ids = ctx["source_firm_ids"]

    # Stable node index across all years.
    all_ids = sorted(set(edges["risk_source_id"].dropna()) |
                     set(edges["risk_target_id"].dropna()))
    node_index = {nid: i for i, nid in enumerate(all_ids)}
    get_attr = attr_getter(master)

    year_nodes = {}
    year_edges = {}
    for yr in config.YEARS:
        yr_edges = edges[edges["Year"] == yr].copy()
        yr_ids = sorted(set(yr_edges["risk_source_id"].dropna()) |
                        set(yr_edges["risk_target_id"].dropna()))

        nodes_df = pd.DataFrame([{
            "node_id":               node_index[nid],
            "canonical_firm_id":     nid,
            "canonical_name":        get_attr(nid, "canonical_name"),
            "node_type":             "source_firm" if nid in source_firm_ids else "target_entity",
            "entity_type":           get_attr(nid, "entity_type"),
            "entity_level":          get_attr(nid, "entity_level"),
            "parent_firm_id":        get_attr(nid, "parent_firm_id"),
            "parent_canonical_name": get_attr(nid, "parent_canonical_name"),
            "year":                  yr,
        } for nid in yr_ids])

        save_csv(nodes_df, config.WRITE_DIR / f"nodes_{yr}.csv", dry_run)
        save_csv(yr_edges, config.WRITE_DIR / f"edges_{yr}_filtered.csv", dry_run)
        year_nodes[yr] = nodes_df
        year_edges[yr] = yr_edges

        n_src = int((nodes_df["node_type"] == "source_firm").sum())
        n_tgt = int((nodes_df["node_type"] == "target_entity").sum())
        print(f"  {yr}: {len(nodes_df)} nodes ({n_src} source_firm, {n_tgt} target_entity), "
              f"{len(yr_edges)} edges")

    ctx["node_index"] = node_index
    ctx["year_nodes"] = year_nodes
    ctx["year_edges"] = year_edges
    return ctx


if __name__ == "__main__":
    from .step1_validate import run as step1
    run(step1())
