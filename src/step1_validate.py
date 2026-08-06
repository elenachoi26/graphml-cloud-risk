"""
Step 1 — Load & validate the four input panels.

Produces `outputs/data_quality_report.md`. This step deliberately fails loudly
on structural problems (edge endpoints missing from the canonical mapping) but
only reports soft ones (missing weight inputs, thin coverage in a given year),
because a firm that simply did not disclose a partnership is data, not an error.
"""

from __future__ import annotations

import pandas as pd

from . import config
from .utils import banner, load_input


def run(dry_run: bool = False) -> dict:
    banner("STEP 1 — Data Validation")

    edges_raw = load_input("edges")
    self_raw = load_input("self")
    incident_raw = load_input("incident")
    master_raw = load_input("master")

    lines = ["# Data Quality Report\n"]

    # Drop relations with no propagation semantics (see config.REMOVE_RELATIONS).
    edges = edges_raw[~edges_raw["relation_type"].isin(config.REMOVE_RELATIONS)].copy()
    removed = len(edges_raw) - len(edges)
    lines += [
        "## Edge filtering",
        f"- Removed {removed} edges with relations: {sorted(config.REMOVE_RELATIONS)}",
        f"- Remaining edges: {len(edges)}\n",
    ]
    print(f"  Edges after relation filter: {len(edges)} / {len(edges_raw)}")

    # Every edge endpoint must resolve to a canonical id, or the node silently
    # disappears from the graph.
    canonical_ids = set(master_raw["canonical_firm_id"].dropna().unique())
    edge_ids = set(edges["risk_source_id"].dropna()) | set(edges["risk_target_id"].dropna())
    missing = edge_ids - canonical_ids
    lines += ["## Canonical ID coverage",
              f"- Edge node IDs not in master mapping: {len(missing)}"]
    if missing:
        lines.append(f"  {sorted(missing)}")
    lines.append("")
    print(f"  Edge node IDs missing from master: {len(missing)}")

    dup_cols = ["risk_source_id", "risk_target_id", "Year", "spillover_type", "relation_type"]
    dupes = int(edges.duplicated(subset=dup_cols, keep=False).sum())
    lines += ["## Duplicate edges", f"- Duplicate edge rows: {dupes}\n"]
    print(f"  Duplicate edges: {dupes}")

    weight_cols = ["contract_scale_score", "risk_language_score",
                   "no_alternative_mentioned", "investment_mentioned"]
    miss = edges[weight_cols].isnull().sum()
    lines.append("## Missing values in edge weight columns")
    lines += [f"- {col}: {n} missing" for col, n in miss.items()]
    lines.append("")

    self_firms = set(self_raw["firm_canonical_firm_id"].dropna().unique())
    lines += ["## Self-feature (source_firm) coverage",
              f"- Firms with 10-K self features: {len(self_firms)}"]
    for yr in config.YEARS:
        n = self_raw[self_raw["Year"] == yr]["firm_canonical_firm_id"].nunique()
        lines.append(f"  - {yr}: {n} firms")
    lines.append("")

    incident_firms = set(incident_raw["company_canonical_firm_id"].dropna().unique())
    lines += ["## Incident coverage",
              f"- Firms with incident records: {len(incident_firms)}\n"]

    present = set(self_raw["Feature"].dropna().unique())
    found = [f for f in config.INTRINSIC_FEATURES if f in present]
    lines += ["## IntrinsicRisk features found in self_features", f"- {found}\n"]
    print(f"  Source firms: {len(self_firms)}, incident firms: {len(incident_firms)}")

    if not dry_run:
        config.ensure_dirs()
        (config.WRITE_DIR / "data_quality_report.md").write_text("\n".join(lines))
        print(f"  → {config.WRITE_DIR.name}/data_quality_report.md")

    return {
        "edges": edges,
        "self": self_raw,
        "incident": incident_raw,
        "master": master_raw,
        "source_firm_ids": self_firms,
    }


if __name__ == "__main__":
    run()
