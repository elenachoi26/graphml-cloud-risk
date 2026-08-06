"""
Step 5 — Decompose exposure into four components an underwriter can act on.

A single risk score is not usable in underwriting: an insurer that has to justify
a premium to a regulator and a client needs to say *why* the score is high, and
what would lower it. So exposure is computed as four separate quantities.

  DirectExposure     Σ over upstream providers of  edge_weight × their vulnerability
                     → "who you depend on, and how badly"

  CascadingExposure  the same, two hops out (your provider's provider)
                     → the risk that never appears in your own filing

  DependencyConcentration  Herfindahl index over parent ecosystems upstream
                     → single point of failure; 1.0 means everything routes
                       through one ecosystem

  StructuralNonSubstitutability  1 / (1 + distinct upstream ecosystems)
                     → recoverability. Three independent providers is a
                       migration path; one is an outage that lasts as long as
                       the provider says it does.

Concentration and substitutability are measured over parent *ecosystems*, not
legal entities — AWS, Bedrock and SageMaker are one failure domain.
"""

from __future__ import annotations

import pandas as pd

from . import config
from .utils import banner, master_lookup, parent_group, print_dist, save_csv

COMPONENT_COLS = list(config.COMPOSITE_WEIGHTS.keys())


def run(ctx: dict, dry_run: bool = False) -> dict:
    banner("STEP 5 — Risk Components")

    parent_lookup = master_lookup(ctx["master"])["parent_firm_id"].to_dict()

    year_components = {}
    for yr in config.YEARS:
        vuln_df = ctx["year_vulnerability"][yr]
        edges_w = ctx["year_edges_weighted"][yr]
        vuln = vuln_df.set_index("canonical_firm_id")["node_vulnerability"].to_dict()

        # preds[node] = [(upstream_id, edge_weight), ...]
        preds = {n: [] for n in vuln_df["canonical_firm_id"]}
        for _, row in edges_w.iterrows():
            if row["risk_target_id"] in preds:
                preds[row["risk_target_id"]].append(
                    (row["risk_source_id"], row["edge_weight"])
                )

        def direct(node_id):
            return sum(w * vuln.get(s, 0.0) for s, w in preds.get(node_id, []))

        def cascading(node_id):
            """2-hop: upstream vulnerability attenuated by both edge weights."""
            return sum(
                vuln.get(up, 0.0) * w_up * w_mid
                for mid, w_mid in preds.get(node_id, [])
                for up, w_up in preds.get(mid, [])
            )

        def concentration(node_id):
            up = preds.get(node_id, [])
            total = sum(w for _, w in up)
            if not up or total == 0:
                return 0.0
            groups: dict = {}
            for s, w in up:
                pg = parent_group(s, parent_lookup)
                groups[pg] = groups.get(pg, 0.0) + w
            return sum((gw / total) ** 2 for gw in groups.values())

        def non_substitutability(node_id):
            up = preds.get(node_id, [])
            if not up:
                return 0.0
            groups = {parent_group(s, parent_lookup) for s, _ in up}
            return 1.0 / (1.0 + len(groups))

        comp_df = pd.DataFrame([{
            "node_id": node["node_id"],
            "canonical_firm_id": node["canonical_firm_id"],
            "canonical_name": node["canonical_name"],
            "node_type": node["node_type"],
            "year": yr,
            "direct_exposure": direct(node["canonical_firm_id"]),
            "cascading_exposure": cascading(node["canonical_firm_id"]),
            "dependency_concentration": concentration(node["canonical_firm_id"]),
            "structural_non_substitutability": non_substitutability(node["canonical_firm_id"]),
        } for _, node in vuln_df.iterrows()])

        save_csv(comp_df, config.WRITE_DIR / f"risk_components_{yr}.csv", dry_run)
        year_components[yr] = comp_df

        name_map = comp_df["canonical_name"].to_dict()
        for col in COMPONENT_COLS:
            print_dist(col, yr, comp_df[col], name_map=name_map)

    ctx["year_components"] = year_components
    return ctx
