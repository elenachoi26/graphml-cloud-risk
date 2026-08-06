#!/usr/bin/env python3
"""
Generate the synthetic sample panels in `data/samples/`.

The real inputs are not distributed (see data/README.md), so these stand in for
them: same column names, same dtypes, same value vocabularies — entirely
invented firms. Enough for `run_pipeline.py` to execute end to end on a clean
clone, and enough to document the schema by example.

The fictional graph is deliberately shaped like the real one: two focal firms
sharing one upstream cloud provider, so the concentration and cascading terms
are non-zero and the pipeline exercises its interesting branches.

    python tools/make_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

YEARS = config.YEARS

# ── Fictional entities ────────────────────────────────────────────────────────
# FIRM_9001/9002 file their own "10-K" (focal firms); the rest are only ever
# mentioned by someone else, like real target entities.
FIRMS = {
    "FIRM_9001": ("Northwind Analytics", "company", "parent", "", ""),
    "FIRM_9002": ("Zephyr Data Systems", "company", "parent", "", ""),
    "FIRM_9003": ("Cirrus Cloud Platform", "cloud_provider", "subsidiary",
                  "Cirrus Holdings", "FIRM_9005"),
    "FIRM_9004": ("Lumen Model Labs", "ai_lab", "parent", "", ""),
    "FIRM_9005": ("Cirrus Holdings", "company", "parent", "", ""),
}
FOCAL = ["FIRM_9001", "FIRM_9002"]

# Both focal firms depend on the same cloud provider — that shared upstream is
# what creates accumulation, and it is the case the pipeline exists to detect.
EDGES = [
    # (source=upstream risk origin, target=downstream exposed, relation, spillover,
    #  risk_language, contract_scale, no_alt, investment)
    ("FIRM_9003", "FIRM_9001", "cloud_provider", "operational", 0.8, 0.7, "Y", "N"),
    ("FIRM_9003", "FIRM_9002", "cloud_provider", "operational", 0.6, 0.5, "Y", "N"),
    ("FIRM_9004", "FIRM_9001", "ai_provider", "governance", 0.7, 0.4, "N", "Y"),
    ("FIRM_9005", "FIRM_9003", "subsidiary", "operational", 0.3, 0.2, "N", "Y"),
    ("FIRM_9002", "FIRM_9001", "software_vendor", "performance", 0.4, 0.3, "N", "N"),
]

SELF_FEATURES = {
    "FIRM_9001": {"cybersecurity_risk": 0.7, "operational_continuity_risk": 0.8,
                  "ai_dependency_risk": 0.9, "regulatory_risk": 0.4,
                  "geopolitical_risk": 0.3, "energy_infra_risk": 0.2},
    "FIRM_9002": {"cybersecurity_risk": 0.5, "operational_continuity_risk": 0.4,
                  "ai_dependency_risk": 0.6, "regulatory_risk": 0.5,
                  "geopolitical_risk": 0.6, "energy_infra_risk": 0.3},
}

INCIDENTS = [
    # (firm, date, severity, reach, novelty)
    ("FIRM_9003", "2023-04-11", 3, 3, 2),
    ("FIRM_9003", "2024-02-02", 2, 2, 1),
    ("FIRM_9004", "2024-08-19", 2, 3, 3),
    ("FIRM_9001", "2023-11-07", 1, 1, 1),
    ("FIRM_9005", "2022-06-30", 2, 1, 1),
]

PLACEHOLDER = "[sample row - synthetic text, not from any filing]"


def make_master() -> pd.DataFrame:
    return pd.DataFrame([{
        "raw_node_id": f"RAW_{i:04d}",
        "raw_name": name,
        "normalized_key": name.lower(),
        "canonical_name": name,
        "canonical_firm_id": fid,
        "entity_level": level,
        "entity_type": etype,
        "parent_raw_name_hint": parent_name,
        "parent_canonical_name": parent_name,
        "parent_firm_id": parent_id,
        "is_parent_hint": "",
        "source_files": "edges_panel",
        "observed_columns": "Target 기업명",
        "years_observed": "; ".join(str(y) for y in YEARS),
        "industries_observed": "Software",
        "raw_occurrence_count": 4,
        "mapping_status": "mapped",
        "mapping_rule": "synthetic_sample",
        "confidence": "high",
        "needs_review": "",
        "notes": PLACEHOLDER,
    } for i, (fid, (name, etype, level, parent_name, parent_id))
        in enumerate(FIRMS.items(), start=1)])


def make_edges() -> pd.DataFrame:
    rows = []
    for year in YEARS:
        for src, tgt, relation, spillover, rls, css, no_alt, inv in EDGES:
            s_name, s_type, s_level, s_parent, s_parent_id = FIRMS[src]
            t_name, t_type, t_level, t_parent, t_parent_id = FIRMS[tgt]
            rows.append({
                "Source 기업명": t_name,   # the filer describing the relationship
                "Source 모기업명": t_parent,
                "Source_산업": "Software",
                "Target 기업명": s_name,   # the counterparty it names
                "Target_모기업": s_parent,
                "Target_산업": "Software",
                "Year": year,
                "relation_type": relation,
                "dependency_direction": "upstream",
                "risk_language_score": rls,
                "investment_mentioned": inv,
                "contract_scale_score": css,
                "no_alternative_mentioned": no_alt,
                "spillover_type": spillover,
                "source_section": "Item1A",
                "근거 문장": PLACEHOLDER,
                "source_canonical_name": t_name,
                "source_canonical_firm_id": tgt,
                "source_entity_level": t_level,
                "source_entity_type": t_type,
                "source_parent_canonical_name": t_parent,
                "source_parent_firm_id": t_parent_id,
                "source_mapping_found": True,
                "target_canonical_name": s_name,
                "target_canonical_firm_id": src,
                "target_entity_level": s_level,
                "target_entity_type": s_type,
                "target_parent_canonical_name": s_parent,
                "target_parent_firm_id": s_parent_id,
                "target_mapping_found": True,
                # risk_source → risk_target is the propagation direction the
                # pipeline consumes: upstream provider → exposed firm.
                "risk_source_id": src,
                "risk_source_name": s_name,
                "risk_target_id": tgt,
                "risk_target_name": t_name,
                "risk_direction_rule": "target_is_upstream_of_source",
                "risk_edge_valid": True,
                "is_reciprocal_expansion": False,
            })
    return pd.DataFrame(rows)


def make_self_features() -> pd.DataFrame:
    rows = []
    for year in YEARS:
        for fid in FOCAL:
            name = FIRMS[fid][0]
            for feature, value in SELF_FEATURES[fid].items():
                rows.append({
                    "Firm_Name": name,
                    "Year": float(year),
                    "Feature": feature,
                    "__": value,
                    "__ __ (__, 100_ __)": PLACEHOLDER,
                    "Feature Type": "Risk",
                    "firm_raw_name": name,
                    "firm_canonical_name": name,
                    "firm_canonical_firm_id": fid,
                    "firm_entity_level": "parent",
                    "firm_entity_type": "company",
                    "firm_parent_canonical_name": "",
                    "firm_parent_firm_id": "",
                    "firm_mapping_found": True,
                })
    return pd.DataFrame(rows)


def make_incidents() -> pd.DataFrame:
    return pd.DataFrame([{
        "incident_id": 90000 + i,
        "incident_date": date,
        "company_name": FIRMS[fid][0],
        "headquarters_country_isocode": "US",
        "related_countries": "US",
        "severity": sev,
        "reach": reach,
        "novelty": nov,
        "ai": "T",
        "cyberattack": "F",
        "energy_management": "F",
        "company_canonical_name": FIRMS[fid][0],
        "company_canonical_firm_id": fid,
        "company_entity_level": FIRMS[fid][2],
        "company_entity_type": FIRMS[fid][1],
        "company_parent_canonical_name": FIRMS[fid][3],
        "company_parent_firm_id": FIRMS[fid][4],
        "company_mapping_found": True,
    } for i, (fid, date, sev, reach, nov) in enumerate(INCIDENTS, start=1)])


def main() -> None:
    config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for key, builder in (("master", make_master), ("edges", make_edges),
                         ("self", make_self_features), ("incident", make_incidents)):
        df = builder()
        path = config.SAMPLE_DIR / config.INPUT_FILES[key]
        df.to_csv(path, index=False)
        print(f"  → {path.name}  ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
