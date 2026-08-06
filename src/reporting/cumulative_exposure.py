"""
Output 2 — provider-level accumulation: the CRO's view of the book.

Output 1 asks "how exposed is this firm?". This asks the inverse and more
dangerous question: "how many of our insured firms fail together?"

This is the catastrophe-accumulation calculation, with the accumulation unit
moved from geography to infrastructure. A cat underwriter asks how many policies
sit in one flood plain. Here we ask how many sit downstream of one cloud
provider — and unlike a flood plain, the exposure is invisible in the policy
documents, because every firm reports its dependency separately and nobody
aggregates them.

Providers are rolled up by *parent ecosystem*, not legal entity: AWS, Bedrock
and SageMaker fail together, so counting them as three independent providers
would understate the concentration exactly where it matters most.

    python -m src.reporting.cumulative_exposure --year 2025

Writes `outputs/interpretation_report/cumulative_exposure_{year}.html` + `.csv`.
"""

from __future__ import annotations

import argparse
import html

import pandas as pd

from .. import config

# Number of distinct focal firms downstream of one provider. Three or more
# firms failing on one provider event is a correlated loss, not a coincidence.
ACCUMULATION_BANDS = [(3, "HIGH"), (2, "MEDIUM"), (0, "LOW")]


def band_for(exposed_firms: int) -> str:
    return next(label for threshold, label in ACCUMULATION_BANDS if exposed_firms >= threshold)


def build_table(year: int) -> pd.DataFrame:
    edges = pd.read_csv(config.OUT_DIR / f"edges_weighted_{year}.csv")
    nodes = pd.read_csv(config.OUT_DIR / f"nodes_{year}.csv")
    preds = pd.read_csv(config.PRED_DIR / f"prediction_{year}_gcn_all_nodes.csv")

    names = nodes.set_index("canonical_firm_id")["canonical_name"].to_dict()
    node_meta = nodes.set_index("canonical_firm_id")
    score = preds.set_index("canonical_firm_id")["gcn_score"].to_dict()
    source_firms = set(nodes[nodes["node_type"] == "source_firm"]["canonical_firm_id"])

    # Only edges landing on a focal firm count: exposure the insurer could
    # actually be holding policies against.
    exposure = edges[edges["risk_target_id"].isin(source_firms)].copy()
    exposure["provider_name"] = exposure["risk_source_id"].map(names).fillna(
        exposure["risk_source_id"])
    exposure["exposed_firm"] = exposure["risk_target_id"].map(names).fillna(
        exposure["risk_target_id"])
    exposure["exposed_score"] = exposure["risk_target_id"].map(score)

    rows = []
    for provider_id, grp in exposure.groupby("risk_source_id"):
        firms = sorted(set(grp["exposed_firm"]))
        meta = node_meta.loc[provider_id] if provider_id in node_meta.index else {}
        rows.append({
            "provider_id": provider_id,
            "provider_name": grp["provider_name"].iloc[0],
            "provider_entity_type": meta.get("entity_type", ""),
            "parent_ecosystem": meta.get("parent_canonical_name", "") or "Independent",
            "exposed_source_firms": len(firms),
            "exposed_firms": " • ".join(firms),
            "avg_predicted_risk_score": grp["exposed_score"].mean(),
            "max_predicted_risk_score": grp["exposed_score"].max(),
            "mean_edge_weight": grp["edge_weight"].mean(),
            "max_edge_weight": grp["edge_weight"].max(),
            "dominant_relation": (grp["relation_type"].mode().iloc[0]
                                  if grp["relation_type"].notna().any() else ""),
            "dominant_spillover": (grp["spillover_type"].mode().iloc[0]
                                   if grp["spillover_type"].notna().any() else ""),
            "accumulation_level": band_for(len(firms)),
        })

    return (pd.DataFrame(rows)
            .sort_values(["exposed_source_firms", "mean_edge_weight"], ascending=False)
            .reset_index(drop=True))


def render_html(table: pd.DataFrame, year: int, top: int = 20) -> str:
    def esc(v):
        return html.escape(str(v))

    top_rows = table.head(top)
    peak = max(top_rows["exposed_source_firms"].max(), 1)

    bars = "".join(
        f'<div class="bar-row"><div class="bar-label">{esc(r.provider_name)}</div>'
        f'<div class="bar-track"><div class="bar bar-{r.accumulation_level.lower()}" '
        f'style="width:{r.exposed_source_firms / peak * 100:.1f}%"></div></div>'
        f'<div class="bar-val">{r.exposed_source_firms}</div></div>'
        for r in top_rows.head(10).itertuples()
    )

    rows = "".join(
        f"<tr><td><b>{esc(r.provider_name)}</b><div class='sub2'>"
        f"{esc(r.provider_entity_type)} · parent: {esc(r.parent_ecosystem)}</div></td>"
        f"<td>{r.exposed_source_firms}</td><td class='firms'>{esc(r.exposed_firms)}</td>"
        f"<td>{r.avg_predicted_risk_score:.3f}</td><td>{r.max_predicted_risk_score:.3f}</td>"
        f"<td>{r.mean_edge_weight:.3f}</td><td>{r.max_edge_weight:.3f}</td>"
        f"<td>{esc(r.dominant_relation)}</td><td>{esc(r.dominant_spillover)}</td>"
        f"<td><span class='lvl lvl-{r.accumulation_level.lower()}'>"
        f"{r.accumulation_level}</span></td></tr>"
        for r in top_rows.itertuples()
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Cumulative Exposure by Provider — {year}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; background: #f5f5f5; color: #16181d; }}
  header {{ background: #101215; color: #fff; padding: 1.4rem 2rem; }}
  header h1 {{ margin: 0; font-size: 1.35rem; }}
  header .sub {{ color: #9aa0a6; font-size: .8rem; margin-top: .3rem; }}
  main {{ padding: 2rem; }}
  .panel {{ background: #fff; border: 1px solid #e3e5e8; border-radius: 10px;
            padding: 1.5rem; margin-bottom: 1.5rem; }}
  .bar-row {{ display: flex; align-items: center; gap: .8rem; margin-bottom: .45rem; }}
  .bar-label {{ width: 190px; text-align: right; font-size: .8rem; }}
  .bar-track {{ flex: 1; background: #f1f3f5; border-radius: 3px; height: 18px; }}
  .bar {{ height: 18px; border-radius: 3px; }}
  .bar-high {{ background: #101215; }} .bar-medium {{ background: #8b9096; }}
  .bar-low {{ background: #c9cdd2; }}
  .bar-val {{ width: 26px; font-size: .8rem; font-weight: 700; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .8rem; }}
  th {{ background: #101215; color: #fff; text-align: left; padding: .6rem .7rem;
        font-size: .72rem; }}
  td {{ padding: .55rem .7rem; border-bottom: 1px solid #eef0f3; vertical-align: top; }}
  .sub2 {{ color: #8b9096; font-size: .68rem; margin-top: .15rem; }}
  .firms {{ color: #5a6068; max-width: 260px; }}
  .lvl {{ font-size: .62rem; font-weight: 700; letter-spacing: .06em; color: #fff;
          padding: .18rem .5rem; border-radius: 3px; }}
  .lvl-high {{ background: #101215; }} .lvl-medium {{ background: #8b9096; }}
  .lvl-low {{ background: #c9cdd2; }}
</style></head><body>
<header>
  <h1>Portfolio Cumulative Exposure by Provider — {year}</h1>
  <div class="sub">Upstream provider concentration · multi-firm exposure accumulation ·
    source: edges_weighted_{year}, prediction_{year}_gcn_all_nodes</div>
</header>
<main>
  <div class="panel"><h2 style="font-size:.95rem;margin-top:0">
    Top providers by exposed focal-firm count</h2>{bars}</div>
  <div class="panel"><table>
    <thead><tr><th>Provider / Ecosystem</th><th>Exposed<br>Firms</th><th>Exposed Firms</th>
      <th>Avg.<br>Score</th><th>Max<br>Score</th><th>Mean<br>Edge Wt</th><th>Max<br>Edge Wt</th>
      <th>Dominant<br>Relation</th><th>Dominant<br>Spillover</th>
      <th>Accumulation</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
</main></body></html>"""


def run(year: int = config.PREDICT_YEAR, top: int = 20) -> pd.DataFrame:
    config.ensure_dirs()
    table = build_table(year)

    (config.REPORT_DIR / f"cumulative_exposure_{year}.html").write_text(
        render_html(table, year, top), encoding="utf-8")
    table.to_csv(config.REPORT_DIR / f"cumulative_exposure_{year}.csv", index=False)

    print("=" * 70)
    print(f"Provider cumulative exposure — {year}")
    print("=" * 70)
    for r in table.head(10).itertuples():
        print(f"  {r.provider_name:<28} {r.exposed_source_firms} firms  "
              f"[{r.accumulation_level}]  mean edge wt {r.mean_edge_weight:.3f}")

    high = table[table["accumulation_level"] == "HIGH"]
    print(f"\n  {len(high)} provider(s) at HIGH accumulation "
          f"(3+ focal firms downstream of a single provider).")
    print(f"  → {config.REPORT_DIR}/cumulative_exposure_{year}.html")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=config.PREDICT_YEAR)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    run(year=args.year, top=args.top)
