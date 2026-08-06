"""
Output 1 — the per-firm risk card an underwriter actually reads.

A single systemic-exposure number is not underwritable. The card exists because
an underwriter has to be able to answer three questions in front of a client:
what is this score made of, which specific relationships drive it, and what
would the client have to change to lower it.

So each card carries the score *and* its provenance: the named upstream
dependencies with their weights, the dominant relation and spillover channel,
and a concrete next action at renewal. That is also the answer to the
explainability objection — a black-box score cannot be priced into a policy an
insurer has to justify to a regulator.

    python -m src.reporting.risk_profile
    python -m src.reporting.risk_profile --year 2025 --top 12

Writes `outputs/interpretation_report/risk_profiles_{year}.html` + `.csv`.
"""

from __future__ import annotations

import argparse
import html

import pandas as pd

from .. import config

# Score bands. Deliberately coarse: the model separates rank well (ρ≈0.85) but
# the absolute scale is a proxy label, not a loss estimate, so presenting more
# than three bands would imply precision the weak supervision cannot support.
BANDS = [(0.60, "HIGH"), (0.45, "MEDIUM"), (0.00, "LOW")]

ACTIONS = {
    "HIGH": ("Review coverage limit; require redundancy evidence; consider "
             "deductible/endorsement adjustment for AI and cloud dependency exposure."),
    "MEDIUM": ("Monitor dependency profile; request vendor resilience and BCP "
               "documentation at next renewal."),
    "LOW": "No additional action; re-screen at renewal.",
}


def band_for(score: float) -> str:
    return next(label for threshold, label in BANDS if score >= threshold)


def build_profiles(year: int, top: int = 6) -> pd.DataFrame:
    """One row per focal firm: score, band, upstream dependencies, exposure channels."""
    preds = pd.read_csv(config.PRED_DIR / f"prediction_{year}_gcn_source_firms.csv")
    edges = pd.read_csv(config.OUT_DIR / f"edges_weighted_{year}.csv")
    nodes = pd.read_csv(config.OUT_DIR / f"nodes_{year}.csv")
    names = nodes.set_index("canonical_firm_id")["canonical_name"].to_dict()

    rows = []
    for _, firm in preds.head(top).iterrows():
        fid = firm["canonical_firm_id"]
        # Edges point upstream-source → downstream-exposed, so this firm's
        # dependencies are the edges where it is the *target*.
        up = edges[edges["risk_target_id"] == fid].copy()
        up["upstream_name"] = up["risk_source_id"].map(names).fillna(up["risk_source_id"])
        up = up.sort_values("edge_weight", ascending=False)

        band = band_for(firm["gcn_score"])
        rows.append({
            "canonical_firm_id": fid,
            "canonical_name": firm["canonical_name"],
            "gcn_score": firm["gcn_score"],
            "rank_within_source_firms": firm["gcn_rank_within_type"],
            "rank_all_nodes": firm["gcn_rank_all"],
            "band": band,
            "n_upstream_edges": len(up),
            "top_dependencies": " | ".join(
                f"{r.upstream_name} {r.edge_weight:.2f}" for r in up.head(8).itertuples()
            ),
            "dominant_relation": (up["relation_type"].mode().iloc[0]
                                  if not up.empty and up["relation_type"].notna().any() else ""),
            "dominant_spillover": (up["spillover_type"].mode().iloc[0]
                                   if not up.empty and up["spillover_type"].notna().any() else ""),
            "avg_edge_weight": up["edge_weight"].mean() if not up.empty else 0.0,
            "max_edge_weight": up["edge_weight"].max() if not up.empty else 0.0,
            "underwriting_action": ACTIONS[band],
        })
    return pd.DataFrame(rows)


def render_html(profiles: pd.DataFrame, year: int) -> str:
    def esc(v):
        return html.escape(str(v))

    cards = []
    for r in profiles.itertuples():
        deps = "".join(
            f'<span class="dep">{esc(d.rsplit(" ", 1)[0])} '
            f'<b>{esc(d.rsplit(" ", 1)[1])}</b></span>'
            for d in (r.top_dependencies.split(" | ") if r.top_dependencies else [])
        )
        cards.append(f"""
        <div class="card">
          <div class="head">
            <div>
              <div class="name">{esc(r.canonical_name)}</div>
              <div class="rank">#{r.rank_within_source_firms} among source firms ·
                All-node rank #{r.rank_all_nodes}</div>
            </div>
            <div class="score">{r.gcn_score:.3f}<span class="max">/ 1.0</span>
              <div class="band band-{r.band.lower()}">{r.band}</div></div>
          </div>
          <div class="section">Top Upstream Dependencies ({r.n_upstream_edges} edges)</div>
          <div class="deps">{deps or '<span class="dep">none disclosed</span>'}</div>
          <div class="section">Exposure Channels</div>
          <div class="meta"><b>Relation:</b> {esc(r.dominant_relation)} &nbsp;|&nbsp;
            <b>Spillover:</b> {esc(r.dominant_spillover)}<br>
            <b>Avg edge weight:</b> {r.avg_edge_weight:.3f} &nbsp;&nbsp;
            <b>Max:</b> {r.max_edge_weight:.3f}</div>
          <div class="action"><b>Underwriting Action:</b> {esc(r.underwriting_action)}</div>
        </div>""")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Firm Risk Profiles — {year}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; background: #f5f5f5; color: #16181d; }}
  header {{ background: #101215; color: #fff; padding: 1.4rem 2rem; }}
  header h1 {{ margin: 0; font-size: 1.35rem; }}
  header .sub {{ color: #9aa0a6; font-size: .8rem; margin-top: .3rem; }}
  main {{ padding: 2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
           gap: 1.2rem; }}
  .card {{ background: #fff; border: 1px solid #e3e5e8; border-radius: 10px; padding: 1.2rem; }}
  .head {{ display: flex; justify-content: space-between; align-items: flex-start;
           border-bottom: 1px solid #eceef0; padding-bottom: .8rem; }}
  .name {{ font-size: 1.15rem; font-weight: 700; }}
  .rank {{ color: #8b9096; font-size: .72rem; margin-top: .2rem; }}
  .score {{ font-size: 1.7rem; font-weight: 800; text-align: right; line-height: 1; }}
  .score .max {{ font-size: .7rem; font-weight: 400; color: #8b9096; margin-left: .2rem; }}
  .band {{ display: inline-block; font-size: .62rem; font-weight: 700; letter-spacing: .07em;
           padding: .18rem .5rem; border-radius: 3px; color: #fff; margin-top: .4rem; }}
  .band-high {{ background: #101215; }} .band-medium {{ background: #8b9096; }}
  .band-low {{ background: #c9cdd2; }}
  .section {{ font-size: .78rem; font-weight: 700; margin: 1rem 0 .5rem; }}
  .deps {{ display: flex; flex-wrap: wrap; gap: .35rem; }}
  .dep {{ background: #f1f3f5; border-radius: 4px; padding: .22rem .5rem; font-size: .74rem; }}
  .meta {{ font-size: .78rem; line-height: 1.6; }}
  .action {{ background: #f8f9fa; border-radius: 6px; padding: .7rem .8rem;
             font-size: .75rem; margin-top: 1rem; line-height: 1.5; }}
</style></head><body>
<header>
  <h1>Individual Firm AI Supply Chain Risk Profiles — {year}</h1>
  <div class="sub">GCN-predicted systemic exposure scores ·
    source: edges_weighted_{year}, prediction_{year}_gcn_source_firms</div>
</header>
<main><div class="grid">{"".join(cards)}</div></main>
</body></html>"""


def run(year: int = config.PREDICT_YEAR, top: int = 6) -> pd.DataFrame:
    config.ensure_dirs()
    profiles = build_profiles(year, top)

    (config.REPORT_DIR / f"risk_profiles_{year}.html").write_text(
        render_html(profiles, year), encoding="utf-8")
    profiles.to_csv(config.REPORT_DIR / f"risk_profiles_{year}.csv", index=False)

    print("=" * 70)
    print(f"Firm risk profiles — {year}")
    print("=" * 70)
    for r in profiles.itertuples():
        print(f"  {r.rank_within_source_firms:>2}. {r.canonical_name:<24} "
              f"{r.gcn_score:.3f}  {r.band:<7} {r.n_upstream_edges} upstream edges")
    print(f"\n  → {config.REPORT_DIR}/risk_profiles_{year}.html")
    return profiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=config.PREDICT_YEAR)
    parser.add_argument("--top", type=int, default=6)
    args = parser.parse_args()
    run(year=args.year, top=args.top)
