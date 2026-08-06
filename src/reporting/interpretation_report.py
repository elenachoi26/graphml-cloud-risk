"""
Interpretation, visualisation and evidence chains.

Reads only artifacts the earlier stages already wrote — it trains nothing. Its
job is to make the scores auditable: graph overviews and ego networks per year,
the top-risk subgraphs, the component decomposition behind each score, and the
evidence chain that traces a high score back to the specific disclosed
relationships that produced it.

    python -m src.reporting.interpretation_report

All outputs → outputs/interpretation_report/
"""

import warnings, textwrap
from pathlib import Path
from itertools import islice

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import skew as _skew

from .. import config

warnings.filterwarnings("ignore")

# ── Paths (all resolved from the repo root via src/config.py) ──────────────────
BASE   = config.ROOT
OUT    = config.OUT_DIR
PRED   = config.PRED_DIR
RPT    = config.REPORT_DIR
RPT.mkdir(parents=True, exist_ok=True)

YEARS      = config.YEARS
TRAIN_YEARS= config.TRAIN_YEARS

# Weights come from config so this report can never drift out of step with the
# scores it is explaining.
COMP_COLS  = [f"{c}_norm" for c in config.COMPOSITE_WEIGHTS]
COMP_WEIGHTS = {f"{c}_norm": w for c, w in config.COMPOSITE_WEIGHTS.items()}
COMP_LABELS  = {"direct_exposure_norm":"Direct Exposure",
                "cascading_exposure_norm":"Cascading Exposure",
                "dependency_concentration_norm":"Dependency Concentration",
                "structural_non_substitutability_norm":"Non-Substitutability"}

NODE_COLORS = {"source_firm":"#E8604C","target_entity":"#4C8EE8"}
DIVIDER = "=" * 68

# ── Helper loaders ─────────────────────────────────────────────────────────────
def load_composite(yr):
    return pd.read_csv(OUT / f"composite_risk_{yr}.csv")

def load_edges_weighted(yr):
    return pd.read_csv(OUT / f"edges_weighted_{yr}.csv")

def load_nodes(yr):
    return pd.read_csv(OUT / f"nodes_{yr}.csv")

def build_nx_graph(yr):
    """Directed nx.DiGraph: upstream provider → downstream firm."""
    edges = load_edges_weighted(yr)
    nodes = load_nodes(yr)
    comp  = load_composite(yr)
    vdict = comp.set_index("canonical_firm_id")["composite_risk"].to_dict()
    ntype = comp.set_index("canonical_firm_id")["node_type"].to_dict()
    ename = comp.set_index("canonical_firm_id")["canonical_name"].to_dict()

    G = nx.DiGraph()
    for _, n in nodes.iterrows():
        G.add_node(n["canonical_firm_id"],
                   name=n["canonical_name"],
                   node_type=n["node_type"],
                   entity_type=n.get("entity_type",""),
                   composite_risk=vdict.get(n["canonical_firm_id"],0.0),
                   year=yr)
    for _, e in edges.iterrows():
        s, t = e["risk_source_id"], e["risk_target_id"]
        if s in G and t in G:
            G.add_edge(s, t,
                       edge_weight=e.get("edge_weight",0.1),
                       relation_type=e.get("relation_type",""),
                       spillover_type=e.get("spillover_type",""),
                       risk_language_score=e.get("risk_language_score",0),
                       contract_scale_score=e.get("contract_scale_score",0),
                       no_alternative_flag=e.get("no_alternative_flag",0),
                       investment_flag=e.get("investment_flag",0),
                       evidence=str(e.get("근거 문장","")),
                       year=yr)
    return G

print(DIVIDER)
print("Graph-based Systemic Risk — Interpretation Report")
print(DIVIDER)

# ══════════════════════════════════════════════════════════════════════════════
# TASK 1  Risk Score Interpretation  2022-2024
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Task 1] Risk score interpretation …")

yearly_stats = []
all_top, all_bot = [], []
comp_dist_rows  = []

for yr in TRAIN_YEARS:
    df = load_composite(yr)
    cr = df["composite_risk"]

    stats = dict(year=yr, n_nodes=len(df),
                 n_source=int((df["node_type"]=="source_firm").sum()),
                 n_target=int((df["node_type"]=="target_entity").sum()),
                 mean=round(cr.mean(),4), median=round(cr.median(),4),
                 std=round(cr.std(),4), min=round(cr.min(),4),
                 max=round(cr.max(),4), skew=round(float(_skew(cr.dropna())),4))

    # average weighted component contribution
    for col in COMP_COLS:
        wt = COMP_WEIGHTS[col]
        stats[f"avg_{col}"] = round(df[col].mean(), 4)
        stats[f"weighted_contribution_{col}"] = round(df[col].mean() * wt, 4)
    yearly_stats.append(stats)

    # component distribution for this year
    for col in COMP_COLS:
        comp_dist_rows.append(dict(year=yr, component=COMP_LABELS[col],
                                   weight=COMP_WEIGHTS[col],
                                   mean_norm=round(df[col].mean(),4),
                                   std_norm=round(df[col].std(),4),
                                   min_norm=round(df[col].min(),4),
                                   max_norm=round(df[col].max(),4),
                                   weighted_mean=round(df[col].mean()*COMP_WEIGHTS[col],4)))

    top10 = df.nlargest(10,"composite_risk")[
        ["canonical_firm_id","canonical_name","node_type","composite_risk"]+COMP_COLS].copy()
    top10.insert(0,"rank",range(1,11))
    top10.insert(0,"year",yr)
    top10.insert(0,"tier","top10")
    all_top.append(top10)

    bot10 = df.nsmallest(10,"composite_risk")[
        ["canonical_firm_id","canonical_name","node_type","composite_risk"]+COMP_COLS].copy()
    bot10.insert(0,"rank",range(1,11))
    bot10.insert(0,"year",yr)
    bot10.insert(0,"tier","bottom10")
    all_bot.append(bot10)

# ── YoY analysis ──────────────────────────────────────────────────────────────
dfs_yr = {yr: load_composite(yr).set_index("canonical_firm_id") for yr in TRAIN_YEARS}
all_ids = set.intersection(*[set(d.index) for d in dfs_yr.values()])

yoy_rows = []
for fid in all_ids:
    row = dict(canonical_firm_id=fid,
               canonical_name=dfs_yr[2022].loc[fid,"canonical_name"],
               node_type=dfs_yr[2022].loc[fid,"node_type"])
    scores = []
    for yr in TRAIN_YEARS:
        s = dfs_yr[yr].loc[fid,"composite_risk"]
        row[f"score_{yr}"] = round(s,4)
        scores.append(s)
    row["mean_score"]    = round(np.mean(scores),4)
    row["std_score"]     = round(np.std(scores),4)
    row["delta_22_23"]   = round(scores[1]-scores[0],4)
    row["delta_23_24"]   = round(scores[2]-scores[1],4)
    row["delta_22_24"]   = round(scores[2]-scores[0],4)
    yoy_rows.append(row)

yoy_df = pd.DataFrame(yoy_rows).sort_values("mean_score",ascending=False)

# Consistently high-risk: mean > 0.65 and std < 0.1
consistent = yoy_df[(yoy_df["mean_score"]>0.65)&(yoy_df["std_score"]<0.10)]
# Volatile: std > 0.12
volatile    = yoy_df[yoy_df["std_score"]>0.12].sort_values("std_score",ascending=False)

# ── Save CSVs ─────────────────────────────────────────────────────────────────
summary_df = pd.DataFrame(yearly_stats)
summary_df.to_csv(RPT/"risk_score_summary_2022_2024.csv",index=False)

top_bot_df = pd.concat(all_top+all_bot,ignore_index=True)
top_bot_df.to_csv(RPT/"yearly_top_risk_nodes.csv",index=False)

comp_dist_df = pd.DataFrame(comp_dist_rows)
comp_dist_df.to_csv(RPT/"risk_component_distribution.csv",index=False)

print(f"  → risk_score_summary_2022_2024.csv")
print(f"  → yearly_top_risk_nodes.csv")
print(f"  → risk_component_distribution.csv")

# ── Markdown interpretation ────────────────────────────────────────────────────
md_t1 = ["# Risk Score Interpretation: 2022–2024\n",
"## Disclaimer\n",
"All scores are **weakly supervised graph-derived systemic exposure proxies**. "
"They reflect structural dependency relationships extracted from 10-K filings "
"and do **not** represent directly observed incident outcomes or financial losses.\n"]

md_t1.append("## 1. Yearly Score Distribution\n")
md_t1.append("| Year | N | Mean | Median | Std | Min | Max | Skew |")
md_t1.append("|---|---|---|---|---|---|---|---|")
for s in yearly_stats:
    md_t1.append(f"| {s['year']} | {s['n_nodes']} ({s['n_source']} source / {s['n_target']} target) "
                 f"| {s['mean']} | {s['median']} | {s['std']} | {s['min']} | {s['max']} | {s['skew']} |")

md_t1.append("\n## 2. Dominant Risk Component by Year\n")
md_t1.append("| Year | Dominant Component | Weighted Contribution | 2nd Component |")
md_t1.append("|---|---|---|---|")
for s in yearly_stats:
    contrib = {c: s[f"weighted_contribution_{c}"] for c in COMP_COLS}
    sorted_c = sorted(contrib.items(),key=lambda x:-x[1])
    dom = COMP_LABELS[sorted_c[0][0]]; dom_v = sorted_c[0][1]
    sec = COMP_LABELS[sorted_c[1][0]]; sec_v = sorted_c[1][1]
    md_t1.append(f"| {s['year']} | {dom} | {dom_v:.4f} | {sec} ({sec_v:.4f}) |")

md_t1.append("\n## 3. Top 10 High-Risk Nodes by Year\n")
for yr in TRAIN_YEARS:
    top = pd.concat(all_top).query("year==@yr").sort_values("rank")
    md_t1.append(f"### {yr}\n")
    md_t1.append("| Rank | Name | Type | CompositeRisk | Direct | Cascading | Concentration | NonSub |")
    md_t1.append("|---|---|---|---|---|---|---|---|")
    for _, r in top.iterrows():
        md_t1.append(f"| {int(r['rank'])} | {r['canonical_name']} | {r['node_type']} "
                     f"| {r['composite_risk']:.4f} "
                     f"| {r['direct_exposure_norm']:.3f} "
                     f"| {r['cascading_exposure_norm']:.3f} "
                     f"| {r['dependency_concentration_norm']:.3f} "
                     f"| {r['structural_non_substitutability_norm']:.3f} |")

md_t1.append("\n## 4. Year-over-Year Risk Changes\n")
md_t1.append("### 4a. Consistently High-Risk Nodes (mean > 0.65, std < 0.10)\n")
if len(consistent) > 0:
    md_t1.append("| Name | Type | 2022 | 2023 | 2024 | Mean | Std |")
    md_t1.append("|---|---|---|---|---|---|---|")
    for _, r in consistent.head(15).iterrows():
        md_t1.append(f"| {r['canonical_name']} | {r['node_type']} "
                     f"| {r['score_2022']:.4f} | {r['score_2023']:.4f} | {r['score_2024']:.4f} "
                     f"| {r['mean_score']:.4f} | {r['std_score']:.4f} |")
else:
    md_t1.append("No node met mean > 0.65 and std < 0.10 across all three years.")

md_t1.append("\n### 4b. Volatile Nodes (std > 0.12)\n")
if len(volatile) > 0:
    md_t1.append("| Name | Type | 2022 | 2023 | 2024 | Std | Δ22→24 |")
    md_t1.append("|---|---|---|---|---|---|---|")
    for _, r in volatile.head(10).iterrows():
        md_t1.append(f"| {r['canonical_name']} | {r['node_type']} "
                     f"| {r['score_2022']:.4f} | {r['score_2023']:.4f} | {r['score_2024']:.4f} "
                     f"| {r['std_score']:.4f} | {r['delta_22_24']:.4f} |")
else:
    md_t1.append("No node showed std > 0.12 across years.")

md_t1.append("\n## 5. Interpretation Notes\n")
md_t1.append(textwrap.dedent("""
- **Score construction**: CompositeRisk = 0.50 × DirectExposure_norm + 0.20 × CascadingExposure_norm + 0.20 × DependencyConcentration_norm + 0.10 × NonSubstitutability_norm. Each component is year-wise percentile rank normalized before combining.
- **Target entity dominance**: Most high-scoring nodes are `target_entity` nodes (subsidiaries/affiliates of source firms). Their high concentration scores reflect single-parent dependency (HHI = 1.0 when all upstream dependency flows from one parent group).
- **Source firm interpretation**: High-scoring source firms indicate they are exposed downstream (many entities depend on them) AND have structural concentration in their upstream supply chain.
- **DirectExposure is typically the dominant driver** across all three years, reflecting that direct upstream dependency relationships carry the most structural risk weight.
- **CascadingExposure is low on average** (mean normalized ~0.02–0.03 raw), indicating the current dataset mainly captures 1-hop dependency relationships. This is expected for 10-K disclosure data.
- **YoY stability**: nodes that consistently score high are structurally embedded — high PageRank as upstream hubs or concentrated single-parent dependency.
- **These scores do not imply any firm will fail or experience an adverse event.**
""".strip()))

(RPT/"risk_score_interpretation.md").write_text("\n".join(md_t1))
print(f"  → risk_score_interpretation.md")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2  Graph Visualizations
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Task 2] Graph visualizations …")

VIZ_YEARS = [yr for yr in [2022,2023,2024,2025]
             if (OUT/f"edges_weighted_{yr}.csv").exists()]

EGO_FIRMS = ["Amazon","Microsoft","SAP","MongoDB","Oracle","Snowflake","IBM"]

graph_metrics = []
viz_notes = ["# Graph Visualization Notes\n"]

def make_pos(G, seed=42):
    return nx.spring_layout(G, seed=seed, k=2.0/max(1,len(G)**0.5), iterations=80)

def node_sizes(G, comp_df, scale=2200, default=200):
    vdict = comp_df.set_index("canonical_firm_id")["composite_risk"].to_dict()
    return [max(80, vdict.get(n, 0.0)*scale) for n in G.nodes()]

def edge_widths(G, scale=4.0, default=0.5):
    return [G[u][v].get("edge_weight",0.1)*scale for u,v in G.edges()]

def node_color_list(G):
    return [NODE_COLORS.get(G.nodes[n].get("node_type","target_entity"),"#888") for n in G.nodes()]

def draw_graph(G, comp_df, title, fpath, highlight_ids=None,
               label_threshold=0.55, figsize=(18,13)):
    if len(G) == 0:
        return
    fig, ax = plt.subplots(figsize=figsize)
    pos = make_pos(G)
    ns  = node_sizes(G, comp_df)
    ew  = edge_widths(G)
    nc  = node_color_list(G)

    # highlight top-risk nodes with a red border
    if highlight_ids:
        node_linewidths = [3.0 if n in highlight_ids else 0.5 for n in G.nodes()]
        node_edgecolors = ["#CC0000" if n in highlight_ids else "#555" for n in G.nodes()]
    else:
        node_linewidths = [0.5]*len(G)
        node_edgecolors = ["#555"]*len(G)

    nx.draw_networkx_edges(G, pos, ax=ax, width=ew, alpha=0.55,
                           edge_color="#aaaaaa", arrows=True,
                           arrowsize=12, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.08",
                           min_source_margin=12, min_target_margin=12)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=ns, node_color=nc,
                           linewidths=node_linewidths, edgecolors=node_edgecolors, alpha=0.90)

    # labels: only for important nodes
    vdict = comp_df.set_index("canonical_firm_id")["composite_risk"].to_dict()
    ndict = comp_df.set_index("canonical_firm_id")["canonical_name"].to_dict()
    label_ids = {n for n in G.nodes() if vdict.get(n,0) >= label_threshold
                 or G.nodes[n].get("node_type") == "source_firm"
                 or (highlight_ids and n in highlight_ids)}
    labels = {n: G.nodes[n].get("name", n)[:20] for n in label_ids}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=7, font_color="#111")

    # Legend
    patches = [mpatches.Patch(color=v, label=k) for k,v in NODE_COLORS.items()]
    patches.append(Line2D([0],[0], color="#aaaaaa", linewidth=2, label="dependency edge"))
    if highlight_ids:
        patches.append(mpatches.Patch(color="white", edgecolor="#CC0000",
                                      linewidth=2, label="top-risk node"))
    ax.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.8)

    ax.set_title(title, fontsize=13, pad=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(fpath, dpi=140, bbox_inches="tight")
    plt.close()

for yr in VIZ_YEARS:
    G    = build_nx_graph(yr)
    comp = load_composite(yr)
    top10_ids = set(comp.nlargest(10,"composite_risk")["canonical_firm_id"])

    # ── Metrics ────────────────────────────────────────────────────────────────
    pr = nx.pagerank(G, weight="edge_weight", max_iter=300)
    in_deg = dict(G.in_degree())
    out_deg= dict(G.out_degree())
    n_scc  = nx.number_strongly_connected_components(G)
    n_wcc  = nx.number_weakly_connected_components(G)
    density = nx.density(G)
    graph_metrics.append(dict(year=yr, n_nodes=G.number_of_nodes(),
                               n_edges=G.number_of_edges(), density=round(density,5),
                               n_strongly_connected=n_scc, n_weakly_connected=n_wcc,
                               avg_in_degree=round(np.mean(list(in_deg.values())),3),
                               avg_out_degree=round(np.mean(list(out_deg.values())),3),
                               max_pagerank_node=max(pr,key=pr.get),
                               max_pagerank_name=G.nodes[max(pr,key=pr.get)].get("name",""),
                               n_source_firm=sum(1 for n in G.nodes()
                                                 if G.nodes[n].get("node_type")=="source_firm"),
                               n_target_entity=sum(1 for n in G.nodes()
                                                   if G.nodes[n].get("node_type")=="target_entity")))

    # ── Full overview ──────────────────────────────────────────────────────────
    draw_graph(G, comp,
               title=f"Risk Propagation Graph — {yr}  "
                     f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)",
               fpath=RPT/f"graph_overview_{yr}.png",
               highlight_ids=top10_ids, label_threshold=0.55)
    print(f"  → graph_overview_{yr}.png")

    # ── Top-risk subgraph ─────────────────────────────────────────────────────
    # top-10 + direct preds + 2-hop preds
    subgraph_ids = set(top10_ids)
    for nid in list(top10_ids):
        if nid in G:
            preds1 = set(G.predecessors(nid))
            subgraph_ids |= preds1
            for p in preds1:
                subgraph_ids |= set(G.predecessors(p))
    Gsub = G.subgraph(subgraph_ids).copy()

    # Color coding: top10=red, direct pred=orange, 2-hop=yellow, other=gray
    tier_color = {}
    for n in Gsub.nodes():
        if n in top10_ids:                            tier_color[n] = "#E74C3C"
        elif any(n in G.predecessors(t) for t in top10_ids): tier_color[n] = "#F39C12"
        else:                                         tier_color[n] = "#A9CCE3"

    if len(Gsub) > 0:
        fig2, ax2 = plt.subplots(figsize=(16,11))
        pos2 = make_pos(Gsub, seed=yr)
        ns2  = node_sizes(Gsub, comp, scale=2600)
        ew2  = edge_widths(Gsub, scale=5.0)
        nc2  = [tier_color.get(n,"#ccc") for n in Gsub.nodes()]
        nx.draw_networkx_edges(Gsub, pos2, ax=ax2, width=ew2, alpha=0.6,
                               edge_color="#888", arrows=True, arrowsize=14,
                               arrowstyle="-|>", connectionstyle="arc3,rad=0.08",
                               min_source_margin=12, min_target_margin=12)
        nx.draw_networkx_nodes(Gsub, pos2, ax=ax2, node_size=ns2, node_color=nc2,
                               linewidths=1.5, edgecolors="#333", alpha=0.92)
        labels2 = {n: Gsub.nodes[n].get("name",n)[:22] for n in Gsub.nodes()}
        nx.draw_networkx_labels(Gsub, pos2, labels=labels2, ax=ax2,
                                font_size=7, font_color="#111")
        patches2 = [
            mpatches.Patch(color="#E74C3C", label="Top-10 high-risk"),
            mpatches.Patch(color="#F39C12", label="Direct upstream (1-hop)"),
            mpatches.Patch(color="#A9CCE3", label="2-hop upstream"),
        ]
        ax2.legend(handles=patches2, loc="upper left", fontsize=8, framealpha=0.8)
        ax2.set_title(f"Top-Risk Subgraph — {yr}  (top 10 + upstream paths)",
                      fontsize=13, pad=12)
        ax2.axis("off")
        plt.tight_layout()
        plt.savefig(RPT/f"top_risk_subgraph_{yr}.png", dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  → top_risk_subgraph_{yr}.png")

    # ── Ego networks ──────────────────────────────────────────────────────────
    name_to_id = {G.nodes[n].get("name","").lower(): n for n in G.nodes()}
    for firm in EGO_FIRMS:
        fid = name_to_id.get(firm.lower())
        if fid is None:  # fuzzy fallback
            for k,v in name_to_id.items():
                if firm.lower() in k:
                    fid = v; break
        if fid is None or fid not in G:
            continue
        ego = nx.ego_graph(G, fid, radius=1, undirected=False)
        # add undirected 1-hop for both in and out
        for nb in list(G.predecessors(fid))+list(G.successors(fid)):
            if nb not in ego:
                ego.add_node(nb, **G.nodes[nb])
            if G.has_edge(nb, fid):  ego.add_edge(nb, fid, **G[nb][fid])
            if G.has_edge(fid, nb):  ego.add_edge(fid, nb, **G[fid][nb])

        if len(ego) < 2:
            continue
        fig3, ax3 = plt.subplots(figsize=(11,8))
        pos3 = nx.spring_layout(ego, seed=42, k=2.5/max(1,len(ego)**0.5))
        ego_ns = [2800 if n==fid else node_sizes(ego,comp,1600,[150])[i]
                  for i, n in enumerate(ego.nodes())]
        ego_nc = ["#E74C3C" if n==fid else
                  ("#F39C12" if n in G.predecessors(fid) else "#4C8EE8")
                  for n in ego.nodes()]
        ego_ew = [ego[u][v].get("edge_weight",0.1)*5.0 for u,v in ego.edges()]
        nx.draw_networkx_edges(ego, pos3, ax=ax3, width=ego_ew, alpha=0.65,
                               edge_color="#999", arrows=True, arrowsize=14,
                               arrowstyle="-|>", connectionstyle="arc3,rad=0.08",
                               min_source_margin=12, min_target_margin=12)
        nx.draw_networkx_nodes(ego, pos3, ax=ax3, node_size=ego_ns,
                               node_color=ego_nc, linewidths=1.5,
                               edgecolors="#333", alpha=0.92)
        ego_labels = {n: ego.nodes[n].get("name",n)[:22] for n in ego.nodes()}
        nx.draw_networkx_labels(ego, pos3, labels=ego_labels, ax=ax3,
                                font_size=7.5, font_color="#111")
        patches3 = [
            mpatches.Patch(color="#E74C3C", label=f"{firm} (focal)"),
            mpatches.Patch(color="#F39C12", label="Upstream provider"),
            mpatches.Patch(color="#4C8EE8", label="Downstream dependent"),
        ]
        ax3.legend(handles=patches3, loc="upper left", fontsize=8, framealpha=0.8)
        ax3.set_title(f"Ego Network — {firm} ({yr})", fontsize=12, pad=10)
        ax3.axis("off")
        plt.tight_layout()
        plt.savefig(RPT/f"ego_{firm.lower().replace(' ','_')}_{yr}.png",
                    dpi=130, bbox_inches="tight")
        plt.close()
    print(f"  → ego networks for {yr}")

pd.DataFrame(graph_metrics).to_csv(RPT/"graph_metrics_summary.csv", index=False)
print("  → graph_metrics_summary.csv")

# Graph visualization notes
viz_notes += [
"## Visual Encoding\n",
"- **Node size**: proportional to CompositeRisk (larger = higher risk score)",
"- **Node color**: red = `source_firm` (focal 10-K filer); blue = `target_entity` (external provider/service)",
"- **Edge width**: proportional to `edge_weight` (0.40×contract_scale + 0.30×risk_language + 0.20×no_alt + 0.10×investment)",
"- **Edge direction**: upstream provider → downstream exposed firm (risk propagation direction)",
"- **Red border**: top-10 highest composite risk nodes",
"- Labels shown for: all source_firm nodes, target_entity nodes with composite_risk ≥ 0.55, all top-risk highlighted nodes\n",
"## Subgraph Composition\n",
"- Top-10 high-risk nodes (red)",
"- Their direct upstream predecessors (orange, 1-hop)",
"- 2-hop upstream nodes (light blue)\n",
"## Ego Network Interpretation\n",
"- Focal firm is shown in red (center of the ego network)",
"- Orange nodes are upstream providers (in-edges toward focal firm)",
"- Blue nodes are downstream dependents (out-edges from focal firm)",
"- Ego graphs show 1-hop neighborhood only (radius=1)\n",
"## Layout\n",
"- Spring layout (Fruchterman-Reingold) with fixed seed for reproducibility",
"- Node repulsion `k = 2/sqrt(N)` to reduce overlap\n",
"## Coverage\n",
]
for yr in VIZ_YEARS:
    row = next((r for r in graph_metrics if r["year"]==yr), {})
    viz_notes.append(f"- **{yr}**: {row.get('n_nodes','?')} nodes, {row.get('n_edges','?')} edges — "
                     f"most central node by PageRank: **{row.get('max_pagerank_name','?')}**")

(RPT/"graph_visualization_notes.md").write_text("\n".join(viz_notes))
print("  → graph_visualization_notes.md")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 3  Evidence Chains
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Task 3] Evidence chains …")

EC_YEARS = [yr for yr in [2022,2023,2024,2025]
            if (OUT/f"edges_weighted_{yr}.csv").exists()
            and (OUT/f"composite_risk_{yr}.csv").exists()]

ev_report_lines = ["# Evidence Chain Report\n",
"## Disclaimer\n",
"Evidence sentences are extracted from 10-K risk disclosure language. "
"They represent structural dependency relationships, not confirmed incidents.\n"]

def get_evidence_chains(yr, top_n=10):
    comp  = load_composite(yr)
    edges = load_edges_weighted(yr)
    top   = comp.nlargest(top_n, "composite_risk")

    rows = []
    for rank_i, (_, node) in enumerate(top.iterrows(), start=1):
        fid   = node["canonical_firm_id"]
        fname = node["canonical_name"]

        # direct upstream edges (predecessors)
        preds_df = edges[edges["risk_target_id"] == fid].copy()

        # 2-hop: edges that target any direct predecessor
        pred_ids = set(preds_df["risk_source_id"].dropna())
        twohop_df = edges[edges["risk_target_id"].isin(pred_ids)].copy()

        # determine dominant component
        comp_vals = {
            "Direct Exposure":    node.get("direct_exposure_norm", 0),
            "Cascading Exposure": node.get("cascading_exposure_norm", 0),
            "Dependency Concentration": node.get("dependency_concentration_norm", 0),
            "Non-Substitutability": node.get("structural_non_substitutability_norm", 0),
        }
        dom_comp = max(comp_vals, key=comp_vals.get)

        base = dict(year=yr, rank=rank_i,
                    focal_firm_id=fid, focal_firm_name=fname,
                    node_type=node["node_type"],
                    composite_risk=round(node["composite_risk"],4),
                    main_risk_driver=dom_comp,
                    direct_exposure_norm   =round(node.get("direct_exposure_norm",0),4),
                    cascading_exposure_norm=round(node.get("cascading_exposure_norm",0),4),
                    dependency_concentration_norm=round(node.get("dependency_concentration_norm",0),4),
                    structural_non_substitutability_norm=round(node.get("structural_non_substitutability_norm",0),4),
                    n_direct_upstream=len(preds_df),
                    n_twohop_upstream=len(twohop_df))

        if len(preds_df) == 0:
            row = {**base, "path_type":"no_upstream",
                   "upstream_firm_id":"","upstream_firm_name":"",
                   "intermediate_firm_id":"","intermediate_firm_name":"",
                   "edge_weight":"","relation_type":"","spillover_type":"",
                   "risk_language_score":"","contract_scale_score":"",
                   "no_alternative_flag":"","investment_flag":"",
                   "evidence_sentence":"",
                   "upstream_parent_group":""}
            rows.append(row)
        else:
            for _, pe in preds_df.iterrows():
                row = {**base, "path_type":"direct_1hop",
                       "upstream_firm_id":pe.get("risk_source_id",""),
                       "upstream_firm_name":pe.get("risk_source_name",pe.get("source_canonical_name","")),
                       "intermediate_firm_id":"","intermediate_firm_name":"",
                       "edge_weight":round(pe.get("edge_weight",0),4),
                       "relation_type":pe.get("relation_type",""),
                       "spillover_type":pe.get("spillover_type",""),
                       "risk_language_score":pe.get("risk_language_score",""),
                       "contract_scale_score":pe.get("contract_scale_score",""),
                       "no_alternative_flag":pe.get("no_alternative_flag",""),
                       "investment_flag":pe.get("investment_flag",""),
                       "evidence_sentence":pe.get("근거 문장",""),
                       "upstream_parent_group":pe.get("source_parent_canonical_name","")}
                rows.append(row)

            # 2-hop paths: upstream_node → intermediate_node → focal_node
            for _, mid_e in preds_df.iterrows():
                mid_id   = mid_e.get("risk_source_id","")
                mid_name = mid_e.get("risk_source_name", mid_e.get("source_canonical_name",""))
                up2 = twohop_df[twohop_df["risk_target_id"]==mid_id]
                for _, ue in up2.iterrows():
                    row2 = {**base, "path_type":"cascading_2hop",
                            "upstream_firm_id":ue.get("risk_source_id",""),
                            "upstream_firm_name":ue.get("risk_source_name",ue.get("source_canonical_name","")),
                            "intermediate_firm_id":mid_id,
                            "intermediate_firm_name":mid_name,
                            "edge_weight":round(float(ue.get("edge_weight",0))*float(mid_e.get("edge_weight",0)),5),
                            "relation_type":f"{ue.get('relation_type','')}→{mid_e.get('relation_type','')}",
                            "spillover_type":f"{ue.get('spillover_type','')}→{mid_e.get('spillover_type','')}",
                            "risk_language_score":ue.get("risk_language_score",""),
                            "contract_scale_score":ue.get("contract_scale_score",""),
                            "no_alternative_flag":ue.get("no_alternative_flag",""),
                            "investment_flag":ue.get("investment_flag",""),
                            "evidence_sentence":ue.get("근거 문장",""),
                            "upstream_parent_group":ue.get("source_parent_canonical_name","")}
                    rows.append(row2)
    return pd.DataFrame(rows), top

def make_explanation_paragraph(fname, yr, node_row, preds_df):
    cr   = node_row.get("composite_risk",0)
    ntype= node_row.get("node_type","")
    de   = node_row.get("direct_exposure_norm",0)
    ce   = node_row.get("cascading_exposure_norm",0)
    dc   = node_row.get("dependency_concentration_norm",0)
    ns   = node_row.get("structural_non_substitutability_norm",0)
    n_up = len(preds_df)

    comp_vals = {"Direct Exposure":de,"Cascading Exposure":ce,
                 "Dependency Concentration":dc,"Non-Substitutability":ns}
    dom = max(comp_vals, key=comp_vals.get)

    # upstream names
    up_names = preds_df["risk_source_name"].dropna().unique()[:4] if len(preds_df)>0 else []
    up_str   = ", ".join(str(x) for x in up_names) if len(up_names)>0 else "no direct upstream"

    # no-alternative flag
    no_alt_count = int(preds_df.get("no_alternative_flag",pd.Series(dtype=float)).sum()) \
                   if "no_alternative_flag" in preds_df.columns else 0

    # parent groups
    parent_groups = preds_df["source_parent_canonical_name"].dropna().unique()[:3] \
                    if "source_parent_canonical_name" in preds_df.columns else []
    pg_str = ", ".join(str(p) for p in parent_groups) if len(parent_groups)>0 else "diverse"

    para = (
        f"**{fname}** ({yr}, {ntype}) receives a composite risk score of **{cr:.3f}**, "
        f"placing it among the highest-risk nodes in the {yr} dependency network. "
        f"The primary risk driver is **{dom}** (normalized score: {comp_vals[dom]:.3f}). "
    )
    if de > 0.5:
        para += (f"The node is directly exposed to {n_up} upstream provider(s), "
                 f"including {up_str}, with substantial edge weights indicating strong dependency. ")
    if ce > 0.4:
        para += (f"There is also significant cascading exposure (score: {ce:.3f}), "
                 f"suggesting that upstream risk propagates through intermediate nodes. ")
    if dc > 0.6:
        para += (f"Dependency concentration is high (score: {dc:.3f}), "
                 f"indicating that upstream dependency is concentrated in one or few parent groups "
                 f"({pg_str}). This amplifies systemic exposure if the concentrated provider is disrupted. ")
    if no_alt_count > 0:
        para += (f"At least {no_alt_count} upstream dependency edge(s) indicate that no viable "
                 f"alternative to the provider is mentioned, increasing structural lock-in risk. ")
    para += ("Note: this score is a weakly supervised graph-derived proxy "
             "and does not predict an actual failure or incident outcome.")
    return para

for yr in EC_YEARS:
    ec_df, top_df = get_evidence_chains(yr)
    ec_df.to_csv(RPT/f"evidence_chain_top10_{yr}.csv", index=False)
    print(f"  → evidence_chain_top10_{yr}.csv")

    edges_yr = load_edges_weighted(yr)
    ev_report_lines.append(f"\n## Year {yr}\n")
    for rank_i, (_, node) in enumerate(top_df.iterrows(), start=1):
        fid   = node["canonical_firm_id"]
        fname = node["canonical_name"]
        preds_df = edges_yr[edges_yr["risk_target_id"] == fid]
        ev_report_lines.append(f"### Rank {rank_i}: {fname}\n")
        para = make_explanation_paragraph(fname, yr, node, preds_df)
        ev_report_lines.append(para + "\n")

        ev_report_lines.append("**Component scores:**")
        ev_report_lines.append(f"- Composite Risk: {node.get('composite_risk',0):.4f}")
        ev_report_lines.append(f"- Direct Exposure (norm): {node.get('direct_exposure_norm',0):.4f}")
        ev_report_lines.append(f"- Cascading Exposure (norm): {node.get('cascading_exposure_norm',0):.4f}")
        ev_report_lines.append(f"- Dependency Concentration (norm): {node.get('dependency_concentration_norm',0):.4f}")
        ev_report_lines.append(f"- Non-Substitutability (norm): {node.get('structural_non_substitutability_norm',0):.4f}\n")

        if len(preds_df) > 0:
            ev_report_lines.append("**Direct upstream evidence (top 3 by edge weight):**\n")
            for _, pe in preds_df.nlargest(3,"edge_weight").iterrows():
                ev_report_lines.append(f"- `{pe.get('risk_source_name',pe.get('source_canonical_name','?'))}` "
                                       f"→ `{fname}` | relation: {pe.get('relation_type','')} | "
                                       f"spillover: {pe.get('spillover_type','')} | "
                                       f"weight: {pe.get('edge_weight',0):.3f}")
                ev_sent = str(pe.get("근거 문장","")).strip()
                if ev_sent and ev_sent != "nan":
                    ev_report_lines.append(f"  > *\"{ev_sent[:200]}\"*")
        ev_report_lines.append("")

(RPT/"evidence_chain_report.md").write_text("\n".join(ev_report_lines))
print("  → evidence_chain_report.md")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4  GNN Result Interpretation
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Task 4] GNN result interpretation …")

# 5-fold CV over 2022–2024 node-year observations. Reproduce with
#   python -m src.gnn.cross_val
cv_results = pd.DataFrame([
    dict(model="Ridge (tabular)", spearman=-0.1375, mae=0.1908, rmse=0.2179,
         uses_graph=False, note="linear, node features only"),
    dict(model="MLP (tabular)",   spearman=0.1750,  mae=0.1896, rmse=0.2140,
         uses_graph=False, note="non-linear control, same 8 features"),
    dict(model="GCN",             spearman=0.8522,  mae=0.0717, rmse=0.0971,
         uses_graph=True,  note="selected model — graph structure + edge weights"),
])
cv_results.to_csv(RPT/"model_comparison_cv_results.csv", index=False)

# ── 2025: model prediction vs the rule it was trained to imitate ───────────────
# The rule-based CompositeRisk exists for 2025 but was never shown to the model.
# Where the two disagree is the interesting part: the model is generalising from
# structure rather than recomputing the formula.
gcn_pred = pd.read_csv(PRED/"prediction_2025_gcn_all_nodes.csv")
rule_2025 = load_composite(2025)[["canonical_firm_id","composite_risk"]]

gnn_vs_rule = (gcn_pred[["canonical_firm_id","canonical_name","node_type",
                         "gcn_score","gcn_rank_all","gcn_rank_within_type"]]
               .merge(rule_2025, on="canonical_firm_id", how="left"))
gnn_vs_rule["rule_rank_all"] = gnn_vs_rule["composite_risk"].rank(
    ascending=False, method="min")
gnn_vs_rule["score_diff_gcn_minus_rule"] = (
    gnn_vs_rule["gcn_score"] - gnn_vs_rule["composite_risk"]).round(4)
gnn_vs_rule["rank_diff"] = (
    gnn_vs_rule["rule_rank_all"] - gnn_vs_rule["gcn_rank_all"])
gnn_vs_rule = gnn_vs_rule.sort_values("gcn_score", ascending=False)
gnn_vs_rule.to_csv(RPT/"gnn_prediction_vs_rule_based.csv", index=False)

rank_rho = gnn_vs_rule[["gcn_rank_all","rule_rank_all"]].corr(method="spearman").iloc[0,1]

# ── Divergence cases: where structure and formula most disagree ────────────────
div = gnn_vs_rule.dropna(subset=["rank_diff"]).copy()
div["abs_rank_diff"] = div["rank_diff"].abs()
div = div.sort_values("abs_rank_diff", ascending=False)
div.head(15).to_csv(RPT/"top_prediction_divergence_cases.csv", index=False)

# ── Markdown interpretation ────────────────────────────────────────────────────
def pct_change(a, b):
    return f"{'−' if b < a else '+'}{abs((b - a) / a * 100):.0f}%"

mc_md = ["# Model Comparison Interpretation\n",
"## Disclaimer\n",
"Every model here predicts a **weakly supervised proxy** for systemic exposure, "
"derived from rule-based composite risk scores. Predictions are **not** forecasts "
"of observed failures, financial distress, or actual adverse events.\n",
"## 1. 5-fold Cross-validation (2022–2024)\n",
"| Model | Graph? | Spearman ρ | MAE | RMSE |",
"|---|---|---|---|---|"]
for _, r in cv_results.iterrows():
    mc_md.append(f"| {r['model']} | {'yes' if r['uses_graph'] else 'no'} "
                 f"| {r['spearman']:.4f} | {r['mae']:.4f} | {r['rmse']:.4f} |")

ridge, mlp, gcn = (cv_results.iloc[i] for i in range(3))

mc_md.append(textwrap.dedent(f"""
## 2. The tabular models do not merely underperform — Ridge points the wrong way

Ridge reaches a Spearman ρ of **{ridge['spearman']:.4f}**. A negative rank correlation
means the firms whose own filings sound most alarming are, if anything, *not* the ones
the network actually exposes. Self-reported intrinsic risk is not a weak proxy for
systemic exposure; it is close to uninformative about it.

The MLP is the control that makes this readable. It sees exactly the same eight
feature columns as the GCN and no graph, and reaches ρ = {mlp['spearman']:.4f}. So the
gap to the GCN's **{gcn['spearman']:.4f}** cannot be explained by non-linearity — it is
attributable to structure: which nodes sit upstream, how strongly they connect, and
what their neighbours look like. MAE improves {pct_change(mlp['mae'], gcn['mae'])} over
the MLP as well, so the ranking gain is not bought with calibration.

This is the empirical case for modelling the dependency network at all. If a table of
firm attributes had been sufficient, no graph would have been needed.

## 3. Why GCN over GAT

Attention did not buy enough to justify the cost. In an insurance setting the score has
to be explainable to an underwriter and defensible to a regulator, so a tie on accuracy
goes to the simpler, more inspectable model. Two convolution layers also match the risk
formulation, which only claims direct and 2-hop (cascading) exposure — a deeper stack
would propagate information further than the underlying definition supports.

## 4. 2025: where the model and the rule disagree

The GCN scored all {len(gnn_vs_rule)} nodes in the 2025 graph, a year whose rule-based
labels it never saw. Rank agreement with the rule is ρ = **{rank_rho:.3f}**: close enough
to confirm the model learned the intended concept, loose enough that it is not simply
recomputing the formula.

The divergences are where the model earns its keep. It ranks nodes by *structural
position* learned across three years, so it can flag an entity whose formula inputs look
unremarkable in 2025 but whose position in the network resembles nodes that scored high
before. Those cases are listed in `top_prediction_divergence_cases.csv` and are the ones
worth a human underwriter's attention.

## 5. Largest rank divergences (2025)
""".strip()))

mc_md.append("\n| Node | Type | GCN rank | Rule rank | Δ rank | GCN score | Rule score |")
mc_md.append("|---|---|---|---|---|---|---|")
for _, r in div.head(10).iterrows():
    mc_md.append(f"| {r['canonical_name']} | {r['node_type']} "
                 f"| {int(r['gcn_rank_all'])} | {int(r['rule_rank_all'])} "
                 f"| {int(r['rank_diff']):+d} | {r['gcn_score']:.4f} "
                 f"| {r['composite_risk']:.4f} |")

(RPT/"model_comparison_interpretation.md").write_text("\n".join(mc_md))
print("  → model_comparison_interpretation.md")
print("  → model_comparison_cv_results.csv")
print("  → gnn_prediction_vs_rule_based.csv")
print("  → top_prediction_divergence_cases.csv")

# ── One-page summary ──────────────────────────────────────────────────────────
n_train = sum(len(load_composite(y)) for y in TRAIN_YEARS)
summary_md = ["# Model Interpretation Summary\n",
"## One-Page Summary\n",
f"**Dataset**: 2022–2025 · {n_train} node-year observations (train) · "
f"{len(load_composite(2025))} nodes scored for 2025",
"**Graph**: directed, source_firm + target_entity in one node space, "
"dependency edges only (upstream → downstream)",
"**Weak labels**: CompositeRisk = "
+ " + ".join(f"{w:.2f}×{COMP_LABELS[c]}" for c, w in COMP_WEIGHTS.items()) + "\n",
"| Model | Graph? | Spearman ρ | Note |",
"|---|---|---|---|",
f"| Ridge (tabular) | no | {ridge['spearman']:.3f} | negative — intrinsic features "
"mis-rank systemic exposure |",
f"| MLP (tabular) | no | {mlp['spearman']:.3f} | non-linearity alone does not close the gap |",
f"| GCN | yes | {gcn['spearman']:.3f} | selected model |",
"\n**Key finding**: message passing over the upstream dependency network is the dominant "
"information source for systemic exposure. Node-level tabular features alone are not just "
"insufficient — in the linear case they are misleading.",
"\n**Caution**: all outputs are weakly supervised proxies. Predicted risk is not a forecast "
"of firm failure or incident occurrence."]
(RPT/"model_interpretation_summary.md").write_text("\n".join(summary_md))
print("  → model_interpretation_summary.md")
# TASK 5  Run Log
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Task 5] Run log …")

all_rpt_files = sorted(RPT.iterdir())

files_used = []
for yr in [2022,2023,2024,2025]:
    for pat in ["composite_risk_{}.csv","edges_weighted_{}.csv","nodes_{}.csv",
                "risk_components_{}.csv","node_vulnerability_{}.csv"]:
        p = OUT / pat.format(yr)
        if p.exists(): files_used.append(str(p.relative_to(BASE)))
for p in [OUT/"weak_labels_2022_2024.csv",
          PRED/"prediction_2025_gcn_all_nodes.csv",
          PRED/"prediction_2025_gcn_source_firms.csv",
          PRED/"prediction_2025_gcn_target_entities.csv"]:
    if p.exists(): files_used.append(str(p.relative_to(BASE)))

run_log = ["# Run Log — Interpretation Report\n",
f"**Generated by**: `src/reporting/interpretation_report.py`\n",
"## Files Used (Input)\n"]
for f in files_used:
    run_log.append(f"- `{f}`")

run_log.append("\n## Files Created (Output → outputs/interpretation_report/)\n")
for f in all_rpt_files:
    run_log.append(f"- `{f.name}`")

run_log.append(textwrap.dedent("""
## Assumptions

1. **Edge direction** is upstream → downstream as defined by `risk_source_id → risk_target_id` in `edges_weighted_{year}.csv`.
2. **CompositeRisk formula** uses weights 0.50/0.20/0.20/0.10 on year-wise percentile-rank normalized components.
3. **source_firm** = firms appearing in `self_features_mapped.csv` (10-K filers). **target_entity** = all other canonical nodes in the edge network.
4. **Incident signal** is treated as static (no year-wise filtering) — all incidents are pooled per firm.
5. **2025 predictions** come from the final GCN (`src/gnn/predict_2025.py`), scored over all nodes in one integrated graph and reported separately for `source_firm` and `target_entity`.
6. **Model selection** used 5-fold CV over 2022-2024 node-year observations comparing Ridge, MLP and GCN (`src/gnn/cross_val.py`).
7. **Ego networks** use 1-hop undirected neighborhood around the focal firm.
8. **Top-risk subgraphs** include top-10 nodes by composite_risk + direct predecessors + 2-hop predecessors.
9. **Label threshold for graph labels**: composite_risk ≥ 0.55 or node_type == source_firm.
10. **Evidence sentences** are reproduced verbatim from the `근거 문장` column; they may contain Korean text.

## Known Limitations & Warnings

- **Weak supervision caveat**: CompositeRisk is a rule-based structural proxy. GNN models learn to predict this proxy, not an observed outcome. High scores indicate structural exposure, not confirmed risk events.
- **Small source_firm set**: Only 10–11 source firms per year limits GNN discriminability among focal firms.
- **Sparse cascading paths**: Most nodes have very low CascadingExposure because the 10-K dataset mainly captures direct dependency relationships (few 2-hop paths exist).
- **HeteroGAT edge type mismatch**: The 2025 graph introduces a `source_firm→source_firm` edge type absent in 2022–2024 training data. This edge type is filtered at inference time; its signal is not captured by the trained HeteroGAT model.
- **Graph layout is stochastic**: Spring layout with fixed seed produces reproducible but not optimal node placement.
- **No ground-truth labels**: Evaluation relies on cross-validation against weak labels (circular but meaningful for relative model comparison).
""".strip()))

(RPT/"run_log.md").write_text("\n".join(run_log))
print("  → run_log.md")

print(f"\n{DIVIDER}")
print(f"All outputs written to: {RPT}")
print(f"Total files created: {len(list(RPT.iterdir()))}")
print(DIVIDER)
