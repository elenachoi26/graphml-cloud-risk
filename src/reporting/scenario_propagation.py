"""
Output 3 — "provider X goes down: who is affected, and how badly?"

The cumulative-exposure table (output 2) is static: it says which providers the
book is concentrated in. This is the dynamic counterpart — pick a shock node and
watch the damage travel.

    path_score = severity × ∏ edge_weight(along path) × DECAY^(hop-1)

Paths are enumerated as *simple* paths (no node repeats within a path) up to
`SCENARIO_MAX_HOPS`. A node can therefore be reached at several hop distances by
different routes, and its total exposure is the sum over all of them — which is
the point: a firm wired to a provider both directly and through two intermediaries
is more exposed than one with a single link, and a shortest-path view would hide
the difference.

    python -m src.reporting.scenario_propagation --shock Microsoft
    python -m src.reporting.scenario_propagation --shock "Amazon Web Services" --severity 0.6

Writes to `outputs/scenario_propagation/`:
    scenario_{year}_{shock}_paths.csv           every propagation path
    scenario_{year}_{shock}_affected_nodes.csv  per-node rollup, ranked
    scenario_{year}_{shock}_hop_summary.csv     damage by hop distance
    scenario_{year}_{shock}_report.html         standalone summary
"""

from __future__ import annotations

import argparse
import html
from collections import defaultdict

import networkx as nx
import pandas as pd

from .. import config


def build_graph(year: int) -> tuple[nx.DiGraph, pd.DataFrame]:
    """Directed graph for `year`, edges pointing upstream-source → downstream-exposed."""
    edges = pd.read_csv(config.OUT_DIR / f"edges_weighted_{year}.csv")
    nodes = pd.read_csv(config.OUT_DIR / f"nodes_{year}.csv")

    G = nx.DiGraph()
    for _, n in nodes.iterrows():
        G.add_node(n["canonical_firm_id"],
                   name=n["canonical_name"],
                   node_type=n["node_type"],
                   entity_type=n["entity_type"],
                   parent=n["parent_canonical_name"])

    for _, e in edges.iterrows():
        src, tgt = e["risk_source_id"], e["risk_target_id"]
        if src not in G or tgt not in G:
            continue
        # Keep the strongest edge when a pair is disclosed more than once —
        # the same relationship described in two 10-K sections is one dependency.
        if G.has_edge(src, tgt) and G[src][tgt]["edge_weight"] >= e["edge_weight"]:
            continue
        G.add_edge(src, tgt,
                   edge_weight=float(e["edge_weight"]),
                   relation_type=e.get("relation_type", ""),
                   spillover_type=e.get("spillover_type", ""),
                   source_section=e.get("source_section", ""))
    return G, nodes


def resolve_shock(G: nx.DiGraph, shock: str) -> str:
    """Accept either a canonical id or a display name."""
    if shock in G:
        return shock
    for nid, attrs in G.nodes(data=True):
        if str(attrs.get("name", "")).lower() == shock.lower():
            return nid
    raise ValueError(f"shock node {shock!r} not found in the graph")


def enumerate_paths(G: nx.DiGraph, shock_id: str, severity: float,
                    max_hops: int, decay: float) -> pd.DataFrame:
    """Depth-first enumeration of simple paths out of the shock node."""
    rows = []

    def walk(path: list, weight_product: float):
        hop = len(path) - 1
        if hop >= max_hops:
            return
        current = path[-1]
        for nxt in G.successors(current):
            if nxt in path:          # simple paths only — no cycles
                continue
            w = G[current][nxt]["edge_weight"]
            product = weight_product * w
            score = severity * product * (decay ** hop)
            new_path = path + [nxt]
            attrs = G.nodes[nxt]
            edge = G[current][nxt]
            rows.append({
                "shock_id": shock_id,
                "shock_name": G.nodes[shock_id].get("name", shock_id),
                "affected_node_id": nxt,
                "affected_node_name": attrs.get("name", nxt),
                "affected_node_type": attrs.get("node_type", ""),
                "affected_entity_type": attrs.get("entity_type", ""),
                "affected_parent": attrs.get("parent", ""),
                "hop": hop + 1,
                "path_score": score,
                "path_ids": " -> ".join(new_path),
                "path_names": " -> ".join(G.nodes[p].get("name", p) for p in new_path),
                "last_edge_weight": w,
                "last_relation_type": edge.get("relation_type", ""),
                "last_spillover_type": edge.get("spillover_type", ""),
                "last_source_section": edge.get("source_section", ""),
            })
            walk(new_path, product)

    walk([shock_id], 1.0)
    return pd.DataFrame(rows)


def rollup_nodes(paths: pd.DataFrame) -> pd.DataFrame:
    """One row per affected node, summed over every route that reaches it."""
    if paths.empty:
        return pd.DataFrame()
    agg = (paths.groupby(["affected_node_id", "affected_node_name", "affected_node_type",
                          "affected_entity_type", "affected_parent"], dropna=False)
                .agg(total_propagation_score=("path_score", "sum"),
                     max_path_score=("path_score", "max"),
                     mean_path_score=("path_score", "mean"),
                     path_count=("path_score", "size"),
                     min_hop=("hop", "min"),
                     max_hop=("hop", "max"))
                .reset_index()
                .sort_values("total_propagation_score", ascending=False))
    # Competition ranking: nodes reached with identical exposure share a rank.
    # Ties are common — a whole subsidiary cluster hanging off one parent is
    # reached by structurally identical paths, and ordering them 22, 23, 24
    # would imply a precedence the data does not support.
    agg["affected_rank"] = (agg["total_propagation_score"]
                            .rank(ascending=False, method="min").astype(int))
    return agg


def rollup_hops(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    return (paths.groupby("hop")
                 .agg(affected_path_count=("path_score", "size"),
                      unique_affected_nodes=("affected_node_id", "nunique"),
                      total_path_score=("path_score", "sum"),
                      mean_path_score=("path_score", "mean"),
                      max_path_score=("path_score", "max"))
                 .reset_index())


def render_html(shock_name: str, year: int, severity: float,
                affected: pd.DataFrame, hops: pd.DataFrame) -> str:
    """Standalone HTML summary — no verbatim filing text is included."""
    def esc(v):
        return html.escape(str(v))

    hop_rows = "".join(
        f"<tr><td>{int(r.hop)}</td><td>{int(r.unique_affected_nodes)}</td>"
        f"<td>{int(r.affected_path_count)}</td><td>{r.total_path_score:.4f}</td>"
        f"<td>{r.mean_path_score:.4f}</td></tr>"
        for r in hops.itertuples()
    )
    node_rows = "".join(
        f"<tr><td>{int(r.affected_rank)}</td><td>{esc(r.affected_node_name)}</td>"
        f"<td>{esc(r.affected_node_type)}</td><td>{esc(r.affected_parent)}</td>"
        f"<td>{r.total_propagation_score:.4f}</td><td>{int(r.path_count)}</td>"
        f"<td>{int(r.min_hop)}</td></tr>"
        for r in affected.itertuples()
    )
    total_score = affected["total_propagation_score"].sum() if not affected.empty else 0.0

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Scenario: {esc(shock_name)} — {year}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 2rem; color: #16181d; background: #fff; }}
  h1 {{ font-size: 1.4rem; margin: 0 0 .25rem; }}
  .sub {{ color: #6b7280; font-size: .85rem; margin-bottom: 1.5rem; }}
  .cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
  .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: .9rem 1.2rem; min-width: 150px; }}
  .card .k {{ font-size: .7rem; letter-spacing: .06em; text-transform: uppercase; color: #6b7280; }}
  .card .v {{ font-size: 1.6rem; font-weight: 700; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .85rem; margin-bottom: 2rem; }}
  th {{ text-align: left; background: #16181d; color: #fff; padding: .5rem .7rem; }}
  td {{ padding: .4rem .7rem; border-bottom: 1px solid #eef0f3; }}
  tr:hover td {{ background: #fafbfc; }}
</style></head><body>
<h1>Shock scenario — {esc(shock_name)}</h1>
<div class="sub">{year} dependency graph · severity {severity} ·
  path_score = severity x product(edge_weight) x {config.SCENARIO_DECAY}^(hop-1) ·
  max {config.SCENARIO_MAX_HOPS} hops</div>
<div class="cards">
  <div class="card"><div class="k">Affected nodes</div><div class="v">{len(affected)}</div></div>
  <div class="card"><div class="k">Propagation paths</div>
    <div class="v">{int(hops["affected_path_count"].sum()) if not hops.empty else 0}</div></div>
  <div class="card"><div class="k">Total score</div><div class="v">{total_score:.3f}</div></div>
</div>
<h2 style="font-size:1rem">Damage by hop distance</h2>
<table><thead><tr><th>Hop</th><th>Nodes</th><th>Paths</th>
  <th>Total score</th><th>Mean score</th></tr></thead><tbody>{hop_rows}</tbody></table>
<h2 style="font-size:1rem">Affected nodes</h2>
<table><thead><tr><th>#</th><th>Node</th><th>Type</th><th>Parent ecosystem</th>
  <th>Total score</th><th>Paths</th><th>Nearest hop</th></tr></thead>
<tbody>{node_rows}</tbody></table>
</body></html>"""


def run(shock: str = "Microsoft", year: int = config.PREDICT_YEAR,
        severity: float = config.SCENARIO_DEFAULT_SEVERITY,
        max_hops: int = config.SCENARIO_MAX_HOPS,
        decay: float = config.SCENARIO_DECAY,
        out_dir=None) -> pd.DataFrame:
    out_dir = config.SCENARIO_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    G, _ = build_graph(year)
    shock_id = resolve_shock(G, shock)
    shock_name = G.nodes[shock_id].get("name", shock_id)

    print("=" * 70)
    print(f"Scenario propagation — {shock_name} ({year})")
    print(f"  severity={severity}  max_hops={max_hops}  decay={decay}")
    print("=" * 70)

    paths = enumerate_paths(G, shock_id, severity, max_hops, decay)
    affected = rollup_nodes(paths)
    hops = rollup_hops(paths)

    if paths.empty:
        print(f"  {shock_name} has no downstream dependents in the {year} graph.")
        return affected

    slug = str(shock_name).replace(" ", "_").replace("/", "-")
    stem = f"scenario_{year}_{slug}"
    paths.to_csv(out_dir / f"{stem}_paths.csv", index=False)
    affected.to_csv(out_dir / f"{stem}_affected_nodes.csv", index=False)
    hops.to_csv(out_dir / f"{stem}_hop_summary.csv", index=False)
    (out_dir / f"{stem}_report.html").write_text(
        render_html(shock_name, year, severity, affected, hops), encoding="utf-8"
    )

    print(f"\n  Affected nodes: {len(affected)}   paths: {len(paths)}")
    for r in hops.itertuples():
        print(f"    hop {int(r.hop)}: {int(r.unique_affected_nodes):>3} nodes, "
              f"{int(r.affected_path_count):>3} paths, total score {r.total_path_score:.4f}")

    print("\n  Most exposed downstream nodes:")
    for r in affected.head(8).itertuples():
        print(f"    {int(r.affected_rank):>2}. {r.affected_node_name:<28} "
              f"{r.total_propagation_score:.4f}  (hop {int(r.min_hop)})")

    n_firms = int((affected["affected_node_type"] == "source_firm").sum())
    print(f"\n  {n_firms} of the affected nodes are focal firms — "
          f"the contracts an insurer would review on this scenario.")
    print(f"  → {out_dir}")
    return affected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shock", default="Microsoft", help="canonical id or display name")
    parser.add_argument("--year", type=int, default=config.PREDICT_YEAR)
    parser.add_argument("--severity", type=float, default=config.SCENARIO_DEFAULT_SEVERITY)
    parser.add_argument("--max-hops", type=int, default=config.SCENARIO_MAX_HOPS)
    parser.add_argument("--decay", type=float, default=config.SCENARIO_DECAY)
    parser.add_argument("--out", default=None, help="output directory override")
    args = parser.parse_args()

    from pathlib import Path
    run(shock=args.shock, year=args.year, severity=args.severity,
        max_hops=args.max_hops, decay=args.decay,
        out_dir=Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
