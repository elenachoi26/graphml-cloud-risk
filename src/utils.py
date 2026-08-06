"""
Shared helpers: input resolution, normalisation, distribution printing.

Every pipeline step imports its normalisers from here so that a change to, say,
percentile ranking propagates to all six steps at once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from . import config


# ── Input resolution ──────────────────────────────────────────────────────────

def load_input(key: str) -> pd.DataFrame:
    """
    Load one of the four input tables named in `config.INPUT_FILES`.

    Looks in `data/raw/` first and falls back to `data/samples/`. The real
    panels are withheld for licensing reasons (see data/README.md), so a clean
    clone transparently runs against the synthetic samples instead of failing.
    """
    filename = config.INPUT_FILES[key]
    raw = config.RAW_DIR / filename
    if raw.exists():
        return pd.read_csv(raw)

    sample = config.SAMPLE_DIR / filename
    if sample.exists():
        print(f"  [samples] {filename} — raw panel absent, using synthetic sample")
        return pd.read_csv(sample)

    raise FileNotFoundError(
        f"{filename} not found in {config.RAW_DIR} or {config.SAMPLE_DIR}"
    )


def using_samples() -> bool:
    """True when the run is reading synthetic samples rather than the real panels."""
    return config.USING_SAMPLES


def save_csv(df: pd.DataFrame, path: Path, dry_run: bool = False) -> None:
    """Write a step artifact, unless this is a weight-tuning dry run."""
    if dry_run:
        print(f"  [dry-run] would write {path.name} ({len(df)} rows)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_output(name: str) -> pd.DataFrame:
    """Read an artifact produced by an earlier step."""
    path = config.WRITE_DIR / name
    if not path.exists():
        path = config.OUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{name} missing — run the earlier pipeline steps first "
            f"(python run_pipeline.py --stage graph)"
        )
    return pd.read_csv(path)


# ── Normalisation ─────────────────────────────────────────────────────────────

def percentile_rank_normalize(series: pd.Series) -> pd.Series:
    """
    Map values onto [0, 1] by percentile rank within the year.

    Used for the composite-risk components, whose raw scales are not comparable
    (a sum of weighted vulnerabilities vs. a Herfindahl share vs. a reciprocal
    count). Rank normalisation makes the weighted blend meaningful and keeps the
    score stable across years even as the graph grows.
    """
    arr = series.values
    ranks = np.array([percentileofscore(arr, v, kind="rank") for v in arr]) / 100.0
    return pd.Series(ranks, index=series.index)


def minmax_norm(series: pd.Series) -> pd.Series:
    """Min-max to [0, 1]; a constant series maps to 0.5 rather than dividing by zero."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_dist(index_name: str, year: int, series: pd.Series,
               name_map: dict | None = None, is_edge: bool = False,
               edge_names: pd.DataFrame | None = None) -> None:
    """
    Print the distribution of a score, plus its top 10 by name.

    The top-10 list is the sanity check that matters: if a scoring change puts a
    small subsidiary above AWS, the weights are wrong regardless of how healthy
    the percentiles look.
    """
    s = series.dropna()
    if len(s) == 0:
        print(f"\n  [{index_name} / {year}] — no data")
        return

    top10_idx = series.nlargest(10).index
    if is_edge and edge_names is not None:
        labels = [f"{edge_names.loc[i, 'src_name']} → {edge_names.loc[i, 'tgt_name']}"
                  for i in top10_idx if i in edge_names.index]
    elif name_map is not None:
        labels = [str(name_map.get(i, i)) for i in top10_idx]
    else:
        labels = [str(i) for i in top10_idx]

    print(f"\n{'─' * 64}")
    print(f"  index_name : {index_name}")
    print(f"  year       : {year}")
    print(f"  N          : {len(s)}")
    for label, value in (
        ("min", s.min()), ("p10", s.quantile(0.10)), ("p25", s.quantile(0.25)),
        ("median", s.median()), ("mean", s.mean()), ("p75", s.quantile(0.75)),
        ("p90", s.quantile(0.90)), ("max", s.max()), ("std", s.std()),
        ("skew", s.skew()), ("zero_ratio", (s == 0).mean()),
    ):
        print(f"  {label:<10} : {value:.4f}")
    print("  top_10     :")
    for label in labels:
        print(f"    {label}")


def banner(title: str, detail: str = "") -> None:
    print("\n" + "=" * 70)
    print(title if not detail else f"{title}  [{detail}]")
    print("=" * 70)


# ── Master mapping ────────────────────────────────────────────────────────────

def master_lookup(master: pd.DataFrame) -> pd.DataFrame:
    """One row per canonical firm id, for attribute lookups."""
    return master.drop_duplicates("canonical_firm_id").set_index("canonical_firm_id")


def attr_getter(master: pd.DataFrame):
    """Return `get(firm_id, column, default)` over the master mapping."""
    lookup = master_lookup(master)

    def get(firm_id, col, default=""):
        try:
            val = lookup.loc[firm_id, col]
            return val if pd.notna(val) else default
        except KeyError:
            return default

    return get


def parent_group(firm_id, parent_lookup: dict):
    """
    Resolve a node to its parent ecosystem, falling back to itself.

    Concentration is measured over ecosystems, not legal entities: depending on
    AWS, Amazon Bedrock and Amazon SageMaker is one point of failure, not three.
    """
    pg = parent_lookup.get(firm_id, firm_id)
    if pd.isna(pg) or pg == "":
        return firm_id
    return pg
