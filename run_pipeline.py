#!/usr/bin/env python3
"""
Single entrypoint for the whole pipeline.

    python run_pipeline.py --stage graph     # steps 1-6: build and score the graph
    python run_pipeline.py --stage model     # cross-validation, final GCN, 2025 prediction
    python run_pipeline.py --stage report    # risk profiles, cumulative exposure, scenarios
    python run_pipeline.py --stage all

    python run_pipeline.py --stage graph --dry-run   # print distributions, write nothing

`--dry-run` exists for weight tuning: it runs every computation and prints the
score distributions but writes no artifacts, so a sweep over the config weights
cannot silently overwrite committed results.

Stages are ordered — `model` needs `graph`'s outputs, `report` needs both.
"""

from __future__ import annotations

import argparse
import sys

from src import config
from src.utils import using_samples


def stage_graph(dry_run: bool) -> None:
    from src import (step1_validate, step2_snapshots, step3_edge_weights,
                     step4_node_vulnerability, step5_risk_components, step6_composite_risk)

    ctx = step1_validate.run(dry_run)
    ctx = step2_snapshots.run(ctx, dry_run)
    ctx = step3_edge_weights.run(ctx, dry_run)
    ctx = step4_node_vulnerability.run(ctx, dry_run)
    ctx = step5_risk_components.run(ctx, dry_run)
    ctx = step6_composite_risk.run(ctx, dry_run)

    print("\n" + "=" * 70)
    print("GRAPH STAGE COMPLETE" + ("  (dry run — nothing written)" if dry_run else ""))
    print(f"  outputs: {config.WRITE_DIR}")
    print("=" * 70)


def stage_model(dry_run: bool) -> None:
    if dry_run:
        print("--dry-run applies to the graph stage only; skipping model stage.")
        return
    from src.gnn import cross_val, train_final, predict_2025

    cross_val.run()
    train_final.run()
    predict_2025.run()


def stage_report(dry_run: bool) -> None:
    if dry_run:
        print("--dry-run applies to the graph stage only; skipping report stage.")
        return
    from src.reporting import risk_profile, cumulative_exposure, scenario_propagation

    risk_profile.run()
    cumulative_exposure.run()
    scenario_propagation.run(shock="Microsoft")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["graph", "model", "report", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute and print, but write no artifacts")
    args = parser.parse_args()

    config.ensure_dirs()

    if using_samples():
        print("\n" + "!" * 70)
        print("  Running against SYNTHETIC SAMPLES (data/samples/).")
        print("  The real 10-K and incident panels are not distributed — see data/README.md.")
        print(f"  Writing to {config.WRITE_DIR.name}/ so the real results in outputs/ stay intact.")
        print("  Numbers produced now are structurally valid and substantively meaningless.")
        print("!" * 70)

    stages = ["graph", "model", "report"] if args.stage == "all" else [args.stage]
    for stage in stages:
        {"graph": stage_graph, "model": stage_model, "report": stage_report}[stage](args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
