#!/usr/bin/env python3
"""
Strip verbatim filing text from result CSVs before they are published.

The pipeline carries the source sentence alongside every edge, because that is
what makes a risk score defensible to an underwriter ("this dependency, this
sentence, this 10-K section"). Those sentences are useful locally and are not
redistributed here — the published results keep the structured scores and drop
the quoted text.

    python tools/scrub_outputs.py                # scrub outputs/ in place
    python tools/scrub_outputs.py --check        # report only, change nothing

Columns removed are listed in `src/config.VERBATIM_TEXT_COLUMNS`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402


def scrub_file(path: Path, check: bool = False) -> list[str]:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:                      # unreadable → leave alone, report
        print(f"  ! {path.name}: {exc}")
        return []

    present = [c for c in config.VERBATIM_TEXT_COLUMNS if c in df.columns]
    if present and not check:
        df.drop(columns=present).to_csv(path, index=False)
    return present


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="report only")
    parser.add_argument("--dir", default=str(config.OUT_DIR))
    args = parser.parse_args()

    root = Path(args.dir)
    hits = 0
    for csv in sorted(root.rglob("*.csv")):
        found = scrub_file(csv, args.check)
        if found:
            hits += 1
            verb = "would remove" if args.check else "removed"
            print(f"  {csv.relative_to(root)}: {verb} {found}")

    if args.check and hits:
        print(f"\n{hits} file(s) still contain verbatim text columns.")
        return 1
    print(f"\n{'Checked' if args.check else 'Scrubbed'} {root} — {hits} file(s) affected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
