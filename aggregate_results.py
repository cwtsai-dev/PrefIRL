#!/usr/bin/env python
"""Aggregate test metrics across all runs into a summary table.

Scans logs/*/test_metrics.csv, groups by (market, policy, tag), and reports
mean +/- std over seeds. Writes results_summary.csv.

Usage:
    python aggregate_results.py [--logdir logs]
"""
import os
import re
import glob
import argparse

import pandas as pd

METRICS = ["ARR", "AVol", "SR", "MDD", "CR", "IR"]
RUN_RE = re.compile(r"^(.+?)_(MLP|HGAT)_(?:(.+)_)?seed(\d+)$")


def parse_run(name):
    m = RUN_RE.match(name)
    if not m:
        return None
    return {"market": m.group(1), "policy": m.group(2),
            "tag": m.group(3) or "", "seed": int(m.group(4))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="logs")
    args = p.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.logdir, "*", "test_metrics.csv"))):
        run = os.path.basename(os.path.dirname(f))
        info = parse_run(run)
        if info is None:
            continue
        try:
            metrics = pd.read_csv(f).iloc[0].to_dict()
        except Exception:
            continue
        info.update({k: metrics.get(k) for k in METRICS})
        rows.append(info)

    if not rows:
        print(f"no test_metrics.csv found under {args.logdir}/")
        return

    df = pd.DataFrame(rows)
    df.to_csv("results_raw.csv", index=False)
    print(f"=== raw runs ({len(df)}) ===")
    print(df.to_string(index=False))

    grp = df.groupby(["market", "policy", "tag"])
    summary = grp[METRICS].agg(["mean", "std"]).round(4)
    summary["n_seeds"] = grp.size()
    summary.to_csv("results_summary.csv")
    print("\n=== summary (mean over seeds) ===")
    mean_only = grp[METRICS].mean().round(4)
    mean_only["n_seeds"] = grp.size()
    print(mean_only.to_string())
    print("\nwrote results_raw.csv and results_summary.csv")


if __name__ == "__main__":
    main()
