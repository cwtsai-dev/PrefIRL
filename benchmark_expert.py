#!/usr/bin/env python
"""Benchmark the deterministic greedy expert (paper Algorithm 1) directly.

If SmartFolio-HGAT is just imitating the expert, the expert's own test
performance is the ceiling. Two variants are reported:

  - oracle : expert ranks by the current sample's labels (= next-day
             return). This is what the IRL is trained against, and it
             cheats — it picks tomorrow's top performers. Treat it as
             the *upper bound of imitation*.

  - causal : expert ranks by the PREVIOUS day's realised return. A
             realistic short-horizon momentum heuristic with no
             look-ahead.

The gap (oracle − agent) tells you how much of the expert's signal the
agent failed to learn from features alone. The gap (agent − causal)
tells you whether the agent beats a no-foresight momentum strategy.

Usage:  python benchmark_expert.py --market sp500
"""
import os
import argparse

import numpy as np
import pandas as pd

from dataloader.data_loader import AllGraphDataSampler
from gen_data.generate_expert import generate_expert_strategy
from baselines import compute_metrics, net_value_curve


def load_test(market, test_start, test_end):
    ds = AllGraphDataSampler(
        base_dir=f"dataset/data_train_predict_{market}/1_hy/",
        date=True, mode="test",
        test_start_date=test_start, test_end_date=test_end,
    )
    return ds.data_all


def run_expert(samples, mode="oracle", top_k=0.1, max_industry_ratio=0.3):
    """Roll the greedy expert forward over the test period.

    `mode`:
        - 'oracle' : rank by the sample's own labels (next-day return)
        - 'causal' : rank by the previous day's realised return
    Returns: daily portfolio return series of length T.
    """
    T = len(samples)
    daily_returns = np.zeros(T)
    prev_labels = None
    for t in range(T):
        s = samples[t]
        labels = s["labels"].numpy()                # [N] - next-day return
        ind = s["industry_matrix"].numpy()          # [N, N]
        corr = s["corr"].numpy()                    # [N, N]
        if mode == "oracle" or prev_labels is None:
            signal = labels                         # cheats: uses tomorrow
        else:
            signal = prev_labels                    # last realised return
        action = generate_expert_strategy(
            returns=signal, industry_relation_matrix=ind,
            correlation_matrix=corr,
            top_k=top_k, max_industry_ratio=max_industry_ratio)
        sel = np.where(action > 0)[0]
        daily_returns[t] = float(labels[sel].mean()) if len(sel) > 0 else 0.0
        prev_labels = labels
    return daily_returns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--market", default="sp500")
    p.add_argument("--test_start", default="2024-01-01")
    p.add_argument("--test_end", default="2024-12-31")
    args = p.parse_args()

    samples = load_test(args.market, args.test_start, args.test_end)
    T = len(samples)
    print(f"{args.market} test: {T} days, {samples[0]['labels'].shape[0]} stocks")

    # equal-weight market return (the IR benchmark used by the agent eval)
    bench = np.array([s["labels"].numpy().mean() for s in samples])

    results, curves = {}, {}
    for mode in ["oracle", "causal"]:
        dr = run_expert(samples, mode=mode)
        m = compute_metrics(dr, benchmark=bench)
        m["strategy"] = f"Expert-{mode}"
        results[f"Expert-{mode}"] = m
        curves[f"Expert-{mode}"] = net_value_curve(dr)

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results).T[["ARR", "AVol", "SR", "MDD", "CR", "IR"]]
    df.to_csv(f"results/expert_metrics_{args.market}.csv")
    pd.DataFrame(curves).to_csv(f"results/expert_curves_{args.market}.csv", index=False)
    print("\n=== greedy expert (Algorithm 1) on the test set ===")
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
