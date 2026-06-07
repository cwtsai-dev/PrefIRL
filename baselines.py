#!/usr/bin/env python
"""Classic (non-learning) baseline portfolio strategies.

Evaluated on exactly the same test set as the SmartFolio agent so we can tell
whether the RL agent actually learns anything:

  - EqualWeight (1/N) : the classic 1/N strategy, rebalanced daily
  - BuyAndHold        : 1/N invested once at t0, never rebalanced
  - Random-topk       : random 10% of stocks each day (same action space as
                        the agent -> the key "did it learn?" reference)
  - Momentum-topk     : top 10% by the previous day's return

Metrics match env.portfolio_env.evaluate (pandas ddof=1, 252 trading days).

Usage:  python baselines.py --market sp500
"""
import os
import argparse

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_metrics(daily_returns, benchmark=None):
    dr = np.asarray(daily_returns, dtype=np.float64)
    m = {"ARR": 0.0, "AVol": 0.0, "SR": 0.0, "MDD": 0.0, "CR": 0.0, "IR": 0.0}
    if len(dr) == 0 or dr.std(ddof=1) == 0:
        return m
    std = dr.std(ddof=1)
    arr = (1 + dr.mean()) ** TRADING_DAYS - 1
    cum = np.cumprod(1 + dr)
    running_max = np.maximum.accumulate(cum)
    mdd = float((cum / running_max - 1).min())
    cr = arr / abs(mdd) if mdd != 0 else 0.0
    ir = 0.0
    if benchmark is not None:
        ex = dr - np.asarray(benchmark, dtype=np.float64)
        if ex.std(ddof=1) != 0:
            ir = ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return {"ARR": float(arr), "AVol": float(std * np.sqrt(TRADING_DAYS)),
            "SR": float(np.sqrt(TRADING_DAYS) * dr.mean() / std),
            "MDD": mdd, "CR": float(cr), "IR": float(ir)}


def net_value_curve(daily_returns):
    nv = [1.0]
    for r in daily_returns:
        nv.append(nv[-1] * (1.0 + r))
    return nv


def load_test_labels(market, test_start="2024-01-01", test_end="2024-12-31"):
    from dataloader.data_loader import AllGraphDataSampler
    ds = AllGraphDataSampler(base_dir=f"dataset/data_train_predict_{market}/1_hy/",
                             date=True, mode="test",
                             test_start_date=test_start, test_end_date=test_end)
    labels = np.stack([d["labels"].numpy() for d in ds.data_all])  # [T, N]
    return labels


# --- strategies: each returns a length-T daily portfolio return series -------
def equal_weight(labels):
    return labels.mean(axis=1)


def buy_and_hold(labels):
    T, N = labels.shape
    val = np.ones(N) / N
    pv = [val.sum()]
    for t in range(T):
        val = val * (1.0 + labels[t])
        pv.append(val.sum())
    pv = np.array(pv)
    return pv[1:] / pv[:-1] - 1.0


def random_topk(labels, k, seed=0):
    rng = np.random.default_rng(seed)
    T, N = labels.shape
    out = np.zeros(T)
    for t in range(T):
        sel = rng.choice(N, size=k, replace=False)
        out[t] = labels[t, sel].mean()
    return out


def momentum_topk(labels, k):
    T, N = labels.shape
    out = np.zeros(T)
    for t in range(T):
        if t == 0:
            out[t] = labels[t].mean()
        else:
            top = np.argsort(-labels[t - 1])[:k]   # yesterday's winners
            out[t] = labels[t, top].mean()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--market", default="sp500")
    args = p.parse_args()

    labels = load_test_labels(args.market)
    T, N = labels.shape
    k = max(1, int(0.1 * N))
    bench = equal_weight(labels)  # equal-weight market proxy == the IR benchmark

    strategies = {
        "EqualWeight(1/N)": equal_weight(labels),
        "BuyAndHold": buy_and_hold(labels),
        "Random-topk": random_topk(labels, k, seed=0),
        "Momentum-topk": momentum_topk(labels, k),
    }

    rows, curves = [], {}
    for name, dr in strategies.items():
        m = compute_metrics(dr, benchmark=bench)
        m["strategy"] = name
        rows.append(m)
        curves[name] = net_value_curve(dr)

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(rows)[["strategy", "ARR", "AVol", "SR", "MDD", "CR", "IR"]]
    df.to_csv(f"results/baselines_metrics_{args.market}.csv", index=False)
    pd.DataFrame(curves).to_csv(f"results/baselines_curves_{args.market}.csv", index=False)
    print(df.round(4).to_string(index=False))
    print(f"\n{args.market} test: {T} days, {N} stocks, top-k={k}")


if __name__ == "__main__":
    main()
