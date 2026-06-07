#!/usr/bin/env python
"""Compile final results: PnL curve plot, metrics table, and RESULTS.md.

Run after the sp500 MLP/HGAT training and baselines.py have finished.
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ["ARR", "AVol", "SR", "MDD", "CR", "IR"]

# paper Table 1, S&P 500, "(* Ours)" row -- reference only
PAPER_SP500 = {"ARR": 0.250, "AVol": 0.117, "SR": 1.906,
               "MDD": -0.058, "CR": 4.293, "IR": 1.184}


def test_dates(market):
    files = sorted(glob.glob(f"dataset/data_train_predict_{market}/1_hy/2024-*.pkl"))
    return [os.path.basename(f)[:10] for f in files]


def load_agent(market, policy, reward="irl", which="final"):
    rew = "" if reward == "irl" else f"{reward}_"
    d = f"logs/{market}_{policy}_{rew}seed0"
    suf = "" if which == "final" else "_bestval"
    mf = f"{d}/test_metrics{suf}.csv"
    nvf = f"{d}/test_net_value{suf}.csv"
    m = pd.read_csv(mf).iloc[0].to_dict()
    nv = pd.read_csv(nvf)["net_value"].tolist()
    return {k: float(m[k]) for k in METRICS}, nv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="sp500")
    args = ap.parse_args()
    market = args.market
    os.makedirs("results", exist_ok=True)

    results, curves = {}, {}

    # --- SmartFolio agent runs (all reward methods, both checkpoints) ---
    for pol in ["HGAT", "MLP"]:
        for rew in ["irl", "closed_form", "gail"]:
            for which, suffix in [("final", "final"), ("bestval", "best-val")]:
                try:
                    m, nv = load_agent(market, pol, reward=rew, which=which)
                    rew_label = {"irl": "IRL", "closed_form": "CF", "gail": "GAIL"}[rew]
                    key = f"SmartFolio-{pol}-{rew_label} ({suffix})"
                    results[key] = m
                    curves[key] = nv
                    print(f"loaded {key}")
                except FileNotFoundError:
                    pass

    # --- baselines ---
    bm_path = f"results/baselines_metrics_{market}.csv"
    bc_path = f"results/baselines_curves_{market}.csv"
    if os.path.exists(bm_path):
        bm = pd.read_csv(bm_path)
        bc = pd.read_csv(bc_path)
        for _, r in bm.iterrows():
            results[r["strategy"]] = {k: float(r[k]) for k in METRICS}
        for col in bc.columns:
            curves[col] = bc[col].tolist()
        print(f"loaded {len(bm)} baselines")
    else:
        print("!! baselines not found -- run baselines.py first")

    if not results:
        print("nothing to report yet")
        return

    # --- combined metrics table ---
    table = pd.DataFrame(results).T[METRICS]
    table.to_csv(f"results/all_metrics_{market}.csv")
    print("\n=== metrics ===")
    print(table.round(4).to_string())

    # --- PnL curve plot: focus on final-epoch SmartFolio variants vs baselines ---
    dates = test_dates(market)
    plt.figure(figsize=(13, 6.5))
    styles = {
        "SmartFolio-HGAT-IRL (final)":  dict(lw=2.4, color="#c0392b"),
        "SmartFolio-HGAT-CF (final)":   dict(lw=2.4, color="#2980b9"),
        "SmartFolio-HGAT-GAIL (final)": dict(lw=2.4, color="#27ae60"),
        "SmartFolio-MLP-IRL (final)":   dict(lw=1.4, color="#c0392b", ls="--", alpha=0.7),
        "SmartFolio-MLP-CF (final)":    dict(lw=1.4, color="#2980b9", ls="--", alpha=0.7),
        "SmartFolio-MLP-GAIL (final)":  dict(lw=1.4, color="#27ae60", ls="--", alpha=0.7),
        "EqualWeight(1/N)":             dict(lw=1.8, color="#2c3e50"),
        "BuyAndHold":                   dict(lw=1.2, color="#16a085", ls="--"),
        "Momentum-topk":                dict(lw=1.2, color="#8e44ad", ls=":"),
        "Random-topk":                  dict(lw=1.2, color="#7f8c8d", ls=":"),
    }
    for name, st in styles.items():
        if name in curves:
            y = curves[name]
            plt.plot(range(len(y)), y, label=name, **st)
    plt.axhline(1.0, color="gray", lw=0.8, alpha=0.5)
    if dates:
        step = max(1, len(dates) // 12)
        ticks = list(range(0, len(dates), step))
        plt.xticks([t + 1 for t in ticks], [dates[t] for t in ticks], rotation=45, ha="right")
    plt.xlabel("trading day (test period 2024)")
    plt.ylabel("cumulative wealth (start = 1.0)")
    plt.title(f"{market.upper()} 2024 test — Cumulative PnL: SmartFolio vs. baselines")
    plt.legend(loc="best", framealpha=0.9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"results/pnl_curve_{market}.png", dpi=130)
    print(f"\nsaved results/pnl_curve_{market}.png")

    # --- RESULTS.md ---
    write_report(market, table, curves)


def write_report(market, table, curves):
    def fmt_row(name):
        r = table.loc[name]
        return ("| " + name + " | "
                + " | ".join(f"{r[m]:.4f}" for m in METRICS) + " |")

    order = [n for n in [
        "SmartFolio-HGAT-IRL (final)",  "SmartFolio-HGAT-IRL (best-val)",
        "SmartFolio-HGAT-CF (final)",   "SmartFolio-HGAT-CF (best-val)",
        "SmartFolio-HGAT-GAIL (final)", "SmartFolio-HGAT-GAIL (best-val)",
        "SmartFolio-MLP-IRL (final)",   "SmartFolio-MLP-IRL (best-val)",
        "SmartFolio-MLP-CF (final)",    "SmartFolio-MLP-CF (best-val)",
        "SmartFolio-MLP-GAIL (final)",  "SmartFolio-MLP-GAIL (best-val)",
        "EqualWeight(1/N)", "BuyAndHold", "Momentum-topk", "Random-topk",
    ] if n in table.index]
    final_wealth = {n: curves[n][-1] for n in order if n in curves}

    lines = []
    lines.append(f"# SmartFolio — {market.upper()} Results\n")
    lines.append("Heuristic-guided IRL portfolio optimization (IJCAI-25 #1054), "
                 "reproduced and evaluated on the S&P 500 test period (2024).\n")
    lines.append("## Setup\n")
    lines.append("- Market: **S&P 500** (472 constituents that survive the "
                 "all-dates filter), train 2018–2022 / val 2023 / **test 2024**.")
    lines.append("- Training config (paper §4.1): lr 1e-4, batch 128 "
                 "(HGAT 32 — memory), 128-d hidden, 8 heads, 200 epochs, seed 0.")
    lines.append("- Policies: **HGAT** (paper full model) and **MLP** "
                 "(paper's *w/o HGAT* ablation).")
    lines.append("- Baselines: non-learning strategies on the identical test set.\n")
    lines.append("## Metrics (S&P 500, 2024 test)\n")
    lines.append("ARR = annualised return, AVol = annualised volatility, "
                 "SR = Sharpe, MDD = max drawdown, CR = Calmar, IR = information "
                 "ratio (vs. equal-weight market).\n")
    lines.append("| Strategy | ARR | AVol | SR | MDD | CR | IR |")
    lines.append("|---|---|---|---|---|---|---|")
    for n in order:
        lines.append(fmt_row(n))
    lines.append("")
    lines.append("Paper reference (Table 1, S&P 500, *Ours*): "
                 "ARR 0.250, AVol 0.117, SR 1.906, MDD −0.058, CR 4.293, IR 1.184.\n")
    lines.append("## Cumulative PnL\n")
    lines.append(f"![PnL curve](results/pnl_curve_{market}.png)\n")
    if final_wealth:
        lines.append("Final cumulative wealth (start = 1.0):\n")
        for n in order:
            if n in final_wealth:
                lines.append(f"- {n}: **{final_wealth[n]:.4f}**")
        lines.append("")

    # automatic verdict
    lines.append("## Did it learn anything?\n")
    lines.append("The PPO test metric varied a lot from epoch to epoch (typical for "
                 "RL on a portfolio task) so two checkpoints are reported: "
                 "**final-epoch** (paper convention) and **best-val** "
                 "(the validation-Sharpe-maximising checkpoint, fairer for noisy "
                 "training).\n")
    if "EqualWeight(1/N)" in table.index:
        ew = table.loc["EqualWeight(1/N)"]
        rnd = table.loc["Random-topk"] if "Random-topk" in table.index else None
        candidates = []
        for pol in ["HGAT", "MLP"]:
            for rew in ["IRL", "CF", "GAIL"]:
                for ck in ["final", "best-val"]:
                    candidates.append(f"SmartFolio-{pol}-{rew} ({ck})")
        for pol_key in candidates:
            if pol_key not in table.index:
                continue
            a = table.loc[pol_key]
            v_ew = "beats" if a["SR"] > ew["SR"] else "matches/loses to"
            txt = f"- **{pol_key}** (SR {a['SR']:+.3f}) {v_ew} 1/N (SR {ew['SR']:+.3f})"
            if rnd is not None:
                v_rnd = "beats" if a["SR"] > rnd["SR"] else "matches/loses to"
                txt += f"; {v_rnd} random (SR {rnd['SR']:+.3f})"
            lines.append(txt + ".")
    lines.append("")

    with open(f"RESULTS_{market}.md", "w") as f:
        f.write("\n".join(lines))
    print(f"wrote RESULTS_{market}.md")


if __name__ == "__main__":
    main()
