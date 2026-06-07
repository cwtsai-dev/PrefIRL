#!/usr/bin/env python
"""Unified dataset builder (run from the project root).

Reproduces the preprocessing of generate_relation.py + train_predict_data.py
but parameterised by market and with correct paths. For a given market it:
  1. computes monthly Pearson correlation matrices -> dataset/corr/{market}/
  2. builds the industry (same-sector) relation matrix
  3. produces per-day .pkl samples -> dataset/data_train_predict_{market}/1_hy/

Industry source:
  * hs300 / zz500  -> dataset/A_stock_industry_matrx.csv (A-share matrix)
  * others         -> dataset/{market}_sector.csv with columns: kdcode,sector
    (a binary same-sector matrix is built, matching paper eq. A_ind)

Usage:
    python gen_data/build_dataset.py --market zz500
"""
import os
import argparse
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DATASET = "dataset"
FEATURE_COLS = ["close", "open", "high", "low", "prev_close", "volume"]
FEATURE_COLS_NORM = [f"{c}_normalized" for c in FEATURE_COLS]
LOOKBACK = 20
CORR_WINDOW = 20
THRESHOLD = 0.2


# --------------------------------------------------------------------------
# 1. correlation matrices
# --------------------------------------------------------------------------
def monthly_corr(df, market):
    """Compute one Pearson correlation matrix per calendar month.

    Stock-pair correlation = mean over the 6 features of the per-feature
    Pearson correlation across the last CORR_WINDOW trading days of the month.
    """
    out_dir = os.path.join(DATASET, "corr", market)
    os.makedirs(out_dir, exist_ok=True)

    trade_days = sorted(df["dt"].unique().tolist())
    df = df.sort_values(["kdcode", "dt"])
    months = sorted({d[:7] for d in trade_days})

    for ym in tqdm(months, desc=f"[{market}] corr"):
        month_days = [d for d in trade_days if d[:7] == ym]
        end_day = month_days[-1]
        end_idx = trade_days.index(end_day)
        if end_idx - CORR_WINDOW + 1 < 0:
            continue
        window = trade_days[end_idx - CORR_WINDOW + 1: end_idx + 1]
        win = df[df["dt"].isin(window)]

        # build {code: [CORR_WINDOW, 6]} keeping only full-window stocks
        ref = {}
        for code, g in win.groupby("kdcode"):
            g = g.sort_values("dt")
            if len(g) == CORR_WINDOW:
                ref[code] = g[FEATURE_COLS].values.astype(np.float64)
        codes = sorted(ref.keys())
        if len(codes) < 2:
            continue
        stacked = np.stack([ref[c] for c in codes])           # [N, W, 6]

        # per-feature Pearson correlation, then average across features
        acc = np.zeros((len(codes), len(codes)))
        for f in range(len(FEATURE_COLS)):
            mat = stacked[:, :, f]                              # [N, W]
            c = np.corrcoef(mat)
            acc += np.nan_to_num(c, nan=0.0)
        corr = acc / len(FEATURE_COLS)
        np.fill_diagonal(corr, 1.0)

        relation_dt = (pd.Timestamp(end_day) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        pd.DataFrame(corr, index=codes, columns=codes).to_csv(
            os.path.join(out_dir, relation_dt + ".csv"))


# --------------------------------------------------------------------------
# 2. industry matrix
# --------------------------------------------------------------------------
def industry_matrix(market, codes):
    """Return an [N, N] industry relation matrix aligned to `codes`."""
    if market in ("hs300", "zz500"):
        astock = pd.read_csv(os.path.join(DATASET, "A_stock_industry_matrx.csv"), index_col=0)
        astock.index = astock.index.astype(str)
        astock.columns = astock.columns.astype(str)
        keep = [c for c in codes if c in astock.index]
        if len(keep) != len(codes):
            missing = set(codes) - set(keep)
            raise ValueError(f"{market}: {len(missing)} codes missing from A-stock matrix")
        return astock.loc[codes, codes].values.astype(np.float32)

    # sector-based binary matrix: A_ind[i,j] = 1 if same sector else 0
    sector_csv = os.path.join(DATASET, f"{market}_sector.csv")
    if not os.path.exists(sector_csv):
        raise FileNotFoundError(
            f"sector file not found: {sector_csv} (columns: kdcode,sector)")
    sec = pd.read_csv(sector_csv, dtype=str)
    smap = dict(zip(sec["kdcode"], sec["sector"]))
    sectors = [smap.get(c, f"__unknown_{c}") for c in codes]
    n = len(codes)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if sectors[i] == sectors[j]:
                mat[i, j] = 1.0
    return mat


# --------------------------------------------------------------------------
# 3. preprocessing helpers (ported from train_predict_data.py)
# --------------------------------------------------------------------------
def get_label(df, horizon=1):
    df = df.sort_values(["kdcode", "dt"]).copy()
    df["label"] = df.groupby("kdcode")["close"].transform(
        lambda x: x.shift(-horizon) / x - 1)
    return df


def cal_rolling_mean_std(df, cols=("close", "volume"), lookback=5):
    df = df.sort_values(["kdcode", "dt"]).copy()
    for col in cols:
        df[f"{col}_mean"] = df.groupby("kdcode")[col].transform(
            lambda x: x.rolling(lookback, min_periods=1).mean())
        df[f"{col}_std"] = df.groupby("kdcode")[col].transform(
            lambda x: x.rolling(lookback, min_periods=1).std())
    return df


def group_and_norm(df, base_cols, n_clusters=4):
    """Per date: k-means cluster stocks, then standardise features inside
    each cluster (intra-group normalisation)."""
    result = []
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    for date, group in df.groupby("dt"):
        group = group.copy()
        feats = group[base_cols].fillna(0)
        if len(group) < n_clusters:
            group["cluster"] = 0
        else:
            scaled = StandardScaler().fit_transform(feats)
            group["cluster"] = kmeans.fit_predict(scaled)
        for f in FEATURE_COLS:
            group[f"{f}_normalized"] = group.groupby("cluster")[f].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8))
        result.append(group)
    return pd.concat(result)


def filter_code(df):
    dts = set(df["dt"])
    by_code = df.groupby("kdcode")["dt"].apply(set)
    return sorted(by_code[by_code.apply(lambda s: s == dts)].index.tolist())


def gen_pos_neg(corr, threshold=THRESHOLD):
    pos = (corr > threshold).astype(np.float32)
    neg = (corr < -threshold).astype(np.float32)
    np.fill_diagonal(pos, 0.0)
    np.fill_diagonal(neg, 0.0)
    return pos, neg


# --------------------------------------------------------------------------
# main build
# --------------------------------------------------------------------------
def build(market, horizon=1, relation_type="hy"):
    org_csv = os.path.join(DATASET, f"{market}_org.csv")
    if not os.path.exists(org_csv):
        raise FileNotFoundError(org_csv)
    print(f"[{market}] reading {org_csv}")
    df_raw = pd.read_csv(org_csv, dtype={"kdcode": str})
    df_raw["dt"] = df_raw["dt"].astype(str)

    # 1. correlation matrices
    corr_dir = os.path.join(DATASET, "corr", market)
    if not os.path.isdir(corr_dir) or len(os.listdir(corr_dir)) == 0:
        monthly_corr(df_raw, market)
    else:
        print(f"[{market}] corr matrices already present, skipping")

    # 2. labels + features
    df = get_label(df_raw, horizon=horizon)
    df = cal_rolling_mean_std(df, cols=("close", "volume"), lookback=5)
    df = group_and_norm(df, base_cols=["close_mean", "close_std",
                                       "volume_mean", "volume_std"], n_clusters=4)
    df_all = df.copy()
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    trade_days_all = sorted(df_all["dt"].unique().tolist())
    trade_days = sorted(df["dt"].unique().tolist())
    codes = filter_code(df)
    print(f"[{market}] {len(codes)} stocks survive the all-dates filter, "
          f"{len(trade_days)} trading days")

    ind = industry_matrix(market, codes)

    out_dir = os.path.join(DATASET, f"data_train_predict_{market}",
                           f"{horizon}_{relation_type}")
    os.makedirs(out_dir, exist_ok=True)

    import torch
    ind_t = torch.nan_to_num(torch.from_numpy(ind).float())

    # pivot to dense [date x code] arrays once (fast per-day slicing afterwards)
    df_all = df_all[df_all["kdcode"].isin(set(codes))]
    piv = {}
    for f in FEATURE_COLS_NORM + ["label"]:
        p = df_all.pivot_table(index="dt", columns="kdcode", values=f)
        p = p.reindex(index=trade_days_all, columns=codes)
        piv[f] = p.values.astype(np.float32)            # [D, N]
    row_of = {d: i for i, d in enumerate(trade_days_all)}

    corr_cache = {}
    n_written = 0
    for dt in tqdm(trade_days, desc=f"[{market}] pkl"):
        r = row_of[dt]
        if r - LOOKBACK + 1 < 0:
            continue

        # correlation matrix of this month
        relation_dt = (pd.Timestamp(dt) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        if relation_dt not in corr_cache:
            cpath = os.path.join(corr_dir, relation_dt + ".csv")
            if not os.path.exists(cpath):
                corr_cache[relation_dt] = None
            else:
                cdf = pd.read_csv(cpath, index_col=0)
                cdf.index = cdf.index.astype(str)
                cdf.columns = cdf.columns.astype(str)
                corr_cache[relation_dt] = cdf
        cdf = corr_cache[relation_dt]
        if cdf is None or not all(c in cdf.index for c in codes):
            continue
        corr = np.nan_to_num(cdf.loc[codes, codes].values.astype(np.float32))
        pos, neg = gen_pos_neg(corr, THRESHOLD)

        # ts window [N, LOOKBACK, 6]  and current features [N, 1, 6]
        ts = np.stack([piv[f][r - LOOKBACK + 1: r + 1] for f in FEATURE_COLS_NORM],
                      axis=-1)                            # [LOOKBACK, N, 6]
        ts = np.nan_to_num(np.transpose(ts, (1, 0, 2)))   # [N, LOOKBACK, 6]
        features = ts[:, -1:, :]                          # [N, 1, 6]
        labels = np.nan_to_num(piv["label"][r])           # [N]

        result = {
            "corr": torch.tensor(corr, dtype=torch.float32),
            "ts_features": torch.tensor(ts, dtype=torch.float32),
            "features": torch.tensor(features, dtype=torch.float32),
            "industry_matrix": ind_t.clone(),
            "pos_matrix": torch.tensor(pos, dtype=torch.float32),
            "neg_matrix": torch.tensor(neg, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "mask": [True] * len(codes),
        }
        with open(os.path.join(out_dir, dt + ".pkl"), "wb") as f:
            pickle.dump(result, f)
        n_written += 1

    print(f"[{market}] wrote {n_written} .pkl samples to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--market", required=True)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--relation_type", default="hy")
    args = p.parse_args()
    build(args.market, horizon=args.horizon, relation_type=args.relation_type)
