#!/usr/bin/env python
"""Build dataset/{market}_sector.csv (kdcode,sector) for US/TW markets.

GICS sectors are needed to build the industry (same-sector) relation matrix
for markets that are not covered by the A-share industry matrix. Sectors are
pulled from Wikipedia constituent tables first, with a yfinance fallback for
any ticker not found.

Usage:
    python gen_data/fetch_sectors.py --market sp500
    python gen_data/fetch_sectors.py --market nd100
"""
import os
import time
import argparse

import pandas as pd

DATASET = "dataset"

WIKI = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "nd100": "https://en.wikipedia.org/wiki/Nasdaq-100",
}


def tickers_of(market):
    df = pd.read_csv(os.path.join(DATASET, f"{market}_org.csv"),
                     usecols=["kdcode"], dtype=str)
    return sorted(df["kdcode"].unique().tolist())


def wiki_sector_map(market):
    """Return {ticker: GICS sector} scraped from the Wikipedia table."""
    url = WIKI.get(market)
    if url is None:
        return {}
    try:
        tables = pd.read_html(url)
    except Exception as e:
        print(f"  wikipedia read failed: {e}")
        return {}
    mapping = {}
    for t in tables:
        cols = {str(c).strip().lower(): c for c in t.columns}
        tick_col = next((cols[c] for c in cols if c in ("ticker", "symbol")), None)
        sec_col = next((cols[c] for c in cols if "gics sector" in c or c == "sector"), None)
        if tick_col is None or sec_col is None:
            continue
        for _, row in t.iterrows():
            tk = str(row[tick_col]).strip().upper().replace(".", "-")
            sc = str(row[sec_col]).strip()
            if tk and sc and sc.lower() != "nan":
                mapping[tk] = sc
    return mapping


def yf_sector(ticker, suffix=""):
    """Best-effort single-ticker sector lookup via yfinance."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker + suffix).info
        return info.get("sector") or info.get("sectorKey")
    except Exception:
        return None


def build(market, yf_fallback=True):
    tickers = tickers_of(market)
    print(f"[{market}] {len(tickers)} tickers")
    wmap = wiki_sector_map(market)
    print(f"[{market}] wikipedia provided {len(wmap)} sector entries")

    suffix = ".TW" if market.startswith("tw") else ""
    rows, missing = [], []
    for tk in tickers:
        key = tk.upper().replace(".", "-")
        sector = wmap.get(key) or wmap.get(key.split("-")[0])
        if sector is None:
            missing.append(tk)
            sector = None
        rows.append([tk, sector])

    if missing and yf_fallback:
        print(f"[{market}] {len(missing)} tickers missing -> yfinance fallback")
        for i, tk in enumerate(missing):
            s = yf_sector(tk.replace("-", "."), suffix)
            for r in rows:
                if r[0] == tk:
                    r[1] = s
            if (i + 1) % 25 == 0:
                print(f"  ...{i + 1}/{len(missing)}")
            time.sleep(0.3)  # be gentle with the API

    still = sum(1 for _, s in rows if not s)
    for r in rows:
        if not r[1]:
            r[1] = "Unknown"
    out = os.path.join(DATASET, f"{market}_sector.csv")
    pd.DataFrame(rows, columns=["kdcode", "sector"]).to_csv(out, index=False)
    print(f"[{market}] wrote {out}  ({still} still unknown)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--market", required=True)
    p.add_argument("--no-yf", action="store_true", help="skip yfinance fallback")
    args = p.parse_args()
    build(args.market, yf_fallback=not args.no_yf)
