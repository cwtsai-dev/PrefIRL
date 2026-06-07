#!/usr/bin/env python
"""Source Taiwan-market data via yfinance and write it in the project schema.

Produces:
  * dataset/tw50_org.csv     -> kdcode,dt,close,open,high,low,prev_close,volume
  * dataset/tw50_sector.csv  -> kdcode,sector

The Taiwan market is an extension beyond the paper's four markets. Constituents
are the FTSE TWSE Taiwan 50 large caps; daily OHLCV for 2018-2024 is downloaded
from Yahoo Finance (.TW tickers). Best-effort: Yahoo may rate-limit.

Usage:
    python gen_data/fetch_taiwan.py
"""
import os
import time

import pandas as pd

DATASET = "dataset"
START, END = "2018-01-01", "2025-01-01"

# FTSE TWSE Taiwan 50 constituents (stable large caps; numeric TWSE codes)
TW50 = [
    "2330", "2317", "2454", "2308", "2382", "2303", "2412", "2881", "2882",
    "2891", "2886", "2884", "2885", "2892", "2880", "2887", "2890", "2883",
    "1216", "1301", "1303", "1326", "2002", "2207", "2301", "2327", "2357",
    "2379", "2395", "2409", "2603", "2609", "2615", "2912", "3008", "3034",
    "3037", "3045", "3711", "4904", "4938", "5871", "5876", "5880", "6505",
    "6669", "9910", "1101", "2408", "3661",
]


def fetch_prices():
    import yfinance as yf
    rows = []
    sectors = {}
    for i, code in enumerate(TW50):
        ticker = f"{code}.TW"
        ok = False
        for attempt in range(3):
            try:
                df = yf.download(ticker, start=START, end=END, progress=False,
                                 auto_adjust=False, threads=False)
                if df is not None and len(df) > 100:
                    ok = True
                    break
            except Exception as e:
                print(f"  {ticker} attempt {attempt+1} failed: {e}")
            time.sleep(2.0)
        if not ok:
            print(f"  SKIP {ticker} (no data)")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df["prev_close"] = df["Close"].shift(1)
        df = df.dropna(subset=["prev_close"])
        for _, r in df.iterrows():
            rows.append([code, pd.Timestamp(r["Date"]).strftime("%Y-%m-%d"),
                         float(r["Close"]), float(r["Open"]), float(r["High"]),
                         float(r["Low"]), float(r["prev_close"]), float(r["Volume"])])
        # sector
        try:
            info = yf.Ticker(ticker).info
            sectors[code] = info.get("sector") or "Unknown"
        except Exception:
            sectors[code] = "Unknown"
        print(f"  [{i+1}/{len(TW50)}] {ticker}: {len(df)} rows, sector={sectors.get(code)}")
        time.sleep(1.0)
    return rows, sectors


def main():
    rows, sectors = fetch_prices()
    if not rows:
        print("no Taiwan data could be downloaded (Yahoo likely rate-limited).")
        return
    org = pd.DataFrame(rows, columns=["kdcode", "dt", "close", "open", "high",
                                      "low", "prev_close", "volume"])
    org = org.sort_values(["kdcode", "dt"])
    org.to_csv(os.path.join(DATASET, "tw50_org.csv"), index=False)
    print(f"wrote dataset/tw50_org.csv  ({len(org)} rows, "
          f"{org['kdcode'].nunique()} stocks)")

    sec = pd.DataFrame([[k, v] for k, v in sectors.items()],
                       columns=["kdcode", "sector"])
    sec.to_csv(os.path.join(DATASET, "tw50_sector.csv"), index=False)
    print(f"wrote dataset/tw50_sector.csv  ({len(sec)} stocks)")


if __name__ == "__main__":
    main()
