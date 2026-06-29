"""
Fetch historical PE/PB valuation data for the leader-stock universe.

Data source: AKShare stock_value_em, which wraps Eastmoney valuation history.

Output CSV schema:
    date, code, pe_ttm, pb, dividend_yield

Dividend yield is left empty by default because AKShare's stock_value_em does
not provide a reliable per-stock historical dividend-yield field. The backtest
can use PE/PB first and will incorporate dividend_yield if you later add it.

Usage:
    python3 fetch_akshare_valuation.py
    python3 fetch_akshare_valuation.py --output valuation.csv --start 2015-01-01
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import akshare as ak
except ImportError as exc:
    raise SystemExit("Missing dependency: akshare. Install with: pip install akshare") from exc


LEADERS = {
    "600900": "Changjiang Power",
    "600036": "China Merchants Bank",
    "601318": "Ping An Insurance",
    "600519": "Kweichow Moutai",
    "000333": "Midea Group",
    "600941": "China Mobile",
    "601088": "China Shenhua",
    "600030": "CITIC Securities",
}


FIELD_ALIASES = {
    "date": ["数据日期", "日期", "TRADE_DATE", "trade_date", "date"],
    "pe_ttm": ["PE(TTM)", "市盈率(TTM)", "PE_TTM", "pe_ttm"],
    "pb": ["市净率", "PB", "pb"],
}


def find_column(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    normalized = {str(col).strip(): col for col in df.columns}
    for name in names:
        if name in normalized:
            return normalized[name]
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for name in names:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def normalize_one(code: str, raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["date", "code", "pe_ttm", "pb", "dividend_yield"])

    date_col = find_column(raw, FIELD_ALIASES["date"])
    pe_col = find_column(raw, FIELD_ALIASES["pe_ttm"])
    pb_col = find_column(raw, FIELD_ALIASES["pb"])
    missing = [
        name
        for name, col in [("date", date_col), ("pe_ttm", pe_col), ("pb", pb_col)]
        if col is None
    ]
    if missing:
        raise ValueError(
            f"{code}: missing columns {missing}; got columns {list(raw.columns)}"
        )

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "code": code,
            "pe_ttm": pd.to_numeric(raw[pe_col], errors="coerce"),
            "pb": pd.to_numeric(raw[pb_col], errors="coerce"),
            "dividend_yield": pd.NA,
        }
    )
    out = out.dropna(subset=["date"]).sort_values("date")
    out = out.drop_duplicates(["date", "code"], keep="last")
    return out.reset_index(drop=True)


def fetch_one(code: str, raw_cache_dir: Path, refresh: bool) -> pd.DataFrame:
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_cache_dir / f"ak_value_{code}.csv"
    if cache_path.exists() and not refresh:
        raw = pd.read_csv(cache_path)
    else:
        raw = ak.stock_value_em(symbol=code)
        raw.to_csv(cache_path, index=False)
    return normalize_one(code, raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch AKShare valuation history.")
    parser.add_argument("--output", default="valuation.csv")
    parser.add_argument("--raw-cache-dir", default="data_cache/akshare_value")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end) if args.end else None
    raw_cache_dir = Path(args.raw_cache_dir)

    frames = []
    for code, name in LEADERS.items():
        print(f"Fetching {code} {name} ...")
        df = fetch_one(code, raw_cache_dir, args.refresh)
        df = df[df["date"] >= start]
        if end is not None:
            df = df[df["date"] <= end]
        print(f"  rows: {len(df)}, {df['date'].min()} -> {df['date'].max()}")
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["code", "date"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"\nSaved {len(result)} rows to {output}")


if __name__ == "__main__":
    main()
