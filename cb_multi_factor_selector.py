"""
Convertible bond multi-factor selector inspired by "持有封基" style ranking.

This script is a current-snapshot selector, not a historical backtest. It uses
AKShare's Jisilu convertible bond table when available, applies risk filters,
then ranks bonds by low price, low conversion premium, higher YTM, acceptable
size, and liquidity.

Usage:
    python3 cb_multi_factor_selector.py --refresh
    python3 cb_multi_factor_selector.py --top 20 --max-price 130 --max-premium 35
    python3 cb_multi_factor_selector.py --cookie "your_jisilu_cookie" --refresh

Outputs:
    cb_factor_output/cb_factor_scores.csv
    cb_factor_output/cb_factor_candidates.csv
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import akshare as ak
except ImportError as exc:
    raise SystemExit("Missing dependency: akshare. Install with: pip install akshare") from exc


FIELD_ALIASES = {
    "code": ["代码", "bond_id", "证券代码", "转债代码", "code"],
    "name": ["转债名称", "名称", "bond_nm", "证券简称", "name"],
    "price": ["现价", "price", "最新价", "trade"],
    "change_pct": ["涨跌幅", "涨跌幅(%)", "changepercent"],
    "stock_code": ["正股代码", "stock_id"],
    "stock_name": ["正股名称", "stock_nm"],
    "stock_price": ["正股价", "sprice"],
    "stock_change_pct": ["正股涨跌", "正股涨跌幅"],
    "stock_pb": ["正股PB", "pb"],
    "convert_price": ["转股价", "convert_price"],
    "convert_value": ["转股价值", "convert_value"],
    "premium_rate": ["转股溢价率", "premium_rt", "溢价率"],
    "rating": ["债券评级", "评级", "rating_cd"],
    "put_trigger_price": ["回售触发价"],
    "redeem_trigger_price": ["强赎触发价"],
    "bond_ratio": ["转债占比"],
    "maturity_date": ["到期时间", "到期日", "maturity_dt"],
    "remaining_years": ["剩余年限", "year_left"],
    "remaining_size": ["剩余规模", "curr_iss_amt"],
    "amount": ["成交额", "amount"],
    "turnover_rate": ["换手率", "turnover_rt"],
    "ytm": ["到期税前收益", "ytm_rt", "税前收益"],
    "double_low": ["双低", "dblow"],
    "redeem_status": ["强赎状态"],
}

RATING_SCORE = {
    "AAA": 6,
    "AA+": 5,
    "AA": 4,
    "AA-": 3,
    "A+": 2,
    "A": 1,
    "A-": 0,
}

DEFAULT_WEIGHTS = {
    "price_rank": 0.32,
    "premium_rank": 0.30,
    "ytm_rank": 0.16,
    "amount_rank": 0.10,
    "size_rank": 0.07,
    "rating_rank": 0.05,
}


def find_column(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    exact = {str(col).strip(): col for col in df.columns}
    for alias in aliases:
        if alias in exact:
            return exact[alias]
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for alias in aliases:
        key = alias.lower()
        if key in lowered:
            return lowered[key]
    return None


def parse_number(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    text = text.str.replace("%", "", regex=False)
    text = text.str.replace(",", "", regex=False)
    text = text.mask(text.isin(["", "-", "None", "nan", "<NA>"]), pd.NA)
    return pd.to_numeric(text, errors="coerce")


def normalize_snapshot(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("Convertible bond snapshot is empty.")

    out = pd.DataFrame()
    for field, aliases in FIELD_ALIASES.items():
        col = find_column(raw, aliases)
        if col is not None:
            out[field] = raw[col]
        else:
            out[field] = np.nan

    text_cols = ["code", "name", "stock_code", "stock_name", "rating", "maturity_date", "redeem_status"]
    for col in text_cols:
        out[col] = out[col].astype(str).str.strip().replace({"nan": ""})

    numeric_cols = [
        "price",
        "change_pct",
        "stock_price",
        "stock_change_pct",
        "stock_pb",
        "convert_price",
        "convert_value",
        "premium_rate",
        "put_trigger_price",
        "redeem_trigger_price",
        "bond_ratio",
        "remaining_years",
        "remaining_size",
        "amount",
        "turnover_rate",
        "ytm",
        "double_low",
    ]
    for col in numeric_cols:
        out[col] = parse_number(out[col])

    missing_required = [
        col for col in ["code", "name", "price", "premium_rate"] if out[col].isna().all()
    ]
    if missing_required:
        raise ValueError(
            f"Snapshot missing required fields {missing_required}; raw columns: {list(raw.columns)}"
        )

    out["rating_score"] = out["rating"].map(RATING_SCORE).fillna(0)
    out["double_low_calc"] = out["price"] + out["premium_rate"]
    out["double_low"] = out["double_low"].fillna(out["double_low_calc"])
    out["snapshot_date"] = date.today().isoformat()
    return out.dropna(subset=["code", "price", "premium_rate"]).reset_index(drop=True)


def fetch_jsl(cookie: Optional[str], cache_path: Path, refresh: bool) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)
    raw = ak.bond_cb_jsl(cookie=cookie)
    raw.to_csv(cache_path, index=False)
    return raw


def fetch_redeem(cache_path: Path, refresh: bool) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)
    raw = ak.bond_cb_redeem_jsl()
    raw.to_csv(cache_path, index=False)
    return raw


def attach_redeem_status(snapshot: pd.DataFrame, redeem_raw: pd.DataFrame) -> pd.DataFrame:
    if redeem_raw.empty:
        return snapshot
    code_col = find_column(redeem_raw, FIELD_ALIASES["code"])
    status_col = find_column(redeem_raw, FIELD_ALIASES["redeem_status"])
    if code_col is None or status_col is None:
        return snapshot
    redeem_cols = pd.DataFrame(
        {
            "code": redeem_raw[code_col].astype(str).str.strip(),
            "redeem_status": redeem_raw[status_col].astype(str).str.strip(),
        }
    )
    redeem_cols = redeem_cols[redeem_cols["redeem_status"] != ""]
    if redeem_cols.empty:
        return snapshot
    out = snapshot.merge(redeem_cols, on="code", how="left", suffixes=("", "_redeem"))
    left_status = out["redeem_status"].astype("string").str.strip()
    right_status = out["redeem_status_redeem"].astype("string").str.strip()
    left_status = left_status.mask(left_status.isin(["", "nan", "<NA>"]), pd.NA)
    right_status = right_status.mask(right_status.isin(["", "nan", "<NA>"]), pd.NA)
    out["redeem_status"] = left_status.fillna(right_status).fillna("").astype(str)
    return out.drop(columns=["redeem_status_redeem"])


def apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["filter_reason"] = ""

    def mark(mask: pd.Series, reason: str) -> None:
        out.loc[mask, "filter_reason"] = (
            out.loc[mask, "filter_reason"].where(out.loc[mask, "filter_reason"] == "", out.loc[mask, "filter_reason"] + "|")
            + reason
        )

    mark(out["price"] < args.min_price, "price_too_low")
    mark(out["price"] > args.max_price, "price_too_high")
    mark(out["premium_rate"] > args.max_premium, "premium_too_high")
    mark(out["premium_rate"] < args.min_premium, "premium_too_low")
    mark(out["remaining_size"] < args.min_remaining_size, "size_too_small")
    mark(out["amount"] < args.min_amount, "liquidity_too_low")
    mark(out["remaining_years"] < args.min_remaining_years, "maturity_too_close")
    mark(out["rating_score"] < args.min_rating_score, "rating_too_low")

    if args.exclude_redeem:
        risky_words = ["强赎", "已公告", "最后交易", "到期赎回", "即将"]
        redeem_text = out["redeem_status"].fillna("").astype(str)
        mark(redeem_text.apply(lambda x: any(word in x for word in risky_words)), "redeem_risk")

    if args.exclude_st:
        stock_name = out["stock_name"].fillna("").astype(str).str.upper()
        mark(stock_name.str.contains("ST", regex=False), "st_stock")

    return out


def pct_rank(series: pd.Series, ascending: bool) -> pd.Series:
    valid = series.replace([np.inf, -np.inf], np.nan)
    ranked = valid.rank(ascending=ascending, pct=True, na_option="bottom")
    return ranked.fillna(1.0)


def add_scores(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    tradable = out["filter_reason"] == ""
    work = out.loc[tradable].copy()

    if work.empty:
        out["score"] = np.nan
        out["rank"] = np.nan
        return out

    # Lower percentile is better for ranks. Score is therefore lower is better.
    work["price_rank"] = pct_rank(work["price"], ascending=True)
    work["premium_rank"] = pct_rank(work["premium_rate"], ascending=True)
    work["ytm_rank"] = pct_rank(work["ytm"], ascending=False)
    work["amount_rank"] = pct_rank(work["amount"], ascending=False)
    work["size_rank"] = pct_rank(work["remaining_size"], ascending=True)
    work["rating_rank"] = pct_rank(work["rating_score"], ascending=False)
    work["double_low_rank"] = pct_rank(work["double_low"], ascending=True)

    score = 0.0
    for col, weight in weights.items():
        score = score + work[col] * weight
    work["score"] = score
    work["rank"] = work["score"].rank(ascending=True, method="first").astype(int)

    rank_cols = [
        "price_rank",
        "premium_rank",
        "ytm_rank",
        "amount_rank",
        "size_rank",
        "rating_rank",
        "double_low_rank",
        "score",
        "rank",
    ]
    out = out.merge(work[["code"] + rank_cols], on="code", how="left")
    return out


def parse_weights(text: str) -> dict[str, float]:
    if not text:
        return DEFAULT_WEIGHTS.copy()
    weights = DEFAULT_WEIGHTS.copy()
    for item in text.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        weights[key.strip()] = float(value)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Factor weights must sum to a positive value.")
    return {key: value / total for key, value in weights.items()}


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "snapshot_date",
        "rank",
        "code",
        "name",
        "price",
        "premium_rate",
        "double_low",
        "ytm",
        "remaining_years",
        "remaining_size",
        "amount",
        "turnover_rate",
        "rating",
        "stock_code",
        "stock_name",
        "stock_price",
        "stock_change_pct",
        "stock_pb",
        "redeem_status",
        "score",
        "filter_reason",
        "price_rank",
        "premium_rank",
        "ytm_rank",
        "amount_rank",
        "size_rank",
        "rating_rank",
    ]
    existing = [col for col in cols if col in df.columns]
    return df[existing]


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cookie = args.cookie or os.environ.get("JISILU_COOKIE")
    today = date.today().isoformat()
    snapshot_cache = cache_dir / f"cb_jsl_{today}.csv"
    redeem_cache = cache_dir / f"cb_redeem_jsl_{today}.csv"

    raw = fetch_jsl(cookie, snapshot_cache, args.refresh)
    snapshot = normalize_snapshot(raw)

    try:
        redeem_raw = fetch_redeem(redeem_cache, args.refresh)
        snapshot = attach_redeem_status(snapshot, redeem_raw)
    except Exception as exc:
        print(f"Warning: failed to fetch/merge redeem table: {exc}")

    filtered = apply_filters(snapshot, args)
    weights = parse_weights(args.weights)
    scored = add_scores(filtered, weights)
    scored = select_columns(scored).sort_values(
        ["filter_reason", "rank", "double_low"], na_position="last"
    )
    candidates = scored[scored["filter_reason"] == ""].head(args.top).copy()

    scores_path = output_dir / "cb_factor_scores.csv"
    candidates_path = output_dir / "cb_factor_candidates.csv"
    history_path = cache_dir / f"cb_factor_scores_{today}.csv"
    scored.to_csv(scores_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    scored.to_csv(history_path, index=False)

    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 220)
    print("\n=== Convertible Bond Multi-Factor Candidates ===")
    if candidates.empty:
        print("No candidates passed filters. Try relaxing price/premium/liquidity filters.")
    else:
        show_cols = [
            "rank",
            "code",
            "name",
            "price",
            "premium_rate",
            "double_low",
            "ytm",
            "remaining_size",
            "amount",
            "rating",
            "stock_name",
            "redeem_status",
            "score",
        ]
        print(candidates[[col for col in show_cols if col in candidates.columns]].to_string(index=False))

    if len(snapshot) < args.min_expected_rows:
        print(
            f"\nWarning: only {len(snapshot)} rows fetched. Jisilu anonymous access may be limited; "
            "pass --cookie or set JISILU_COOKIE for a fuller universe."
        )

    print("\nSaved:")
    print(f"  {scores_path}")
    print(f"  {candidates_path}")
    print(f"  {history_path}")
    print("\nNote: This is a ranking/selection aid, not investment advice or a backtest.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convertible bond multi-factor selector.")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--cache-dir", default="data_cache/convertible_bonds")
    parser.add_argument("--output-dir", default="cb_factor_output")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cookie", default=None, help="Optional Jisilu cookie. Can also use JISILU_COOKIE env var.")
    parser.add_argument("--min-expected-rows", type=int, default=100)

    parser.add_argument("--min-price", type=float, default=95.0)
    parser.add_argument("--max-price", type=float, default=130.0)
    parser.add_argument("--min-premium", type=float, default=-20.0)
    parser.add_argument("--max-premium", type=float, default=35.0)
    parser.add_argument("--min-remaining-size", type=float, default=1.0)
    parser.add_argument("--min-amount", type=float, default=1000.0)
    parser.add_argument("--min-remaining-years", type=float, default=0.5)
    parser.add_argument("--min-rating-score", type=float, default=0.0)
    parser.add_argument("--exclude-redeem", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-st", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--weights",
        default="",
        help=(
            "Comma-separated factor weights, e.g. "
            "price_rank=0.35,premium_rank=0.35,ytm_rank=0.15,amount_rank=0.1,size_rank=0.05"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
