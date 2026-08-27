"""
Generate daily signals for the convertible-bond multi-factor strategy.

Inputs:
    - Latest scored snapshot from cb_multi_factor_selector.py
    - Optional holdings CSV when you want account-aware sell/hold hints

Holdings CSV schema:
    code,name,shares,cost_price

Usage:
    python3 cb_factor_signal.py
    python3 cb_factor_signal.py --capital 40000
    python3 cb_factor_signal.py --holdings my_cb_holdings.csv --capital 40000

Outputs:
    cb_factor_signal_output/cb_daily_signals.csv
    cb_factor_signal_output/cb_target_portfolio.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HOLDING_COLUMNS = ["code", "name", "shares", "cost_price"]


def normalize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.zfill(6)


def load_scores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"code": str, "stock_code": str})
    required = {"snapshot_date", "code", "name", "price", "rank", "filter_reason"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Scores CSV missing columns: {sorted(missing)}")
    df["code"] = normalize_code(df["code"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["filter_reason"] = df["filter_reason"].fillna("").astype(str)
    return df.sort_values(["rank", "double_low"], na_position="last").reset_index(drop=True)


def load_or_create_holdings(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, dtype={"code": str})
        for col in HOLDING_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[HOLDING_COLUMNS].copy()
        df["code"] = normalize_code(df["code"])
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
        df["cost_price"] = pd.to_numeric(df["cost_price"], errors="coerce")
        return df[df["shares"] > 0].reset_index(drop=True)

    template = pd.DataFrame(columns=HOLDING_COLUMNS)
    template.to_csv(path, index=False)
    return template


def signal_for_holding(row: pd.Series, args: argparse.Namespace) -> tuple[str, str]:
    if row.get("missing_snapshot", False):
        return "REVIEW", "holding_missing_from_snapshot"
    filter_reason = str(row.get("filter_reason", ""))
    if filter_reason:
        return "SELL", f"risk_filter:{filter_reason}"
    price = float(row["price"])
    rank = float(row["rank"]) if pd.notna(row["rank"]) else np.inf
    if price >= args.hard_exit_price:
        return "SELL", "hard_exit_price"
    if rank > args.sell_rank:
        return "SELL", "rank_buffer_exit"
    if rank <= args.buy_rank:
        return "HOLD_STRONG", "still_in_buy_zone"
    return "HOLD", "inside_buffer"


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = load_scores(args.scores)
    holdings = load_or_create_holdings(Path(args.holdings)) if args.holdings else pd.DataFrame(columns=HOLDING_COLUMNS)
    held_codes = set(holdings["code"])

    tradable = scores[(scores["filter_reason"] == "") & scores["rank"].notna()].copy()
    target = tradable.head(args.positions).copy()
    target_codes = set(target["code"])

    rows = []
    score_by_code = {row["code"]: row for _, row in scores.iterrows()}

    for _, holding in holdings.iterrows():
        code = holding["code"]
        score_row = score_by_code.get(code)
        if score_row is None:
            merged = holding.to_dict()
            merged.update({"missing_snapshot": True, "price": np.nan, "rank": np.nan, "filter_reason": ""})
        else:
            merged = score_row.to_dict()
            merged.update(
                {
                    "shares": holding["shares"],
                    "cost_price": holding["cost_price"],
                    "missing_snapshot": False,
                }
            )
        action, reason = signal_for_holding(pd.Series(merged), args)
        market_value = merged.get("shares", 0) * merged.get("price", 0) if pd.notna(merged.get("price", np.nan)) else np.nan
        pnl_pct = (
            merged["price"] / merged["cost_price"] - 1
            if pd.notna(merged.get("price", np.nan)) and pd.notna(merged.get("cost_price", np.nan)) and merged["cost_price"] > 0
            else np.nan
        )
        rows.append(
            {
                "snapshot_date": merged.get("snapshot_date", ""),
                "action": action,
                "reason": reason,
                "code": code,
                "name": merged.get("name", holding.get("name", code)),
                "price": merged.get("price", np.nan),
                "rank": merged.get("rank", np.nan),
                "shares": merged.get("shares", 0),
                "market_value": round(market_value, 2) if pd.notna(market_value) else np.nan,
                "pnl_pct": round(pnl_pct * 100, 2) if pd.notna(pnl_pct) else np.nan,
                "filter_reason": merged.get("filter_reason", ""),
            }
        )

    buy_slots = max(args.positions - len([r for r in rows if r["action"] in {"HOLD", "HOLD_STRONG"}]), 0)
    buy_candidates = target[~target["code"].isin(held_codes)].head(buy_slots)
    target_value = args.capital / args.positions if args.capital and args.positions > 0 else np.nan

    for _, row in buy_candidates.iterrows():
        shares = target_value / row["price"] if pd.notna(target_value) and row["price"] > 0 else np.nan
        rows.append(
            {
                "snapshot_date": row["snapshot_date"],
                "action": "BUY",
                "reason": "ranked_entry",
                "code": row["code"],
                "name": row["name"],
                "price": row["price"],
                "rank": row["rank"],
                "shares": round(shares, 3) if pd.notna(shares) else np.nan,
                "market_value": round(target_value, 2) if pd.notna(target_value) else np.nan,
                "pnl_pct": np.nan,
                "filter_reason": "",
            }
        )

    watch = tradable[(~tradable["code"].isin(held_codes | target_codes)) & (tradable["rank"] <= args.buy_rank)].head(args.watch)
    for _, row in watch.iterrows():
        rows.append(
            {
                "snapshot_date": row["snapshot_date"],
                "action": "WATCH",
                "reason": "buy_zone_but_no_slot",
                "code": row["code"],
                "name": row["name"],
                "price": row["price"],
                "rank": row["rank"],
                "shares": np.nan,
                "market_value": np.nan,
                "pnl_pct": np.nan,
                "filter_reason": "",
            }
        )

    signals = pd.DataFrame(rows)
    if signals.empty:
        signals = pd.DataFrame(columns=["snapshot_date", "action", "reason", "code", "name", "price", "rank"])
    action_order = {"SELL": 0, "BUY": 1, "HOLD_STRONG": 2, "HOLD": 3, "WATCH": 4, "REVIEW": 5}
    signals["action_order"] = signals["action"].map(action_order).fillna(9)
    signals = signals.sort_values(["action_order", "rank"], na_position="last").drop(columns=["action_order"])

    target_portfolio = target.copy()
    if args.capital:
        target_portfolio["target_weight_pct"] = round(100 / args.positions, 2)
        target_portfolio["target_value"] = round(target_value, 2)
        target_portfolio["target_shares"] = (target_value / target_portfolio["price"]).round(3)

    signals_path = output_dir / "cb_daily_signals.csv"
    target_path = output_dir / "cb_target_portfolio.csv"
    signals.to_csv(signals_path, index=False)
    target_portfolio.to_csv(target_path, index=False)

    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 180)
    print("\n=== CB Daily Signals ===")
    show_cols = ["action", "reason", "code", "name", "price", "rank", "shares", "market_value", "pnl_pct"]
    print(signals[[col for col in show_cols if col in signals.columns]].to_string(index=False))
    print("\nSaved:")
    print(f"  {signals_path}")
    print(f"  {target_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily signals for CB factor strategy.")
    parser.add_argument("--scores", default="cb_factor_output/cb_factor_scores.csv")
    parser.add_argument("--holdings", default=None)
    parser.add_argument("--output-dir", default="cb_factor_signal_output")
    parser.add_argument("--capital", type=float, default=40000.0)
    parser.add_argument("--positions", type=int, default=10)
    parser.add_argument("--buy-rank", type=float, default=20)
    parser.add_argument("--sell-rank", type=float, default=40)
    parser.add_argument("--hard-exit-price", type=float, default=135.0)
    parser.add_argument("--watch", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
