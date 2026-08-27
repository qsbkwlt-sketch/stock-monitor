"""
Backtest a convertible-bond multi-factor rotation from daily scored snapshots.

The expected input is one or more CSV files produced by cb_multi_factor_selector.py:
    data_cache/convertible_bonds/cb_factor_scores_YYYY-MM-DD.csv

Strategy:
    - Daily risk check: sell filtered bonds immediately.
    - Scheduled rebalance: every N available snapshots, target equal-weight top ranks.
    - Buffer: keep current holdings while rank <= sell-rank.
    - Buy: fill empty slots from rank <= buy-rank.

Usage:
    python3 cb_factor_backtest.py
    python3 cb_factor_backtest.py --snapshot-glob "data_cache/convertible_bonds/cb_factor_scores_*.csv"
    python3 cb_factor_backtest.py --rebalance-steps 10 --positions 12 --buy-rank 20 --sell-rank 40

Outputs:
    cb_factor_backtest_output/cb_factor_backtest_summary.csv
    cb_factor_backtest_output/cb_factor_backtest_trades.csv
    cb_factor_backtest_output/cb_factor_backtest_equity.csv
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Position:
    code: str
    name: str
    shares: float
    entry_price: float
    entry_date: pd.Timestamp


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def load_snapshots(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    frames = []
    for path in files:
        df = pd.read_csv(path, dtype={"code": str, "stock_code": str})
        if "snapshot_date" not in df.columns:
            continue
        df["snapshot_date"] = normalize_date(df["snapshot_date"])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No snapshot files matched: {pattern}")
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["snapshot_date", "code", "price"])
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["rank"] = pd.to_numeric(out.get("rank"), errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["filter_reason"] = out.get("filter_reason", "").fillna("").astype(str)
    return out.sort_values(["snapshot_date", "rank", "code"]).reset_index(drop=True)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min()) if len(dd) else 0.0


def latest_prices(day: pd.DataFrame) -> dict[str, float]:
    return {
        str(row.code).zfill(6): float(row.price)
        for row in day.itertuples(index=False)
        if pd.notna(row.price) and float(row.price) > 0
    }


def portfolio_value(cash: float, positions: dict[str, Position], prices: dict[str, float]) -> float:
    value = cash
    for code, pos in positions.items():
        price = prices.get(code)
        if price is not None:
            value += pos.shares * price
    return float(value)


def sell_position(
    code: str,
    date: pd.Timestamp,
    price: float,
    cash: float,
    positions: dict[str, Position],
    trades: list[dict],
    reason: str,
    cost_rate: float,
) -> float:
    pos = positions.pop(code)
    fill = price * (1 - cost_rate)
    proceeds = pos.shares * fill
    cash += proceeds
    trades.append(
        {
            "date": date.strftime("%Y-%m-%d"),
            "code": code,
            "name": pos.name,
            "side": "SELL",
            "price": round(fill, 3),
            "shares": round(pos.shares, 3),
            "value": round(proceeds, 2),
            "pnl_pct": round((fill / pos.entry_price - 1) * 100, 2),
            "reason": reason,
        }
    )
    return cash


def buy_position(
    row: pd.Series,
    date: pd.Timestamp,
    target_value: float,
    cash: float,
    positions: dict[str, Position],
    trades: list[dict],
    cost_rate: float,
) -> float:
    price = float(row["price"])
    fill = price * (1 + cost_rate)
    invest = min(cash, target_value)
    if invest <= 0 or fill <= 0:
        return cash
    shares = invest / fill
    cash -= invest
    code = str(row["code"]).zfill(6)
    name = str(row.get("name", code))
    positions[code] = Position(
        code=code,
        name=name,
        shares=shares,
        entry_price=fill,
        entry_date=date,
    )
    trades.append(
        {
            "date": date.strftime("%Y-%m-%d"),
            "code": code,
            "name": name,
            "side": "BUY",
            "price": round(fill, 3),
            "shares": round(shares, 3),
            "value": round(invest, 2),
            "pnl_pct": np.nan,
            "reason": "ranked_entry",
        }
    )
    return cash


def row_by_code(day: pd.DataFrame) -> dict[str, pd.Series]:
    return {str(row["code"]).zfill(6): row for _, row in day.iterrows()}


def run_backtest(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    snapshots = load_snapshots(args.snapshot_glob)
    dates = list(snapshots["snapshot_date"].drop_duplicates().sort_values())
    if len(dates) < 2:
        raise ValueError(
            f"Need at least 2 snapshot dates for a backtest; found {len(dates)}. "
            "Run cb_multi_factor_selector.py daily or provide historical snapshots."
        )

    cash = args.initial_cash
    positions: dict[str, Position] = {}
    trades: list[dict] = []
    equity_rows: list[dict] = []

    for idx, dt in enumerate(dates):
        day = snapshots[snapshots["snapshot_date"] == dt].copy()
        day["code"] = day["code"].astype(str).str.zfill(6)
        prices = latest_prices(day)
        rows = row_by_code(day)

        # Daily risk sell: filter reason, missing price, or hard price cap.
        for code in list(positions):
            row = rows.get(code)
            price = prices.get(code)
            if row is None or price is None:
                continue
            filter_reason = str(row.get("filter_reason", ""))
            reason: Optional[str] = None
            if filter_reason:
                reason = f"risk_filter:{filter_reason}"
            elif float(row.get("price", 0)) >= args.hard_exit_price:
                reason = "hard_exit_price"
            if reason:
                cash = sell_position(code, dt, price, cash, positions, trades, reason, args.cost_rate)

        is_rebalance = idx == 0 or idx % args.rebalance_steps == 0
        if is_rebalance:
            tradable = day[(day["filter_reason"] == "") & day["rank"].notna()].copy()
            tradable = tradable.sort_values(["rank", "score", "double_low"], na_position="last")

            # Scheduled sell: only keep holdings that remain inside the buffer.
            for code in list(positions):
                row = rows.get(code)
                price = prices.get(code)
                rank = float(row["rank"]) if row is not None and pd.notna(row.get("rank")) else np.inf
                if price is not None and rank > args.sell_rank:
                    cash = sell_position(code, dt, price, cash, positions, trades, "rank_buffer_exit", args.cost_rate)

            keep_codes = set(positions)
            target_codes = []
            for _, row in tradable.iterrows():
                code = str(row["code"]).zfill(6)
                rank = float(row["rank"])
                if code in keep_codes or rank <= args.buy_rank:
                    target_codes.append(code)
                if len(target_codes) >= args.positions:
                    break

            equity_now = portfolio_value(cash, positions, prices)
            target_value = equity_now / args.positions if args.positions > 0 else 0
            for code in target_codes:
                if len(positions) >= args.positions:
                    break
                if code in positions:
                    continue
                row = rows.get(code)
                if row is None:
                    continue
                cash = buy_position(row, dt, target_value, cash, positions, trades, args.cost_rate)

        equity = portfolio_value(cash, positions, prices)
        equity_rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "equity": round(equity, 2),
                "cash": round(cash, 2),
                "positions": len(positions),
                "holdings": ",".join(sorted(positions)),
            }
        )

    # Liquidate at final snapshot for realized trade stats.
    final_dt = dates[-1]
    final_day = snapshots[snapshots["snapshot_date"] == final_dt].copy()
    final_day["code"] = final_day["code"].astype(str).str.zfill(6)
    final_prices = latest_prices(final_day)
    for code in list(positions):
        price = final_prices.get(code)
        if price is not None:
            cash = sell_position(code, final_dt, price, cash, positions, trades, "end_of_test", args.cost_rate)

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    summary = summarize(equity_df, trades_df, args)
    return summary, trades_df, equity_df


def summarize(equity: pd.DataFrame, trades: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    total_return = equity["equity"].iloc[-1] / args.initial_cash - 1 if not equity.empty else 0.0
    start = pd.to_datetime(equity["date"].iloc[0])
    end = pd.to_datetime(equity["date"].iloc[-1])
    years = (end - start).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else np.nan
    sells = trades[trades["side"] == "SELL"] if not trades.empty else pd.DataFrame()
    wins = sells[sells["pnl_pct"] > 0] if not sells.empty else pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "snapshots": len(equity),
                "initial_cash": args.initial_cash,
                "total_return_pct": round(total_return * 100, 2),
                "cagr_pct": round(cagr * 100, 2) if not pd.isna(cagr) else np.nan,
                "max_drawdown_pct": round(max_drawdown(equity["equity"]) * 100, 2) if not equity.empty else 0.0,
                "buy_count": int((trades["side"] == "BUY").sum()) if not trades.empty else 0,
                "sell_count": int((trades["side"] == "SELL").sum()) if not trades.empty else 0,
                "win_rate_pct": round(len(wins) / len(sells) * 100, 1) if len(sells) else 0.0,
                "avg_sell_pnl_pct": round(float(sells["pnl_pct"].mean()), 2) if len(sells) else 0.0,
                "rebalance_steps": args.rebalance_steps,
                "positions": args.positions,
                "buy_rank": args.buy_rank,
                "sell_rank": args.sell_rank,
            }
        ]
    )


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary, trades, equity = run_backtest(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    summary_path = output_dir / "cb_factor_backtest_summary.csv"
    trades_path = output_dir / "cb_factor_backtest_trades.csv"
    equity_path = output_dir / "cb_factor_backtest_equity.csv"
    summary.to_csv(summary_path, index=False)
    trades.to_csv(trades_path, index=False)
    equity.to_csv(equity_path, index=False)

    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 180)
    print("\n=== CB Factor Backtest Summary ===")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {trades_path}")
    print(f"  {equity_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest convertible-bond factor rotation from snapshots.")
    parser.add_argument("--snapshot-glob", default="data_cache/convertible_bonds/cb_factor_scores_*.csv")
    parser.add_argument("--output-dir", default="cb_factor_backtest_output")
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--positions", type=int, default=12)
    parser.add_argument("--buy-rank", type=float, default=20)
    parser.add_argument("--sell-rank", type=float, default=40)
    parser.add_argument("--rebalance-steps", type=int, default=10, help="10 trading snapshots is roughly two weeks.")
    parser.add_argument("--hard-exit-price", type=float, default=135.0)
    parser.add_argument("--cost-rate", type=float, default=0.001)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
