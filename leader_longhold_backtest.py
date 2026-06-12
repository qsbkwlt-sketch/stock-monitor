"""
Low-frequency leader-stock backtest.

This script is designed for a "buy good leaders at reasonable points and hold"
style:

1. Check signals monthly, not daily.
2. Buy only when a multi-factor score is strong enough.
3. Hold by default.
4. Exit gradually when long-term trend breaks or real valuation percentile is high.

Historical valuation is optional. If no valuation CSV is provided, the script
uses the stock's 3-year price percentile only as a scoring aid. Price
percentile will not trigger valuation-only exits.

Usage:
    python3 leader_longhold_backtest.py --end 2026-06-12
    python3 leader_longhold_backtest.py --no-download --end 2026-06-12
    python3 leader_longhold_backtest.py --valuation-csv valuation.csv

Expected valuation CSV columns:
    date, code, pe_ttm, pb

Optional valuation CSV column:
    dividend_yield
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: yfinance. Install with: pip install yfinance pandas numpy"
    ) from exc


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

MARKET_INDEX = "000300.SS"


@dataclass(frozen=True)
class CostModel:
    buy_commission: float = 0.0003
    sell_commission: float = 0.0003
    stamp_tax: float = 0.0005
    slippage: float = 0.001


@dataclass
class Holding:
    shares: float
    entry_date: str
    entry_price: float
    trend_bad_months: int = 0
    trend_trim_done: bool = False
    valuation_trim_done: bool = False


def to_yahoo_ticker(code: str) -> str:
    if code.startswith(("6", "9")):
        return f"{code}.SS"
    return f"{code}.SZ"


def normalize_yf_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index()
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
    if "adj_close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"adj_close": "close"})
    cols = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Downloaded data missing columns: {missing}")
    df = df[cols].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def download_or_load(
    ticker: str,
    start: str,
    end: Optional[str],
    cache_dir: Path,
    no_download: bool,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker.replace('.', '_')}_{start}_{end or 'today'}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["date"])
    if no_download:
        raise FileNotFoundError(f"Cache not found for {ticker}: {cache_path}")
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    df = normalize_yf_frame(raw)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df.to_csv(cache_path, index=False)
    return df


def rolling_percentile_last(series: pd.Series, window: int) -> pd.Series:
    def pct_rank(values: np.ndarray) -> float:
        current = values[-1]
        if np.isnan(current):
            return np.nan
        valid = values[~np.isnan(values)]
        if len(valid) < max(60, window // 4):
            return np.nan
        return float((valid <= current).sum() / len(valid))

    return series.rolling(window, min_periods=max(60, window // 4)).apply(pct_rank, raw=True)


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_price_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for n in [20, 60, 120, 250]:
        out[f"ma{n}"] = out["close"].rolling(n).mean()
    out["ma120_slope60"] = out["ma120"] / out["ma120"].shift(60) - 1
    out["ma250_slope60"] = out["ma250"] / out["ma250"].shift(60) - 1
    out["ret_6m"] = out["close"] / out["close"].shift(126) - 1
    out["ret_12m"] = out["close"] / out["close"].shift(252) - 1
    out["high_1y"] = out["high"].rolling(252).max()
    out["drawdown_1y"] = out["close"] / out["high_1y"] - 1
    out["vol_60"] = out["close"].pct_change().rolling(60).std() * np.sqrt(252)
    out["rsi14"] = rsi(out["close"])
    out["price_pct_5y"] = rolling_percentile_last(out["close"], 1250)
    out["price_pct_3y"] = rolling_percentile_last(out["close"], 756)
    return out


def load_valuation(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    val = pd.read_csv(path, dtype={"code": str}, parse_dates=["date"])
    required = {"date", "code", "pe_ttm", "pb"}
    missing = required - set(val.columns)
    if missing:
        raise ValueError(f"Valuation CSV missing columns: {sorted(missing)}")
    if "dividend_yield" not in val.columns:
        val["dividend_yield"] = np.nan
    val = val.sort_values(["code", "date"]).copy()
    val["pe_pct_5y"] = val.groupby("code")["pe_ttm"].transform(
        lambda s: rolling_percentile_last(s, 1250)
    )
    val["pb_pct_5y"] = val.groupby("code")["pb"].transform(
        lambda s: rolling_percentile_last(s, 1250)
    )
    val["dy_pct_5y"] = val.groupby("code")["dividend_yield"].transform(
        lambda s: rolling_percentile_last(s, 1250)
    )
    val["dy_cheap_pct"] = 1 - val["dy_pct_5y"]
    val["valuation_pct"] = val[["pe_pct_5y", "pb_pct_5y", "dy_cheap_pct"]].mean(axis=1)
    return val[["date", "code", "valuation_pct", "pe_ttm", "pb", "dividend_yield"]]


def attach_valuation(
    code: str,
    df: pd.DataFrame,
    valuations: Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    if valuations is None:
        out["valuation_pct"] = out["price_pct_3y"]
        out["valuation_is_real"] = False
        return out, "price_percentile_proxy"

    one = valuations[valuations["code"] == code].sort_values("date")
    if one.empty:
        out["valuation_pct"] = out["price_pct_3y"]
        out["valuation_is_real"] = False
        return out, "price_percentile_proxy"

    out = pd.merge_asof(
        out.sort_values("date"),
        one.sort_values("date"),
        on="date",
        direction="backward",
    )
    out["valuation_pct"] = out["valuation_pct"].fillna(out["price_pct_3y"])
    out["valuation_is_real"] = out["pe_ttm"].notna() | out["pb"].notna()
    return out, "pe_pb_percentile"


def merge_market(stock: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    m = market[
        ["date", "close", "ma250", "ma250_slope60", "ret_6m", "ret_12m"]
    ].rename(
        columns={
            "close": "mkt_close",
            "ma250": "mkt_ma250",
            "ma250_slope60": "mkt_ma250_slope60",
            "ret_6m": "mkt_ret_6m",
            "ret_12m": "mkt_ret_12m",
        }
    )
    return stock.merge(m, on="date", how="left").ffill()


def entry_score(row: pd.Series) -> tuple[float, list[str]]:
    score = 0.0
    notes: list[str] = []

    market_ok = True
    if "mkt_filter_enabled" in row and row["mkt_filter_enabled"]:
        market_ok = row["mkt_close"] >= row["mkt_ma250"] * 0.92
    if not market_ok:
        return 0.0, ["market_weak"]

    if row["close"] > row["ma250"] and row["ma250_slope60"] > 0:
        score += 25
        notes.append("long_trend_up")
    elif row["close"] > row["ma250"] * 0.95 and row["ma120_slope60"] >= -0.02:
        score += 15
        notes.append("trend_neutral")

    dd = row["drawdown_1y"]
    close_to_ma120 = row["close"] / row["ma120"] - 1
    if -0.25 <= dd <= -0.08 and -0.10 <= close_to_ma120 <= 0.08:
        score += 25
        notes.append("reasonable_pullback")
    elif -0.35 <= dd < -0.25 and row["close"] > row["ma250"] * 0.92:
        score += 15
        notes.append("deep_pullback")
    elif -0.08 < dd <= -0.02 and row["close"] <= row["ma120"] * 1.08:
        score += 12
        notes.append("shallow_pullback")
    elif row["close"] > row["ma120"] and row["ma120_slope60"] > 0.02 and row["close"] <= row["ma120"] * 1.12:
        score += 16
        notes.append("trend_continuation")

    val_pct = row["valuation_pct"]
    valuation_is_real = bool(row.get("valuation_is_real", False))
    if valuation_is_real:
        if val_pct <= 0.35:
            score += 25
            notes.append("valuation_low")
        elif val_pct <= 0.55:
            score += 18
            notes.append("valuation_fair")
        elif val_pct <= 0.70:
            score += 10
            notes.append("valuation_acceptable")
    else:
        if val_pct <= 0.35:
            score += 15
            notes.append("price_percentile_low")
        elif val_pct <= 0.55:
            score += 10
            notes.append("price_percentile_fair")
        elif val_pct <= 0.70:
            score += 5
            notes.append("price_percentile_acceptable")

    mkt_ret_6m = row["mkt_ret_6m"] if not pd.isna(row.get("mkt_ret_6m", np.nan)) else 0.0
    rel_6m = row["ret_6m"] - mkt_ret_6m
    if rel_6m > 0 and row["vol_60"] <= 0.35:
        score += 15
        notes.append("relative_strength_low_vol")
    elif rel_6m > -0.05:
        score += 8
        notes.append("relative_strength_ok")

    if row["rsi14"] > 75 or row["close"] > row["ma120"] * 1.25:
        score -= 20
        notes.append("overheated_penalty")

    return max(score, 0.0), notes


def exit_decision(row: pd.Series, holding: Holding, args: argparse.Namespace) -> tuple[float, str]:
    if row["close"] < row["ma250"]:
        holding.trend_bad_months += 1
    else:
        holding.trend_bad_months = 0
        holding.trend_trim_done = False

    hard_stop = 0.78 if args.core_hold else 0.85
    trend_crash = row["close"] < row["ma250"] * hard_stop
    trend_broken = row["close"] < row["ma250"] * 0.97 and row["ma250_slope60"] < -0.03
    trend_confirmed = holding.trend_bad_months >= 4 and row["ma120"] < row["ma250"] * 0.98

    if trend_crash:
        return 1.0, "trend_hard_exit"
    if args.core_hold and args.core_trend_trim > 0 and trend_confirmed and not holding.trend_trim_done:
        holding.trend_trim_done = True
        return args.core_trend_trim, f"core_trend_confirmed_trim_{int(args.core_trend_trim * 100)}pct"

    val_pct = row["valuation_pct"]
    valuation_is_real = bool(row.get("valuation_is_real", False))
    if valuation_is_real:
        if val_pct >= args.valuation_extreme_pct:
            return 1.0, "valuation_extreme_exit"
        if val_pct >= args.valuation_high_pct and not holding.valuation_trim_done:
            holding.valuation_trim_done = True
            return args.valuation_trim, f"valuation_high_trim_{int(args.valuation_trim * 100)}pct"
        if val_pct < args.valuation_reset_pct:
            holding.valuation_trim_done = False
    else:
        if val_pct < 0.75 or row["close"] > row["ma120"]:
            holding.valuation_trim_done = False

    if args.core_hold:
        return 0.0, ""

    if trend_confirmed:
        return 1.0, "trend_confirmed_exit"
    if trend_broken and not holding.trend_trim_done:
        holding.trend_trim_done = True
        return 0.5, "trend_break_trim_50pct"

    return 0.0, ""


def is_earnings_avoid_month(signal_date: pd.Timestamp, args: argparse.Namespace) -> bool:
    if not args.avoid_earnings:
        return False
    months = {
        int(item.strip())
        for item in args.avoid_earnings_months.split(",")
        if item.strip()
    }
    return signal_date.month in months


def period_end_dates(df: pd.DataFrame, frequency: str) -> list[pd.Timestamp]:
    freq_map = {
        "daily": None,
        "weekly": "W-FRI",
        "monthly": "M",
    }
    if frequency not in freq_map:
        raise ValueError(f"Unsupported signal frequency: {frequency}")
    valid = df.dropna(subset=["ma250"]).copy()
    if valid.empty:
        return []
    if frequency == "daily":
        return list(valid["date"])
    return list(valid.groupby(valid["date"].dt.to_period(freq_map[frequency]))["date"].max())


def month_end_dates(market: pd.DataFrame) -> list[pd.Timestamp]:
    m = market.dropna(subset=["ma250"]).copy()
    return list(m.groupby(m["date"].dt.to_period("M"))["date"].max())


def union_period_end_dates(data_by_code: dict[str, pd.DataFrame], frequency: str) -> list[pd.Timestamp]:
    dates = []
    for df in data_by_code.values():
        dates.extend(period_end_dates(df, frequency))
    return sorted(set(dates))


def latest_row_on_or_before(df: pd.DataFrame, date: pd.Timestamp) -> Optional[pd.Series]:
    loc = df["date"].searchsorted(date, side="right") - 1
    if loc < 0:
        return None
    return df.iloc[loc]


def execute_price_on_or_after(df: pd.DataFrame, date: pd.Timestamp, side: str) -> tuple[pd.Timestamp, float]:
    loc = df["date"].searchsorted(date, side="right")
    if loc >= len(df):
        loc = len(df) - 1
        price = df.iloc[loc]["close"]
    else:
        price = df.iloc[loc]["open"]
    return df.iloc[loc]["date"], float(price)


def portfolio_value(
    cash: float,
    holdings: dict[str, Holding],
    data_by_code: dict[str, pd.DataFrame],
    date: pd.Timestamp,
) -> float:
    value = cash
    for code, holding in holdings.items():
        row = latest_row_on_or_before(data_by_code[code], date)
        if row is not None:
            value += holding.shares * row["close"]
    return float(value)


def run_backtest(
    data_by_code: dict[str, pd.DataFrame],
    market: pd.DataFrame,
    args: argparse.Namespace,
    cost: CostModel,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cash = 1.0
    holdings: dict[str, Holding] = {}
    trades: list[dict] = []
    equity_rows: list[dict] = []
    score_rows: list[dict] = []

    signal_dates = (
        period_end_dates(market, args.signal_frequency)
        if args.use_market_filter
        else union_period_end_dates(data_by_code, args.signal_frequency)
    )
    for signal_date in signal_dates:
        equity_before = portfolio_value(cash, holdings, data_by_code, signal_date)

        for code in list(holdings):
            row = latest_row_on_or_before(data_by_code[code], signal_date)
            if row is None or pd.isna(row["valuation_pct"]):
                continue
            sell_frac, reason = exit_decision(row, holdings[code], args)
            if sell_frac <= 0:
                continue
            exec_date, raw_price = execute_price_on_or_after(data_by_code[code], signal_date, "sell")
            sell_price = raw_price * (1 - cost.slippage - cost.sell_commission - cost.stamp_tax)
            shares_to_sell = holdings[code].shares * sell_frac
            cash += shares_to_sell * sell_price
            trades.append(
                {
                    "date": exec_date.strftime("%Y-%m-%d"),
                    "code": code,
                    "name": LEADERS[code],
                    "side": "SELL",
                    "price": round(sell_price, 4),
                    "shares": round(shares_to_sell, 6),
                    "reason": reason,
                    "score": np.nan,
                    "valuation_pct": round(row["valuation_pct"] * 100, 1),
                    "valuation_is_real": bool(row.get("valuation_is_real", False)),
                }
            )
            holdings[code].shares -= shares_to_sell
            if holdings[code].shares <= 1e-9:
                del holdings[code]

        candidates = []
        avoid_new_buy = is_earnings_avoid_month(signal_date, args)
        for code, df in data_by_code.items():
            if code in holdings:
                continue
            row = latest_row_on_or_before(df, signal_date)
            if row is None:
                continue
            needed = ["ma250", "ma120", "valuation_pct", "ret_6m"]
            if args.use_market_filter:
                needed.extend(["mkt_ma250", "mkt_ret_6m"])
            if any(pd.isna(row[col]) for col in needed):
                continue
            row = row.copy()
            row["mkt_filter_enabled"] = args.use_market_filter
            score, notes = entry_score(row)
            if avoid_new_buy:
                notes = notes + ["earnings_window_no_new_buy"]
            score_rows.append(
                {
                    "date": signal_date.strftime("%Y-%m-%d"),
                    "code": code,
                    "name": LEADERS[code],
                    "score": round(score, 1),
                    "valuation_pct": round(row["valuation_pct"] * 100, 1),
                    "valuation_is_real": bool(row.get("valuation_is_real", False)),
                    "close": round(row["close"], 3),
                    "notes": "|".join(notes),
                }
            )
            if not avoid_new_buy and score >= args.fallback_score:
                candidates.append((score, code, row, notes))

        candidates.sort(reverse=True, key=lambda item: item[0])
        available_slots = max(args.max_positions - len(holdings), 0)
        strong = [item for item in candidates if item[0] >= args.buy_score]
        selected = strong[:available_slots]
        min_needed = max(args.min_positions - len(holdings) - len(selected), 0)
        if min_needed > 0:
            selected_codes = {item[1] for item in selected}
            fallback = [item for item in candidates if item[1] not in selected_codes]
            selected.extend(fallback[: min(min_needed, available_slots - len(selected))])

        for score, code, row, notes in selected:
            equity_now = portfolio_value(cash, holdings, data_by_code, signal_date)
            target_value = equity_now / args.max_positions
            invest = min(cash, target_value)
            if invest < equity_now * 0.02:
                continue
            exec_date, raw_price = execute_price_on_or_after(data_by_code[code], signal_date, "buy")
            buy_price = raw_price * (1 + cost.slippage + cost.buy_commission)
            shares = invest / buy_price
            cash -= invest
            holdings[code] = Holding(
                shares=shares,
                entry_date=exec_date.strftime("%Y-%m-%d"),
                entry_price=buy_price,
            )
            trades.append(
                {
                    "date": exec_date.strftime("%Y-%m-%d"),
                    "code": code,
                    "name": LEADERS[code],
                    "side": "BUY",
                    "price": round(buy_price, 4),
                    "shares": round(shares, 6),
                    "reason": "|".join(
                        notes + (["fallback_ranked_entry"] if score < args.buy_score else [])
                    ),
                    "score": round(score, 1),
                    "valuation_pct": round(row["valuation_pct"] * 100, 1),
                    "valuation_is_real": bool(row.get("valuation_is_real", False)),
                }
            )

        equity_after = portfolio_value(cash, holdings, data_by_code, signal_date)
        equity_rows.append(
            {
                "date": signal_date.strftime("%Y-%m-%d"),
                "equity": equity_after,
                "cash": cash,
                "positions": len(holdings),
                "holdings": ",".join(sorted(holdings)),
            }
        )

    return pd.DataFrame(trades), pd.DataFrame(equity_rows), pd.DataFrame(score_rows)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min()) if len(dd) else 0.0


def summarize(equity: pd.DataFrame, trades: pd.DataFrame) -> dict:
    if equity.empty:
        return {}
    total_return = equity["equity"].iloc[-1] - 1
    start = pd.to_datetime(equity["date"].iloc[0])
    end = pd.to_datetime(equity["date"].iloc[-1])
    years = (end - start).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else np.nan
    sells = trades[trades["side"] == "SELL"] if not trades.empty else pd.DataFrame()
    return {
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "total_return_pct": round(total_return * 100, 1),
        "cagr_pct": round(cagr * 100, 1) if not pd.isna(cagr) else np.nan,
        "max_drawdown_pct": round(max_drawdown(equity["equity"]) * 100, 1),
        "buy_count": int((trades["side"] == "BUY").sum()) if not trades.empty else 0,
        "sell_count": int(len(sells)),
        "final_positions": int(equity["positions"].iloc[-1]),
        "final_cash_pct": round(equity["cash"].iloc[-1] / equity["equity"].iloc[-1] * 100, 1),
    }


def buy_hold_benchmark(
    data_by_code: dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
) -> dict:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    returns = []
    for code, df in data_by_code.items():
        start_row = latest_row_on_or_before(df, start)
        end_row = latest_row_on_or_before(df, end)
        if start_row is None or end_row is None or start_row["close"] <= 0:
            continue
        returns.append(float(end_row["close"] / start_row["close"] - 1))
    if not returns:
        return {"benchmark_return_pct": np.nan}
    return {"benchmark_return_pct": round(float(np.mean(returns)) * 100, 1)}


def run(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cost = CostModel(
        buy_commission=args.buy_commission,
        sell_commission=args.sell_commission,
        stamp_tax=args.stamp_tax,
        slippage=args.slippage,
    )

    valuations = load_valuation(args.valuation_csv)
    market = download_or_load(MARKET_INDEX, args.start, args.end, cache_dir, args.no_download)
    market = add_price_indicators(market)

    data_by_code = {}
    valuation_sources = {}
    for code, name in LEADERS.items():
        ticker = to_yahoo_ticker(code)
        print(f"Loading {code} {name} ...")
        df = download_or_load(ticker, args.start, args.end, cache_dir, args.no_download)
        df = add_price_indicators(df)
        df, source = attach_valuation(code, df, valuations)
        df = merge_market(df, market)
        required = ["ma250", "mkt_ma250"] if args.use_market_filter else ["ma250"]
        df = df.dropna(subset=required).reset_index(drop=True)
        data_by_code[code] = df
        valuation_sources[code] = source

    trades, equity, scores = run_backtest(data_by_code, market, args, cost)
    summary = summarize(equity, trades)
    if summary:
        summary.update(buy_hold_benchmark(data_by_code, summary["start"], summary["end"]))

    trades_path = output_dir / "leader_longhold_trades.csv"
    equity_path = output_dir / "leader_longhold_equity.csv"
    scores_path = output_dir / "leader_longhold_scores.csv"
    summary_path = output_dir / "leader_longhold_summary.csv"
    source_path = output_dir / "leader_longhold_valuation_sources.csv"

    trades.to_csv(trades_path, index=False)
    equity.to_csv(equity_path, index=False)
    scores.to_csv(scores_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    pd.DataFrame(
        [{"code": c, "name": LEADERS[c], "valuation_source": s} for c, s in valuation_sources.items()]
    ).to_csv(source_path, index=False)

    print("\n=== Long-hold strategy summary ===")
    print(pd.DataFrame([summary]).to_string(index=False))
    print("\nValuation source:")
    for code, source in valuation_sources.items():
        print(f"  {code} {LEADERS[code]}: {source}")
    print("\nRecent trades:")
    if trades.empty:
        print("  No trades.")
    else:
        print(trades.tail(12).to_string(index=False))
    print("\nSaved:")
    for path in [summary_path, trades_path, equity_path, scores_path, source_path]:
        print(f"  {path}")


def run_longhold_analysis(
    start: str = "2015-01-01",
    end: Optional[str] = None,
    valuation_csv: Optional[str] = None,
    cache_dir: str = "data_cache",
    no_download: bool = False,
    signal_frequency: str = "monthly",
    max_positions: int = 4,
    min_positions: int = 0,
    buy_score: float = 65.0,
    fallback_score: float = 60.0,
    avoid_earnings: bool = True,
    avoid_earnings_months: str = "4,8,10",
    use_market_filter: bool = False,
    core_hold: bool = True,
    core_trend_trim: float = 0.0,
    valuation_high_pct: float = 0.95,
    valuation_extreme_pct: float = 0.99,
    valuation_reset_pct: float = 0.75,
    valuation_trim: float = 0.20,
    buy_commission: float = 0.0003,
    sell_commission: float = 0.0003,
    stamp_tax: float = 0.0005,
    slippage: float = 0.001,
) -> dict:
    args = argparse.Namespace(
        start=start,
        end=end,
        cache_dir=cache_dir,
        output_dir="",
        valuation_csv=valuation_csv,
        no_download=no_download,
        signal_frequency=signal_frequency,
        max_positions=max_positions,
        min_positions=min_positions,
        buy_score=buy_score,
        fallback_score=fallback_score,
        avoid_earnings=avoid_earnings,
        avoid_earnings_months=avoid_earnings_months,
        use_market_filter=use_market_filter,
        core_hold=core_hold,
        core_trend_trim=core_trend_trim,
        valuation_high_pct=valuation_high_pct,
        valuation_extreme_pct=valuation_extreme_pct,
        valuation_reset_pct=valuation_reset_pct,
        valuation_trim=valuation_trim,
        buy_commission=buy_commission,
        sell_commission=sell_commission,
        stamp_tax=stamp_tax,
        slippage=slippage,
    )
    cost = CostModel(
        buy_commission=buy_commission,
        sell_commission=sell_commission,
        stamp_tax=stamp_tax,
        slippage=slippage,
    )
    cache_path = Path(cache_dir)
    valuations = load_valuation(valuation_csv)
    market = download_or_load(MARKET_INDEX, start, end, cache_path, no_download)
    market = add_price_indicators(market)

    data_by_code = {}
    valuation_sources = {}
    for code in LEADERS:
        df = download_or_load(to_yahoo_ticker(code), start, end, cache_path, no_download)
        df = add_price_indicators(df)
        df, source = attach_valuation(code, df, valuations)
        df = merge_market(df, market)
        required = ["ma250", "mkt_ma250"] if use_market_filter else ["ma250"]
        data_by_code[code] = df.dropna(subset=required).reset_index(drop=True)
        valuation_sources[code] = source

    trades, equity, scores = run_backtest(data_by_code, market, args, cost)
    summary = summarize(equity, trades)
    if summary:
        summary.update(buy_hold_benchmark(data_by_code, summary["start"], summary["end"]))

    sources = pd.DataFrame(
        [{"code": c, "name": LEADERS[c], "valuation_source": s} for c, s in valuation_sources.items()]
    )
    return {
        "summary": pd.DataFrame([summary]) if summary else pd.DataFrame(),
        "trades": trades,
        "equity": equity,
        "scores": scores,
        "sources": sources,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-frequency multi-factor leader backtest.")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-06-12")
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--output-dir", default="backtest_output")
    parser.add_argument("--valuation-csv", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--signal-frequency",
        choices=["daily", "weekly", "monthly"],
        default="monthly",
    )
    parser.add_argument("--max-positions", type=int, default=4)
    parser.add_argument("--min-positions", type=int, default=0)
    parser.add_argument("--buy-score", type=float, default=65.0)
    parser.add_argument("--fallback-score", type=float, default=60.0)
    parser.add_argument("--use-market-filter", action="store_true")
    parser.add_argument("--core-hold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--core-trend-trim", type=float, default=0.0)
    parser.add_argument("--valuation-high-pct", type=float, default=0.95)
    parser.add_argument("--valuation-extreme-pct", type=float, default=0.99)
    parser.add_argument("--valuation-reset-pct", type=float, default=0.75)
    parser.add_argument("--valuation-trim", type=float, default=0.20)
    parser.add_argument("--avoid-earnings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--avoid-earnings-months",
        default="4,8,10",
        help="Month-end signals in these months will not open new positions.",
    )
    parser.add_argument("--buy-commission", type=float, default=0.0003)
    parser.add_argument("--sell-commission", type=float, default=0.0003)
    parser.add_argument("--stamp-tax", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.001)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
