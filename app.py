"""
A股信号监控 & 回测  |  Streamlit Web App
运行：streamlit run app.py
手机访问：局域网 http://你的IP:8501  或 部署到 Streamlit Cloud
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
from pathlib import Path

# ─── 页面配置 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A股信号监控","""
A股信号监控 & 回测  |  Streamlit Web App
运行：streamlit run app.py
手机访问：局域网 http://你的IP:8501  或 部署到 Streamlit Cloud
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
from pathlib import Path

def normalize_date_series(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    try:
        dates = dates.dt.tz_localize(None)
    except TypeError:
        dates = dates.dt.tz_convert(None)
    return pd.Series(dates.dt.normalize().to_numpy(dtype="datetime64[ns]"), index=series.index)

# ─── 页面配置 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A股信号监控",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── 常用股票快捷按钮 ─────────────────────────────────────────────────────────
PRESETS = [
    ("紫金", "601899"), ("洛阳钼", "603993"), ("宁德", "300750"),
    ("长江电", "600900"), ("招商银", "600036"), ("小米",  "1810"),
    ("腾讯", "0700"), ("美团", "3690"),
    ("阿里", "BABA"), ("拼多多", "PDD"), ("京东", "JD"),
]

# ─── 代码 → Yahoo Ticker ─────────────────────────────────────────────────────
def to_ticker(code: str) -> str:
    code = code.strip().upper()
    if "." in code:
        return code
    if code.isalpha():
        return code
    if len(code) <= 4:                        # 港股
        return code.zfill(4) + ".HK"
    if code.startswith(("6", "9")):
        return code + ".SS"
    return code + ".SZ"

def guess_name(code: str) -> str:
    known = {"601899": "紫金矿业", "603993": "洛阳钼业", "300750": "宁德时代",
             "600900": "长江电力", "600036": "招商银行", "1810":   "小米集团",
             "0700": "腾讯控股", "700": "腾讯控股", "3690": "美团",
             "BABA": "阿里巴巴", "PDD": "拼多多", "JD": "京东"}
    return known.get(code.strip().upper(), code)

# ─── 数据获取 ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_data(ticker: str, period: str = "3y") -> pd.DataFrame | None:
    try:
        raw = yf.download(ticker, period=period, interval="1d",
                          progress=False, auto_adjust=True)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.reset_index()
        df.columns = [str(c).lower().strip() for c in df.columns]
        df = df.rename(columns={"date": "date"})
        df["date"]   = df["date"].astype(str).str[:10]
        df["amount"] = df["close"] * df["volume"]
        df["pct"]    = df["close"].pct_change() * 100
        df = df[["date","open","high","low","close","volume","amount","pct"]]
        return df.dropna(subset=["close"]).reset_index(drop=True)
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_pe_percentile(ticker: str, period: str = "5y") -> dict | None:
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval="1d", auto_adjust=True)
        earnings = tk.get_earnings_dates(limit=32)
        if hist.empty or earnings is None or earnings.empty:
            return None
        hist = hist.reset_index()
        hist["date"] = normalize_date_series(hist["Date"])
        hist = hist[["date", "Close"]].rename(columns={"Close": "close"}).dropna()

        eps_col = None
        for col in ["Reported EPS", "reportedEPS"]:
            if col in earnings.columns:
                eps_col = col
                break
        if eps_col is None:
            return None
        eps = earnings.reset_index()
        date_col = "Earnings Date" if "Earnings Date" in eps.columns else eps.columns[0]
        eps["date"] = normalize_date_series(eps[date_col])
        eps["eps"] = pd.to_numeric(eps[eps_col], errors="coerce")
        eps = eps[["date", "eps"]].dropna().sort_values("date")
        if len(eps) < 4:
            return None
        eps["eps_ttm"] = eps["eps"].rolling(4).sum()
        left = hist.sort_values("date").copy()
        right = eps[["date", "eps_ttm"]].sort_values("date").copy()
        left["date"] = normalize_date_series(left["date"])
        right["date"] = normalize_date_series(right["date"])
        merged = pd.merge_asof(left, right, on="date", direction="backward")
        merged = merged.dropna(subset=["eps_ttm"])
        merged = merged[merged["eps_ttm"] > 0].copy()
        if len(merged) < 60:
            return None
        merged["pe_ttm"] = merged["close"] / merged["eps_ttm"]
        merged = merged.replace([float("inf"), -float("inf")], pd.NA).dropna(subset=["pe_ttm"])
        if merged.empty:
            return None
        current = float(merged["pe_ttm"].iloc[-1])
        pct = float((merged["pe_ttm"] <= current).mean() * 100)
        return {"pe_ttm": current, "pe_pct": pct, "samples": len(merged)}
    except Exception:
        return None

# ─── 信号计算 ─────────────────────────────────────────────────────────────────
def compute_signal(df: pd.DataFrame, ma: int = 20, vol_thr: float = 1.5):
    n = len(df)
    if n < ma + 3:
        return None
    t1, t2, t3 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    s20  = df.tail(ma)
    s60  = df.tail(60)
    ma20 = s20["close"].mean()
    ma60 = s60["close"].mean()
    vol_ma   = s20["volume"].mean()
    vol_ratio = t1["volume"] / vol_ma if vol_ma > 0 else None

    diff     = (t1["close"] - ma20) / ma20 * 100
    above    = t1["close"] > ma20
    bH       = max(t1["open"], t1["close"])
    bL       = min(t1["open"], t1["close"])
    body     = bH - bL or t1["close"] * 0.001
    lsh      = bL - t1["low"]
    ush      = t1["high"] - bH
    green    = t1["close"] >= t1["open"]

    hammer   = lsh >= body * 2 and ush <= body * 0.5 and lsh > 0
    engulf   = (green and t2["close"] < t2["open"]
                and t1["open"] <= t2["close"] and t1["close"] >= t2["open"])
    vbreak   = bool(vol_ratio) and vol_ratio >= vol_thr and green and -1 <= diff <= 6
    shrink   = bool(vol_ratio) and vol_ratio < 0.8 and 0 < diff < 8 and not green
    bkdown   = not green and not above and (not vol_ratio or vol_ratio >= 1.0)
    below3   = not above and t2["close"] < ma20 and t3["close"] < ma20

    pts, tags = 0, []
    if hammer:  pts += 2; tags.append("长下影线")
    if engulf:  pts += 2; tags.append("阳包阴")
    if vbreak:  pts += 3; tags.append("放量突破均线")
    elif vol_ratio and vol_ratio >= vol_thr and green: pts += 1; tags.append("放量阳线")
    if above and t1["close"] > ma60: pts += 1
    if shrink:  pts += 1; tags.append("缩量回踩")

    if below3 and (not vol_ratio or vol_ratio >= 0.8):
        sig, reason = "STOP",  "连续三日均线下方，破位风险高"
    elif bkdown:
        sig, reason = "STOP",  "有量下穿均线，注意止损"
    elif pts >= 4:
        sig, reason = "BUY",   " + ".join(tags)
    elif shrink or pts >= 2:
        sig, reason = "WATCH", (" + ".join(tags) + "，" if tags else "") + "等待放量确认"
    elif above:
        sig, reason = "HOLD",  f"均线上方 +{diff:.1f}%，趋势完好"
    else:
        sig, reason = "WATCH", f"均线下方 {abs(diff):.1f}%，观察企稳"

    return dict(date=t1["date"], close=t1["close"], open=t1["open"],
                high=t1["high"], low=t1["low"], volume=t1["volume"],
                pct=t1["pct"], vol_ratio=vol_ratio, vol_ma=vol_ma,
                ma20=ma20, ma60=ma60, diff_ma20=diff, above_ma20=above,
                signal=sig, reason=reason, closes20=s20["close"].tolist())

# ─── 回测引擎 ─────────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, stop_pct=8.0, max_hold=40,
                 ma=20, vol_thr=1.5) -> list:
    trades, pos, n = [], None, len(df)
    for i in range(ma + 5, n - 1):
        ind = compute_signal(df.iloc[:i+1], ma, vol_thr)
        if ind is None:
            continue
        sig = ind["signal"]
        if pos is None:
            if sig == "BUY":
                px = float(df.iloc[i+1]["open"])
                pos = {"px": px, "date": str(df.iloc[i+1]["date"]),
                       "i": i+1, "peak": px}
        else:
            close = float(df.iloc[i]["close"])
            pnl   = (close - pos["px"]) / pos["px"] * 100
            held  = i - pos["i"]
            pos["peak"] = max(pos["peak"], close)
            reason = None
            if pnl <= -stop_pct:           reason = f"硬止损-{stop_pct:.0f}%"
            elif sig == "STOP":            reason = "信号止损"
            elif held >= max_hold:         reason = f"超{max_hold}天出场"
            if reason:
                ex    = float(df.iloc[i+1]["open"]) if i+1 < n else close
                final = (ex - pos["px"]) / pos["px"] * 100
                peak  = (pos["peak"] - pos["px"]) / pos["px"] * 100
                trades.append(dict(entry_date=pos["date"],
                                   exit_date=str(df.iloc[i+1]["date"] if i+1<n else df.iloc[i]["date"]),
                                   entry_px=pos["px"], exit_px=ex,
                                   pnl=round(final,2), days=held,
                                   reason=reason, peak_ret=round(peak,1)))
                pos = None
    if pos:
        ex   = float(df.iloc[-1]["close"])
        peak = (pos["peak"] - pos["px"]) / pos["px"] * 100
        trades.append(dict(entry_date=pos["date"],
                           exit_date=str(df.iloc[-1]["date"]),
                           entry_px=pos["px"], exit_px=ex,
                           pnl=round((ex-pos["px"])/pos["px"]*100,2),
                           days=n-pos["i"], reason="回测结束",
                           peak_ret=round(peak,1)))
    return trades

def calc_metrics(trades):
    if not trades:
        return None
    pnls  = [t["pnl"] for t in trades]
    wins  = [p for p in pnls if p > 0]
    loss  = [p for p in pnls if p <= 0]
    cum, peak, mdd, eq = 1.0, 1.0, 0.0, []
    for p in pnls:
        cum *= (1 + p/100)
        eq.append(round((cum-1)*100, 2))
        if cum > peak: peak = cum
        dd = (peak-cum)/peak*100
        if dd > mdd: mdd = dd
    return dict(n=len(trades), wins=len(wins), losses=len(loss),
                win_rate=round(len(wins)/len(trades)*100,1),
                avg_win=round(sum(wins)/len(wins),2)  if wins else 0,
                avg_loss=round(sum(loss)/len(loss),2) if loss else 0,
                pf=round(sum(wins)/abs(sum(loss)),2)  if loss and sum(loss)!=0 else 99,
                total_ret=round((cum-1)*100,1), max_dd=round(mdd,1),
                avg_days=round(sum(t["days"] for t in trades)/len(trades),1),
                equity=eq)

# ─── 交易窗口 ─────────────────────────────────────────────────────────────────
def trading_window():
    m, d = date.today().month, date.today().day
    if m in [3,4] or (m==5 and d<=15): return "danger", "❌ 禁入期（年报+一季报）"
    if m in [7,8]:                     return "danger", "❌ 禁入期（半年报）"
    if m == 10:                        return "danger", "❌ 禁入期（三季报）"
    if (m==5 and d>=16) or (m==6 and d<=25): return "safe", "✅ 安全窗口①"
    if m == 9:                         return "safe",   "✅ 安全窗口②"
    if m in [11,12,1] or (m==2 and d<20):    return "safe", "✅ 安全窗口③"
    return "caution", "⚠️ 过渡期，谨慎操作"

# ─── K线图 + 均线 + 成交量 ───────────────────────────────────────────────────
def make_chart(df: pd.DataFrame, name: str) -> go.Figure:
    tail = df.tail(60).copy()
    s20  = df["close"].rolling(20).mean().tail(60)
    s60  = df["close"].rolling(60).mean().tail(60)
    vol_ma = df["volume"].rolling(20).mean().tail(60)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=tail["date"], open=tail["open"], high=tail["high"],
        low=tail["low"], close=tail["close"],
        increasing_line_color="#d93025", decreasing_line_color="#188038",
        name="K线", showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(x=tail["date"], y=s20, name="MA20",
        line=dict(color="#f5c400", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=tail["date"], y=s60, name="MA60",
        line=dict(color="#4d9fff", width=1.5)), row=1, col=1)

    # Volume bars
    colors = ["#d93025" if c >= o else "#188038"
              for c, o in zip(tail["close"], tail["open"])]
    fig.add_trace(go.Bar(x=tail["date"], y=tail["volume"], name="成交量",
        marker_color=colors, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=tail["date"], y=vol_ma, name="均量",
        line=dict(color="#f5c400", width=1.2, dash="dot")), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"{name}  近60日", font_size=14),
        xaxis_rangeslider_visible=False,
        height=480, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.02, x=0),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font_color="#202124",
    )
    fig.update_xaxes(gridcolor="#e8eaed", showgrid=True)
    fig.update_yaxes(gridcolor="#e8eaed", showgrid=True)
    return fig

# ─── 权益曲线 ─────────────────────────────────────────────────────────────────
def make_equity_chart(equity: list) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity, mode="lines+markers",
        line=dict(color="#d93025", width=2),
        marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(217,48,37,0.08)",
        name="累计收益%"))
    fig.add_hline(y=0, line_dash="dash", line_color="#555")
    fig.update_layout(
        title="权益曲线（每笔累计收益%）",
        height=260, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font_color="#202124", showlegend=False,
    )
    fig.update_xaxes(gridcolor="#e8eaed")
    fig.update_yaxes(gridcolor="#e8eaed")
    return fig

def make_longhold_equity_chart(equity_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if equity_df is None or equity_df.empty:
        return fig
    plot_df = equity_df.copy()
    plot_df["ret"] = (plot_df["equity"] - 1) * 100
    fig.add_trace(go.Scatter(
        x=plot_df["date"], y=plot_df["ret"], mode="lines",
        line=dict(color="#d93025", width=2),
        fill="tozeroy", fillcolor="rgba(217,48,37,0.08)",
        name="策略收益%"))
    fig.add_hline(y=0, line_dash="dash", line_color="#555")
    fig.update_layout(
        title="龙头长期策略权益曲线",
        height=320, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font_color="#202124", showlegend=False,
    )
    fig.update_xaxes(gridcolor="#e8eaed")
    fig.update_yaxes(gridcolor="#e8eaed", ticksuffix="%")
    return fig

@st.cache_resource(show_spinner=False)
def load_longhold_module():
    from leader_longhold_backtest import LEADERS, run_longhold_analysis
    return LEADERS, run_longhold_analysis

def render_longhold_backtest():
    try:
        leaders, run_analysis = load_longhold_module()
    except Exception as exc:
        st.error(
            "无法加载龙头长期回测模块。请确认 GitHub 仓库中包含 "
            "`leader_longhold_backtest.py`，且 requirements.txt 已包含 numpy。"
        )
        st.exception(exc)
        return

    st.title("🏛️ 龙头长期回测")
    st.caption("多因子择机买入 · 核心持有 · 极端估值/严重破位退出")

    st.markdown("**股票池**")
    st.caption(" · ".join([f"{code} {name}" for code, name in leaders.items()]))

    with st.expander("参数", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            start = st.text_input("开始日期", "2015-01-01")
            end = st.text_input("结束日期", str(date.today()))
            freq = st.selectbox("信号频率", ["monthly", "weekly", "daily"], index=0)
            max_positions = st.slider("最多持仓", 2, 8, 4)
        with c2:
            buy_score = st.slider("买入分数", 50, 90, 65)
            val_high = st.slider("估值减仓分位", 80, 99, 95) / 100
            val_extreme = st.slider("估值清仓分位", 90, 100, 99) / 100
            avoid_earnings = st.checkbox("财报密集月不新开仓", value=True)

        valuation_file = st.file_uploader(
            "估值CSV（可选：date, code, pe_ttm, pb, dividend_yield）",
            type=["csv"],
        )
        bundled_valuation = Path("valuation.csv")
        if valuation_file is None and bundled_valuation.exists():
            st.caption("已检测到仓库内置 valuation.csv，将自动使用真实 PE/PB 分位。")
        elif valuation_file is None:
            st.caption("未检测到 valuation.csv；不上传时将使用价格分位代理估值。")
        use_market_filter = st.checkbox("启用沪深300大盘过滤", value=False)

    if st.button("运行龙头回测", type="primary", use_container_width=True):
        tmp_val_path = None
        if valuation_file is not None:
            tmp_dir = Path(".streamlit_tmp")
            tmp_dir.mkdir(exist_ok=True)
            tmp_val_path = tmp_dir / "valuation_uploaded.csv"
            tmp_val_path.write_bytes(valuation_file.getvalue())
        elif Path("valuation.csv").exists():
            tmp_val_path = Path("valuation.csv")

        with st.spinner("正在拉取数据并回测..."):
            try:
                result = run_analysis(
                    start=start,
                    end=end,
                    valuation_csv=str(tmp_val_path) if tmp_val_path else None,
                    cache_dir="data_cache",
                    signal_frequency=freq,
                    max_positions=max_positions,
                    buy_score=float(buy_score),
                    fallback_score=float(buy_score - 5),
                    avoid_earnings=avoid_earnings,
                    use_market_filter=use_market_filter,
                    valuation_high_pct=val_high,
                    valuation_extreme_pct=val_extreme,
                )
            except Exception as exc:
                st.error(f"回测失败：{exc}")
                st.stop()
        st.session_state["longhold_result"] = result

    result = st.session_state.get("longhold_result")
    if not result:
        st.info("点击上方按钮运行。上传估值CSV后会使用真实PE/PB分位；不上传则使用价格分位代理。")
        return

    summary = result["summary"]
    trades = result["trades"]
    equity = result["equity"]
    scores = result["scores"]
    sources = result["sources"]

    if summary.empty:
        st.warning("没有生成回测结果")
        return

    s = summary.iloc[0]
    st.divider()
    r1, r2, r3, r4 = st.columns(4)
    metrics = [
        (r1, "累计收益", f"{s['total_return_pct']:.1f}%"),
        (r2, "年化收益", f"{s['cagr_pct']:.1f}%"),
        (r3, "最大回撤", f"{s['max_drawdown_pct']:.1f}%"),
        (r4, "等权基准", f"{s['benchmark_return_pct']:.1f}%"),
    ]
    for col, label, value in metrics:
        col.markdown(f"""<div class="metric-box">
            <div style="font-size:10px;color:#888">{label}</div>
            <div style="color:#202124;font-weight:700;font-size:16px">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.caption(
        f"区间 {s['start']} 至 {s['end']} · 买入 {int(s['buy_count'])} 次 · "
        f"卖出 {int(s['sell_count'])} 次 · 期末持仓 {int(s['final_positions'])} 只 · "
        f"现金 {s['final_cash_pct']:.1f}%"
    )
    st.plotly_chart(make_longhold_equity_chart(equity), use_container_width=True)

    latest_holdings = equity.iloc[-1]["holdings"] if not equity.empty else ""
    st.markdown("**当前持仓**")
    st.caption(latest_holdings if latest_holdings else "无")

    latest_date = scores["date"].max() if not scores.empty else None
    if latest_date:
        latest_scores = scores[scores["date"] == latest_date].sort_values("score", ascending=False)
        latest_scores = latest_scores.copy()
        latest_scores["notes"] = latest_scores["notes"].map(translate_signal_text)
        st.markdown(f"**最新信号（{latest_date}）**")
        st.dataframe(
            latest_scores[["code", "name", "score", "valuation_pct", "notes"]],
            use_container_width=True, hide_index=True,
        )

    with st.expander("交易明细", expanded=False):
        if trades.empty:
            st.caption("无交易")
        else:
            show_trades = trades.copy()
            show_trades["side"] = show_trades["side"].map({"BUY": "买入", "SELL": "卖出"}).fillna(show_trades["side"])
            show_trades["reason"] = show_trades["reason"].map(translate_signal_text)
            st.dataframe(show_trades, use_container_width=True, hide_index=True)

    with st.expander("估值来源", expanded=False):
        st.dataframe(sources, use_container_width=True, hide_index=True)

    st.caption("数据仅供参考 · 不构成投资建议 · 历史表现不代表未来收益")

SIGNAL_TEXT_MAP = {
    "market_weak": "大盘偏弱",
    "long_trend_up": "长期趋势向上",
    "trend_neutral": "趋势中性",
    "reasonable_pullback": "合理回撤",
    "deep_pullback": "深度回撤",
    "shallow_pullback": "浅回撤",
    "trend_continuation": "趋势延续",
    "valuation_low": "估值偏低",
    "valuation_fair": "估值合理",
    "valuation_acceptable": "估值可接受",
    "price_percentile_low": "价格分位偏低",
    "price_percentile_fair": "价格分位合理",
    "price_percentile_acceptable": "价格分位可接受",
    "relative_strength_low_vol": "相对强且波动较低",
    "relative_strength_ok": "相对强度尚可",
    "overheated_penalty": "过热扣分",
    "earnings_window_no_new_buy": "财报窗口不新开仓",
    "fallback_ranked_entry": "候选补位买入",
    "trend_hard_exit": "严重破位退出",
    "valuation_extreme_exit": "估值极端退出",
}

def translate_signal_text(text):
    if pd.isna(text):
        return text
    parts = str(text).split("|")
    translated = []
    for part in parts:
        if part.startswith("valuation_high_trim_"):
            translated.append("估值高位减仓")
        elif part.startswith("core_trend_confirmed_trim_"):
            translated.append("长期趋势转弱减仓")
        elif part.startswith("trend_break_trim_"):
            translated.append("趋势破位减仓")
        else:
            translated.append(SIGNAL_TEXT_MAP.get(part, part))
    return " + ".join(translated)

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.stApp { background-color: #f8fafc; color:#202124; }
.signal-buy   { background:#d9302514; border:1px solid #d93025;
                border-radius:8px; padding:12px; text-align:center; }
.signal-watch { background:#fbbc041c; border:1px solid #fbbc04;
                border-radius:8px; padding:12px; text-align:center; }
.signal-hold  { background:#1a73e814; border:1px solid #1a73e8;
                border-radius:8px; padding:12px; text-align:center; }
.signal-stop  { background:#18803814; border:1px solid #188038;
                border-radius:8px; padding:12px; text-align:center; }
.metric-box   { background:#ffffff; border:1px solid #e0e3e7;
                border-radius:8px; padding:10px; text-align:center; }
</style>
""", unsafe_allow_html=True)

st.title("📈 A股信号监控")
st.caption(f"数据来源：Yahoo Finance · T-1日线 · {date.today()}")

view = st.radio("视图", ["单股信号监控", "龙头长期回测"], horizontal=True, label_visibility="collapsed")
if view == "龙头长期回测":
    render_longhold_backtest()
    st.stop()

# 交易窗口状态
w_status, w_label = trading_window()
color = {"safe": "🟢", "danger": "🔴", "caution": "🟡"}[w_status]
st.info(f"{color} 交易窗口：{w_label}")

# ─── 快捷按钮 ─────────────────────────────────────────────────────────────────
st.markdown("**快捷选股**")
cols = st.columns(len(PRESETS))
selected_preset = None
for i, (label, code) in enumerate(PRESETS):
    if cols[i].button(label, key=f"btn_{code}", use_container_width=True):
        selected_preset = code

# ─── 输入框 ───────────────────────────────────────────────────────────────────
col_input, col_period = st.columns([3, 1])
with col_input:
    default_code = selected_preset or st.session_state.get("last_code", "")
    code_input = st.text_input(
        "股票代码",
        value=default_code,
        placeholder="601899 / 300750 / 1810",
        label_visibility="collapsed",
    )
with col_period:
    period = st.selectbox("回测", ["3y", "1y", "5y"], label_visibility="collapsed")

if code_input:
    st.session_state["last_code"] = code_input
    ticker = to_ticker(code_input)
    name   = guess_name(code_input)

    with st.spinner(f"拉取 {name}（{ticker}）数据..."):
        df = get_data(ticker, period)

    if df is None or len(df) < 25:
        st.error(f"❌ 无法获取 {ticker} 数据，请检查代码是否正确\n\n"
                 f"上交所：6位数字（如 601899）\n"
                 f"深交所：6位数字（如 300750）\n"
                 f"港  股：4位数字（如 0700 / 3690）\n"
                 f"美  股：英文代码（如 BABA / PDD / JD）")
        st.stop()

    ind = compute_signal(df)

    if ind is None:
        st.warning("数据不足，无法计算信号")
        st.stop()

    # ─── T-1 信号卡片 ──────────────────────────────────────────────────────
    st.divider()
    sig_class = {"BUY":"buy","WATCH":"watch","HOLD":"hold","STOP":"stop"}.get(ind["signal"],"hold")
    sig_label = {"BUY":"买  入","WATCH":"观  察","HOLD":"持  有","STOP":"⚡ 止损"}.get(ind["signal"])
    sig_color = {"BUY":"#d93025","WATCH":"#b06000","HOLD":"#1a73e8","STOP":"#188038"}.get(ind["signal"])

    c1, c2 = st.columns([2, 1])
    with c1:
        pct_color = "#d93025" if ind["pct"] >= 0 else "#188038"
        pct_str = f"+{ind['pct']:.2f}%" if ind["pct"] >= 0 else f"{ind['pct']:.2f}%"
        st.markdown(f"### {name} `{code_input}`")
        st.markdown(f"**T-1 {ind['date']}**")
        st.markdown(f"## ¥{ind['close']:.2f}  "
                    f"<span style='color:{pct_color}'>{pct_str}</span>",
                    unsafe_allow_html=True)
        st.caption(f"开 {ind['open']:.2f}  高 {ind['high']:.2f}  低 {ind['low']:.2f}")
    with c2:
        st.markdown(f"""<div class="signal-{sig_class}">
            <div style="font-size:11px;color:#888">信号</div>
            <div style="font-size:20px;font-weight:700;color:{sig_color}">{sig_label}</div>
        </div>""", unsafe_allow_html=True)

    # 指标行
    vr = ind["vol_ratio"]
    vr_str = f"{vr:.2f}x" if vr else "—"
    diff_str = f"+{ind['diff_ma20']:.1f}%" if ind["diff_ma20"] >= 0 else f"{ind['diff_ma20']:.1f}%"
    pe_info = get_pe_percentile(ticker)
    pe_str = f"{pe_info['pe_ttm']:.1f}x" if pe_info else "—"
    pe_pct_str = f"{pe_info['pe_pct']:.0f}%" if pe_info else "—"

    m1, m2, m3, m4, m5 = st.columns(5)
    for col, label, value, color_cond in [
        (m1, "量比",   vr_str,                 vr and vr >= 1.5),
        (m2, "20日均", f"¥{ind['ma20']:.2f}",  False),
        (m3, "偏离",   diff_str,               ind["diff_ma20"] >= 0),
        (m4, "60日均", f"¥{ind['ma60']:.2f}",  False),
        (m5, "PE分位", f"{pe_str} / {pe_pct_str}", pe_info and pe_info["pe_pct"] <= 35),
    ]:
        c = "#d93025" if color_cond else ("#188038" if not color_cond and label=="偏离" and ind["diff_ma20"]<0 else "#202124")
        col.markdown(f"""<div class="metric-box">
            <div style="font-size:10px;color:#888">{label}</div>
            <div style="color:{c};font-weight:600">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.caption(f"💡 {ind['reason']}")

    # ─── K线图 ─────────────────────────────────────────────────────────────
    st.plotly_chart(make_chart(df, name), use_container_width=True)

    # ─── 回测 ──────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(f"### 📊 回测结果（{period}）")

    trades = run_backtest(df)
    m      = calc_metrics(trades)

    if not m or m["n"] < 3:
        st.warning("交易次数不足 3 笔，回测结果参考意义有限")
    else:
        # 核心指标
        r1, r2, r3, r4, r5 = st.columns(5)
        for col, label, value, good in [
            (r1, "交易笔数", str(m["n"]),           None),
            (r2, "胜率",     f"{m['win_rate']}%",   m["win_rate"] >= 55),
            (r3, "累计收益", f"+{m['total_ret']}%"  if m["total_ret"]>=0 else f"{m['total_ret']}%", m["total_ret"] > 0),
            (r4, "最大回撤", f"-{m['max_dd']}%",    m["max_dd"] < 15),
            (r5, "盈亏比",   str(m["pf"]),           m["pf"] >= 1.5),
        ]:
            c = "#d93025" if good else ("#188038" if good is False else "#202124")
            col.markdown(f"""<div class="metric-box">
                <div style="font-size:10px;color:#888">{label}</div>
                <div style="color:{c};font-weight:700;font-size:16px">{value}</div>
            </div>""", unsafe_allow_html=True)

        st.caption(f"平均盈利 +{m['avg_win']}%  |  平均亏损 {m['avg_loss']}%  |  平均持仓 {m['avg_days']}天")

        # 权益曲线
        st.plotly_chart(make_equity_chart(m["equity"]), use_container_width=True)

        # 交易明细
        with st.expander(f"交易明细（{m['n']}笔）"):
            rows = []
            for t in trades:
                rows.append({
                    "进场": t["entry_date"], "出场": t["exit_date"],
                    "盈亏": f"{'+' if t['pnl']>=0 else ''}{t['pnl']}%",
                    "峰值": f"+{t['peak_ret']}%",
                    "天数": t["days"], "原因": t["reason"],
                })
            tdf = pd.DataFrame(rows)
            # Color the pnl column
            def color_pnl(val):
                c = "#d93025" if "+" in str(val) and val != "+0.0%" else "#188038"
                return f"color: {c}"
            st.dataframe(
                tdf.style.map(color_pnl, subset=["盈亏"]),
                use_container_width=True, hide_index=True,
            )

    # 免责
    st.caption("数据仅供参考 · 不构成投资建议 · 历史表现不代表未来收益")

    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── 常用股票快捷按钮 ─────────────────────────────────────────────────────────
PRESETS = [
    ("紫金", "601899"), ("洛阳钼", "603993"), ("宁德", "300750"),
    ("长江电", "600900"), ("招商银", "600036"), ("小米",  "1810"),
    ("腾讯", "0700"), ("美团", "3690"),
    ("阿里", "BABA"), ("拼多多", "PDD"), ("京东", "JD"),
]

# ─── 代码 → Yahoo Ticker ─────────────────────────────────────────────────────
def to_ticker(code: str) -> str:
    code = code.strip().upper()
    if "." in code:
        return code
    if code.isalpha():
        return code
    if len(code) <= 4:                        # 港股
        return code.zfill(4) + ".HK"
    if code.startswith(("6", "9")):
        return code + ".SS"
    return code + ".SZ"

def guess_name(code: str) -> str:
    known = {"601899": "紫金矿业", "603993": "洛阳钼业", "300750": "宁德时代",
             "600900": "长江电力", "600036": "招商银行", "1810":   "小米集团",
             "0700": "腾讯控股", "700": "腾讯控股", "3690": "美团",
             "BABA": "阿里巴巴", "PDD": "拼多多", "JD": "京东"}
    return known.get(code.strip().upper(), code)

# ─── 数据获取 ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_data(ticker: str, period: str = "3y") -> pd.DataFrame | None:
    try:
        raw = yf.download(ticker, period=period, interval="1d",
                          progress=False, auto_adjust=True)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.reset_index()
        df.columns = [str(c).lower().strip() for c in df.columns]
        df = df.rename(columns={"date": "date"})
        df["date"]   = df["date"].astype(str).str[:10]
        df["amount"] = df["close"] * df["volume"]
        df["pct"]    = df["close"].pct_change() * 100
        df = df[["date","open","high","low","close","volume","amount","pct"]]
        return df.dropna(subset=["close"]).reset_index(drop=True)
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_pe_percentile(ticker: str, period: str = "5y") -> dict | None:
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval="1d", auto_adjust=True)
        earnings = tk.get_earnings_dates(limit=32)
        if hist.empty or earnings is None or earnings.empty:
            return None
        hist = hist.reset_index()
        hist["date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
        hist = hist[["date", "Close"]].rename(columns={"Close": "close"}).dropna()

        eps_col = None
        for col in ["Reported EPS", "reportedEPS"]:
            if col in earnings.columns:
                eps_col = col
                break
        if eps_col is None:
            return None
        eps = earnings.reset_index()
        date_col = "Earnings Date" if "Earnings Date" in eps.columns else eps.columns[0]
        eps["date"] = pd.to_datetime(eps[date_col]).dt.tz_localize(None)
        eps["eps"] = pd.to_numeric(eps[eps_col], errors="coerce")
        eps = eps[["date", "eps"]].dropna().sort_values("date")
        if len(eps) < 4:
            return None
        eps["eps_ttm"] = eps["eps"].rolling(4).sum()
        merged = pd.merge_asof(hist.sort_values("date"), eps[["date", "eps_ttm"]], on="date", direction="backward")
        merged = merged.dropna(subset=["eps_ttm"])
        merged = merged[merged["eps_ttm"] > 0].copy()
        if len(merged) < 60:
            return None
        merged["pe_ttm"] = merged["close"] / merged["eps_ttm"]
        merged = merged.replace([float("inf"), -float("inf")], pd.NA).dropna(subset=["pe_ttm"])
        if merged.empty:
            return None
        current = float(merged["pe_ttm"].iloc[-1])
        pct = float((merged["pe_ttm"] <= current).mean() * 100)
        return {"pe_ttm": current, "pe_pct": pct, "samples": len(merged)}
    except Exception:
        return None

# ─── 信号计算 ─────────────────────────────────────────────────────────────────
def compute_signal(df: pd.DataFrame, ma: int = 20, vol_thr: float = 1.5):
    n = len(df)
    if n < ma + 3:
        return None
    t1, t2, t3 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    s20  = df.tail(ma)
    s60  = df.tail(60)
    ma20 = s20["close"].mean()
    ma60 = s60["close"].mean()
    vol_ma   = s20["volume"].mean()
    vol_ratio = t1["volume"] / vol_ma if vol_ma > 0 else None

    diff     = (t1["close"] - ma20) / ma20 * 100
    above    = t1["close"] > ma20
    bH       = max(t1["open"], t1["close"])
    bL       = min(t1["open"], t1["close"])
    body     = bH - bL or t1["close"] * 0.001
    lsh      = bL - t1["low"]
    ush      = t1["high"] - bH
    green    = t1["close"] >= t1["open"]

    hammer   = lsh >= body * 2 and ush <= body * 0.5 and lsh > 0
    engulf   = (green and t2["close"] < t2["open"]
                and t1["open"] <= t2["close"] and t1["close"] >= t2["open"])
    vbreak   = bool(vol_ratio) and vol_ratio >= vol_thr and green and -1 <= diff <= 6
    shrink   = bool(vol_ratio) and vol_ratio < 0.8 and 0 < diff < 8 and not green
    bkdown   = not green and not above and (not vol_ratio or vol_ratio >= 1.0)
    below3   = not above and t2["close"] < ma20 and t3["close"] < ma20

    pts, tags = 0, []
    if hammer:  pts += 2; tags.append("长下影线")
    if engulf:  pts += 2; tags.append("阳包阴")
    if vbreak:  pts += 3; tags.append("放量突破均线")
    elif vol_ratio and vol_ratio >= vol_thr and green: pts += 1; tags.append("放量阳线")
    if above and t1["close"] > ma60: pts += 1
    if shrink:  pts += 1; tags.append("缩量回踩")

    if below3 and (not vol_ratio or vol_ratio >= 0.8):
        sig, reason = "STOP",  "连续三日均线下方，破位风险高"
    elif bkdown:
        sig, reason = "STOP",  "有量下穿均线，注意止损"
    elif pts >= 4:
        sig, reason = "BUY",   " + ".join(tags)
    elif shrink or pts >= 2:
        sig, reason = "WATCH", (" + ".join(tags) + "，" if tags else "") + "等待放量确认"
    elif above:
        sig, reason = "HOLD",  f"均线上方 +{diff:.1f}%，趋势完好"
    else:
        sig, reason = "WATCH", f"均线下方 {abs(diff):.1f}%，观察企稳"

    return dict(date=t1["date"], close=t1["close"], open=t1["open"],
                high=t1["high"], low=t1["low"], volume=t1["volume"],
                pct=t1["pct"], vol_ratio=vol_ratio, vol_ma=vol_ma,
                ma20=ma20, ma60=ma60, diff_ma20=diff, above_ma20=above,
                signal=sig, reason=reason, closes20=s20["close"].tolist())

# ─── 回测引擎 ─────────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, stop_pct=8.0, max_hold=40,
                 ma=20, vol_thr=1.5) -> list:
    trades, pos, n = [], None, len(df)
    for i in range(ma + 5, n - 1):
        ind = compute_signal(df.iloc[:i+1], ma, vol_thr)
        if ind is None:
            continue
        sig = ind["signal"]
        if pos is None:
            if sig == "BUY":
                px = float(df.iloc[i+1]["open"])
                pos = {"px": px, "date": str(df.iloc[i+1]["date"]),
                       "i": i+1, "peak": px}
        else:
            close = float(df.iloc[i]["close"])
            pnl   = (close - pos["px"]) / pos["px"] * 100
            held  = i - pos["i"]
            pos["peak"] = max(pos["peak"], close)
            reason = None
            if pnl <= -stop_pct:           reason = f"硬止损-{stop_pct:.0f}%"
            elif sig == "STOP":            reason = "信号止损"
            elif held >= max_hold:         reason = f"超{max_hold}天出场"
            if reason:
                ex    = float(df.iloc[i+1]["open"]) if i+1 < n else close
                final = (ex - pos["px"]) / pos["px"] * 100
                peak  = (pos["peak"] - pos["px"]) / pos["px"] * 100
                trades.append(dict(entry_date=pos["date"],
                                   exit_date=str(df.iloc[i+1]["date"] if i+1<n else df.iloc[i]["date"]),
                                   entry_px=pos["px"], exit_px=ex,
                                   pnl=round(final,2), days=held,
                                   reason=reason, peak_ret=round(peak,1)))
                pos = None
    if pos:
        ex   = float(df.iloc[-1]["close"])
        peak = (pos["peak"] - pos["px"]) / pos["px"] * 100
        trades.append(dict(entry_date=pos["date"],
                           exit_date=str(df.iloc[-1]["date"]),
                           entry_px=pos["px"], exit_px=ex,
                           pnl=round((ex-pos["px"])/pos["px"]*100,2),
                           days=n-pos["i"], reason="回测结束",
                           peak_ret=round(peak,1)))
    return trades

def calc_metrics(trades):
    if not trades:
        return None
    pnls  = [t["pnl"] for t in trades]
    wins  = [p for p in pnls if p > 0]
    loss  = [p for p in pnls if p <= 0]
    cum, peak, mdd, eq = 1.0, 1.0, 0.0, []
    for p in pnls:
        cum *= (1 + p/100)
        eq.append(round((cum-1)*100, 2))
        if cum > peak: peak = cum
        dd = (peak-cum)/peak*100
        if dd > mdd: mdd = dd
    return dict(n=len(trades), wins=len(wins), losses=len(loss),
                win_rate=round(len(wins)/len(trades)*100,1),
                avg_win=round(sum(wins)/len(wins),2)  if wins else 0,
                avg_loss=round(sum(loss)/len(loss),2) if loss else 0,
                pf=round(sum(wins)/abs(sum(loss)),2)  if loss and sum(loss)!=0 else 99,
                total_ret=round((cum-1)*100,1), max_dd=round(mdd,1),
                avg_days=round(sum(t["days"] for t in trades)/len(trades),1),
                equity=eq)

# ─── 交易窗口 ─────────────────────────────────────────────────────────────────
def trading_window():
    m, d = date.today().month, date.today().day
    if m in [3,4] or (m==5 and d<=15): return "danger", "❌ 禁入期（年报+一季报）"
    if m in [7,8]:                     return "danger", "❌ 禁入期（半年报）"
    if m == 10:                        return "danger", "❌ 禁入期（三季报）"
    if (m==5 and d>=16) or (m==6 and d<=25): return "safe", "✅ 安全窗口①"
    if m == 9:                         return "safe",   "✅ 安全窗口②"
    if m in [11,12,1] or (m==2 and d<20):    return "safe", "✅ 安全窗口③"
    return "caution", "⚠️ 过渡期，谨慎操作"

# ─── K线图 + 均线 + 成交量 ───────────────────────────────────────────────────
def make_chart(df: pd.DataFrame, name: str) -> go.Figure:
    tail = df.tail(60).copy()
    s20  = df["close"].rolling(20).mean().tail(60)
    s60  = df["close"].rolling(60).mean().tail(60)
    vol_ma = df["volume"].rolling(20).mean().tail(60)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=tail["date"], open=tail["open"], high=tail["high"],
        low=tail["low"], close=tail["close"],
        increasing_line_color="#d93025", decreasing_line_color="#188038",
        name="K线", showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(x=tail["date"], y=s20, name="MA20",
        line=dict(color="#f5c400", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=tail["date"], y=s60, name="MA60",
        line=dict(color="#4d9fff", width=1.5)), row=1, col=1)

    # Volume bars
    colors = ["#d93025" if c >= o else "#188038"
              for c, o in zip(tail["close"], tail["open"])]
    fig.add_trace(go.Bar(x=tail["date"], y=tail["volume"], name="成交量",
        marker_color=colors, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=tail["date"], y=vol_ma, name="均量",
        line=dict(color="#f5c400", width=1.2, dash="dot")), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"{name}  近60日", font_size=14),
        xaxis_rangeslider_visible=False,
        height=480, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.02, x=0),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font_color="#202124",
    )
    fig.update_xaxes(gridcolor="#e8eaed", showgrid=True)
    fig.update_yaxes(gridcolor="#e8eaed", showgrid=True)
    return fig

# ─── 权益曲线 ─────────────────────────────────────────────────────────────────
def make_equity_chart(equity: list) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity, mode="lines+markers",
        line=dict(color="#d93025", width=2),
        marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(217,48,37,0.08)",
        name="累计收益%"))
    fig.add_hline(y=0, line_dash="dash", line_color="#555")
    fig.update_layout(
        title="权益曲线（每笔累计收益%）",
        height=260, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font_color="#202124", showlegend=False,
    )
    fig.update_xaxes(gridcolor="#e8eaed")
    fig.update_yaxes(gridcolor="#e8eaed")
    return fig

def make_longhold_equity_chart(equity_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if equity_df is None or equity_df.empty:
        return fig
    plot_df = equity_df.copy()
    plot_df["ret"] = (plot_df["equity"] - 1) * 100
    fig.add_trace(go.Scatter(
        x=plot_df["date"], y=plot_df["ret"], mode="lines",
        line=dict(color="#d93025", width=2),
        fill="tozeroy", fillcolor="rgba(217,48,37,0.08)",
        name="策略收益%"))
    fig.add_hline(y=0, line_dash="dash", line_color="#555")
    fig.update_layout(
        title="龙头长期策略权益曲线",
        height=320, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font_color="#202124", showlegend=False,
    )
    fig.update_xaxes(gridcolor="#e8eaed")
    fig.update_yaxes(gridcolor="#e8eaed", ticksuffix="%")
    return fig

@st.cache_resource(show_spinner=False)
def load_longhold_module():
    from leader_longhold_backtest import LEADERS, run_longhold_analysis
    return LEADERS, run_longhold_analysis

def render_longhold_backtest():
    try:
        leaders, run_analysis = load_longhold_module()
    except Exception as exc:
        st.error(
            "无法加载龙头长期回测模块。请确认 GitHub 仓库中包含 "
            "`leader_longhold_backtest.py`，且 requirements.txt 已包含 numpy。"
        )
        st.exception(exc)
        return

    st.title("🏛️ 龙头长期回测")
    st.caption("多因子择机买入 · 核心持有 · 极端估值/严重破位退出")

    st.markdown("**股票池**")
    st.caption(" · ".join([f"{code} {name}" for code, name in leaders.items()]))

    with st.expander("参数", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            start = st.text_input("开始日期", "2015-01-01")
            end = st.text_input("结束日期", str(date.today()))
            freq = st.selectbox("信号频率", ["monthly", "weekly", "daily"], index=0)
            max_positions = st.slider("最多持仓", 2, 8, 4)
        with c2:
            buy_score = st.slider("买入分数", 50, 90, 65)
            val_high = st.slider("估值减仓分位", 80, 99, 95) / 100
            val_extreme = st.slider("估值清仓分位", 90, 100, 99) / 100
            avoid_earnings = st.checkbox("财报密集月不新开仓", value=True)

        valuation_file = st.file_uploader(
            "估值CSV（可选：date, code, pe_ttm, pb, dividend_yield）",
            type=["csv"],
        )
        bundled_valuation = Path("valuation.csv")
        if valuation_file is None and bundled_valuation.exists():
            st.caption("已检测到仓库内置 valuation.csv，将自动使用真实 PE/PB 分位。")
        elif valuation_file is None:
            st.caption("未检测到 valuation.csv；不上传时将使用价格分位代理估值。")
        use_market_filter = st.checkbox("启用沪深300大盘过滤", value=False)

    if st.button("运行龙头回测", type="primary", use_container_width=True):
        tmp_val_path = None
        if valuation_file is not None:
            tmp_dir = Path(".streamlit_tmp")
            tmp_dir.mkdir(exist_ok=True)
            tmp_val_path = tmp_dir / "valuation_uploaded.csv"
            tmp_val_path.write_bytes(valuation_file.getvalue())
        elif Path("valuation.csv").exists():
            tmp_val_path = Path("valuation.csv")

        with st.spinner("正在拉取数据并回测..."):
            try:
                result = run_analysis(
                    start=start,
                    end=end,
                    valuation_csv=str(tmp_val_path) if tmp_val_path else None,
                    cache_dir="data_cache",
                    signal_frequency=freq,
                    max_positions=max_positions,
                    buy_score=float(buy_score),
                    fallback_score=float(buy_score - 5),
                    avoid_earnings=avoid_earnings,
                    use_market_filter=use_market_filter,
                    valuation_high_pct=val_high,
                    valuation_extreme_pct=val_extreme,
                )
            except Exception as exc:
                st.error(f"回测失败：{exc}")
                st.stop()
        st.session_state["longhold_result"] = result

    result = st.session_state.get("longhold_result")
    if not result:
        st.info("点击上方按钮运行。上传估值CSV后会使用真实PE/PB分位；不上传则使用价格分位代理。")
        return

    summary = result["summary"]
    trades = result["trades"]
    equity = result["equity"]
    scores = result["scores"]
    sources = result["sources"]

    if summary.empty:
        st.warning("没有生成回测结果")
        return

    s = summary.iloc[0]
    st.divider()
    r1, r2, r3, r4 = st.columns(4)
    metrics = [
        (r1, "累计收益", f"{s['total_return_pct']:.1f}%"),
        (r2, "年化收益", f"{s['cagr_pct']:.1f}%"),
        (r3, "最大回撤", f"{s['max_drawdown_pct']:.1f}%"),
        (r4, "等权基准", f"{s['benchmark_return_pct']:.1f}%"),
    ]
    for col, label, value in metrics:
        col.markdown(f"""<div class="metric-box">
            <div style="font-size:10px;color:#888">{label}</div>
            <div style="color:#202124;font-weight:700;font-size:16px">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.caption(
        f"区间 {s['start']} 至 {s['end']} · 买入 {int(s['buy_count'])} 次 · "
        f"卖出 {int(s['sell_count'])} 次 · 期末持仓 {int(s['final_positions'])} 只 · "
        f"现金 {s['final_cash_pct']:.1f}%"
    )
    st.plotly_chart(make_longhold_equity_chart(equity), use_container_width=True)

    latest_holdings = equity.iloc[-1]["holdings"] if not equity.empty else ""
    st.markdown("**当前持仓**")
    st.caption(latest_holdings if latest_holdings else "无")

    latest_date = scores["date"].max() if not scores.empty else None
    if latest_date:
        latest_scores = scores[scores["date"] == latest_date].sort_values("score", ascending=False)
        latest_scores = latest_scores.copy()
        latest_scores["notes"] = latest_scores["notes"].map(translate_signal_text)
        st.markdown(f"**最新信号（{latest_date}）**")
        st.dataframe(
            latest_scores[["code", "name", "score", "valuation_pct", "notes"]],
            use_container_width=True, hide_index=True,
        )

    with st.expander("交易明细", expanded=False):
        if trades.empty:
            st.caption("无交易")
        else:
            show_trades = trades.copy()
            show_trades["side"] = show_trades["side"].map({"BUY": "买入", "SELL": "卖出"}).fillna(show_trades["side"])
            show_trades["reason"] = show_trades["reason"].map(translate_signal_text)
            st.dataframe(show_trades, use_container_width=True, hide_index=True)

    with st.expander("估值来源", expanded=False):
        st.dataframe(sources, use_container_width=True, hide_index=True)

    st.caption("数据仅供参考 · 不构成投资建议 · 历史表现不代表未来收益")

SIGNAL_TEXT_MAP = {
    "market_weak": "大盘偏弱",
    "long_trend_up": "长期趋势向上",
    "trend_neutral": "趋势中性",
    "reasonable_pullback": "合理回撤",
    "deep_pullback": "深度回撤",
    "shallow_pullback": "浅回撤",
    "trend_continuation": "趋势延续",
    "valuation_low": "估值偏低",
    "valuation_fair": "估值合理",
    "valuation_acceptable": "估值可接受",
    "price_percentile_low": "价格分位偏低",
    "price_percentile_fair": "价格分位合理",
    "price_percentile_acceptable": "价格分位可接受",
    "relative_strength_low_vol": "相对强且波动较低",
    "relative_strength_ok": "相对强度尚可",
    "overheated_penalty": "过热扣分",
    "earnings_window_no_new_buy": "财报窗口不新开仓",
    "fallback_ranked_entry": "候选补位买入",
    "trend_hard_exit": "严重破位退出",
    "valuation_extreme_exit": "估值极端退出",
}

def translate_signal_text(text):
    if pd.isna(text):
        return text
    parts = str(text).split("|")
    translated = []
    for part in parts:
        if part.startswith("valuation_high_trim_"):
            translated.append("估值高位减仓")
        elif part.startswith("core_trend_confirmed_trim_"):
            translated.append("长期趋势转弱减仓")
        elif part.startswith("trend_break_trim_"):
            translated.append("趋势破位减仓")
        else:
            translated.append(SIGNAL_TEXT_MAP.get(part, part))
    return " + ".join(translated)

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.stApp { background-color: #f8fafc; color:#202124; }
.signal-buy   { background:#d9302514; border:1px solid #d93025;
                border-radius:8px; padding:12px; text-align:center; }
.signal-watch { background:#fbbc041c; border:1px solid #fbbc04;
                border-radius:8px; padding:12px; text-align:center; }
.signal-hold  { background:#1a73e814; border:1px solid #1a73e8;
                border-radius:8px; padding:12px; text-align:center; }
.signal-stop  { background:#18803814; border:1px solid #188038;
                border-radius:8px; padding:12px; text-align:center; }
.metric-box   { background:#ffffff; border:1px solid #e0e3e7;
                border-radius:8px; padding:10px; text-align:center; }
</style>
""", unsafe_allow_html=True)

st.title("📈 A股信号监控")
st.caption(f"数据来源：Yahoo Finance · T-1日线 · {date.today()}")

view = st.radio("视图", ["单股信号监控", "龙头长期回测"], horizontal=True, label_visibility="collapsed")
if view == "龙头长期回测":
    render_longhold_backtest()
    st.stop()

# 交易窗口状态
w_status, w_label = trading_window()
color = {"safe": "🟢", "danger": "🔴", "caution": "🟡"}[w_status]
st.info(f"{color} 交易窗口：{w_label}")

# ─── 快捷按钮 ─────────────────────────────────────────────────────────────────
st.markdown("**快捷选股**")
cols = st.columns(len(PRESETS))
selected_preset = None
for i, (label, code) in enumerate(PRESETS):
    if cols[i].button(label, key=f"btn_{code}", use_container_width=True):
        selected_preset = code

# ─── 输入框 ───────────────────────────────────────────────────────────────────
col_input, col_period = st.columns([3, 1])
with col_input:
    default_code = selected_preset or st.session_state.get("last_code", "")
    code_input = st.text_input(
        "股票代码",
        value=default_code,
        placeholder="601899 / 300750 / 1810",
        label_visibility="collapsed",
    )
with col_period:
    period = st.selectbox("回测", ["3y", "1y", "5y"], label_visibility="collapsed")

if code_input:
    st.session_state["last_code"] = code_input
    ticker = to_ticker(code_input)
    name   = guess_name(code_input)

    with st.spinner(f"拉取 {name}（{ticker}）数据..."):
        df = get_data(ticker, period)

    if df is None or len(df) < 25:
        st.error(f"❌ 无法获取 {ticker} 数据，请检查代码是否正确\n\n"
                 f"上交所：6位数字（如 601899）\n"
                 f"深交所：6位数字（如 300750）\n"
                 f"港  股：4位数字（如 0700 / 3690）\n"
                 f"美  股：英文代码（如 BABA / PDD / JD）")
        st.stop()

    ind = compute_signal(df)

    if ind is None:
        st.warning("数据不足，无法计算信号")
        st.stop()

    # ─── T-1 信号卡片 ──────────────────────────────────────────────────────
    st.divider()
    sig_class = {"BUY":"buy","WATCH":"watch","HOLD":"hold","STOP":"stop"}.get(ind["signal"],"hold")
    sig_label = {"BUY":"买  入","WATCH":"观  察","HOLD":"持  有","STOP":"⚡ 止损"}.get(ind["signal"])
    sig_color = {"BUY":"#d93025","WATCH":"#b06000","HOLD":"#1a73e8","STOP":"#188038"}.get(ind["signal"])

    c1, c2 = st.columns([2, 1])
    with c1:
        pct_color = "#d93025" if ind["pct"] >= 0 else "#188038"
        pct_str = f"+{ind['pct']:.2f}%" if ind["pct"] >= 0 else f"{ind['pct']:.2f}%"
        st.markdown(f"### {name} `{code_input}`")
        st.markdown(f"**T-1 {ind['date']}**")
        st.markdown(f"## ¥{ind['close']:.2f}  "
                    f"<span style='color:{pct_color}'>{pct_str}</span>",
                    unsafe_allow_html=True)
        st.caption(f"开 {ind['open']:.2f}  高 {ind['high']:.2f}  低 {ind['low']:.2f}")
    with c2:
        st.markdown(f"""<div class="signal-{sig_class}">
            <div style="font-size:11px;color:#888">信号</div>
            <div style="font-size:20px;font-weight:700;color:{sig_color}">{sig_label}</div>
        </div>""", unsafe_allow_html=True)

    # 指标行
    vr = ind["vol_ratio"]
    vr_str = f"{vr:.2f}x" if vr else "—"
    diff_str = f"+{ind['diff_ma20']:.1f}%" if ind["diff_ma20"] >= 0 else f"{ind['diff_ma20']:.1f}%"
    pe_info = get_pe_percentile(ticker)
    pe_str = f"{pe_info['pe_ttm']:.1f}x" if pe_info else "—"
    pe_pct_str = f"{pe_info['pe_pct']:.0f}%" if pe_info else "—"

    m1, m2, m3, m4, m5 = st.columns(5)
    for col, label, value, color_cond in [
        (m1, "量比",   vr_str,                 vr and vr >= 1.5),
        (m2, "20日均", f"¥{ind['ma20']:.2f}",  False),
        (m3, "偏离",   diff_str,               ind["diff_ma20"] >= 0),
        (m4, "60日均", f"¥{ind['ma60']:.2f}",  False),
        (m5, "PE分位", f"{pe_str} / {pe_pct_str}", pe_info and pe_info["pe_pct"] <= 35),
    ]:
        c = "#d93025" if color_cond else ("#188038" if not color_cond and label=="偏离" and ind["diff_ma20"]<0 else "#202124")
        col.markdown(f"""<div class="metric-box">
            <div style="font-size:10px;color:#888">{label}</div>
            <div style="color:{c};font-weight:600">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.caption(f"💡 {ind['reason']}")

    # ─── K线图 ─────────────────────────────────────────────────────────────
    st.plotly_chart(make_chart(df, name), use_container_width=True)

    # ─── 回测 ──────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(f"### 📊 回测结果（{period}）")

    trades = run_backtest(df)
    m      = calc_metrics(trades)

    if not m or m["n"] < 3:
        st.warning("交易次数不足 3 笔，回测结果参考意义有限")
    else:
        # 核心指标
        r1, r2, r3, r4, r5 = st.columns(5)
        for col, label, value, good in [
            (r1, "交易笔数", str(m["n"]),           None),
            (r2, "胜率",     f"{m['win_rate']}%",   m["win_rate"] >= 55),
            (r3, "累计收益", f"+{m['total_ret']}%"  if m["total_ret"]>=0 else f"{m['total_ret']}%", m["total_ret"] > 0),
            (r4, "最大回撤", f"-{m['max_dd']}%",    m["max_dd"] < 15),
            (r5, "盈亏比",   str(m["pf"]),           m["pf"] >= 1.5),
        ]:
            c = "#d93025" if good else ("#188038" if good is False else "#202124")
            col.markdown(f"""<div class="metric-box">
                <div style="font-size:10px;color:#888">{label}</div>
                <div style="color:{c};font-weight:700;font-size:16px">{value}</div>
            </div>""", unsafe_allow_html=True)

        st.caption(f"平均盈利 +{m['avg_win']}%  |  平均亏损 {m['avg_loss']}%  |  平均持仓 {m['avg_days']}天")

        # 权益曲线
        st.plotly_chart(make_equity_chart(m["equity"]), use_container_width=True)

        # 交易明细
        with st.expander(f"交易明细（{m['n']}笔）"):
            rows = []
            for t in trades:
                rows.append({
                    "进场": t["entry_date"], "出场": t["exit_date"],
                    "盈亏": f"{'+' if t['pnl']>=0 else ''}{t['pnl']}%",
                    "峰值": f"+{t['peak_ret']}%",
                    "天数": t["days"], "原因": t["reason"],
                })
            tdf = pd.DataFrame(rows)
            # Color the pnl column
            def color_pnl(val):
                c = "#d93025" if "+" in str(val) and val != "+0.0%" else "#188038"
                return f"color: {c}"
            st.dataframe(
                tdf.style.map(color_pnl, subset=["盈亏"]),
                use_container_width=True, hide_index=True,
            )

    # 免责
    st.caption("数据仅供参考 · 不构成投资建议 · 历史表现不代表未来收益")
