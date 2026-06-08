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
]

# ─── 代码 → Yahoo Ticker ─────────────────────────────────────────────────────
def to_ticker(code: str) -> str:
    code = code.strip().upper()
    if "." in code:
        return code
    if len(code) <= 4:                        # 港股
        return code.zfill(4) + ".HK"
    if code.startswith(("6", "9")):
        return code + ".SS"
    return code + ".SZ"

def guess_name(code: str) -> str:
    known = {"601899": "紫金矿业", "603993": "洛阳钼业", "300750": "宁德时代",
             "600900": "长江电力", "600036": "招商银行", "1810":   "小米集团"}
    return known.get(code.strip(), code)

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
        increasing_line_color="#00d68f", decreasing_line_color="#ff3d71",
        name="K线", showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(x=tail["date"], y=s20, name="MA20",
        line=dict(color="#f5c400", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=tail["date"], y=s60, name="MA60",
        line=dict(color="#4d9fff", width=1.5)), row=1, col=1)

    # Volume bars
    colors = ["#00d68f" if c >= o else "#ff3d71"
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
        plot_bgcolor="#0c1220", paper_bgcolor="#0c1220",
        font_color="#c0d0e0",
    )
    fig.update_xaxes(gridcolor="#1a2535", showgrid=True)
    fig.update_yaxes(gridcolor="#1a2535", showgrid=True)
    return fig

# ─── 权益曲线 ─────────────────────────────────────────────────────────────────
def make_equity_chart(equity: list) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity, mode="lines+markers",
        line=dict(color="#00d68f", width=2),
        marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(0,214,143,0.1)",
        name="累计收益%"))
    fig.add_hline(y=0, line_dash="dash", line_color="#555")
    fig.update_layout(
        title="权益曲线（每笔累计收益%）",
        height=260, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#0c1220", paper_bgcolor="#0c1220",
        font_color="#c0d0e0", showlegend=False,
    )
    fig.update_xaxes(gridcolor="#1a2535")
    fig.update_yaxes(gridcolor="#1a2535")
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
body { background-color: #080c18; }
.stApp { background-color: #080c18; }
.signal-buy   { background:#00d68f22; border:1px solid #00d68f;
                border-radius:8px; padding:12px; text-align:center; }
.signal-watch { background:#f5c40022; border:1px solid #f5c400;
                border-radius:8px; padding:12px; text-align:center; }
.signal-hold  { background:#4d9fff22; border:1px solid #4d9fff;
                border-radius:8px; padding:12px; text-align:center; }
.signal-stop  { background:#ff3d7122; border:1px solid #ff3d71;
                border-radius:8px; padding:12px; text-align:center; }
.metric-box   { background:#0c1220; border:1px solid #1a2535;
                border-radius:8px; padding:10px; text-align:center; }
</style>
""", unsafe_allow_html=True)

st.title("📈 A股信号监控")
st.caption(f"数据来源：Yahoo Finance · T-1日线 · {date.today()}")

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
                 f"港  股：4位数字（如 1810）")
        st.stop()

    ind = compute_signal(df)

    if ind is None:
        st.warning("数据不足，无法计算信号")
        st.stop()

    # ─── T-1 信号卡片 ──────────────────────────────────────────────────────
    st.divider()
    sig_class = {"BUY":"buy","WATCH":"watch","HOLD":"hold","STOP":"stop"}.get(ind["signal"],"hold")
    sig_label = {"BUY":"买  入","WATCH":"观  察","HOLD":"持  有","STOP":"⚡ 止损"}.get(ind["signal"])
    sig_color = {"BUY":"#00d68f","WATCH":"#f5c400","HOLD":"#4d9fff","STOP":"#ff3d71"}.get(ind["signal"])

    c1, c2 = st.columns([2, 1])
    with c1:
        pct_color = "green" if ind["pct"] >= 0 else "red"
        pct_str = f"+{ind['pct']:.2f}%" if ind["pct"] >= 0 else f"{ind['pct']:.2f}%"
        st.markdown(f"### {name} `{code_input}`")
        st.markdown(f"**T-1 {ind['date']}**")
        st.markdown(f"## ¥{ind['close']:.2f}  "
                    f"<span style='color:{'#00d68f' if ind['pct']>=0 else '#ff3d71'}'>{pct_str}</span>",
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

    m1, m2, m3, m4 = st.columns(4)
    for col, label, value, color_cond in [
        (m1, "量比",   vr_str,                 vr and vr >= 1.5),
        (m2, "20日均", f"¥{ind['ma20']:.2f}",  False),
        (m3, "偏离",   diff_str,               ind["diff_ma20"] >= 0),
        (m4, "60日均", f"¥{ind['ma60']:.2f}",  False),
    ]:
        c = "#00d68f" if color_cond else ("#ff3d71" if not color_cond and label=="偏离" and ind["diff_ma20"]<0 else "#c0d0e0")
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
            c = "#00d68f" if good else ("#ff3d71" if good is False else "#c0d0e0")
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
                c = "#00d68f" if "+" in str(val) and val != "+0.0%" else "#ff3d71"
                return f"color: {c}"
            st.dataframe(
                tdf.style.map(color_pnl, subset=["盈亏"]),
                use_container_width=True, hide_index=True,
            )

    # 免责
    st.caption("数据仅供参考 · 不构成投资建议 · 历史表现不代表未来收益")
