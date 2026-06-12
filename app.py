"""
A股 / 港股 / 美股中概  信号监控 & 回测  |  Streamlit Web App
运行：streamlit run app.py
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

# ─── 页面配置 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="信号监控", page_icon="📈",
    layout="centered", initial_sidebar_state="collapsed",
)

# ─── 交易成本 & 预热（与回测脚本保持一致）──────────────────────────────────────
BUY_COST    = 0.0003   # 买入佣金 0.03%
SELL_COST   = 0.0013   # 卖出佣金 0.03% + 印花税 0.10%
WARMUP_DAYS = 60       # 前 60 根 K 线不产生信号

# ─── 快捷按钮 ─────────────────────────────────────────────────────────────────
PRESETS = [
    ("紫金",  "601899"), ("洛阳钼", "603993"), ("宁德",  "300750"),
    ("长江电","600900"), ("招商银", "600036"), ("小米",  "1810"),
    ("腾讯",  "0700"),  ("美团",   "3690"),
    ("阿里",  "BABA"),  ("拼多多", "PDD"),    ("京东",  "JD"),
]

# ─── 代码识别 → Yahoo Ticker ──────────────────────────────────────────────────
def to_ticker(code: str) -> str:
    code = code.strip().upper()
    if "." in code:                          # 已含后缀，直接返回
        return code
    if code.isalpha():                       # 纯字母 → 美股（PDD / BABA / JD）
        return code
    if code.isdigit() and len(code) <= 5:   # 纯数字 ≤5位 → 港股
        return code.zfill(4) + ".HK"
    if code.startswith(("6", "9")):          # 6位数字 6/9开头 → 上交所
        return code + ".SS"
    return code + ".SZ"                      # 其余 → 深交所

def guess_name(code: str) -> str:
    known = {
        "601899":"紫金矿业","603993":"洛阳钼业","300750":"宁德时代",
        "600900":"长江电力","600036":"招商银行","1810":"小米集团",
        "0700":"腾讯控股","700":"腾讯控股","3690":"美团",
        "BABA":"阿里巴巴","PDD":"拼多多","JD":"京东",
        "BIDU":"百度","NIO":"蔚来",
    }
    return known.get(code.strip().upper(), code)

# ─── 数据获取 ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_data(ticker: str, period: str = "3y"):
    try:
        raw = yf.download(ticker, period=period, interval="1d",
                          progress=False, auto_adjust=True)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.reset_index()
        df.columns = [str(c).lower().strip() for c in df.columns]
        df["date"]   = df["date"].astype(str).str[:10]
        df["amount"] = df["close"] * df["volume"]
        df["pct"]    = df["close"].pct_change() * 100
        return df[["date","open","high","low","close","volume","amount","pct"]]\
               .dropna(subset=["close"]).reset_index(drop=True)
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_pe(ticker: str):
    """返回 (当前PE, 市值亿, PE所在历史分位数%)"""
    try:
        info = yf.Ticker(ticker).info
        pe   = info.get("trailingPE") or info.get("forwardPE")
        mktcap = info.get("marketCap")
        mktcap_str = None
        if mktcap:
            if mktcap >= 1e12:
                mktcap_str = f"{mktcap/1e12:.1f}万亿"
            elif mktcap >= 1e8:
                mktcap_str = f"{mktcap/1e8:.0f}亿"
        return pe, mktcap_str
    except Exception:
        return None, None

# ─── 信号计算 ─────────────────────────────────────────────────────────────────
def compute_signal(df, ma=20, vol_thr=1.5):
    n = len(df)
    if n < ma + 3: return None
    t1,t2,t3 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    s20  = df.tail(ma);  ma20 = s20["close"].mean()
    s60  = df.tail(60);  ma60 = s60["close"].mean()
    vol_ma    = s20["volume"].mean()
    vol_ratio = t1["volume"] / vol_ma if vol_ma > 0 else None

    diff  = (t1["close"] - ma20) / ma20 * 100
    above = t1["close"] > ma20
    bH    = max(t1["open"], t1["close"])
    bL    = min(t1["open"], t1["close"])
    body  = bH - bL or t1["close"] * 0.001
    lsh   = bL - t1["low"];  ush = t1["high"] - bH
    green = t1["close"] >= t1["open"]

    hammer = lsh >= body*2 and ush <= body*0.5 and lsh > 0
    engulf = (green and t2["close"]<t2["open"]
              and t1["open"]<=t2["close"] and t1["close"]>=t2["open"])
    vbreak = bool(vol_ratio) and vol_ratio>=vol_thr and green and -1<=diff<=6
    shrink = bool(vol_ratio) and vol_ratio<0.8 and 0<diff<8 and not green
    bkdown = not green and not above and (not vol_ratio or vol_ratio>=1.0)
    below3 = not above and t2["close"]<ma20 and t3["close"]<ma20

    pts, tags = 0, []
    if hammer: pts+=2; tags.append("长下影线")
    if engulf: pts+=2; tags.append("阳包阴")
    if vbreak: pts+=3; tags.append("放量突破均线")
    elif vol_ratio and vol_ratio>=vol_thr and green: pts+=1; tags.append("放量阳线")
    if above and t1["close"]>ma60: pts+=1
    if shrink: pts+=1; tags.append("缩量回踩")

    if   below3 and (not vol_ratio or vol_ratio>=0.8): sig,reason="STOP","连续三日均线下方，破位风险高"
    elif bkdown:                                        sig,reason="STOP","有量下穿均线，注意止损"
    elif pts>=4:                                        sig,reason="BUY", " + ".join(tags)
    elif shrink or pts>=2:                              sig,reason="WATCH",(" + ".join(tags)+"，" if tags else "")+"等待放量确认"
    elif above:                                         sig,reason="HOLD",f"均线上方 +{diff:.1f}%，趋势完好"
    else:                                               sig,reason="WATCH",f"均线下方 {abs(diff):.1f}%，观察企稳"

    # 价格所在历史分位（用现有数据）
    prices = df["close"].values
    price_pct = round((prices < t1["close"]).mean() * 100, 1)

    return dict(
        date=t1["date"], close=t1["close"], open=t1["open"],
        high=t1["high"], low=t1["low"], volume=t1["volume"],
        pct=t1["pct"], vol_ratio=vol_ratio, vol_ma=vol_ma,
        ma20=ma20, ma60=ma60, diff_ma20=diff, above_ma20=above,
        signal=sig, reason=reason, price_pct=price_pct,
    )

# ─── 回测 ─────────────────────────────────────────────────────────────────────
def run_backtest(df, stop_pct=8.0, max_hold=40, ma=20, vol_thr=1.5):
    trades, pos, n = [], None, len(df)
    for i in range(ma+5, n-1):
        ind = compute_signal(df.iloc[:i+1], ma, vol_thr)
        if ind is None: continue
        sig = ind["signal"]
        if pos is None:
            if sig == "BUY":
                px  = float(df.iloc[i+1]["open"])
                pos = {"px":px,"date":str(df.iloc[i+1]["date"]),"i":i+1,"peak":px}
        else:
            close = float(df.iloc[i]["close"])
            pnl   = (close - pos["px"]) / pos["px"] * 100
            held  = i - pos["i"]
            pos["peak"] = max(pos["peak"], close)
            reason = None
            if   pnl <= -stop_pct:  reason = f"硬止损-{stop_pct:.0f}%"
            elif sig == "STOP":     reason = "信号止损"
            elif held >= max_hold:  reason = f"超{max_hold}天"
            if reason:
                ex    = float(df.iloc[i+1]["open"]) if i+1<n else close
                final = (ex - pos["px"]) / pos["px"] * 100
                peak  = (pos["peak"] - pos["px"]) / pos["px"] * 100
                trades.append(dict(
                    entry_date=pos["date"],
                    exit_date=str(df.iloc[i+1]["date"] if i+1<n else df.iloc[i]["date"]),
                    entry_px=pos["px"], exit_px=ex,
                    pnl=round(final,2), days=held,
                    reason=reason, peak_ret=round(peak,1)))
                pos = None
    if pos:
        ex   = float(df.iloc[-1]["close"])
        peak = (pos["peak"]-pos["px"])/pos["px"]*100
        trades.append(dict(entry_date=pos["date"],exit_date=str(df.iloc[-1]["date"]),
                           entry_px=pos["px"],exit_px=ex,
                           pnl=round((ex-pos["px"])/pos["px"]*100,2),
                           days=n-pos["i"],reason="回测结束",peak_ret=round(peak,1)))
    return trades

def calc_metrics(trades):
    if not trades: return None
    pnls  = [t["pnl"] for t in trades]
    wins  = [p for p in pnls if p>0]
    loss  = [p for p in pnls if p<=0]
    cum,peak,mdd,eq = 1.0,1.0,0.0,[]
    for p in pnls:
        cum *= (1+p/100); eq.append(round((cum-1)*100,2))
        if cum>peak: peak=cum
        dd=(peak-cum)/peak*100
        if dd>mdd: mdd=dd
    return dict(n=len(trades),wins=len(wins),losses=len(loss),
                win_rate=round(len(wins)/len(trades)*100,1),
                avg_win=round(sum(wins)/len(wins),2)   if wins else 0,
                avg_loss=round(sum(loss)/len(loss),2)  if loss else 0,
                pf=round(sum(wins)/abs(sum(loss)),2)   if loss and sum(loss)!=0 else 99,
                total_ret=round((cum-1)*100,1), max_dd=round(mdd,1),
                avg_days=round(sum(t["days"] for t in trades)/len(trades),1),
                equity=eq)

# ─── 交易窗口 ─────────────────────────────────────────────────────────────────
def trading_window():
    m,d = date.today().month, date.today().day
    if m in [3,4] or (m==5 and d<=15): return "danger",  "❌ 禁入期（年报+一季报）"
    if m in [7,8]:                     return "danger",  "❌ 禁入期（半年报）"
    if m==10:                          return "danger",  "❌ 禁入期（三季报）"
    if (m==5 and d>=16) or (m==6 and d<=25): return "safe","✅ 安全窗口①"
    if m==9:                           return "safe",    "✅ 安全窗口②"
    if m in [11,12,1] or (m==2 and d<20):    return "safe","✅ 安全窗口③"
    return "caution", "⚠️ 过渡期，谨慎操作"

# ─── 图表（红涨绿跌，浅色背景）────────────────────────────────────────────────
RISE_COLOR = "#e84040"   # 红 = 涨
FALL_COLOR = "#1aab5e"   # 绿 = 跌
CHART_BG   = "#ffffff"
GRID_COLOR = "#e8e8e8"
TEXT_COLOR = "#333333"

def make_chart(df, name):
    tail   = df.tail(60).copy()
    s20    = df["close"].rolling(20).mean().tail(60)
    s60    = df["close"].rolling(60).mean().tail(60)
    vol_ma = df["volume"].rolling(20).mean().tail(60)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(
        x=tail["date"], open=tail["open"], high=tail["high"],
        low=tail["low"], close=tail["close"],
        increasing_line_color=RISE_COLOR, increasing_fillcolor=RISE_COLOR,
        decreasing_line_color=FALL_COLOR, decreasing_fillcolor=FALL_COLOR,
        name="K线", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=tail["date"], y=s20, name="MA20",
        line=dict(color="#f5a623", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=tail["date"], y=s60, name="MA60",
        line=dict(color="#4a90d9", width=1.5)), row=1, col=1)

    bar_colors = [RISE_COLOR if c>=o else FALL_COLOR
                  for c,o in zip(tail["close"], tail["open"])]
    fig.add_trace(go.Bar(x=tail["date"], y=tail["volume"],
        marker_color=bar_colors, name="成交量", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=tail["date"], y=vol_ma, name="均量",
        line=dict(color="#f5a623", width=1.2, dash="dot")), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"{name}  近60日", font_size=14, font_color=TEXT_COLOR),
        xaxis_rangeslider_visible=False,
        height=460, margin=dict(l=10,r=10,t=40,b=10),
        legend=dict(orientation="h", y=1.02, x=0, font_color=TEXT_COLOR),
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG, font_color=TEXT_COLOR,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, linecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, linecolor=GRID_COLOR)
    return fig

def make_equity_chart(equity):
    pos = [v for v in equity if v > 0]
    line_color = RISE_COLOR if (equity[-1] > 0 if equity else False) else FALL_COLOR
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity, mode="lines+markers",
        line=dict(color=line_color, width=2),
        marker=dict(size=4),
        fill="tozeroy", fillcolor=f"rgba({int(line_color[1:3],16)},{int(line_color[3:5],16)},{int(line_color[5:7],16)},0.1)",
        name="累计收益%"))
    fig.add_hline(y=0, line_dash="dash", line_color="#aaaaaa")
    fig.update_layout(
        title="权益曲线", height=240,
        margin=dict(l=10,r=10,t=40,b=10),
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
        font_color=TEXT_COLOR, showlegend=False,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.sig-buy   {background:#fff0f0;border:1.5px solid #e84040;border-radius:8px;padding:12px;text-align:center}
.sig-watch {background:#fffbe6;border:1.5px solid #f5a623;border-radius:8px;padding:12px;text-align:center}
.sig-hold  {background:#f0f6ff;border:1.5px solid #4a90d9;border-radius:8px;padding:12px;text-align:center}
.sig-stop  {background:#f0fff4;border:1.5px solid #1aab5e;border-radius:8px;padding:12px;text-align:center}
.mbox      {background:#f8f9fa;border:1px solid #e9ecef;border-radius:8px;padding:10px;text-align:center}
.rise      {color:#e84040;font-weight:700}
.fall      {color:#1aab5e;font-weight:700}
</style>
""", unsafe_allow_html=True)

st.title("📈 A股 / 港股 / 美股信号监控")
st.caption(f"数据来源：Yahoo Finance · T-1日线 · {date.today()}")

w_status, w_label = trading_window()
icon = {"safe":"🟢","danger":"🔴","caution":"🟡"}[w_status]
st.info(f"{icon} 交易窗口：{w_label}")

# 快捷按钮
st.markdown("**快捷选股**")
rows = [PRESETS[:6], PRESETS[6:]]
for row in rows:
    cols = st.columns(len(row))
    for i,(label,code) in enumerate(row):
        if cols[i].button(label, key=f"b_{code}", use_container_width=True):
            st.session_state["selected"] = code

# 输入框
c1, c2 = st.columns([3,1])
with c1:
    default = st.session_state.get("selected", st.session_state.get("last",""))
    code_input = st.text_input("代码", value=default, placeholder="601899 / 0700 / PDD",
                               label_visibility="collapsed")
with c2:
    period = st.selectbox("回测", ["3y","1y","5y"], label_visibility="collapsed")

if code_input:
    st.session_state["last"] = code_input
    ticker = to_ticker(code_input)
    name   = guess_name(code_input)

    with st.spinner(f"拉取 {name}（{ticker}）..."):
        df = get_data(ticker, period)
        pe, mktcap = get_pe(ticker)

    if df is None or len(df) < 25:
        st.error(
            f"❌ 无法获取 **{ticker}** 数据，请检查代码格式\n\n"
            "| 市场 | 格式 | 示例 |\n|---|---|---|\n"
            "| 上交所 | 6位数字，6/9开头 | 601899 |\n"
            "| 深交所 | 6位数字，0/3开头 | 300750 |\n"
            "| 港 股 | 4位数字 | 0700 |\n"
            "| 美 股 | 英文字母 | PDD |"
        )
        st.stop()

    ind = compute_signal(df)
    if ind is None:
        st.warning("数据不足，无法计算信号"); st.stop()

    # ─── 信号卡片 ──────────────────────────────────────────────────────────
    st.divider()
    sig_map = {"BUY":("buy","买  入","#e84040"),
               "WATCH":("watch","观  察","#f5a623"),
               "HOLD":("hold","持  有","#4a90d9"),
               "STOP":("stop","卖  出","#1aab5e")}
    sig_cls, sig_lbl, sig_col = sig_map.get(ind["signal"], ("hold","—","#888"))

    pct_cls = "rise" if ind["pct"] >= 0 else "fall"
    pct_str = f"+{ind['pct']:.2f}%" if ind["pct"] >= 0 else f"{ind['pct']:.2f}%"

    c1, c2 = st.columns([2,1])
    with c1:
        st.markdown(f"### {name} `{code_input.upper()}`")
        st.caption(f"T-1  {ind['date']}")
        st.markdown(
            f"<span style='font-size:28px;font-weight:700'>¥{ind['close']:.2f}</span> "
            f"<span class='{pct_cls}' style='font-size:16px'>{pct_str}</span>",
            unsafe_allow_html=True)
        st.caption(f"开 {ind['open']:.2f}  高 {ind['high']:.2f}  低 {ind['low']:.2f}")
    with c2:
        st.markdown(
            f'<div class="sig-{sig_cls}">'
            f'<div style="font-size:11px;color:#888;margin-bottom:4px">信号</div>'
            f'<div style="font-size:22px;font-weight:700;color:{sig_col}">{sig_lbl}</div>'
            f'</div>', unsafe_allow_html=True)

    # ─── 指标行 ────────────────────────────────────────────────────────────
    vr      = ind["vol_ratio"]
    vr_str  = f"{vr:.2f}x" if vr else "—"
    vr_cls  = "rise" if (vr and vr>=1.5) else ("fall" if (vr and vr<0.7) else "")
    diff_cls= "rise" if ind["diff_ma20"]>=0 else "fall"
    diff_str= f"+{ind['diff_ma20']:.1f}%" if ind["diff_ma20"]>=0 else f"{ind['diff_ma20']:.1f}%"
    pp      = ind["price_pct"]
    pp_cls  = "rise" if pp>=70 else ("fall" if pp<=30 else "")

    # PE 分位
    pe_str  = f"{pe:.1f}x" if pe else "—"
    pe_level= ""
    if pe:
        if   pe < 15: pe_level = "低"
        elif pe < 30: pe_level = "中"
        elif pe < 50: pe_level = "偏高"
        else:         pe_level = "高"
    pe_cls  = "fall" if pe and pe<15 else ("rise" if pe and pe>40 else "")

    cols = st.columns(5)
    items = [
        ("量比",         vr_str,   vr_cls),
        ("均线偏离",     diff_str, diff_cls),
        (f"价格分位({period})", f"{pp}%", pp_cls),
        ("市盈率(PE)",   pe_str,   pe_cls),
        ("市值",         mktcap or "—", ""),
    ]
    for col,(label,val,cls) in zip(cols, items):
        col.markdown(
            f'<div class="mbox">'
            f'<div style="font-size:10px;color:#888">{label}</div>'
            f'<div class="{cls}" style="font-size:14px;{"font-weight:700" if cls else ""}'
            f'">{val}</div>'
            f'</div>', unsafe_allow_html=True)

    st.caption(f"💡 {ind['reason']}")

    # ─── K线图 ─────────────────────────────────────────────────────────────
    st.plotly_chart(make_chart(df, name), use_container_width=True)

    # ─── 回测结果 ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown(f"### 📊 回测（{period}）")

    stop_col, hold_col, vthr_col = st.columns(3)
    stop_pct = stop_col.slider("止损%", 5, 15, 8)
    max_hold = hold_col.slider("最长持仓天", 20, 60, 40)
    vol_thr  = vthr_col.slider("量比阈值", 1.0, 2.5, 1.5, step=0.1)

    trades = run_backtest(df, stop_pct=stop_pct, max_hold=max_hold, vol_thr=vol_thr)
    m      = calc_metrics(trades)

    if not m or m["n"] < 3:
        st.warning("交易次数不足 3 笔，回测结果参考意义有限")
    else:
        r1,r2,r3,r4,r5 = st.columns(5)
        metrics_items = [
            (r1, "交易笔数",  str(m["n"]),  None),
            (r2, "胜率",      f"{m['win_rate']}%", m["win_rate"]>=55),
            (r3, "累计收益",  f"{'+' if m['total_ret']>=0 else ''}{m['total_ret']}%", m["total_ret"]>0),
            (r4, "最大回撤",  f"-{m['max_dd']}%",  m["max_dd"]<15),
            (r5, "盈亏比",    str(m["pf"]),  m["pf"]>=1.5),
        ]
        for col,label,val,good in metrics_items:
            c = RISE_COLOR if good else (FALL_COLOR if good is False else "#333")
            col.markdown(
                f'<div class="mbox">'
                f'<div style="font-size:10px;color:#888">{label}</div>'
                f'<div style="color:{c};font-weight:700;font-size:18px">{val}</div>'
                f'</div>', unsafe_allow_html=True)

        st.caption(f"平均盈利 +{m['avg_win']}%  ·  平均亏损 {m['avg_loss']}%  ·  平均持仓 {m['avg_days']}天")
        st.plotly_chart(make_equity_chart(m["equity"]), use_container_width=True)

        with st.expander(f"交易明细（{m['n']}笔）"):
            rows = [{"进场":t["entry_date"],"出场":t["exit_date"],
                     "盈亏":f"{'+' if t['pnl']>=0 else ''}{t['pnl']}%",
                     "峰值":f"+{t['peak_ret']}%","天数":t["days"],
                     "原因":t["reason"]} for t in trades]
            tdf = pd.DataFrame(rows)
            def color_pnl(val):
                if isinstance(val, str) and "+" in val and val!= "+0.0%":
                    return f"color:{RISE_COLOR}"
                elif isinstance(val, str) and val.startswith("-"):
                    return f"color:{FALL_COLOR}"
                return ""
            st.dataframe(tdf.style.map(color_pnl, subset=["盈亏"]),
                         use_container_width=True, hide_index=True)

    st.caption("仅供参考 · 不构成投资建议 · 历史表现不代表未来")
