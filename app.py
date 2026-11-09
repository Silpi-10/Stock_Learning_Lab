"""Stock Learning Lab — AI-based stock prediction & virtual trading (Streamlit)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import trading
from src.data import fetch_history
from src.indicators import add_indicators
from src.models import train_all

st.set_page_config(page_title="Stock Learning Lab", page_icon="📈", layout="wide")
trading.init_db()

POPULAR = ["AAPL", "MSFT", "GOOGL", "TSLA", "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]


@st.cache_data(ttl=900, show_spinner="Fetching market data…")
def load(ticker, period):
    df, src = fetch_history(ticker, period)
    return add_indicators(df), src


# ---------------- Sidebar ----------------
st.sidebar.title("📈 Stock Learning Lab")
user = st.sidebar.text_input("Trader name", "student").strip() or "student"
trading.ensure_account(user)
page = st.sidebar.radio("Navigate", ["Dashboard", "Model Lab", "Virtual Trading", "Portfolio"])
choice = st.sidebar.selectbox("Ticker", POPULAR + ["Custom…"])
ticker = (st.sidebar.text_input("Custom ticker", "NVDA") if choice == "Custom…" else choice).upper()
period = st.sidebar.selectbox("History", ["1y", "2y", "5y", "10y"], index=2)

data, source = load(ticker, period)
if source == "synthetic":
    st.sidebar.warning("Live data unavailable — showing synthetic demo data.")
price = float(data["Close"].iloc[-1])
st.sidebar.metric(ticker, f"{price:,.2f}", f"{data['Return_1d'].iloc[-1]*100:+.2f}%")


# ---------------- Dashboard ----------------
def page_dashboard():
    st.header(f"Market Explorer — {ticker}")
    opts = st.multiselect("Overlays", ["SMA_20", "SMA_50", "EMA_12", "EMA_26", "Bollinger"],
                          default=["SMA_20", "SMA_50"])
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[.5, .15, .2, .15],
                        vertical_spacing=.03,
                        subplot_titles=("Price", "Volume", "MACD", "RSI (14)"))
    fig.add_trace(go.Candlestick(x=data.index, open=data.Open, high=data.High, low=data.Low,
                                 close=data.Close, name="OHLC"), 1, 1)
    for o in opts:
        if o == "Bollinger":
            for col in ("BB_Upper", "BB_Lower"):
                fig.add_trace(go.Scatter(x=data.index, y=data[col], name=col,
                                         line=dict(width=1, dash="dot")), 1, 1)
        else:
            fig.add_trace(go.Scatter(x=data.index, y=data[o], name=o, line=dict(width=1.3)), 1, 1)
    fig.add_trace(go.Bar(x=data.index, y=data.Volume, name="Volume"), 2, 1)
    fig.add_trace(go.Scatter(x=data.index, y=data.MACD, name="MACD"), 3, 1)
    fig.add_trace(go.Scatter(x=data.index, y=data.MACD_Signal, name="Signal"), 3, 1)
    fig.add_trace(go.Bar(x=data.index, y=data.MACD_Hist, name="Hist"), 3, 1)
    fig.add_trace(go.Scatter(x=data.index, y=data.RSI_14, name="RSI"), 4, 1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
    fig.update_layout(height=850, xaxis_rangeslider_visible=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    last = data.iloc[-1]
    c1, c2, c3 = st.columns(3)
    rsi_v = last.RSI_14
    c1.metric("RSI (14)", f"{rsi_v:.1f}", "Overbought" if rsi_v > 70 else "Oversold" if rsi_v < 30 else "Neutral")
    c2.metric("MACD vs Signal", f"{last.MACD:.3f}", "Bullish" if last.MACD > last.MACD_Signal else "Bearish")
    c3.metric("Price vs SMA-50", f"{(last.Close/last.SMA_50-1)*100:+.1f}%")
    with st.expander("Raw data"):
        st.dataframe(data.tail(200))


# ---------------- Model Lab ----------------
def page_models():
    st.header(f"Model Lab — {ticker}")
    st.caption("Models predict next-day return; price = close × (1 + return). "
               "Chronological 80/20 train/test split (no shuffling).")
    use_lstm = st.checkbox("Include LSTM (TensorFlow, slower)", True)
    if st.button("Train & compare models", type="primary"):
        status = st.empty()
        results, test_df, fc = train_all(data.dropna(), use_lstm, progress=status.info)
        status.empty()
        st.session_state[f"res_{ticker}"] = (results, test_df, fc)
    if f"res_{ticker}" not in st.session_state:
        st.info("Click the button to train all models."); return
    results, test_df, fc = st.session_state[f"res_{ticker}"]

    st.subheader("Performance on test set")
    st.dataframe(results.style.highlight_min(subset=["MAE", "RMSE"], color="#c8f7c5")
                 .highlight_max(subset=["Accuracy (%)", "Directional Accuracy (%)"], color="#c8f7c5"),
                 use_container_width=True)
    bar = go.Figure([go.Bar(x=results.index, y=results["Directional Accuracy (%)"], name="Directional acc.")])
    bar.add_hline(y=50, line_dash="dash", annotation_text="coin-flip baseline")
    bar.update_layout(title="Directional accuracy (%)", height=300)
    st.plotly_chart(bar, use_container_width=True)

    st.subheader("Predicted vs actual price (test period)")
    models = list(results.index)
    show = st.multiselect("Models", models, default=models)
    fig = go.Figure(go.Scatter(x=test_df.index, y=test_df["Actual"], name="Actual",
                               line=dict(color="black", width=2)))
    for m in show:
        fig.add_trace(go.Scatter(x=test_df.index, y=test_df[m], name=m, line=dict(width=1)))
    fig.update_layout(height=450, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Next-day forecast")
    cols = st.columns(len(fc))
    for col, (m, v) in zip(cols, fc.items()):
        col.metric(m, f"{v['price']:,.2f}", f"{v['return_%']:+.2f}%")
    st.caption("⚠️ Educational use only. Markets are noisy; directional accuracy near 50% is normal.")


# ---------------- Virtual Trading ----------------
def page_trade():
    st.header(f"Virtual Trading — {ticker}")
    cash = trading.get_cash(user)
    c1, c2 = st.columns(2)
    c1.metric("Cash available", f"{cash:,.2f}")
    c2.metric(f"Market price ({ticker})", f"{price:,.2f}")
    h = trading.holdings(user)
    owned = int(h.loc[h.ticker == ticker, "qty"].sum()) if not h.empty else 0
    st.write(f"You own **{owned}** shares of {ticker}.")
    qty = st.number_input("Quantity", min_value=1, value=10, step=1)
    st.write(f"Order value: **{qty*price:,.2f}**")
    b, s = st.columns(2)
    for col, side in ((b, "BUY"), (s, "SELL")):
        if col.button(side, use_container_width=True, type="primary" if side == "BUY" else "secondary"):
            ok, msg = trading.trade(user, ticker, side, qty, price)
            (st.success if ok else st.error)(msg)
            if ok:
                pass
    st.subheader("Transaction history")
    st.dataframe(trading.transactions(user), use_container_width=True, hide_index=True)


# ---------------- Portfolio ----------------
def page_portfolio():
    st.header("Portfolio")
    cash = trading.get_cash(user)
    h = trading.holdings(user)
    rows, mv_total = [], 0.0
    for _, r in h.iterrows():
        try:
            px = float(load(r.ticker, "1y")[0]["Close"].iloc[-1])
        except Exception:
            px = r.avg_cost
        mv = px * r.qty; mv_total += mv
        rows.append({"Ticker": r.ticker, "Qty": r.qty, "Avg cost": r.avg_cost, "Price": px,
                     "Market value": mv, "P/L": mv - r.avg_cost * r.qty,
                     "P/L %": (px / r.avg_cost - 1) * 100})
    equity = cash + mv_total
    pl = equity - trading.START_CASH
    trading.snapshot(user, equity)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total value", f"{equity:,.2f}", f"{pl:+,.2f}")
    c2.metric("Cash", f"{cash:,.2f}")
    c3.metric("Invested", f"{mv_total:,.2f}")
    c4.metric("Return", f"{pl/trading.START_CASH*100:+.2f}%")
    if rows:
        df = pd.DataFrame(rows).round(2)
        st.dataframe(df, use_container_width=True, hide_index=True)
        pie = go.Figure(go.Pie(labels=list(df.Ticker) + ["Cash"], values=list(df["Market value"]) + [cash], hole=.4))
        pie.update_layout(title="Allocation", height=350)
        st.plotly_chart(pie, use_container_width=True)
    else:
        st.info("No holdings yet — head to Virtual Trading.")
    snaps = trading.snapshots(user)
    if len(snaps) > 1:
        st.plotly_chart(go.Figure(go.Scatter(x=snaps.ts, y=snaps.equity, mode="lines+markers"))
                        .update_layout(title="Portfolio value over time", height=320), use_container_width=True)
    with st.expander("Danger zone"):
        if st.button("Reset account"):
            trading.reset(user); st.success("Account reset."); st.rerun()


{"Dashboard": page_dashboard, "Model Lab": page_models,
 "Virtual Trading": page_trade, "Portfolio": page_portfolio}[page]()
