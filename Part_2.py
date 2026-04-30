import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="StockSight",
    page_icon="📈",
    layout="wide"
)

# ── Clean Title UI ──────────────────────────────────────
st.markdown("""
<h1 style='text-align: center; font-size: 48px;'>📈 StockSight</h1>
<p style='text-align: center; color: gray; margin-top: -10px;'>
Real-time Stock Analytics & Portfolio Dashboard
</p>
<hr>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Part 1: Individual Stock Analysis", "💼 Part 2: Portfolio Dashboard"])


# ════════════════════════════════════════════════════════
#  SHARED HELPERS
# ════════════════════════════════════════════════════════

@st.cache_data(ttl=1800)
def get_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)
    return df

def safe_fetch(ticker, start, end):
    for i in range(3):
        try:
            return get_data(ticker, start, end)
        except Exception as e:
            if "Too Many Requests" in str(e):
                time.sleep(2 * (i + 1))
            else:
                raise e
    raise Exception("Rate limited. Try again.")

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ════════════════════════════════════════════════════════
#  PART 1 — Individual Stock Analysis
# ════════════════════════════════════════════════════════

with tab1:
    # ── Sidebar-style controls inside tab ───────────────
    with st.sidebar:
        st.header("⚙️ Part 1 Settings")
        ticker = st.text_input("Ticker", "AAPL").upper()
        six_months_ago = datetime.today() - timedelta(days=182)
        start_date = st.date_input("Start Date", six_months_ago)
        end_date   = st.date_input("End Date",   datetime.today())
        forecast_days = st.slider("Forecast Days", 5, 60, 30)
        ma_short = st.slider("Short MA", 5, 50, 20)
        ma_long  = st.slider("Long MA", 30, 200, 50)
        run_btn  = st.button("▶ Analyze Stock")

    if run_btn or "p1_loaded" not in st.session_state:
        st.session_state.p1_loaded = True

        with st.spinner(f"Fetching {ticker} data..."):
            try:
                df = safe_fetch(ticker, start_date, end_date)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        if df.empty:
            st.error("Invalid ticker or no data.")
            st.stop()

        df = df.reset_index()

        # ── Moving averages ────────────────────────────
        df["MA20"]  = df["Close"].rolling(20).mean()
        df["MA50"]  = df["Close"].rolling(50).mean()
        df[f"MA_Short"] = df["Close"].rolling(ma_short).mean()
        df[f"MA_Long"]  = df["Close"].rolling(ma_long).mean()

        # ── RSI ────────────────────────────────────────
        df["RSI"] = compute_rsi(df["Close"], 14)

        # ── Volatility (20-day annualized) ─────────────
        df["Returns"]    = df["Close"].pct_change()
        df["Volatility"] = df["Returns"].rolling(20).std() * np.sqrt(252) * 100

        # ── Golden / Death Cross ───────────────────────
        df["Signal"]    = 0
        df.loc[ma_short:, "Signal"] = np.where(
            df["MA_Short"][ma_short:] > df["MA_Long"][ma_short:], 1, 0
        )
        df["Crossover"] = df["Signal"].diff()
        golden_cross = df[df["Crossover"] ==  1]
        death_cross  = df[df["Crossover"] == -1]

        # ── Latest values ──────────────────────────────
        latest       = df.dropna(subset=["MA20","MA50","RSI","Volatility"]).iloc[-1]
        price        = latest["Close"]
        ma20_val     = latest["MA20"]
        ma50_val     = latest["MA50"]
        rsi_val      = latest["RSI"]
        vol_val      = latest["Volatility"]

        # ── Trend classification ───────────────────────
        if price > ma20_val > ma50_val:
            trend_label = "📈 Strong Uptrend"
            trend_color = "green"
        elif price < ma20_val < ma50_val:
            trend_label = "📉 Strong Downtrend"
            trend_color = "red"
        else:
            trend_label = "↔️ Mixed Trend"
            trend_color = "orange"

        # ── RSI classification ─────────────────────────
        if rsi_val > 70:
            rsi_label = "Overbought 🔴 (Possible Sell Signal)"
        elif rsi_val < 30:
            rsi_label = "Oversold 🟢 (Possible Buy Signal)"
        else:
            rsi_label = "Neutral ⚪"

        # ── Volatility classification ──────────────────
        if vol_val > 40:
            vol_label = "High ⚠️"
        elif vol_val >= 25:
            vol_label = "Medium 🟡"
        else:
            vol_label = "Low 🟢"

        # ── Trading recommendation ─────────────────────
        buy_signals  = 0
        sell_signals = 0

        if price > ma20_val > ma50_val:
            buy_signals += 1
        elif price < ma20_val < ma50_val:
            sell_signals += 1

        if rsi_val < 30:
            buy_signals += 1
        elif rsi_val > 70:
            sell_signals += 1

        if len(golden_cross) > 0 and len(death_cross) > 0:
            if golden_cross.index[-1] > death_cross.index[-1]:
                buy_signals += 1
            else:
                sell_signals += 1
        elif len(golden_cross) > 0:
            buy_signals += 1
        elif len(death_cross) > 0:
            sell_signals += 1

        if buy_signals >= 2:
            rec = "BUY"
            rec_color = "green"
            rec_reason = (
                f"Multiple bullish signals: trend={trend_label}, "
                f"RSI={rsi_val:.1f}, crossover history supports upside."
            )
        elif sell_signals >= 2:
            rec = "SELL"
            rec_color = "red"
            rec_reason = (
                f"Multiple bearish signals: trend={trend_label}, "
                f"RSI={rsi_val:.1f}, crossover history suggests downside."
            )
        else:
            rec = "HOLD"
            rec_color = "orange"
            rec_reason = (
                f"Mixed signals: trend={trend_label}, RSI={rsi_val:.1f}. "
                "No strong directional conviction."
            )

        # ══════════════════════════════════════════════
        #  STEP 1  — Key Metrics Row
        # ══════════════════════════════════════════════
        st.subheader(f"Step 1: {ticker} — Key Metrics (last 6 months)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price",  f"${price:.2f}")
        c2.metric("20-Day MA",      f"${ma20_val:.2f}")
        c3.metric("50-Day MA",      f"${ma50_val:.2f}")
        c4.metric("14-Day RSI",     f"{rsi_val:.1f}")

        # ══════════════════════════════════════════════
        #  STEP 2  — Trend + Price Chart
        # ══════════════════════════════════════════════
        st.subheader("Step 2: Trend Analysis")

        if trend_color == "green":
            st.success(f"Trend: {trend_label}")
        elif trend_color == "red":
            st.error(f"Trend: {trend_label}")
        else:
            st.warning(f"Trend: {trend_label}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"],     name="Price",    line=dict(width=2)))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"],      name="20-Day MA", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"],      name="50-Day MA", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=df["Date"], y=df[f"MA_Short"], name=f"{ma_short}-Day MA (custom)", line=dict(dash="dot", color="cyan")))
        fig.add_trace(go.Scatter(x=df["Date"], y=df[f"MA_Long"],  name=f"{ma_long}-Day MA (custom)",  line=dict(dash="dot", color="yellow")))

        # Golden / Death cross markers
        fig.add_trace(go.Scatter(
            x=golden_cross["Date"], y=golden_cross["Close"],
            mode="markers", marker=dict(size=12, color="green", symbol="triangle-up"),
            name="Golden Cross (BUY)"
        ))
        fig.add_trace(go.Scatter(
            x=death_cross["Date"], y=death_cross["Close"],
            mode="markers", marker=dict(size=12, color="red", symbol="triangle-down"),
            name="Death Cross (SELL)"
        ))

        fig.update_layout(title=f"{ticker} Price & Moving Averages", template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)

        # ══════════════════════════════════════════════
        #  STEP 3  — RSI
        # ══════════════════════════════════════════════
        st.subheader("Step 3: Momentum — RSI (14)")
        st.info(f"RSI: **{rsi_val:.1f}** → {rsi_label}")

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI", line=dict(color="violet")))
        fig_rsi.add_hline(y=70, line_color="red",   line_dash="dash", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_color="green", line_dash="dash", annotation_text="Oversold (30)")
        fig_rsi.update_layout(template="plotly_dark", height=300, yaxis=dict(range=[0,100]))
        st.plotly_chart(fig_rsi, use_container_width=True)

        # ══════════════════════════════════════════════
        #  STEP 4  — Volatility
        # ══════════════════════════════════════════════
        st.subheader("Step 4: Volatility (20-Day Annualized)")
        st.info(f"Volatility: **{vol_val:.1f}%** → {vol_label}")

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=df["Date"], y=df["Volatility"], name="Volatility", fill="tozeroy", line=dict(color="orange")))
        fig_vol.add_hline(y=40, line_color="red",    line_dash="dash", annotation_text="High (40%)")
        fig_vol.add_hline(y=25, line_color="yellow", line_dash="dash", annotation_text="Medium (25%)")
        fig_vol.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_vol, use_container_width=True)

        # ══════════════════════════════════════════════
        #  STEP 5  — Trading Recommendation
        # ══════════════════════════════════════════════
        st.subheader("Step 5: Trading Recommendation")
        if rec == "BUY":
            st.success(f"🟢 Recommendation: **{rec}**\n\n{rec_reason}")
        elif rec == "SELL":
            st.error(f"🔴 Recommendation: **{rec}**\n\n{rec_reason}")
        else:
            st.warning(f"🟡 Recommendation: **{rec}**\n\n{rec_reason}")

        # ══════════════════════════════════════════════
        #  ML Prediction (existing feature — kept)
        # ══════════════════════════════════════════════
        st.subheader("📐 ML Price Forecast (Linear Regression)")

        df_ml = df.dropna().copy()
        df_ml["Days"] = (df_ml["Date"] - df_ml["Date"].min()).dt.days

        X = df_ml[["Days"]]
        y = df_ml["Close"]

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        future_days_range = np.arange(
            X["Days"].max() + 1,
            X["Days"].max() + forecast_days + 1
        ).reshape(-1, 1)
        future_pred  = model.predict(future_days_range)
        future_dates = [df_ml["Date"].max() + timedelta(days=i+1) for i in range(forecast_days)]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_ml["Date"],              y=df_ml["Close"], name="Actual"))
        fig2.add_trace(go.Scatter(x=df_ml["Date"].iloc[split:], y=y_pred,         name="Predicted"))
        fig2.add_trace(go.Scatter(x=future_dates,               y=future_pred,    name="Forecast"))
        fig2.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig2, use_container_width=True)
        st.metric("Model RMSE", f"{rmse:.2f}")


# ════════════════════════════════════════════════════════
#  PART 2 — Portfolio Dashboard
# ════════════════════════════════════════════════════════

with tab2:
    st.subheader("💼 Portfolio Performance Dashboard")

    # ── Portfolio setup controls ─────────────────────
    st.markdown("**Step 1: Configure Your Portfolio**")

    default_tickers  = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    default_weights  = [0.25, 0.25, 0.20, 0.15, 0.15]
    benchmark_ticker = "SPY"

    col_a, col_b = st.columns([2, 1])
    with col_a:
        raw_tickers = st.text_input(
            "Portfolio Tickers (comma-separated, exactly 5)",
            ", ".join(default_tickers)
        )
        portfolio_tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]

    with col_b:
        benchmark = st.text_input("Benchmark ETF", benchmark_ticker).upper()

    st.markdown("**Assign Weights (must sum to 1.00)**")
    weight_cols = st.columns(5)
    weights = []
    for i, col in enumerate(weight_cols):
        label = portfolio_tickers[i] if i < len(portfolio_tickers) else f"Stock {i+1}"
        w = col.number_input(label, min_value=0.0, max_value=1.0,
                             value=default_weights[i] if i < len(default_weights) else 0.20,
                             step=0.05, format="%.2f")
        weights.append(w)

    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 0.001:
        st.warning(f"⚠️ Weights sum to {weight_sum:.2f} — must equal 1.00.")

    run_p2 = st.button("▶ Run Portfolio Analysis")

    if run_p2:
        if len(portfolio_tickers) != 5:
            st.error("Please enter exactly 5 tickers.")
            st.stop()
        if abs(weight_sum - 1.0) > 0.001:
            st.error("Weights must sum to 1.00.")
            st.stop()

        one_year_ago = datetime.today() - timedelta(days=365)
        today        = datetime.today()

        # ── Fetch all data ───────────────────────────
        all_tickers = portfolio_tickers + [benchmark]
        price_data  = {}

        with st.spinner("Fetching 1 year of price data..."):
            for t in all_tickers:
                try:
                    d = safe_fetch(t, one_year_ago, today)
                    if not d.empty:
                        price_data[t] = d["Close"]
                    else:
                        st.warning(f"No data for {t}, skipping.")
                except Exception as e:
                    st.warning(f"Could not fetch {t}: {e}")

        valid_portfolio = [t for t in portfolio_tickers if t in price_data]
        if len(valid_portfolio) < 2:
            st.error("Not enough valid tickers to build portfolio.")
            st.stop()

        # ── Build prices DataFrame ───────────────────
        prices_df  = pd.DataFrame({t: price_data[t] for t in valid_portfolio + ([benchmark] if benchmark in price_data else [])})
        prices_df  = prices_df.dropna()
        returns_df = prices_df.pct_change().dropna()

        # Align weights to valid tickers
        valid_weights = np.array([weights[portfolio_tickers.index(t)] for t in valid_portfolio])
        valid_weights = valid_weights / valid_weights.sum()  # re-normalize in case tickers were dropped

        portfolio_returns  = returns_df[valid_portfolio].dot(valid_weights)
        benchmark_returns  = returns_df[benchmark] if benchmark in returns_df.columns else None

        portfolio_cumret   = (1 + portfolio_returns).cumprod() - 1
        benchmark_cumret   = (1 + benchmark_returns).cumprod() - 1 if benchmark_returns is not None else None

        # ══════════════════════════════════════════════
        #  STEP 3-4 — Cumulative Return Chart
        # ══════════════════════════════════════════════
        st.subheader("Steps 3–4: Cumulative Returns vs Benchmark")
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=portfolio_cumret.index,
            y=portfolio_cumret.values * 100,
            name="My Portfolio", line=dict(color="cyan", width=2)
        ))
        if benchmark_cumret is not None:
            fig_p.add_trace(go.Scatter(
                x=benchmark_cumret.index,
                y=benchmark_cumret.values * 100,
                name=benchmark, line=dict(color="orange", width=2)
            ))
        fig_p.update_layout(
            template="plotly_dark", height=420,
            yaxis_title="Cumulative Return (%)",
            title="Portfolio vs Benchmark — 1 Year"
        )
        st.plotly_chart(fig_p, use_container_width=True)

        # ══════════════════════════════════════════════
        #  STEP 5 — Performance Metrics
        # ══════════════════════════════════════════════
        st.subheader("Step 5: Performance Metrics")

        total_return      = portfolio_cumret.iloc[-1] * 100
        bench_return      = (benchmark_cumret.iloc[-1] * 100) if benchmark_cumret is not None else None
        outperformance    = (total_return - bench_return) if bench_return is not None else None

        ann_vol           = portfolio_returns.std() * np.sqrt(252) * 100
        risk_free         = 0.05  # 5% annual risk-free rate
        ann_return        = (1 + portfolio_returns.mean()) ** 252 - 1
        sharpe            = (ann_return - risk_free) / (portfolio_returns.std() * np.sqrt(252))

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Return",         f"{total_return:.2f}%")
        m2.metric("Benchmark Return",     f"{bench_return:.2f}%" if bench_return is not None else "N/A")
        m3.metric("Outperformance",       f"{outperformance:+.2f}%" if outperformance is not None else "N/A",
                  delta=f"{outperformance:+.2f}%" if outperformance is not None else None)
        m4.metric("Annualized Volatility",f"{ann_vol:.2f}%")
        m5.metric("Sharpe Ratio",         f"{sharpe:.2f}")

        # ── Per-stock returns bar chart ──────────────
        stock_total_returns = ((prices_df[valid_portfolio].iloc[-1] / prices_df[valid_portfolio].iloc[0]) - 1) * 100
        colors = ["green" if r >= 0 else "red" for r in stock_total_returns]

        fig_bar = go.Figure(go.Bar(
            x=stock_total_returns.index,
            y=stock_total_returns.values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in stock_total_returns.values],
            textposition="outside"
        ))
        fig_bar.update_layout(
            template="plotly_dark", height=350,
            title="Individual Stock Returns (1 Year)",
            yaxis_title="Return (%)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Portfolio weights pie ────────────────────
        fig_pie = go.Figure(go.Pie(
            labels=valid_portfolio,
            values=valid_weights,
            hole=0.4
        ))
        fig_pie.update_layout(template="plotly_dark", height=350, title="Portfolio Weights")

        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Rolling volatility comparison ────────────
        with col_right:
            roll_vol_port  = portfolio_returns.rolling(20).std() * np.sqrt(252) * 100
            fig_rv = go.Figure()
            fig_rv.add_trace(go.Scatter(x=roll_vol_port.index, y=roll_vol_port, name="Portfolio Vol", line=dict(color="cyan")))
            if benchmark_returns is not None:
                roll_vol_bench = benchmark_returns.rolling(20).std() * np.sqrt(252) * 100
                fig_rv.add_trace(go.Scatter(x=roll_vol_bench.index, y=roll_vol_bench, name=f"{benchmark} Vol", line=dict(color="orange")))
            fig_rv.update_layout(template="plotly_dark", height=350, title="Rolling 20-Day Volatility (%)")
            st.plotly_chart(fig_rv, use_container_width=True)

        # ══════════════════════════════════════════════
        #  STEP 6 — Interpretation
        # ══════════════════════════════════════════════
        st.subheader("Step 6: Interpretation")

        # Outperformance
        if outperformance is not None:
            if outperformance > 0:
                st.success(f"✅ Your portfolio **outperformed** {benchmark} by **{outperformance:.2f}%** over the past year.")
            else:
                st.error(f"❌ Your portfolio **underperformed** {benchmark} by **{abs(outperformance):.2f}%** over the past year.")
        
        # Benchmark volatility comparison
        if benchmark_returns is not None:
            bench_vol = benchmark_returns.std() * np.sqrt(252) * 100
            if ann_vol > bench_vol:
                st.warning(f"⚠️ Portfolio volatility ({ann_vol:.1f}%) is **higher** than {benchmark} ({bench_vol:.1f}%) — more risk taken.")
            else:
                st.success(f"✅ Portfolio volatility ({ann_vol:.1f}%) is **lower** than {benchmark} ({bench_vol:.1f}%) — less risk taken.")
        
        # Sharpe ratio
        if sharpe >= 1.5:
            st.success(f"📊 Sharpe Ratio of **{sharpe:.2f}** — Excellent risk-adjusted returns.")
        elif sharpe >= 1.0:
            st.success(f"📊 Sharpe Ratio of **{sharpe:.2f}** — Good risk-adjusted returns.")
        elif sharpe >= 0.5:
            st.warning(f"📊 Sharpe Ratio of **{sharpe:.2f}** — Moderate efficiency; room to improve.")
        else:
            st.error(f"📊 Sharpe Ratio of **{sharpe:.2f}** — Poor risk-adjusted returns; consider rebalancing.")

        # ── Summary table ────────────────────────────
        st.markdown("**Summary Table**")
        summary = pd.DataFrame({
            "Metric": ["Total Return", "Benchmark Return", "Outperformance",
                       "Annualized Volatility", "Sharpe Ratio"],
            "Portfolio": [
                f"{total_return:.2f}%",
                f"{bench_return:.2f}%" if bench_return is not None else "N/A",
                f"{outperformance:+.2f}%" if outperformance is not None else "N/A",
                f"{ann_vol:.2f}%",
                f"{sharpe:.2f}"
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)
