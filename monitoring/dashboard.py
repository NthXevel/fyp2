"""
monitoring/dashboard.py — Streamlit real-time dashboard.
Usage: streamlit run monitoring/dashboard.py
"""
import sys
sys.path.insert(0, ".")

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from sqlalchemy import text

from streamlit_autorefresh import st_autorefresh

from config.settings import (
    STOCK_SYMBOL, CONFIDENCE_THRESHOLD,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    TARGET_ACCURACY, TARGET_SHARPE, TARGET_MAX_DRAWDOWN,
    MODEL_PATH,
)
from execution.alpaca_executor import TradingExecutor
from monitoring.bot_manager import is_running, start, stop, get_log
from utils.db_connector import get_engine, init_database

st.set_page_config(page_title="QuantLearn", layout="wide", page_icon="📈")


@st.cache_resource
def get_executor():
    return TradingExecutor()


def _load_model_signal():
    """
    Load the trained model, fetch recent data, compute features,
    and return the latest prediction signal + feature values.
    Returns None on any failure.
    """
    try:
        from models.trainer import ModelTrainer
        from utils.data_fetcher import DataFetcher
        from strategies.feature_engineering import FeatureEngineer
        from strategies.signal_generator import SignalGenerator

        trainer = ModelTrainer()
        if not trainer.load():
            return None

        fetcher = DataFetcher()
        engineer = FeatureEngineer()
        signal_gen = SignalGenerator()

        # Fetch enough bars for indicator warm-up
        df = fetcher.get_historical_data(days=59)
        if df is None or df.empty:
            return None

        df_feat = engineer.create_features(df)
        feature_cols = trainer.feature_cols
        if feature_cols is None:
            return None

        # Latest row of features
        latest = df_feat[feature_cols].dropna().iloc[-1:]
        if latest.empty:
            return None

        # Prediction
        probabilities = trainer.predict(latest)  # [[prob_down, prob_up]]
        action, confidence = signal_gen.decide_trade(probabilities)

        prob_up = float(probabilities[0][1])
        prob_down = float(probabilities[0][0])

        # Feature values for display
        feat_vals = latest.iloc[0].to_dict()

        # OHLCV data for candlestick chart
        ohlcv = df_feat[['open', 'high', 'low', 'close', 'volume']].dropna().tail(60)

        # Latest price
        last_price = float(df_feat['close'].dropna().iloc[-1])

        return {
            'action': action,
            'confidence': confidence,
            'prob_up': prob_up,
            'prob_down': prob_down,
            'features': feat_vals,
            'feature_cols': feature_cols,
            'last_price': last_price,
            'ohlcv': ohlcv,
            'timestamp': str(df_feat.index[-1]),
        }
    except Exception as e:
        st.error(f"Error computing signal: {e}")
        return None


def _query_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run SQL and return DataFrame; returns empty on errors for resilient UI."""
    try:
        with get_engine().connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})
    except Exception:
        return pd.DataFrame()


def _load_trade_log(limit: int = 200) -> pd.DataFrame:
    return _query_df(
        """
        SELECT event_time AS timestamp, action, symbol, qty, price, confidence,
               investment, capital, order_id, status, venue, mode, run_id, notes
        FROM trade_logs
        ORDER BY event_time DESC
        LIMIT :limit
        """,
        {"limit": limit},
    )


def _load_equity_curve(limit: int = 400) -> pd.DataFrame:
    df = _query_df(
        """
        SELECT event_time, capital
        FROM trade_logs
        WHERE capital IS NOT NULL
        ORDER BY event_time ASC
        LIMIT :limit
        """,
        {"limit": limit},
    )
    if not df.empty:
        df["event_time"] = pd.to_datetime(df["event_time"])
    return df


def _load_predictions(limit: int = 120) -> pd.DataFrame:
    return _query_df(
        """
        SELECT prediction_time, symbol, timeframe, model_name, signal,
               prob_up, prob_down, confidence
        FROM predictions
        ORDER BY prediction_time DESC
        LIMIT :limit
        """,
        {"limit": limit},
    )


def _load_backtest_metrics(limit: int = 20) -> pd.DataFrame:
    return _query_df(
        """
        SELECT created_at, run_id, symbol, timeframe, final_capital,
               cumulative_return, sharpe_ratio, max_drawdown, win_rate,
               num_trades, test_accuracy
        FROM backtest_metrics
        ORDER BY created_at DESC
        LIMIT :limit
        """,
        {"limit": limit},
    )


def render_market_chart(df, symbol):
    """Render a Plotly candlestick chart with EMA overlays and volume."""
    from plotly.subplots import make_subplots

    ema_10 = df['close'].ewm(span=10).mean()
    ema_20 = df['close'].ewm(span=20).mean()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Price',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=ema_10, mode='lines', name='EMA 10',
        line=dict(color='#ff9800', width=1.2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=ema_20, mode='lines', name='EMA 20',
        line=dict(color='#2196f3', width=1.2),
    ), row=1, col=1)

    colors = ['#26a69a' if c >= o else '#ef5350'
              for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df.index, y=df['volume'], name='Volume',
        marker_color=colors, opacity=0.5,
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"{symbol} — Recent Price (last {len(df)} bars)", y=0.98),
        height=450, xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.06, x=0.25),
    )
    fig.update_yaxes(title_text='Price ($)', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    return fig


def main():
    init_database()
    st.title("📈 QuantLearn — Live Dashboard")
    st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    executor = get_executor()

    # ── Account Summary ────────────────────────────────────────────────
    account = executor.get_account_info()
    if account:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Portfolio Value", f"${account['portfolio_value']:,.2f}")
        col2.metric("Cash",           f"${account['cash']:,.2f}")
        col3.metric("Buying Power",   f"${account['buying_power']:,.2f}")
        col4.metric("Equity",         f"${account['equity']:,.2f}")
    else:
        st.error("Unable to fetch account information.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════
    # ── Live Model Signal  ────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════
    st.subheader(f"🤖 Live Signal — {STOCK_SYMBOL}")

    signal_data = _load_model_signal()

    if signal_data:
        action = signal_data['action']
        confidence = signal_data['confidence']
        prob_up = signal_data['prob_up']
        prob_down = signal_data['prob_down']
        last_price = signal_data['last_price']

        # Action badge colour
        action_colours = {'buy': '🟢', 'sell': '🔴', 'hold': '🟡'}
        badge = action_colours.get(action, '⚪')

        # Top-level signal metrics
        sig1, sig2, sig3, sig4 = st.columns(4)
        sig1.metric("Current Signal", f"{badge} {action.upper()}")
        sig2.metric("Confidence", f"{confidence:.2%}")
        sig3.metric("Last Price", f"${last_price:,.2f}")
        sig4.metric("Data as of", signal_data['timestamp'][:19])

        # Probability gauge
        col_prob, col_chart = st.columns([1, 2])

        with col_prob:
            st.markdown("**Probability Breakdown**")
            st.progress(prob_up, text=f"Prob Up: {prob_up:.2%}")
            st.progress(prob_down, text=f"Prob Down: {prob_down:.2%}")

            st.markdown(f"**Confidence Threshold:** {CONFIDENCE_THRESHOLD:.0%}")
            if prob_up >= CONFIDENCE_THRESHOLD:
                st.success(f"Above threshold -> BUY signal")
            elif prob_down >= CONFIDENCE_THRESHOLD:
                st.error(f"Below threshold -> SELL signal")
            else:
                st.warning(f"Within threshold -> HOLD")

        with col_chart:
            ohlcv = signal_data.get('ohlcv')
            if ohlcv is not None and not ohlcv.empty:
                fig_price = render_market_chart(ohlcv, STOCK_SYMBOL)
                st.plotly_chart(fig_price, use_container_width=True)

        # Feature values table
        with st.expander("📊 Current Feature Values", expanded=False):
            feat_df = pd.DataFrame(
                list(signal_data['features'].items()),
                columns=['Feature', 'Value']
            )
            feat_df['Value'] = feat_df['Value'].apply(lambda v: f"{v:.6f}")
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

        # Risk management info
        with st.expander("⚙️ Risk / Model Settings", expanded=False):
            r1, r2, r3 = st.columns(3)
            r1.metric("Stop-Loss", f"{STOP_LOSS_PCT:.1%}")
            r2.metric("Take-Profit", f"{TAKE_PROFIT_PCT:.1%}")
            r3.metric("Conf. Threshold", f"{CONFIDENCE_THRESHOLD:.0%}")

            t1, t2, t3 = st.columns(3)
            t1.metric("Target Accuracy", f"{TARGET_ACCURACY:.0%}")
            t2.metric("Target Sharpe", f"{TARGET_SHARPE}")
            t3.metric("Target Max DD", f"{TARGET_MAX_DRAWDOWN:.0%}")

            model_exists = Path(MODEL_PATH).exists()
            st.markdown(f"**Model file:** `{MODEL_PATH}` — "
                        f"{'found' if model_exists else 'NOT FOUND'}")
    else:
        st.warning(
            "Could not compute live signal. Make sure the model is trained "
            "(`python scripts/train.py`) and market data is available."
        )

    st.divider()

    # ── Positions ─────────────────────────────────────────────────────
    st.subheader("🟢 Open Positions")
    try:
        positions = executor.get_all_positions()
        if positions:
            pos_df = (
                pd.DataFrame(positions)
                .T.reset_index()
                .rename(columns={"index": "symbol"})
            )
            st.dataframe(pos_df, use_container_width=True)
        else:
            st.info("No open positions.")
    except Exception as e:
        st.error(f"Error fetching positions: {e}")

    st.divider()

    # ── Trade Log ─────────────────────────────────────────────────────
    st.subheader("📒 Trade Log")
    trade_log = _load_trade_log()
    if trade_log is not None and not trade_log.empty:
        trade_log['timestamp'] = pd.to_datetime(trade_log['timestamp'])
        # Show summary stats
        n_buys = (trade_log['action'] == 'BUY').sum()
        n_sells = trade_log['action'].str.startswith('SELL').sum()
        tl1, tl2, tl3 = st.columns(3)
        tl1.metric("Total Trades", len(trade_log))
        tl2.metric("Buys", int(n_buys))
        tl3.metric("Sells", int(n_sells))

        # Table (most recent first)
        st.dataframe(
            trade_log.sort_values('timestamp', ascending=False),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No trades logged yet.")

    st.divider()

    # ── Equity Curve (DB) ─────────────────────────────────────────────
    st.subheader("📉 Equity Curve")
    equity_df = _load_equity_curve()
    if not equity_df.empty:
        fig_eq = go.Figure()
        fig_eq.add_trace(
            go.Scatter(
                x=equity_df['event_time'],
                y=equity_df['capital'],
                mode='lines+markers',
                name='Capital',
                line=dict(color='#1565c0', width=2),
            )
        )
        fig_eq.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("No equity curve data in database yet.")

    st.divider()

    # ── Model Predictions (DB) ────────────────────────────────────────
    st.subheader("🧠 Recent Predictions")
    pred_df = _load_predictions()
    if not pred_df.empty:
        pred_df['prediction_time'] = pd.to_datetime(pred_df['prediction_time'])
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
    else:
        st.info("No prediction records found yet.")

    st.divider()

    # ── Backtest Metrics (DB) ─────────────────────────────────────────
    st.subheader("🧪 Backtest Metrics")
    metrics_df = _load_backtest_metrics()
    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.info("No backtest metrics found yet.")

    st.divider()

    # ── Open Orders (only shown when orders exist) ─────────────────────
    try:
        open_orders = executor.get_open_orders()
    except Exception as e:
        open_orders = None
        st.error(f"Error fetching open orders: {e}")

    has_open_orders = bool(open_orders)

    if has_open_orders:
        st.subheader("📋 Open Orders")
        orders_data = [
            {
                "id": str(o.id)[:8],
                "symbol": o.symbol,
                "side": o.side,
                "qty": o.qty,
                "type": o.type,
                "status": o.status,
            }
            for o in open_orders
        ]
        st.dataframe(pd.DataFrame(orders_data), use_container_width=True)

    # ── Recent Order History ───────────────────────────────────────────
    st.divider()
    st.subheader("📜 Recent Orders (all symbols)")
    try:
        history = executor.get_order_history(limit=20)
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True)
        else:
            st.info("No recent orders found.")
    except Exception as e:
        st.error(f"Error fetching order history: {e}")

    # ── Trading Bot Control ──────────────────────────────────────────
    st.divider()
    st.subheader("🤖 Trading Bot")

    bot_running = is_running()
    status_text = "🟢 Running" if bot_running else "🔴 Stopped"
    st.markdown(f"**Status:** {status_text}")

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("▶️ Start Bot", disabled=bot_running, type="primary"):
            ok, msg = start()
            if ok:
                st.success(msg)
            else:
                st.warning(msg)
            st.rerun()
    with col_stop:
        if st.button("⏹️ Stop Bot", disabled=not bot_running, type="secondary"):
            ok, msg = stop()
            if ok:
                st.success(msg)
            else:
                st.warning(msg)
            st.rerun()

    # Show bot log output
    with st.expander("📄 Bot Log (last 80 lines)", expanded=bot_running):
        log_text = get_log(tail=80)
        if log_text:
            st.code(log_text, language="text")
        else:
            st.info("No log output yet.")

    # ── Controls ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("⚙️ Controls")
    if has_open_orders:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 Cancel All Orders", type="secondary"):
                executor.cancel_all_orders()
                st.success("All orders cancelled.")
                st.rerun()
        with col_b:
            if st.button("🛑 Close All Positions", type="primary"):
                confirm = st.checkbox("Confirm close all?")
                if confirm:
                    executor.close_all_positions()
                    st.warning("All positions closed!")
    else:
        if st.button("🛑 Close All Positions", type="primary"):
            confirm = st.checkbox("Confirm close all?")
            if confirm:
                executor.close_all_positions()
                st.warning("All positions closed!")

    # Auto-refresh every 5 minutes (300 000 ms)
    st.divider()
    st_autorefresh(interval=5 * 60 * 1000, key="dashboard_autorefresh")
    st.caption("Dashboard auto-refreshes every 5 minutes.")
    if st.button("🔃 Refresh Now"):
        st.rerun()


if __name__ == "__main__":
    main()
