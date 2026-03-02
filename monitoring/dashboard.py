"""
monitoring/dashboard.py — Streamlit real-time dashboard.
Usage: streamlit run monitoring/dashboard.py
"""
import sys
sys.path.insert(0, ".")

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from config.settings import STOCK_SYMBOL
from execution.alpaca_executor import TradingExecutor

st.set_page_config(page_title="Quant Bot Dashboard", layout="wide", page_icon="📈")


@st.cache_resource
def get_executor():
    return TradingExecutor()


def main():
    st.title("📈 XGBoost Quant Bot — Live Dashboard")
    st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")

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
            pos_df["unrealized_pl"] = pos_df["unrealized_pl"].astype(float)
            st.dataframe(pos_df, use_container_width=True)

            # PnL bar chart
            fig = px.bar(
                pos_df,
                x="symbol",
                y="unrealized_pl",
                color="unrealized_pl",
                color_continuous_scale="RdYlGn",
                title="Unrealized P&L by Position",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No open positions.")
    except Exception as e:
        st.error(f"Error fetching positions: {e}")

    st.divider()

    # ── Open Orders ────────────────────────────────────────────────────
    st.subheader("📋 Open Orders")
    try:
        open_orders = executor.get_open_orders()
        if open_orders:
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
        else:
            st.info("No open orders.")
    except Exception as e:
        st.error(f"Error fetching open orders: {e}")

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

    # ── Controls ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("⚙️ Controls")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Cancel All Orders", type="secondary"):
            executor.cancel_all_orders()
            st.success("All orders cancelled.")
    with col_b:
        if st.button("🛑 Close All Positions", type="primary"):
            confirm = st.checkbox("Confirm close all?")
            if confirm:
                executor.close_all_positions()
                st.warning("All positions closed!")

    # Auto-refresh
    st.divider()
    if st.button("🔃 Refresh"):
        st.rerun()


if __name__ == "__main__":
    main()
