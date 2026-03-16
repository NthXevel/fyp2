"""
Database connector and helper utilities for PostgreSQL.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

# Import models from your existing project structure
from models.db_models import BacktestMetric, Base, MarketData, Prediction, TradeLog

load_dotenv()

def _read_streamlit_secrets() -> dict:
    """Best-effort read of Streamlit secrets without hard dependency at runtime."""
    try:
        import streamlit as st
        return dict(st.secrets)
    except Exception:
        return {}

def get_database_url() -> str:
    """Build DB URL from Streamlit secrets or environment variables."""
    secrets = _read_streamlit_secrets()
    
    # 1. Check for a direct 'url' in a [postgres] section (Recommended)
    pg_secrets = secrets.get("postgres", {})
    if isinstance(pg_secrets, dict) and pg_secrets.get("url"):
        url = pg_secrets["url"]
    else:
        # 2. Check for top-level DATABASE_URL or POSTGRES_URL
        url = (os.getenv("DATABASE_URL") or 
               secrets.get("DATABASE_URL") or 
               os.getenv("POSTGRES_URL") or 
               secrets.get("POSTGRES_URL"))

    if url:
        # SQLAlchemy requires 'postgresql://' or 'postgresql+psycopg2://'
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg2://", 1)
        elif not url.startswith("postgresql+psycopg2://"):
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        # Remove channel_binding param (not supported by psycopg2, handled via connect_args)
        url = url.replace("&channel_binding=require", "").replace("?channel_binding=require&", "?").replace("?channel_binding=require", "")
        return url

    # 3. Fallback to building from discrete variables
    user = os.getenv("POSTGRES_USER") or pg_secrets.get("user") or "postgres"
    password = os.getenv("POSTGRES_PASSWORD") or pg_secrets.get("password") or "postgres"
    host = os.getenv("POSTGRES_HOST") or pg_secrets.get("host") or "localhost"
    port = os.getenv("POSTGRES_PORT") or pg_secrets.get("port") or "5432"
    db_name = os.getenv("POSTGRES_DB") or pg_secrets.get("database") or "quantlearn"
    
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"

@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create and cache SQLAlchemy engine."""
    secrets = _read_streamlit_secrets()

    return create_engine(
        get_database_url(),
        pool_size=int(os.getenv("DB_POOL_SIZE") or secrets.get("DB_POOL_SIZE", 5)),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW") or secrets.get("DB_MAX_OVERFLOW", 10)),
        pool_recycle=int(os.getenv("DB_POOL_RECYCLE") or secrets.get("DB_POOL_RECYCLE", 300)),
        pool_timeout=int(os.getenv("DB_POOL_TIMEOUT") or secrets.get("DB_POOL_TIMEOUT", 30)),
        pool_pre_ping=True,
        future=True,
        connect_args={"sslmode": "require"},
    )

@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, future=True)

@contextmanager
def session_scope() -> Iterable[Session]:
    """Transactional session context manager."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def init_database() -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(bind=get_engine())

def healthcheck() -> bool:
    """Simple DB connectivity check."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False

# --- Helper functions for Data Operations ---

def _normalize_market_df(df: pd.DataFrame, symbol: str, timeframe: str, source: str) -> list[dict]:
    if df is None or df.empty:
        return []
    temp = df.copy()
    if "date" in temp.columns:
        temp["timestamp"] = pd.to_datetime(temp["date"], utc=True)
    else:
        temp = temp.reset_index().rename(columns={"date": "timestamp"})
        temp["timestamp"] = pd.to_datetime(temp["timestamp"], utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        temp[col] = pd.to_numeric(temp[col], errors="coerce")
    temp = temp.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    temp["symbol"], temp["timeframe"], temp["source"] = symbol, timeframe, source
    return temp[["symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume", "source"]].to_dict("records")

def upsert_market_data(df: pd.DataFrame, symbol: str, timeframe: str = "15m", source: str = "yahoo") -> int:
    records = _normalize_market_df(df, symbol=symbol, timeframe=timeframe, source=source)
    if not records: return 0
    stmt = insert(MarketData).values(records)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_market_data_symbol_tf_ts",
        set_={col: getattr(stmt.excluded, col) for col in ["open", "high", "low", "close", "volume", "source"]},
    )
    with session_scope() as session:
        session.execute(stmt)
    return len(records)

def insert_trade_log(**kwargs) -> None:
    with session_scope() as session:
        session.add(TradeLog(**kwargs))

def insert_prediction(**kwargs) -> None:
    with session_scope() as session:
        session.add(Prediction(**kwargs))

def upsert_backtest_metric(**kwargs) -> None:
    # Map ORM attribute names to actual column names for excluded reference
    attr_to_col = {a.key: a.name for a in BacktestMetric.__table__.columns}
    stmt = insert(BacktestMetric.__table__).values(
        {attr_to_col.get(k, k): v for k, v in kwargs.items()}
    )
    update_cols = {attr_to_col.get(k, k) for k in kwargs if k not in ("run_id", "symbol", "timeframe")}
    stmt = stmt.on_conflict_do_update(
        constraint="uq_backtest_run_symbol_tf",
        set_={col: getattr(stmt.excluded, col) for col in update_cols},
    )
    with session_scope() as session:
        session.execute(stmt)

def load_market_data(symbol: str, timeframe: str = "15m", days: int = 59) -> pd.DataFrame:
    sql = text("SELECT timestamp AS date, open, high, low, close, volume FROM market_data WHERE symbol = :symbol AND timeframe = :timeframe AND timestamp >= (NOW() AT TIME ZONE 'UTC') - (:days || ' day')::interval ORDER BY timestamp ASC")
    with get_engine().connect() as conn:
        df = pd.read_sql(sql, conn, params={"symbol": symbol, "timeframe": timeframe, "days": days})
    if df.empty: return df
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date")