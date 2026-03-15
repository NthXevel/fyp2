"""
Database connector and helper utilities for PostgreSQL.
Updated for Supabase Cloud Deployment.
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

from models.db_models import BacktestMetric, Base, MarketData, Prediction, TradeLog

load_dotenv()

def _read_streamlit_secrets() -> dict:
    """Best-effort read of Streamlit secrets without hard dependency at runtime."""
    try:
        import streamlit as st
        return dict(st.secrets)
    except Exception:
        return {}

def _get_db_config() -> dict:
    """
    Retrieves database configuration from secrets or environment variables.
    Prioritizes the 'postgres' section in Streamlit secrets.
    """
    secrets = _read_streamlit_secrets()
    
    # 1. Check for a nested [postgres] section (Recommended for Streamlit)
    if "postgres" in secrets and isinstance(secrets["postgres"], dict):
        return secrets["postgres"]
    
    # 2. Check for top-level env vars/secrets
    return {
        "url": os.getenv("DATABASE_URL") or secrets.get("DATABASE_URL"),
        "user": os.getenv("POSTGRES_USER") or secrets.get("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD") or secrets.get("POSTGRES_PASSWORD"),
        "host": os.getenv("POSTGRES_HOST") or secrets.get("POSTGRES_HOST"),
        "port": os.getenv("POSTGRES_PORT") or secrets.get("POSTGRES_PORT", "5432"),
        "database": os.getenv("POSTGRES_DB") or secrets.get("POSTGRES_DB", "postgres")
    }

def get_database_url() -> str:
    """Build DB URL ensuring the correct driver and credentials."""
    config = _get_db_config()
    
    # Use direct URL if provided, but ensure it uses the psycopg2 driver
    if config.get("url"):
        url = config["url"]
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg2://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return url

    # Fallback to discrete variables
    user = config.get("user")
    password = config.get("password")
    host = config.get("host")
    port = config.get("port")
    db_name = config.get("database")
    
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"

@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create and cache SQLAlchemy engine with SSL enforced for Cloud DBs."""
    secrets = _read_streamlit_secrets()
    
    # Supabase/Cloud DBs REQUIRE sslmode=require
    connect_args = {"sslmode": "require"}
    
    return create_engine(
        get_database_url(),
        connect_args=connect_args,
        pool_size=int(os.getenv("DB_POOL_SIZE") or secrets.get("DB_POOL_SIZE", 8)),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW") or secrets.get("DB_MAX_OVERFLOW", 16)),
        pool_recycle=int(os.getenv("DB_POOL_RECYCLE") or secrets.get("DB_POOL_RECYCLE", 1800)),
        pool_timeout=int(os.getenv("DB_POOL_TIMEOUT") or secrets.get("DB_POOL_TIMEOUT", 30)),
        pool_pre_ping=True,
        future=True,
    )

# --- Remaining helper functions kept identical to original for compatibility ---

@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, future=True)

@contextmanager
def session_scope() -> Iterable[Session]:
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
    Base.metadata.create_all(bind=get_engine())

def healthcheck() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False

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

def _normalize_market_df(df: pd.DataFrame, symbol: str, timeframe: str, source: str) -> list[dict]:
    if df is None or df.empty: return []
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

def insert_trade_log(**kwargs) -> None:
    with session_scope() as session:
        session.add(TradeLog(**kwargs))

def insert_prediction(**kwargs) -> None:
    with session_scope() as session:
        session.add(Prediction(**kwargs))

def upsert_backtest_metric(**kwargs) -> None:
    stmt = insert(BacktestMetric).values(**kwargs)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_backtest_run_symbol_tf",
        set_={k: getattr(stmt.excluded, k) for k in kwargs.keys() if k not in ["run_id", "symbol", "timeframe"]},
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