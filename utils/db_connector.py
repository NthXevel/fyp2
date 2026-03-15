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

from models.db_models import BacktestMetric, Base, MarketData, Prediction, TradeLog

load_dotenv()


def _read_streamlit_secrets() -> dict:
    """Best-effort read of Streamlit secrets without hard dependency at runtime."""
    try:
        import streamlit as st

        secrets = dict(st.secrets)
        return secrets
    except Exception:
        return {}


def _secret_or_env(key: str, default: str | None = None) -> str | None:
    val = os.getenv(key)
    if val:
        return val

    secrets = _read_streamlit_secrets()
    if key in secrets:
        return str(secrets[key])

    postgres = secrets.get("postgres") if isinstance(secrets, dict) else None
    if isinstance(postgres, dict) and key in postgres:
        return str(postgres[key])

    return default


def get_database_url() -> str:
    """Build DB URL from DATABASE_URL/POSTGRES_URL or discrete PG* variables."""
    direct_url = _secret_or_env("DATABASE_URL") or _secret_or_env("POSTGRES_URL")
    if direct_url:
        return direct_url

    user = _secret_or_env("POSTGRES_USER", "postgres")
    password = _secret_or_env("POSTGRES_PASSWORD", "postgres")
    host = _secret_or_env("POSTGRES_HOST", "localhost")
    port = _secret_or_env("POSTGRES_PORT", "5432")
    db_name = _secret_or_env("POSTGRES_DB", "quantlearn")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create and cache SQLAlchemy engine with pool settings suitable for app + jobs."""
    return create_engine(
        get_database_url(),
        pool_size=int(_secret_or_env("DB_POOL_SIZE", "8")),
        max_overflow=int(_secret_or_env("DB_MAX_OVERFLOW", "16")),
        pool_recycle=int(_secret_or_env("DB_POOL_RECYCLE", "1800")),
        pool_timeout=int(_secret_or_env("DB_POOL_TIMEOUT", "30")),
        pool_pre_ping=True,
        future=True,
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
    temp["symbol"] = symbol
    temp["timeframe"] = timeframe
    temp["source"] = source

    cols = ["symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume", "source"]
    return temp[cols].to_dict("records")


def upsert_market_data(df: pd.DataFrame, symbol: str, timeframe: str = "15m", source: str = "yahoo") -> int:
    """Upsert market OHLCV rows keyed by (symbol, timeframe, timestamp)."""
    records = _normalize_market_df(df, symbol=symbol, timeframe=timeframe, source=source)
    if not records:
        return 0

    stmt = insert(MarketData).values(records)
    update_cols = {
        "open": stmt.excluded.open,
        "high": stmt.excluded.high,
        "low": stmt.excluded.low,
        "close": stmt.excluded.close,
        "volume": stmt.excluded.volume,
        "source": stmt.excluded.source,
    }
    stmt = stmt.on_conflict_do_update(
        constraint="uq_market_data_symbol_tf_ts",
        set_=update_cols,
    )

    with session_scope() as session:
        session.execute(stmt)
    return len(records)


def insert_trade_log(
    *,
    event_time: datetime,
    symbol: str,
    action: str,
    qty: float,
    price: float,
    confidence: float | None = None,
    investment: float | None = None,
    capital: float | None = None,
    order_id: str | None = None,
    status: str | None = None,
    venue: str = "alpaca",
    mode: str = "live",
    run_id: str | None = None,
    notes: str | None = None,
) -> None:
    with session_scope() as session:
        session.add(
            TradeLog(
                event_time=event_time,
                symbol=symbol,
                action=action,
                qty=qty,
                price=price,
                confidence=confidence,
                investment=investment,
                capital=capital,
                order_id=order_id,
                status=status,
                venue=venue,
                mode=mode,
                run_id=run_id,
                notes=notes,
            )
        )


def insert_prediction(
    *,
    prediction_time: datetime,
    symbol: str,
    signal: str,
    prob_up: float,
    prob_down: float,
    confidence: float,
    timeframe: str = "15m",
    model_name: str = "xgboost",
    features: dict | None = None,
) -> None:
    with session_scope() as session:
        session.add(
            Prediction(
                prediction_time=prediction_time,
                symbol=symbol,
                timeframe=timeframe,
                model_name=model_name,
                signal=signal,
                prob_up=prob_up,
                prob_down=prob_down,
                confidence=confidence,
                features=features,
            )
        )


def upsert_backtest_metric(
    *,
    run_id: str,
    symbol: str,
    timeframe: str,
    initial_capital: float,
    final_capital: float,
    cumulative_return: float,
    sharpe_ratio: float,
    max_drawdown: float,
    win_rate: float,
    num_trades: int,
    test_accuracy: float,
    metadata: dict | None = None,
) -> None:
    stmt = insert(BacktestMetric).values(
        run_id=run_id,
        symbol=symbol,
        timeframe=timeframe,
        initial_capital=initial_capital,
        final_capital=final_capital,
        cumulative_return=cumulative_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        num_trades=num_trades,
        test_accuracy=test_accuracy,
        meta_payload=metadata,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_backtest_run_symbol_tf",
        set_={
            "initial_capital": stmt.excluded.initial_capital,
            "final_capital": stmt.excluded.final_capital,
            "cumulative_return": stmt.excluded.cumulative_return,
            "sharpe_ratio": stmt.excluded.sharpe_ratio,
            "max_drawdown": stmt.excluded.max_drawdown,
            "win_rate": stmt.excluded.win_rate,
            "num_trades": stmt.excluded.num_trades,
            "test_accuracy": stmt.excluded.test_accuracy,
            "metadata": stmt.excluded.metadata,
        },
    )
    with session_scope() as session:
        session.execute(stmt)


def load_market_data(symbol: str, timeframe: str = "15m", days: int = 59) -> pd.DataFrame:
    """Load historical market data from PostgreSQL into DataFrame."""
    sql = text(
        """
        SELECT timestamp AS date, open, high, low, close, volume
        FROM market_data
        WHERE symbol = :symbol
          AND timeframe = :timeframe
          AND timestamp >= (NOW() AT TIME ZONE 'UTC') - (:days || ' day')::interval
        ORDER BY timestamp ASC
        """
    )
    with get_engine().connect() as conn:
        df = pd.read_sql(sql, conn, params={"symbol": symbol, "timeframe": timeframe, "days": days})
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date")
