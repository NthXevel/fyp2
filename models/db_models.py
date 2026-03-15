"""
SQLAlchemy ORM models for QuantLearn PostgreSQL storage.
"""
from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class MarketData(Base):
    """15-minute OHLCV market bars."""

    __tablename__ = "market_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(24), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(12), nullable=False, default="15m")
    timestamp: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    high: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    low: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    close: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    volume: Mapped[float] = mapped_column(Numeric(24, 8), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="yahoo")
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_market_data_symbol_tf_ts"),
        Index("ix_market_data_symbol_tf_ts", "symbol", "timeframe", "timestamp"),
    )


class TradeLog(Base):
    """Live execution and backtest trade events."""

    __tablename__ = "trade_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_time: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False)
    symbol: Mapped[str] = mapped_column(String(24), nullable=False)
    action: Mapped[str] = mapped_column(String(24), nullable=False)
    qty: Mapped[float] = mapped_column(Numeric(24, 8), nullable=False)
    price: Mapped[float] = mapped_column(Numeric(24, 8), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    investment: Mapped[float | None] = mapped_column(Numeric(24, 8), nullable=True)
    capital: Mapped[float | None] = mapped_column(Numeric(24, 8), nullable=True)
    order_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    venue: Mapped[str] = mapped_column(String(24), nullable=False, default="alpaca")
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="live")
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_trade_logs_symbol_time", "symbol", "event_time"),
        Index("ix_trade_logs_mode_time", "mode", "event_time"),
    )


class Prediction(Base):
    """Model prediction snapshots used for monitoring and audit."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    prediction_time: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False)
    symbol: Mapped[str] = mapped_column(String(24), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(12), nullable=False, default="15m")
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, default="xgboost")
    signal: Mapped[str] = mapped_column(String(16), nullable=False)
    prob_up: Mapped[float] = mapped_column(Float, nullable=False)
    prob_down: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    features: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_predictions_symbol_time", "symbol", "prediction_time"),
    )


class BacktestMetric(Base):
    """Backtest summary metrics and run metadata."""

    __tablename__ = "backtest_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(24), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(12), nullable=False, default="15m")
    initial_capital: Mapped[float] = mapped_column(Numeric(24, 8), nullable=False)
    final_capital: Mapped[float] = mapped_column(Numeric(24, 8), nullable=False)
    cumulative_return: Mapped[float] = mapped_column(Float, nullable=False)
    sharpe_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=False)
    win_rate: Mapped[float] = mapped_column(Float, nullable=False)
    num_trades: Mapped[int] = mapped_column(Integer, nullable=False)
    test_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    meta_payload: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("run_id", "symbol", "timeframe", name="uq_backtest_run_symbol_tf"),
        Index("ix_backtest_metrics_symbol_time", "symbol", "created_at"),
    )
