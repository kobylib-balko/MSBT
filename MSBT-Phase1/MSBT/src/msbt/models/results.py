"""Simulation result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional


# Status / rejection reason constants
ACCEPTED = "Accepted"
PARTIAL_INSUFFICIENT = "Partially Filled - Insufficient Capital"
REJECTED_INSUFFICIENT = "Rejected - Insufficient Capital"
REJECTED_MAX_SYMBOL = "Rejected - Maximum Symbol Exposure"
REJECTED_MAX_OPEN = "Rejected - Max Open Positions"
REJECTED_SAME_SYMBOL = "Rejected - Same Symbol"
REJECTED_BELOW_MIN = "Rejected - Below Minimum Allocation"
REJECTED_DATE_FILTER = "Rejected - Filtered by Date Range"


@dataclass
class TradeResult:
    trade_id: str
    symbol: str
    strategy: str
    exchange: str
    source_file: str
    source_trade_number: int
    buy_date: date
    sell_date: Optional[date]
    gross_return: Optional[float]  # decimal
    net_return: Optional[float]
    allocated_capital: float
    qty: float
    cost_basis: float
    gross_pnl: Optional[float]
    net_pnl: Optional[float]
    entry_fee: float
    exit_fee: float
    entry_slippage: float
    exit_slippage: float
    status: str
    rejection_reason: Optional[str] = None
    duration_bars: Optional[int] = None
    is_open: bool = False
    buy_price: Optional[float] = None
    sell_price: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "Trade ID": self.trade_id,
            "Symbol": self.symbol,
            "Strategy": self.strategy,
            "Exchange": self.exchange,
            "Source file": self.source_file,
            "Source trade #": self.source_trade_number,
            "Buy date": self.buy_date.isoformat() if self.buy_date else None,
            "Sell date": self.sell_date.isoformat() if self.sell_date else None,
            "Gross return": self.gross_return,
            "Net return": self.net_return,
            "Allocated capital": self.allocated_capital,
            "Qty": self.qty,
            "Cost basis": self.cost_basis,
            "Gross PnL": self.gross_pnl,
            "Net PnL": self.net_pnl,
            "Entry fee": self.entry_fee,
            "Exit fee": self.exit_fee,
            "Entry slippage": self.entry_slippage,
            "Exit slippage": self.exit_slippage,
            "Status": self.status,
            "Rejection reason": self.rejection_reason,
            "Duration bars": self.duration_bars,
            "Is open": self.is_open,
            "Buy price": self.buy_price,
            "Sell price": self.sell_price,
        }


@dataclass
class EquitySnapshot:
    timestamp: date
    equity: float
    cash: float
    invested_cost_basis: float
    open_positions: int
    exposure_pct: float
    cumulative_return: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "Date": self.timestamp.isoformat(),
            "Equity": self.equity,
            "Cash": self.cash,
            "Invested cost basis": self.invested_cost_basis,
            "# Open positions": self.open_positions,
            "Exposure %": self.exposure_pct,
            "Cumulative return": self.cumulative_return,
        }


@dataclass
class PerformanceSummary:
    initial_capital: float
    final_equity: float
    absolute_pnl: float
    total_return_pct: float
    realized_pnl: float
    unrealized_pnl: float  # Phase 1: 0 under cost-basis valuation
    equity_including_opens: float
    equity_realized_only: float  # cash + 0 for opens marked at cost → same as equity when cost basis
    candidates: int
    accepted: int
    partial: int
    rejected: int
    rejected_by_reason: dict[str, int] = field(default_factory=dict)
    win_rate: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_holding_bars: Optional[float] = None
    median_holding_bars: Optional[float] = None
    note: str = (
        "Average trade return ≠ portfolio return due to sizing, overlap, "
        "cash constraints, and rejects."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "Initial capital": self.initial_capital,
            "Final equity": self.final_equity,
            "Absolute P&L": self.absolute_pnl,
            "Total return %": self.total_return_pct,
            "Realized P&L": self.realized_pnl,
            "Unrealized P&L (cost-basis view)": self.unrealized_pnl,
            "Equity including opens": self.equity_including_opens,
            "Equity realized-only view": self.equity_realized_only,
            "Candidates": self.candidates,
            "Accepted": self.accepted,
            "Partial": self.partial,
            "Rejected": self.rejected,
            "Rejected by reason": self.rejected_by_reason,
            "Win rate (net)": self.win_rate,
            "Avg win (net)": self.avg_win,
            "Avg loss (net)": self.avg_loss,
            "Profit factor (net)": self.profit_factor,
            "Avg holding bars": self.avg_holding_bars,
            "Median holding bars": self.median_holding_bars,
            "Note": self.note,
        }


@dataclass
class SimulationResult:
    portfolio_timeseries: list[EquitySnapshot]
    trade_results: list[TradeResult]
    performance_metrics: PerformanceSummary
    source_metrics: list[dict[str, Any]]
    symbol_metrics: list[dict[str, Any]]
    rejected_trades: list[TradeResult]
    simulation_config: dict[str, Any]
    validation_report: list[dict[str, Any]] = field(default_factory=list)
    simulation_id: str = ""
