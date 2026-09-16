"""Simulation result models (Phase 1 + Phase 2/2.5 extensions)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional


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
    gross_return: Optional[float]
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
    market_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    market_return: Optional[float] = None
    source_return_ref: Optional[float] = None
    discrepancy_flag: bool = False
    discrepancy_detail: Optional[str] = None
    valuation_notes: Optional[str] = None
    entry_signal: Optional[str] = None
    exit_signal: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "Trade ID": self.trade_id,
            "Symbol": self.symbol,
            "Strategy": self.strategy,
            "Exchange": self.exchange,
            "Source file": self.source_file,
            "Source trade #": self.source_trade_number,
            "Buy date": self.buy_date.isoformat() if self.buy_date else None,
            "Sell date": self.sell_date.isoformat() if self.sell_date else None,
            "Signal": _format_signal(self.entry_signal, self.exit_signal),
            "Entry signal": self.entry_signal,
            "Exit signal": self.exit_signal,
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
        if self.market_price is not None or self.is_open:
            d.update(
                {
                    "Market price": self.market_price,
                    "Market value": self.market_value,
                    "Unrealized PnL": self.unrealized_pnl,
                    "Market return": self.market_return,
                    "Source return ref": self.source_return_ref,
                    "Discrepancy flag": self.discrepancy_flag,
                    "Discrepancy detail": self.discrepancy_detail,
                    "Valuation notes": self.valuation_notes,
                }
            )
        return d


def _format_signal(entry: Optional[str], exit_: Optional[str]) -> Optional[str]:
    parts = [p for p in (entry, exit_) if p]
    return " / ".join(parts) if parts else None


@dataclass
class EquitySnapshot:
    timestamp: date
    equity: float
    cash: float
    invested_cost_basis: float
    open_positions: int
    exposure_pct: float
    cumulative_return: float
    market_value_opens: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    missing_price_symbols: tuple[str, ...] = ()
    is_daily_mtm: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = {
            "Date": self.timestamp.isoformat(),
            "Equity": self.equity,
            "Cash": self.cash,
            "Invested cost basis": self.invested_cost_basis,
            "# Open positions": self.open_positions,
            "Exposure %": self.exposure_pct,
            "Cumulative return": self.cumulative_return,
        }
        if self.is_daily_mtm or self.market_value_opens is not None:
            d["Market value opens"] = self.market_value_opens
            d["Unrealized PnL"] = self.unrealized_pnl
            d["Missing price symbols"] = ",".join(self.missing_price_symbols)
            d["Daily MTM"] = self.is_daily_mtm
        return d


@dataclass
class PerformanceSummary:
    initial_capital: float
    final_equity: float
    absolute_pnl: float
    total_return_pct: float
    realized_pnl: float
    unrealized_pnl: float
    equity_including_opens: float
    equity_realized_only: float
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
        "Average trade return != portfolio return due to sizing, overlap, "
        "cash constraints, and rejects."
    )
    ann_volatility: Optional[float] = None
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_drawdown_duration_days: Optional[int] = None
    risk_unavailable: dict[str, str] = field(default_factory=dict)
    cagr: Optional[float] = None
    cash_interest_pnl: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = {
            "Initial capital": self.initial_capital,
            "Final equity": self.final_equity,
            "Absolute P&L": self.absolute_pnl,
            "Total return %": self.total_return_pct,
            "CAGR": self.cagr,
            "Cash interest P&L": self.cash_interest_pnl,
            "Realized P&L": self.realized_pnl,
            "Unrealized P&L": self.unrealized_pnl,
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
        if self.ann_volatility is not None or self.risk_unavailable:
            d.update(
                {
                    "Ann. volatility": self.ann_volatility,
                    "Sharpe": self.sharpe,
                    "Sortino": self.sortino,
                    "Max drawdown": self.max_drawdown,
                    "Max DD duration (days)": self.max_drawdown_duration_days,
                    "Risk unavailable": self.risk_unavailable or None,
                }
            )
        return d


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
    daily_mtm_timeseries: list[EquitySnapshot] = field(default_factory=list)
    risk_metrics: Optional[Any] = None
    benchmark_metrics: Optional[Any] = None
    market_data_report: list[dict[str, Any]] = field(default_factory=list)
    open_valuations: list[dict[str, Any]] = field(default_factory=list)
    affected_dates: list[str] = field(default_factory=list)
    trading_stats: Optional[Any] = None
    portfolio_stats: Optional[Any] = None
    cash_earn_notes: Optional[str] = None
