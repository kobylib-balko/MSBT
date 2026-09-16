"""Trading and portfolio statistics for UX results (Phase UX / 2.5).

Invested bars definition (locked preference):
  Prefer count of equity-snapshot days where invested capital > 0
  (invested_cost_basis > 0), using daily MTM when available else the
  event equity curve. Fallback: sum of duration_bars on executed closed
  trades when no usable snapshots exist.

Average Return per bar = portfolio total return / invested_bars
  (None when invested_bars == 0).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from statistics import median
from typing import Any, Optional, Sequence

from msbt.analytics.grid import cagr as equity_cagr
from msbt.models.results import (
    ACCEPTED,
    PARTIAL_INSUFFICIENT,
    EquitySnapshot,
    PerformanceSummary,
    TradeResult,
)


@dataclass
class TradingStats:
    number_of_trades: int
    executed: int
    rejected: int
    winning: int
    losing: int
    win_rate: Optional[float]
    avg_win: Optional[float]
    avg_loss: Optional[float]
    largest_win: Optional[float]
    largest_loss: Optional[float]
    profit_factor: Optional[float]
    avg_holding_bars: Optional[float]
    median_holding_bars: Optional[float]
    max_consecutive_wins: int
    max_consecutive_losses: int
    invested_bars: int
    invested_bars_definition: str
    avg_return_per_bar: Optional[float]
    portfolio_total_return: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioStats:
    avg_positions_per_day: Optional[float]
    max_exposure_pct: Optional[float]
    avg_exposure_pct: Optional[float]
    pct_time_invested: Optional[float]
    pct_time_in_cash: Optional[float]
    n_days: int
    source: str  # "daily_mtm" | "event" | "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_executed(r: TradeResult) -> bool:
    return r.status in (ACCEPTED, PARTIAL_INSUFFICIENT) and r.allocated_capital > 0


def _is_rejected(r: TradeResult) -> bool:
    return bool(r.rejection_reason)


def _closed_executed(trade_results: Sequence[TradeResult]) -> list[TradeResult]:
    return [
        r
        for r in trade_results
        if _is_executed(r) and not r.is_open and r.net_pnl is not None
    ]


def _max_consecutive(flags: Sequence[bool]) -> int:
    best = cur = 0
    for f in flags:
        if f:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def count_invested_bars(
    snapshots: Sequence[EquitySnapshot],
    closed_executed: Sequence[TradeResult],
) -> tuple[int, str]:
    """Prefer days with invested_cost_basis > 0; else sum duration_bars."""
    if snapshots:
        n = sum(1 for s in snapshots if (s.invested_cost_basis or 0) > 0)
        return n, (
            "days with invested_cost_basis > 0 from equity snapshots "
            "(daily MTM preferred when available)"
        )
    bars = [r.duration_bars for r in closed_executed if r.duration_bars is not None]
    if bars:
        return int(sum(bars)), "sum of duration_bars on executed closed trades"
    return 0, "no invested bars (no snapshots or duration_bars)"


def compute_trading_stats(
    trade_results: Sequence[TradeResult],
    *,
    portfolio_total_return: float,
    snapshots: Optional[Sequence[EquitySnapshot]] = None,
) -> TradingStats:
    executed = [r for r in trade_results if _is_executed(r)]
    rejected = [r for r in trade_results if _is_rejected(r)]
    closed = _closed_executed(trade_results)

    nets = [r.net_pnl for r in closed if r.net_pnl is not None]
    wins = [n for n in nets if n > 0]
    losses = [n for n in nets if n < 0]
    win_flags = [n > 0 for n in nets]
    loss_flags = [n < 0 for n in nets]

    gross_wins = sum(wins) if wins else 0.0
    gross_losses = abs(sum(losses)) if losses else 0.0
    if gross_losses > 0:
        profit_factor: Optional[float] = gross_wins / gross_losses
    elif gross_wins > 0:
        profit_factor = float("inf")
    else:
        profit_factor = None

    bars = [r.duration_bars for r in closed if r.duration_bars is not None]
    avg_bars = (sum(bars) / len(bars)) if bars else None
    med_bars = float(median(bars)) if bars else None

    invested_bars, inv_def = count_invested_bars(snapshots or [], closed)
    avg_ret_per_bar = (
        portfolio_total_return / invested_bars if invested_bars > 0 else None
    )

    return TradingStats(
        number_of_trades=len(trade_results),
        executed=len(executed),
        rejected=len(rejected),
        winning=len(wins),
        losing=len(losses),
        win_rate=(len(wins) / len(nets)) if nets else None,
        avg_win=(sum(wins) / len(wins)) if wins else None,
        avg_loss=(sum(losses) / len(losses)) if losses else None,
        largest_win=max(wins) if wins else None,
        largest_loss=min(losses) if losses else None,
        profit_factor=profit_factor,
        avg_holding_bars=avg_bars,
        median_holding_bars=med_bars,
        max_consecutive_wins=_max_consecutive(win_flags),
        max_consecutive_losses=_max_consecutive(loss_flags),
        invested_bars=invested_bars,
        invested_bars_definition=inv_def,
        avg_return_per_bar=avg_ret_per_bar,
        portfolio_total_return=portfolio_total_return,
    )


def compute_portfolio_stats(
    snapshots: Sequence[EquitySnapshot],
    *,
    source: str = "event",
) -> PortfolioStats:
    if not snapshots:
        return PortfolioStats(
            avg_positions_per_day=None,
            max_exposure_pct=None,
            avg_exposure_pct=None,
            pct_time_invested=None,
            pct_time_in_cash=None,
            n_days=0,
            source="none",
        )
    n = len(snapshots)
    avg_pos = sum(s.open_positions for s in snapshots) / n
    exposures = [s.exposure_pct for s in snapshots]
    invested_days = sum(1 for s in snapshots if (s.invested_cost_basis or 0) > 0)
    cash_days = sum(1 for s in snapshots if (s.invested_cost_basis or 0) <= 0)
    return PortfolioStats(
        avg_positions_per_day=avg_pos,
        max_exposure_pct=max(exposures) if exposures else None,
        avg_exposure_pct=(sum(exposures) / n) if exposures else None,
        pct_time_invested=invested_days / n,
        pct_time_in_cash=cash_days / n,
        n_days=n,
        source=source,
    )


def compute_cagr_from_snapshots(
    snapshots: Sequence[EquitySnapshot],
) -> Optional[float]:
    if not snapshots or len(snapshots) < 2:
        return None
    first, last = snapshots[0], snapshots[-1]
    return equity_cagr(first.equity, last.equity, first.timestamp, last.timestamp)


def attach_stats_to_result(
    *,
    trade_results: Sequence[TradeResult],
    performance: PerformanceSummary,
    event_snapshots: Sequence[EquitySnapshot],
    daily_mtm: Sequence[EquitySnapshot],
) -> tuple[TradingStats, PortfolioStats, Optional[float]]:
    """Compute trading/portfolio stats and preferred CAGR series."""
    series = list(daily_mtm) if daily_mtm else list(event_snapshots)
    source = "daily_mtm" if daily_mtm else ("event" if event_snapshots else "none")
    trading = compute_trading_stats(
        trade_results,
        portfolio_total_return=performance.total_return_pct,
        snapshots=series,
    )
    portfolio = compute_portfolio_stats(series, source=source)
    cagr = compute_cagr_from_snapshots(series)
    return trading, portfolio, cagr
