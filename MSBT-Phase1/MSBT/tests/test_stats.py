"""Unit tests for trading / portfolio stats analytics."""

from datetime import date

from msbt.analytics.stats import (
    compute_portfolio_stats,
    compute_trading_stats,
    count_invested_bars,
)
from msbt.models.config import SimulationConfig
from msbt.models.results import EquitySnapshot
from msbt.simulation.engine import run_simulation
from tests.helpers import make_trade


def _phase1_cfg(**kwargs):
    base = dict(
        initial_capital=100_000,
        buy_pct_of_equity=0.20,
        max_pct_per_symbol=0.40,
        entry_fee_pct=0.0,
        exit_fee_pct=0.0,
        slippage_pct=0.0,
        cash_earn_mode="none",
    )
    base.update(kwargs)
    return SimulationConfig(**base)


def test_trading_stats_win_loss_and_holding():
    trades = [
        make_trade("W", "AAA", "2020-01-01", "2020-01-10", 0.10, duration_bars=5),
        make_trade("L", "BBB", "2020-02-01", "2020-02-10", -0.05, duration_bars=7),
    ]
    result = run_simulation(trades, _phase1_cfg(buy_pct_of_equity=0.10, max_pct_per_symbol=0.50))
    ts = result.trading_stats
    assert ts is not None
    assert ts.executed == 2
    assert ts.winning == 1
    assert ts.losing == 1
    assert abs(ts.win_rate - 0.5) < 1e-9
    assert ts.avg_holding_bars == 6.0
    assert ts.median_holding_bars == 6.0
    assert ts.max_consecutive_wins == 1
    assert ts.max_consecutive_losses == 1
    assert ts.profit_factor is not None and ts.profit_factor > 0


def test_invested_bars_prefer_snapshots():
    snaps = [
        EquitySnapshot(date(2020, 1, 1), 100, 50, 50, 1, 0.5, 0.0),
        EquitySnapshot(date(2020, 1, 2), 100, 100, 0, 0, 0.0, 0.0),
        EquitySnapshot(date(2020, 1, 3), 110, 40, 70, 1, 0.6, 0.1),
    ]
    n, defn = count_invested_bars(snaps, [])
    assert n == 2
    assert "invested_cost_basis" in defn


def test_avg_return_per_bar_uses_invested_days():
    trades = [
        make_trade("T1", "AAPL", "2020-01-10", "2020-01-20", 0.10, source_trade_number=1),
        make_trade("T2", "MSFT", "2020-01-15", "2020-01-25", 0.20, source_trade_number=2),
    ]
    result = run_simulation(trades, _phase1_cfg())
    ts = result.trading_stats
    assert ts.invested_bars > 0
    assert ts.avg_return_per_bar is not None
    assert abs(ts.avg_return_per_bar - (0.06 / ts.invested_bars)) < 1e-9


def test_portfolio_stats_exposure():
    snaps = [
        EquitySnapshot(date(2020, 1, 1), 100, 40, 60, 2, 0.6, 0.0),
        EquitySnapshot(date(2020, 1, 2), 100, 100, 0, 0, 0.0, 0.0),
        EquitySnapshot(date(2020, 1, 3), 100, 20, 80, 1, 0.8, 0.0),
    ]
    ps = compute_portfolio_stats(snaps, source="event")
    assert ps.n_days == 3
    assert abs(ps.avg_positions_per_day - 1.0) < 1e-9
    assert abs(ps.max_exposure_pct - 0.8) < 1e-9
    assert abs(ps.avg_exposure_pct - (0.6 + 0.0 + 0.8) / 3) < 1e-9
    assert abs(ps.pct_time_invested - 2 / 3) < 1e-9
    assert abs(ps.pct_time_in_cash - 1 / 3) < 1e-9


def test_cagr_surfaced_on_performance():
    trades = [
        make_trade("T1", "AAPL", "2020-01-10", "2020-01-20", 0.10, source_trade_number=1),
        make_trade("T2", "MSFT", "2020-01-15", "2020-01-25", 0.20, source_trade_number=2),
    ]
    result = run_simulation(trades, _phase1_cfg())
    assert result.performance_metrics.cagr is not None
    assert result.performance_metrics.cagr > 0
