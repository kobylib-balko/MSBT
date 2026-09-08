"""Unit tests for simulation engine edge cases."""

from datetime import date

from msbt.models.config import SimulationConfig
from msbt.models.results import (
    ACCEPTED,
    REJECTED_INSUFFICIENT,
    REJECTED_MAX_SYMBOL,
    REJECTED_SAME_SYMBOL,
)
from msbt.simulation.accounting import compute_entry, compute_exit, money
from msbt.simulation.engine import run_simulation
from msbt.simulation.priority import rank_entries, score_symbol
from tests.helpers import make_trade


def test_one_win_one_loss():
    trades = [
        make_trade("W", "AAA", "2020-01-01", "2020-01-10", 0.10),
        make_trade("L", "BBB", "2020-02-01", "2020-02-10", -0.05),
    ]
    cfg = SimulationConfig(
        initial_capital=100_000,
        buy_pct_of_equity=0.10,
        max_pct_per_symbol=0.50,
    )
    result = run_simulation(trades, cfg)
    # First: allocate 10k, exit 11k → equity 101000
    # Second: allocate 10100, exit 10100*0.95=9595 → equity 101000-10100+9595=100495
    assert result.performance_metrics.final_equity == 100495.0


def test_same_day_sell_before_buy():
    trades = [
        make_trade("T1", "AAPL", "2020-01-01", "2020-01-15", 0.10),
        make_trade("T2", "MSFT", "2020-01-15", "2020-01-20", 0.0),
    ]
    cfg = SimulationConfig(initial_capital=100_000, buy_pct_of_equity=0.5, max_pct_per_symbol=1.0)
    result = run_simulation(trades, cfg)
    snap = next(s for s in result.portfolio_timeseries if s.timestamp == date(2020, 1, 15))
    # Exit AAPL first: cost 50k → 55k; cash was 50k + 55k = 105k; then buy MSFT 50% of 105k = 52500
    assert snap.equity == 105000.0
    t2 = next(r for r in result.trade_results if r.trade_id == "T2")
    assert t2.allocated_capital == 52500.0


def test_insufficient_capital_reject_no_partial():
    trades = [
        make_trade("T1", "A", "2020-01-01", "2020-01-10", 0.0),
        make_trade("T2", "B", "2020-01-01", "2020-01-10", 0.0),
        make_trade("T3", "C", "2020-01-01", "2020-01-10", 0.0),
    ]
    cfg = SimulationConfig(
        initial_capital=100_000,
        buy_pct_of_equity=0.5,
        max_pct_per_symbol=1.0,
        allow_partial_fills=False,
        entry_priority="highest_avg_trade_return",
        min_history_trades=1,
    )
    # No history → all scores null → tie-break symbol then trade_id: A, B, C
    result = run_simulation(trades, cfg)
    statuses = {r.trade_id: r.status for r in result.trade_results}
    assert statuses["T1"] == ACCEPTED
    assert statuses["T2"] == ACCEPTED
    assert statuses["T3"] == REJECTED_INSUFFICIENT


def test_max_symbol_exposure():
    trades = [
        make_trade("T1", "AAPL", "2020-01-01", "2020-02-01", 0.0, source_trade_number=1),
        make_trade("T2", "AAPL", "2020-01-02", "2020-02-01", 0.0, source_trade_number=2),
    ]
    cfg = SimulationConfig(
        initial_capital=100_000,
        buy_pct_of_equity=0.30,
        max_pct_per_symbol=0.30,
        allow_multiple_positions_same_symbol=True,
        allow_partial_fills=False,
    )
    result = run_simulation(trades, cfg)
    by_id = {r.trade_id: r for r in result.trade_results}
    assert by_id["T1"].status == ACCEPTED
    assert by_id["T2"].status == REJECTED_MAX_SYMBOL


def test_same_symbol_disallowed():
    trades = [
        make_trade("T1", "AAPL", "2020-01-01", "2020-02-01", 0.0, source_trade_number=1),
        make_trade("T2", "AAPL", "2020-01-02", "2020-02-01", 0.0, source_trade_number=2),
    ]
    cfg = SimulationConfig(
        initial_capital=100_000,
        buy_pct_of_equity=0.10,
        max_pct_per_symbol=0.50,
        allow_multiple_positions_same_symbol=False,
    )
    result = run_simulation(trades, cfg)
    by_id = {r.trade_id: r for r in result.trade_results}
    assert by_id["T1"].status == ACCEPTED
    assert by_id["T2"].status == REJECTED_SAME_SYMBOL


def test_fees_and_slippage_cash_math():
    entry = compute_entry(10_000, 100.0, entry_fee_pct=0.001, slippage_pct=0.0005)
    assert entry.entry_fee == 10.0
    assert entry.entry_slippage == 5.0
    assert entry.cash_outflow == 10015.0
    ex = compute_exit(
        cost_basis=10_000,
        return_pct=0.10,
        exit_fee_pct=0.001,
        slippage_pct=0.0005,
        entry_fee=entry.entry_fee,
        entry_slippage=entry.entry_slippage,
    )
    assert ex.gross_exit_value == 11000.0
    assert ex.exit_fee == 11.0
    assert ex.exit_slippage == 5.5
    assert ex.cash_inflow == 10983.5
    # net = 1000 - 10 - 11 - 5 - 5.5 = 968.5
    assert ex.net_pnl == 968.5


def test_open_position_at_end():
    trades = [
        make_trade("O1", "XYZ", "2020-01-01", None, 0.10),
    ]
    cfg = SimulationConfig(initial_capital=100_000, buy_pct_of_equity=0.2, max_pct_per_symbol=0.5)
    result = run_simulation(trades, cfg)
    assert result.performance_metrics.final_equity == 100000.0  # cost basis valuation
    open_res = [r for r in result.trade_results if r.is_open]
    assert len(open_res) == 1
    assert open_res[0].allocated_capital == 20000.0


def test_lookahead_priority_ignores_future():
    """A trade's own return must not affect ranking at entry time."""
    history = [
        make_trade("H1", "AAA", "2019-01-01", "2019-06-01", 0.50),
        make_trade("H2", "BBB", "2019-01-01", "2019-06-01", 0.01),
    ]
    candidates = [
        make_trade("C1", "AAA", "2020-01-01", "2020-06-01", -0.99),  # terrible eventual
        make_trade("C2", "BBB", "2020-01-01", "2020-06-01", 0.99),  # great eventual
    ]
    cfg = SimulationConfig(entry_priority="highest_avg_trade_return", min_history_trades=1)
    ranked = rank_entries(candidates, date(2020, 1, 1), history, cfg)
    assert ranked[0].symbol == "AAA"  # higher hist avg, despite own future loss
    # Own return not used:
    assert score_symbol("AAA", date(2020, 1, 1), history + candidates, cfg) == 0.50


def test_accounting_invariant_snapshots(acceptance1_trades, acceptance1_config):
    result = run_simulation(acceptance1_trades, acceptance1_config)
    for s in result.portfolio_timeseries:
        assert abs(s.equity - money(s.cash + s.invested_cost_basis)) < 0.01


def test_overlapping_trades(acceptance1_trades, acceptance1_config):
    result = run_simulation(acceptance1_trades, acceptance1_config)
    # Overlap window 01-15 to 01-20: 2 opens
    snap = next(s for s in result.portfolio_timeseries if s.timestamp == date(2020, 1, 15))
    assert snap.open_positions == 2
