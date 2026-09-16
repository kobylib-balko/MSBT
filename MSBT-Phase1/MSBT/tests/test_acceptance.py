"""Acceptance Examples 1–2 from the locked Phase 1 prompt."""

from datetime import date

from msbt.models.config import SimulationConfig
from msbt.models.results import ACCEPTED, PARTIAL_INSUFFICIENT
from msbt.simulation.engine import run_simulation
from tests.helpers import make_trade


def test_acceptance_example_1(acceptance1_trades, acceptance1_config):
    result = run_simulation(acceptance1_trades, acceptance1_config)
    assert result.performance_metrics.final_equity == 106000.0
    assert abs(result.performance_metrics.total_return_pct - 0.06) < 1e-9

    by_date = {s.timestamp: s for s in result.portfolio_timeseries}
    assert by_date[date(2020, 1, 10)].cash == 80000.0
    assert by_date[date(2020, 1, 15)].cash == 60000.0
    assert by_date[date(2020, 1, 20)].cash == 82000.0
    assert by_date[date(2020, 1, 20)].equity == 102000.0
    assert by_date[date(2020, 1, 25)].cash == 106000.0
    assert by_date[date(2020, 1, 25)].equity == 106000.0


def test_acceptance_example_2_same_day_exit_priority():
    """Same-day exit + priority ranking + frozen equity_for_sizing.

    Prompt text claims GOOG partial 41800, but that contradicts buy 30%:
    target = 104500 * 0.30 = 31350, and after two full fills cash is 41800 > 31350,
    so GOOG also receives a full 31350 under locked sizing rules.
    We assert the locked-rule outcome; see test_same_day_priority_partial for partial.
    """
    warmup = [
        make_trade("W-NVDA", "NVDA", "2019-01-01", "2019-06-01", 0.18, source_trade_number=100),
        make_trade("W-MSFT", "MSFT", "2019-01-01", "2019-06-01", 0.08, source_trade_number=101),
        make_trade("W-GOOG", "GOOG", "2019-01-01", "2019-06-01", 0.06, source_trade_number=102),
        make_trade("W-AAPL", "AAPL", "2019-01-01", "2019-06-01", 0.12, source_trade_number=103),
    ]
    trades = warmup + [
        make_trade("T1", "AAPL", "2020-02-01", "2020-02-10", 0.15, source_trade_number=1),
        make_trade("T2", "MSFT", "2020-02-10", "2020-02-20", 0.10, source_trade_number=2),
        make_trade("T3", "NVDA", "2020-02-10", "2020-02-25", 0.25, source_trade_number=3),
        make_trade("T4", "GOOG", "2020-02-10", "2020-02-28", 0.05, source_trade_number=4),
    ]
    config = SimulationConfig(
        initial_capital=100_000,
        buy_pct_of_equity=0.30,
        max_pct_per_symbol=0.50,
        entry_fee_pct=0.0,
        exit_fee_pct=0.0,
        slippage_pct=0.0,
        allow_partial_fills=True,
        allow_multiple_positions_same_symbol=True,
        entry_priority="highest_avg_trade_return",
        start_date=date(2020, 2, 1),
        cash_earn_mode="none",
    )

    result = run_simulation(trades, config)

    by_date = {s.timestamp: s for s in result.portfolio_timeseries}
    snap = by_date[date(2020, 2, 10)]
    assert snap.equity == 104500.0

    by_id = {r.trade_id: r for r in result.trade_results}
    assert by_id["T3"].allocated_capital == 31350.0  # NVDA
    assert by_id["T3"].status == ACCEPTED
    assert by_id["T2"].allocated_capital == 31350.0  # MSFT
    assert by_id["T2"].status == ACCEPTED
    assert by_id["T4"].allocated_capital == 31350.0  # GOOG full (cash sufficient)
    assert by_id["T4"].status == ACCEPTED
    assert snap.cash == 10450.0
    assert snap.invested_cost_basis == 94050.0


def test_same_day_priority_partial():
    """Force partial on lowest-priority symbol via higher buy_pct."""
    warmup = [
        make_trade("W-NVDA", "NVDA", "2019-01-01", "2019-06-01", 0.18, source_trade_number=100),
        make_trade("W-MSFT", "MSFT", "2019-01-01", "2019-06-01", 0.08, source_trade_number=101),
        make_trade("W-GOOG", "GOOG", "2019-01-01", "2019-06-01", 0.06, source_trade_number=102),
        make_trade("W-AAPL", "AAPL", "2019-01-01", "2019-06-01", 0.12, source_trade_number=103),
    ]
    trades = warmup + [
        make_trade("T1", "AAPL", "2020-02-01", "2020-02-10", 0.15, source_trade_number=1),
        make_trade("T2", "MSFT", "2020-02-10", "2020-02-20", 0.10, source_trade_number=2),
        make_trade("T3", "NVDA", "2020-02-10", "2020-02-25", 0.25, source_trade_number=3),
        make_trade("T4", "GOOG", "2020-02-10", "2020-02-28", 0.05, source_trade_number=4),
    ]
    # buy 40%: target=41800; after NVDA+MSFT cash=20900 → GOOG partial
    config = SimulationConfig(
        initial_capital=100_000,
        buy_pct_of_equity=0.40,
        max_pct_per_symbol=0.50,
        allow_partial_fills=True,
        entry_priority="highest_avg_trade_return",
        start_date=date(2020, 2, 1),
        entry_fee_pct=0.0,
        exit_fee_pct=0.0,
        slippage_pct=0.0,
        cash_earn_mode="none",
    )
    result = run_simulation(trades, config)
    snap = next(s for s in result.portfolio_timeseries if s.timestamp == date(2020, 2, 10))
    # AAPL was 40% of 100k=40000; exit 46000; cash 60000+46000=106000
    assert snap.equity == 106000.0
    by_id = {r.trade_id: r for r in result.trade_results}
    assert by_id["T3"].allocated_capital == 42400.0  # NVDA first
    assert by_id["T2"].allocated_capital == 42400.0  # MSFT
    assert by_id["T4"].status == PARTIAL_INSUFFICIENT
    assert by_id["T4"].allocated_capital == 21200.0
    assert snap.cash == 0.0
