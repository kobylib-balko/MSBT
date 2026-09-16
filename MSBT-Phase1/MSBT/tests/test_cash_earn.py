"""Phase 2.5 cash earn tests."""

from datetime import date
from pathlib import Path

from msbt.cash_earn import (
    SyntheticRateProvider,
    build_total_return_index,
    load_rates_csv,
)
from msbt.models.config import SimulationConfig
from msbt.simulation.engine import run_simulation
from tests.helpers import make_trade

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_rf_rates_sample.csv"


def _cfg(**kwargs):
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


def test_index_growth_known_rate():
    rates = {date(2020, 1, 1): 0.365}  # high so growth is visible
    # only one day of rates; last-known applies thereafter
    idx = build_total_return_index(rates, date(2020, 1, 1), date(2020, 1, 3))
    seed = date(2019, 12, 31)
    assert abs(idx[seed] - 100.0) < 1e-12
    expected = 100.0 * ((1.0 + 0.365) ** (1.0 / 365.0))
    assert abs(idx[date(2020, 1, 1)] - expected) < 1e-9
    assert idx[date(2020, 1, 3)] > idx[date(2020, 1, 1)]


def test_load_fixture_csv():
    rates = load_rates_csv(FIXTURE)
    assert date(2020, 1, 1) in rates
    assert abs(rates[date(2020, 1, 1)] - 0.0365) < 1e-12


def test_cash_grows_with_known_rates():
    """Idle cash earns overnight; known constant rate."""
    provider = SyntheticRateProvider(
        {date(2020, 1, 1): 0.365, date(2020, 1, 10): 0.365}
    )
    # No trades → all cash every day; start/end need events — use a flat 0-return trade
    # that opens and closes same capital quickly so cash is mostly idle after.
    # Simpler: one trade with 0 return short hold, cash earn on leftover.
    trades = [
        make_trade("T1", "AAA", "2020-01-02", "2020-01-03", 0.0, source_trade_number=1),
    ]
    cfg = _cfg(
        buy_pct_of_equity=0.10,
        max_pct_per_symbol=0.50,
        cash_earn_mode="synthetic_rf",
    )
    result = run_simulation(trades, cfg, rate_provider=provider)
    assert result.performance_metrics.cash_interest_pnl > 0
    assert result.performance_metrics.final_equity > 100_000.0
    # Closed trade net pnl should be ~0 (0 return, no fees)
    closed = [r for r in result.trade_results if r.net_pnl is not None]
    assert len(closed) == 1
    assert abs(closed[0].net_pnl) < 0.01


def test_closed_trade_pnl_unchanged_by_rf():
    """RF may change sizing via equity growth, but Return-% economics are untouched.

    gross_return stays the source return; net_return (fraction) matches the fee
    formula on that return — RF never rewrites closed-trade exit math.
    """
    trades = [
        make_trade("T1", "AAPL", "2020-01-10", "2020-01-20", 0.10, source_trade_number=1),
        make_trade("T2", "MSFT", "2020-01-15", "2020-01-25", 0.20, source_trade_number=2),
    ]
    cfg_none = _cfg(cash_earn_mode="none")
    cfg_rf = _cfg(cash_earn_mode="synthetic_rf")
    provider = SyntheticRateProvider(load_rates_csv(FIXTURE))
    r0 = run_simulation(trades, cfg_none)
    r1 = run_simulation(trades, cfg_rf, rate_provider=provider)
    closed0 = {r.trade_id: r for r in r0.trade_results if r.net_pnl is not None}
    closed1 = {r.trade_id: r for r in r1.trade_results if r.net_pnl is not None}
    assert set(closed0) == set(closed1) == {"T1", "T2"}
    for tid in ("T1", "T2"):
        assert closed0[tid].gross_return == closed1[tid].gross_return
        assert abs((closed0[tid].net_return or 0) - (closed1[tid].net_return or 0)) < 1e-6
    assert r1.performance_metrics.cash_interest_pnl > 0
    assert r1.performance_metrics.final_equity > r0.performance_metrics.final_equity


def test_no_lookahead_future_rate_ignored():
    """Rate dated after day T must not affect I_T."""
    rates_early = {date(2020, 1, 1): 0.01}
    rates_with_future = {
        date(2020, 1, 1): 0.01,
        date(2020, 1, 10): 1.0,  # huge future rate
    }
    idx_a = build_total_return_index(rates_early, date(2020, 1, 1), date(2020, 1, 5))
    idx_b = build_total_return_index(rates_with_future, date(2020, 1, 1), date(2020, 1, 5))
    for d in (date(2020, 1, 1), date(2020, 1, 3), date(2020, 1, 5)):
        assert abs(idx_a[d] - idx_b[d]) < 1e-12


def test_phase1_acceptance_with_cash_earn_none():
    trades = [
        make_trade("T1", "AAPL", "2020-01-10", "2020-01-20", 0.10, source_trade_number=1),
        make_trade("T2", "MSFT", "2020-01-15", "2020-01-25", 0.20, source_trade_number=2),
    ]
    result = run_simulation(trades, _cfg())
    assert result.performance_metrics.final_equity == 106000.0
    assert result.performance_metrics.cash_interest_pnl == 0.0


def test_symbol_mode_stub_no_earn():
    trades = [
        make_trade("T1", "AAPL", "2020-01-10", "2020-01-20", 0.10, source_trade_number=1),
    ]
    cfg = _cfg(cash_earn_mode="symbol", cash_earn_symbol="BIL")
    result = run_simulation(trades, cfg)
    assert result.performance_metrics.cash_interest_pnl == 0.0
    assert result.cash_earn_notes and "stubbed" in result.cash_earn_notes.lower()


def test_signals_carried_on_trade_results():
    trades = [
        make_trade(
            "T1",
            "AAPL",
            "2020-01-10",
            "2020-01-20",
            0.10,
            entry_signal="Long1",
            exit_signal="Exitlong3",
        ),
    ]
    result = run_simulation(trades, _cfg())
    closed = [r for r in result.trade_results if r.net_pnl is not None][0]
    assert closed.entry_signal == "Long1"
    assert closed.exit_signal == "Exitlong3"
