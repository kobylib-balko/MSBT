"""Tests for MFE/MAE excursion import and analytics."""

from datetime import date
from pathlib import Path

from msbt.analytics.mfe_mae import compute_mfe_mae_stats, _pctile
from msbt.importers.csv_importer import import_trade_file
from msbt.models.results import ACCEPTED, TradeResult
from msbt.models.trade import TradeStatus
from msbt.simulation.engine import run_simulation
from msbt.models.config import SimulationConfig

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "TripleStrategy_AMEX_SAMPLE_2026-09-05.csv"


def _tr(
    tid: str,
    *,
    net_pnl: float,
    mfe: float | None,
    mae: float | None,
    gross: float | None = None,
    buy: str = "2020-01-01",
) -> TradeResult:
    if gross is None:
        gross = net_pnl / 1000.0  # rough; override in tests as needed
    return TradeResult(
        trade_id=tid,
        symbol="TEST",
        strategy="S",
        exchange="X",
        source_file="f.csv",
        source_trade_number=1,
        buy_date=date.fromisoformat(buy),
        sell_date=date.fromisoformat(buy),
        gross_return=gross,
        net_return=gross,
        allocated_capital=1000.0,
        qty=10.0,
        cost_basis=1000.0,
        gross_pnl=net_pnl,
        net_pnl=net_pnl,
        entry_fee=0.0,
        exit_fee=0.0,
        entry_slippage=0.0,
        exit_slippage=0.0,
        status=ACCEPTED,
        is_open=False,
        mfe_pct=mfe,
        mae_pct=mae,
    )


def test_fixture_import_has_mfe_mae():
    trades, _ = import_trade_file(FIXTURE)
    completed = [t for t in trades if t.status == TradeStatus.COMPLETED]
    assert completed, "expected completed trades in fixture"
    with_exc = [t for t in completed if t.mfe_pct is not None and t.mae_pct is not None]
    assert with_exc, "fixture should populate Favorable/Adverse excursion %"
    # First trade in fixture: 15.80% → 0.1580, -5.51% → -0.0551
    t1 = next(t for t in completed if t.source_trade_number == 1)
    assert t1.mfe_pct is not None
    assert abs(t1.mfe_pct - 0.1580) < 1e-9
    assert t1.mae_pct is not None
    assert abs(t1.mae_pct - (-0.0551)) < 1e-9
    assert t1.mfe_usd is not None
    assert t1.mae_usd is not None


def test_compute_mfe_mae_known_medians():
    # Two winners, one loser — known MFE/MAE
    results = [
        _tr("W1", net_pnl=100, mfe=0.20, mae=-0.04, gross=0.10),  # giveback 0.10
        _tr("W2", net_pnl=50, mfe=0.12, mae=-0.06, gross=0.05),   # giveback 0.07
        _tr("L1", net_pnl=-80, mfe=0.05, mae=-0.15, gross=-0.08),
    ]
    stats = compute_mfe_mae_stats(results)
    assert stats.n_with_data == 3
    assert stats.n_closed_executed == 3
    assert abs(stats.median_mfe - 0.12) < 1e-9
    assert abs(stats.median_mae - (-0.06)) < 1e-9
    # givebacks: 0.10, 0.07, 0.05+0.08=0.13 → median 0.10
    assert abs(stats.median_giveback - 0.10) < 1e-9
    assert stats.winners_n == 2
    assert stats.losers_n == 1
    assert stats.winners_median_mfe is not None
    assert abs(stats.winners_median_mfe - 0.16) < 1e-9  # median of 0.20, 0.12
    # suggested stop = p25 of winners MAE [-0.04, -0.06] → more adverse side
    assert stats.suggested_stop_pct is not None
    assert stats.suggested_target_pct is not None
    assert abs(stats.suggested_target_pct - 0.16) < 1e-9
    assert stats.recommendations
    assert any("look-ahead" in r.lower() or "own MFE/MAE" in r for r in stats.recommendations)
    assert "Full sample" in stats.scope_note
    d = stats.to_dict()
    assert d["n_with_data"] == 3


def test_empty_excursion_recommendation():
    results = [
        _tr("A", net_pnl=10, mfe=None, mae=None, gross=0.01),
        _tr("B", net_pnl=-5, mfe=None, mae=None, gross=-0.01),
    ]
    stats = compute_mfe_mae_stats(results)
    assert stats.n_with_data == 0
    assert stats.suggested_stop_pct is None
    assert stats.suggested_target_pct is None
    assert len(stats.recommendations) == 1
    assert "missing" in stats.recommendations[0].lower() or "empty" in stats.recommendations[0].lower()


def test_train_cutoff_scopes_sample():
    results = [
        _tr("OLD", net_pnl=10, mfe=0.20, mae=-0.05, gross=0.10, buy="2019-01-01"),
        _tr("NEW", net_pnl=10, mfe=0.50, mae=-0.01, gross=0.40, buy="2021-06-01"),
    ]
    stats = compute_mfe_mae_stats(results, train_cutoff=date(2020, 1, 1))
    assert stats.n_with_data == 1
    assert abs(stats.median_mfe - 0.20) < 1e-9
    assert "Train sample" in stats.scope_note


def test_pctile_helper():
    assert _pctile([1.0], 0.5) == 1.0
    assert abs(_pctile([0.0, 1.0], 0.5) - 0.5) < 1e-9
    assert _pctile([], 0.5) is None


def test_engine_attaches_mfe_mae_from_fixture():
    trades, _ = import_trade_file(FIXTURE)
    valid = [t for t in trades if t.is_valid_for_sim]
    cfg = SimulationConfig(
        initial_capital=100_000,
        buy_pct_of_equity=0.05,
        max_pct_per_symbol=0.50,
        entry_fee_pct=0.0,
        exit_fee_pct=0.0,
        slippage_pct=0.0,
        cash_earn_mode="none",
    )
    result = run_simulation(valid, cfg)
    assert result.mfe_mae_stats is not None
    assert result.mfe_mae_stats.n_with_data > 0
    executed = [r for r in result.trade_results if r.status == ACCEPTED and not r.is_open]
    assert any(r.mfe_pct is not None for r in executed)
    d = executed[0].to_dict()
    assert "MFE %" in d or executed[0].mfe_pct is None
