"""Phase 4: train/test split, locked OOS, walk-forward, stress pack."""

from __future__ import annotations

from datetime import date

from msbt.analytics.robustness import (
    buy_date_span,
    default_cutoff,
    drop_symbol,
    run_locked_oos,
    run_stress_pack,
    run_train_vs_test,
    run_walk_forward,
    select_best_grid_row,
    split_by_buy_date,
    top_symbol_by_net_pnl,
    walk_forward_windows,
)
from msbt.models.config import EntryPriority, SimulationConfig
from tests.helpers import make_trade


def _multi_year_trades():
    """Synthetic trades spanning 2016–2023 across several symbols."""
    rows = [
        ("T1", "AAA", "2016-03-01", "2016-04-01", 0.10),
        ("T2", "BBB", "2016-06-01", "2016-07-01", 0.05),
        ("T3", "AAA", "2017-02-01", "2017-03-01", 0.08),
        ("T4", "CCC", "2017-08-01", "2017-09-01", 0.12),
        ("T5", "BBB", "2018-01-15", "2018-02-15", 0.06),
        ("T6", "AAA", "2018-09-01", "2018-10-01", 0.15),  # strong AAA
        ("T7", "CCC", "2019-03-01", "2019-04-01", 0.04),
        ("T8", "BBB", "2019-11-01", "2019-12-01", 0.07),
        ("T9", "AAA", "2020-04-01", "2020-05-01", 0.20),  # strong AAA
        ("T10", "DDD", "2020-10-01", "2020-11-01", 0.03),
        ("T11", "CCC", "2021-02-01", "2021-03-01", 0.05),
        ("T12", "BBB", "2021-07-01", "2021-08-01", 0.09),
        ("T13", "AAA", "2022-01-10", "2022-02-10", 0.11),
        ("T14", "DDD", "2022-06-01", "2022-07-01", 0.02),
        ("T15", "CCC", "2023-03-01", "2023-04-01", 0.06),
        ("T16", "BBB", "2023-09-01", "2023-10-01", 0.04),
    ]
    return [
        make_trade(tid, sym, buy, sell, ret, source_trade_number=i + 1)
        for i, (tid, sym, buy, sell, ret) in enumerate(rows)
    ]


def _cfg(**kwargs) -> SimulationConfig:
    base = dict(
        initial_capital=100_000,
        buy_pct_of_equity=0.10,
        max_pct_per_symbol=0.50,
        entry_priority=EntryPriority.HIGHEST_AVG_TRADE_RETURN.value,
        entry_fee_pct=0.001,
        exit_fee_pct=0.001,
        slippage_pct=0.0005,
        allow_partial_fills=True,
        cash_earn_mode="none",
    )
    base.update(kwargs)
    return SimulationConfig(**base)


def test_split_counts_by_buy_date():
    trades = _multi_year_trades()
    cutoff = date(2020, 1, 1)
    train, test = split_by_buy_date(trades, cutoff)
    assert len(train) + len(test) == len(trades)
    assert all(t.buy_date < cutoff for t in train)
    assert all(t.buy_date >= cutoff for t in test)
    assert len(train) == 8
    assert len(test) == 8

    # Default median / 70-30 produce a usable split
    med = default_cutoff(trades, mode="median")
    assert med is not None
    tr, te = split_by_buy_date(trades, med)
    assert tr and te

    c70 = default_cutoff(trades, mode="70_30")
    assert c70 is not None
    tr70, te70 = split_by_buy_date(trades, c70)
    assert tr70 and te70
    assert len(tr70) >= len(te70)


def test_train_vs_test_side_by_side():
    trades = _multi_year_trades()
    out = run_train_vs_test(trades, _cfg(), cutoff=date(2020, 1, 1), market_data=None)
    assert out["n_train"] == 8
    assert out["n_test"] == 8
    assert len(out["side_by_side"]) == 2
    assert out["side_by_side"][0]["slice"] == "Train"
    assert out["side_by_side"][1]["slice"] == "Test"
    assert "return_pct" in out["train"]
    assert "degradation" in out


def test_locked_oos_does_not_use_test_for_selection():
    trades = _multi_year_trades()
    cfg = _cfg()
    grid = {
        "buy_pct_of_equity": [0.05, 0.10, 0.20],
        "entry_priority": [EntryPriority.HIGHEST_AVG_TRADE_RETURN.value],
    }
    out = run_locked_oos(
        trades,
        cfg,
        grid=grid,
        cutoff=date(2020, 1, 1),
        market_data=None,
    )
    assert out["selection_used_test"] is False
    assert out["best_params"]
    assert "buy_pct_of_equity" in out["best_params"]
    assert out["train_metrics"]["return_pct"] is not None
    assert out["test_metrics"]["return_pct"] is not None
    assert "delta_return_pct" in out["degradation"]

    # Best params must match a Train-only grid winner (recompute Train grid)
    from msbt.analytics.grid import run_parameter_grid
    from msbt.analytics.robustness import split_by_buy_date, select_best_grid_row

    train, test = split_by_buy_date(trades, date(2020, 1, 1))
    train_rows = run_parameter_grid(train, cfg, grid, market_data=None)
    best = select_best_grid_row(train_rows)
    assert out["best_params"]["buy_pct_of_equity"] == best["buy_pct_of_equity"]

    # Sanity: running the same grid on Test alone can pick a different cell —
    # we only assert selection source is Train (already checked equality above).
    test_rows = run_parameter_grid(test, cfg, grid, market_data=None)
    assert len(test_rows) == len(train_rows)
    # Ensure Test was never consulted: flag + train_grid length
    assert len(out["train_grid"]) == 3


def test_stress_removes_top_symbol():
    trades = _multi_year_trades()
    cfg = _cfg(buy_pct_of_equity=0.15, entry_fee_pct=0.0, exit_fee_pct=0.0, slippage_pct=0.0)
    # Force cash earn on so cash_earn_none variant appears
    cfg_earn = _cfg(
        buy_pct_of_equity=0.15,
        entry_fee_pct=0.001,
        exit_fee_pct=0.001,
        slippage_pct=0.0005,
        cash_earn_mode="synthetic_rf",
    )
    pack = run_stress_pack(
        trades,
        cfg_earn,
        cutoff=date(2020, 1, 1),
        use_test_set=True,
        drop_symbol_from="train",
        market_data=None,
    )
    variants = {r["variant"] for r in pack["rows"]}
    assert "baseline" in variants
    assert "costs_x3" in variants
    assert "partial_fills_off" in variants
    assert "cash_earn_none" in variants
    assert pack["top_symbol"] is not None
    drop_rows = [r for r in pack["rows"] if r["variant"].startswith("drop_top_symbol:")]
    assert len(drop_rows) == 1
    assert drop_rows[0]["dropped_symbol"] == pack["top_symbol"]

    # drop_symbol helper
    dropped = drop_symbol(trades, pack["top_symbol"])
    assert all(t.symbol != pack["top_symbol"] for t in dropped)
    assert len(dropped) < len(trades)

    top = top_symbol_by_net_pnl(trades, cfg, market_data=None)
    assert top in {t.symbol for t in trades}


def test_walk_forward_produces_n_windows():
    trades = _multi_year_trades()
    span = buy_date_span(trades)
    assert span is not None
    min_b, max_b = span
    # 2016–2023 ≈ 7+ years; train 3y, test 1y, step 1y → several windows
    wins = walk_forward_windows(min_b, max_b, train_years=3, test_years=1, step_years=1, mode="expanding")
    assert len(wins) >= 3

    cfg = _cfg(entry_fee_pct=0.0, exit_fee_pct=0.0, slippage_pct=0.0)
    grid = {"buy_pct_of_equity": [0.05, 0.10]}
    out = run_walk_forward(
        trades,
        cfg,
        train_years=3,
        test_years=1,
        step_years=1,
        mode="expanding",
        grid=grid,
        market_data=None,
    )
    assert out["n_windows"] == len(wins)
    assert out["n_windows"] >= 3
    assert out["n_evaluated"] >= 1
    assert "oos_summary" in out
    assert out["oos_summary"]["n_test_windows"] >= 1

    rolling = walk_forward_windows(
        min_b, max_b, train_years=3, test_years=1, step_years=1, mode="rolling"
    )
    assert len(rolling) >= 3
    # Rolling train_start moves forward after the first full window
    assert rolling[0]["train_start"] == min_b


def test_select_best_grid_row_deterministic():
    rows = [
        {"scenario": 1, "return_pct": 0.10, "cagr": None, "max_drawdown": None, "params": {"buy_pct_of_equity": 0.05}},
        {"scenario": 2, "return_pct": 0.20, "cagr": None, "max_drawdown": None, "params": {"buy_pct_of_equity": 0.10}},
        {"scenario": 3, "return_pct": 0.15, "cagr": None, "max_drawdown": None, "params": {"buy_pct_of_equity": 0.15}},
    ]
    best = select_best_grid_row(rows)
    assert best["params"]["buy_pct_of_equity"] == 0.10
