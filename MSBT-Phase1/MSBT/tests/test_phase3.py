"""Phase 3: parameter grid, run history, TradingView hooks. No live Yahoo."""

from __future__ import annotations

from msbt.analytics.grid import run_parameter_grid
from msbt.analytics.history import (
    compare_runs,
    delete_run,
    get_run,
    list_runs,
    save_run,
    update_run_meta,
)
from msbt.ingestion.tradingview import (
    import_symbol_list,
    map_strategy_names,
    receive_exported_signals,
)
from msbt.models.config import EntryPriority, SimulationConfig
from msbt.simulation.engine import run_simulation
from tests.helpers import make_trade


def _trades():
    return [
        make_trade("T1", "AAA", "2020-01-02", "2020-02-03", 0.10, source_trade_number=1),
        make_trade("T2", "BBB", "2020-01-02", "2020-03-02", 0.05, source_trade_number=2),
    ]


def test_grid_two_buy_pcts_independent_and_trades_unchanged():
    trades = _trades()
    before = [(t.trade_id, t.return_pct, t.symbol, t.buy_date, t.sell_date) for t in trades]
    ids = [id(t) for t in trades]
    length = len(trades)

    base = SimulationConfig(
        initial_capital=100_000,
        buy_pct_of_equity=0.05,
        max_pct_per_symbol=0.50,
        entry_priority=EntryPriority.HIGHEST_AVG_TRADE_RETURN.value,
    )
    rows = run_parameter_grid(
        trades,
        base,
        {"buy_pct_of_equity": [0.05, 0.10]},
        market_data=None,
    )

    assert [(t.trade_id, t.return_pct, t.symbol, t.buy_date, t.sell_date) for t in trades] == before
    assert [id(t) for t in trades] == ids
    assert len(trades) == length

    assert len(rows) == 2
    assert rows[0]["buy_pct_of_equity"] == 0.05
    assert rows[1]["buy_pct_of_equity"] == 0.10
    assert rows[0]["final_equity"] != rows[1]["final_equity"]
    assert rows[0]["return_pct"] != rows[1]["return_pct"]
    assert rows[0]["simulation_id"] != rows[1]["simulation_id"]
    # No daily MTM → max DD omitted / NA
    assert rows[0]["max_drawdown"] is None
    assert rows[1]["max_drawdown"] is None
    assert "accepted" in rows[0] and "rejected" in rows[0]

    # Independent of calling run_simulation directly with the same variants.
    alone = run_simulation(
        trades,
        SimulationConfig(
            initial_capital=100_000,
            buy_pct_of_equity=0.10,
            max_pct_per_symbol=0.50,
            entry_priority=EntryPriority.HIGHEST_AVG_TRADE_RETURN.value,
        ),
        market_data=None,
    )
    assert rows[1]["final_equity"] == alone.performance_metrics.final_equity
    assert rows[1]["accepted"] == alone.performance_metrics.accepted


def test_grid_priority_axis_does_not_require_market_data():
    trades = _trades()
    base = SimulationConfig(initial_capital=100_000, buy_pct_of_equity=0.05, max_pct_per_symbol=0.50)
    rows = run_parameter_grid(
        trades,
        base,
        {
            "entry_priority": [
                EntryPriority.HIGHEST_AVG_TRADE_RETURN.value,
                EntryPriority.HIGHEST_WIN_RATE.value,
            ]
        },
        market_data=None,
    )
    assert len(rows) == 2
    assert {r["entry_priority"] for r in rows} == {
        EntryPriority.HIGHEST_AVG_TRADE_RETURN.value,
        EntryPriority.HIGHEST_WIN_RATE.value,
    }


def test_history_save_list_get_update_delete_and_compare(tmp_path):
    db = tmp_path / "runs.sqlite"
    trades = _trades()
    cfg_a = SimulationConfig(initial_capital=100_000, buy_pct_of_equity=0.05, max_pct_per_symbol=0.50)
    cfg_b = SimulationConfig(initial_capital=100_000, buy_pct_of_equity=0.10, max_pct_per_symbol=0.50)
    result_a = run_simulation(trades, cfg_a, market_data=None)
    result_b = run_simulation(trades, cfg_b, market_data=None)

    id_a = save_run(result_a, name="five", tags="alpha", notes="first", db_path=db)
    id_b = save_run(result_b, name="ten", tags=["beta", "grid"], notes="second", db_path=db)
    assert id_a == result_a.simulation_id
    assert id_b == result_b.simulation_id

    listed = list_runs(db)
    assert {r["id"] for r in listed} == {id_a, id_b}
    assert all("final_equity" in r for r in listed)

    got = get_run(id_a, db_path=db)
    assert got["name"] == "five"
    assert got["notes"] == "first"
    assert got["tags"] == ["alpha"]
    assert got["config"]["buy_pct_of_equity"] == 0.05
    assert got["summary"]["final_equity"] == result_a.performance_metrics.final_equity
    assert got["ledger_counts"]["accepted"] == result_a.performance_metrics.accepted
    assert got["equity"]["event"]
    assert got["created_at"].endswith("Z")

    update_run_meta(id_a, "five-b", "zeta", "updated notes", db_path=db)
    got2 = get_run(id_a, db_path=db)
    assert got2["name"] == "five-b"
    assert got2["notes"] == "updated notes"
    assert got2["tags"] == ["zeta"]
    # Outputs unchanged
    assert got2["summary"]["final_equity"] == result_a.performance_metrics.final_equity

    table = compare_runs([id_a, id_b], db_path=db)
    assert [row["id"] for row in table] == [id_a, id_b]
    assert table[0]["notes"] == "updated notes"
    assert table[0]["final_equity"] != table[1]["final_equity"]
    assert table[0]["return_pct"] != table[1]["return_pct"]
    assert table[0]["buy_pct_of_equity"] == 0.05
    assert table[1]["buy_pct_of_equity"] == 0.10
    # Aligned columns present even when max DD is unavailable
    assert "max_drawdown" in table[0]
    assert "cagr" in table[0]
    assert "accepted" in table[0]
    assert "rejected" in table[0]

    delete_run(id_a, db_path=db)
    remaining = list_runs(db)
    assert [r["id"] for r in remaining] == [id_b]


def test_history_saves_grid_comparison(tmp_path):
    db = tmp_path / "runs.sqlite"
    trades = _trades()
    base = SimulationConfig(initial_capital=100_000, buy_pct_of_equity=0.05, max_pct_per_symbol=0.50)
    rows = run_parameter_grid(trades, base, {"buy_pct_of_equity": [0.05, 0.20]}, market_data=None)
    rid = save_run(
        name="grid-demo",
        tags="grid",
        notes="two cells",
        grid_comparison=rows,
        config=base,
        db_path=db,
    )
    rec = get_run(rid, db_path=db)
    assert rec["kind"] == "grid"
    assert rec["grid_comparison"] is not None
    assert len(rec["grid_comparison"]) == 2
    assert rec["notes"] == "two cells"


def test_tradingview_hooks_are_local_only():
    symbols = import_symbol_list("NASDAQ:AAPL, NYSE:IBM\nAMEX:IVV\nAAPL")
    assert symbols == ["AAPL", "IBM", "IVV"]

    csv_payload = "symbol,strategy,time,action,price\nNASDAQ:MSFT,Triple,2020-01-02,buy,100\n"
    signals = receive_exported_signals(csv_payload)
    assert signals[0]["symbol"] == "MSFT"
    assert signals[0]["strategy"] == "Triple"
    assert signals[0]["action"] == "buy"

    mapped = map_strategy_names(["Triple", "Other"], {"triple": "TripleStrategy"})
    assert mapped == ["TripleStrategy", "Other"]
    assert map_strategy_names("triple", {"triple": "TripleStrategy"}) == "TripleStrategy"
