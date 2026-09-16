"""Phase 2 tests: daily MTM, risk, cache, look-ahead — synthetic market only."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from msbt.analytics.risk import compute_risk_metrics, daily_returns_from_equity
from msbt.market_data.cache import DiskOHLCVCache
from msbt.market_data.base import OHLCVBar
from msbt.market_data.synthetic import SyntheticMarketDataProvider
from msbt.models.config import SimulationConfig
from msbt.models.results import ACCEPTED
from msbt.simulation.engine import run_simulation
from tests.helpers import make_trade


def _weekday_prices(ticker: str, start: date, end: date, base: float = 100.0, step: float = 1.0):
    """Deterministic increasing closes on weekdays."""
    prices = {}
    d = start
    i = 0
    while d <= end:
        if d.weekday() < 5:
            prices[d] = base + i * step
            i += 1
        d += timedelta(days=1)
    return {ticker: prices}


def test_daily_mtm_open_position_known_prices():
    """Open position MTM across days with known closes."""
    # Buy 2020-01-06 (Mon) open trade; valuation through Fri
    trades = [make_trade("O1", "AAA", "2020-01-06", None, 0.50, buy_price=100.0)]
    # Allocate 20% of 100k = 20k → qty = 200
    cfg = SimulationConfig(
        initial_capital=100_000,
        buy_pct_of_equity=0.20,
        max_pct_per_symbol=0.50,
        open_valuation_mode="market",
        valuation_date=date(2020, 1, 10),
        discrepancy_threshold_pct=0.02,
        entry_fee_pct=0.0,
        exit_fee_pct=0.0,
        slippage_pct=0.0,
        cash_earn_mode="none",
    )
    # Closes: Mon 100, Tue 110, Wed 120, Thu 130, Fri 140
    prices = {
        "AAA": {
            date(2020, 1, 6): 100.0,
            date(2020, 1, 7): 110.0,
            date(2020, 1, 8): 120.0,
            date(2020, 1, 9): 130.0,
            date(2020, 1, 10): 140.0,
        }
    }
    provider = SyntheticMarketDataProvider(prices)
    result = run_simulation(trades, cfg, market_data=provider)

    assert result.daily_mtm_timeseries, "expected daily MTM series"
    by_d = {s.timestamp: s for s in result.daily_mtm_timeseries}
    # qty = 20000/100 = 200
    assert by_d[date(2020, 1, 6)].cash == 80000.0
    assert by_d[date(2020, 1, 6)].equity == 80000.0 + 200 * 100.0
    assert by_d[date(2020, 1, 8)].equity == 80000.0 + 200 * 120.0
    assert by_d[date(2020, 1, 10)].equity == 80000.0 + 200 * 140.0
    assert by_d[date(2020, 1, 10)].unrealized_pnl == 200 * 140.0 - 20000.0

    # Source return 50% vs market (140/100-1)=0.40 → discrepancy if threshold 0.02
    assert result.open_valuations
    ov = result.open_valuations[0]
    assert ov["discrepancy_flag"] is True
    assert abs(ov["market_return"] - 0.40) < 1e-9


def test_closed_trade_pnl_unchanged_by_market_prices():
    """Yahoo/market prices must NOT rewrite closed trade P&L."""
    trades = [
        make_trade("T1", "AAPL", "2020-01-06", "2020-01-10", 0.10, buy_price=100.0),
    ]
    cfg = SimulationConfig(
        initial_capital=100_000,
        buy_pct_of_equity=0.20,
        max_pct_per_symbol=0.50,
        open_valuation_mode="market",
        valuation_date=date(2020, 1, 10),
        entry_fee_pct=0.0,
        exit_fee_pct=0.0,
        slippage_pct=0.0,
        cash_earn_mode="none",
    )
    # Wildly different market path — should not affect closed net_pnl
    prices = {
        "AAPL": {
            date(2020, 1, 6): 50.0,
            date(2020, 1, 7): 200.0,
            date(2020, 1, 8): 10.0,
            date(2020, 1, 9): 500.0,
            date(2020, 1, 10): 1.0,
        }
    }
    provider = SyntheticMarketDataProvider(prices)

    r0 = run_simulation(trades, cfg, market_data=None)
    r1 = run_simulation(trades, cfg, market_data=provider)

    closed0 = [t for t in r0.trade_results if t.trade_id == "T1"][0]
    closed1 = [t for t in r1.trade_results if t.trade_id == "T1"][0]
    assert closed0.net_pnl == closed1.net_pnl == 2000.0  # 20000 * 0.10
    assert closed0.gross_pnl == closed1.gross_pnl == 2000.0
    assert closed0.status == closed1.status == ACCEPTED
    # Event final equity (realized) same
    assert r0.portfolio_timeseries[-1].equity == r1.portfolio_timeseries[-1].equity == 102000.0


def test_lookahead_future_prices_do_not_affect_selection():
    """Future prices must not affect accept/reject/sizing."""
    warmup = [
        make_trade("W-A", "AAA", "2019-01-01", "2019-06-01", 0.20, source_trade_number=100),
        make_trade("W-B", "BBB", "2019-01-01", "2019-06-01", 0.05, source_trade_number=101),
    ]
    trades = warmup + [
        make_trade("C1", "AAA", "2020-01-06", "2020-01-10", -0.50, source_trade_number=1),
        make_trade("C2", "BBB", "2020-01-06", "2020-01-10", 0.90, source_trade_number=2),
    ]
    cfg = SimulationConfig(
        initial_capital=100_000,
        buy_pct_of_equity=0.60,
        max_pct_per_symbol=1.0,
        allow_partial_fills=False,
        entry_priority="highest_avg_trade_return",
        start_date=date(2020, 1, 1),
        open_valuation_mode="market",
        valuation_date=date(2020, 1, 10),
        entry_fee_pct=0.0,
        exit_fee_pct=0.0,
        slippage_pct=0.0,
        cash_earn_mode="none",
    )
    # Future huge prices for BBB — must not change ranking (AAA has higher hist)
    prices = {}
    d = date(2020, 1, 6)
    while d <= date(2020, 1, 10):
        if d.weekday() < 5:
            prices.setdefault("AAA", {})[d] = 100.0
            prices.setdefault("BBB", {})[d] = 10000.0  # tempting future
        d += timedelta(days=1)
    # also warmup range unused
    provider = SyntheticMarketDataProvider(prices)

    r_none = run_simulation(trades, cfg, market_data=None)
    r_mkt = run_simulation(trades, cfg, market_data=provider)

    by0 = {t.trade_id: t.status for t in r_none.trade_results if t.trade_id in ("C1", "C2")}
    by1 = {t.trade_id: t.status for t in r_mkt.trade_results if t.trade_id in ("C1", "C2")}
    assert by0 == by1
    # AAA ranked first → accepted; BBB may be rejected for capital
    assert by0["C1"] == ACCEPTED
    alloc0 = {t.trade_id: t.allocated_capital for t in r_none.trade_results if t.trade_id in ("C1", "C2")}
    alloc1 = {t.trade_id: t.allocated_capital for t in r_mkt.trade_results if t.trade_id in ("C1", "C2")}
    assert alloc0 == alloc1


def test_risk_metrics_deterministic_known_values():
    """Risk metrics on a small equity series with known expected values."""
    # Equity: 100, 110, 105, 115 → returns: 0.10, -0.045454..., 0.095238...
    equity = [100.0, 110.0, 105.0, 115.0]
    dates = [date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8), date(2020, 1, 9)]
    rets = daily_returns_from_equity(equity)
    assert abs(rets[0] - 0.10) < 1e-12
    assert abs(rets[1] - (105 / 110 - 1)) < 1e-12
    assert abs(rets[2] - (115 / 105 - 1)) < 1e-12

    m = compute_risk_metrics(equity, dates=dates, risk_free_rate=0.0, sortino_target=0.0)
    assert m.ann_volatility is not None and m.ann_volatility > 0
    assert m.sharpe is not None
    assert m.max_drawdown is not None
    # Peak 110 → trough 105: DD = (105-110)/110
    assert abs(m.max_drawdown - (105 / 110 - 1)) < 1e-12

    # Zero vol → Sharpe undefined
    flat = [100.0, 100.0, 100.0]
    m2 = compute_risk_metrics(flat, risk_free_rate=0.0)
    assert m2.ann_volatility == 0.0 or abs(m2.ann_volatility) < 1e-15
    assert m2.sharpe is None
    assert "sharpe" in m2.unavailable_reasons

    # Insufficient data
    m3 = compute_risk_metrics([100.0])
    assert "all" in m3.unavailable_reasons


def test_cache_reuse_and_missing_dates(tmp_path: Path):
    """Disk cache stores bars; synthetic reports missing dates."""
    cache = DiskOHLCVCache(tmp_path / "cache")
    bars = [
        OHLCVBar(bar_date=date(2020, 1, 6), open=1, high=1, low=1, close=10, volume=0),
        OHLCVBar(bar_date=date(2020, 1, 7), open=1, high=1, low=1, close=11, volume=0),
    ]
    cache.store("TEST", "adjusted", bars)
    loaded = cache.load("TEST", "adjusted", date(2020, 1, 6), date(2020, 1, 10))
    assert date(2020, 1, 6) in loaded
    assert loaded[date(2020, 1, 6)].close == 10
    # Re-store merge
    cache.store(
        "TEST",
        "adjusted",
        [OHLCVBar(bar_date=date(2020, 1, 8), open=1, high=1, low=1, close=12, volume=0)],
    )
    loaded2 = cache.load_all("TEST", "adjusted")
    assert len(loaded2) == 3

    # Synthetic missing dates
    provider = SyntheticMarketDataProvider(
        {"ZZZ": {date(2020, 1, 6): 100.0, date(2020, 1, 8): 102.0}}
    )
    res = provider.get_daily_ohlcv("ZZZ", date(2020, 1, 6), date(2020, 1, 10))
    assert date(2020, 1, 7) in res.missing_dates  # Tue missing
    assert date(2020, 1, 9) in res.missing_dates
    assert date(2020, 1, 10) in res.missing_dates
    assert len(res.bars) == 2
    # fetch count increments
    provider.get_daily_ohlcv("ZZZ", date(2020, 1, 6), date(2020, 1, 10))
    assert provider.fetch_count["ZZZ"] == 2


def test_phase1_unchanged_without_market_data(acceptance1_trades, acceptance1_config):
    result = run_simulation(acceptance1_trades, acceptance1_config, market_data=None)
    assert result.performance_metrics.final_equity == 106000.0
    assert result.daily_mtm_timeseries == []
    assert result.risk_metrics is None


def test_source_valuation_mode():
    trades = [make_trade("O1", "AAA", "2020-01-06", None, 0.25, buy_price=100.0)]
    cfg = SimulationConfig(
        initial_capital=100_000,
        buy_pct_of_equity=0.10,
        max_pct_per_symbol=0.50,
        open_valuation_mode="source",
        valuation_date=date(2020, 1, 8),
        entry_fee_pct=0.0,
        exit_fee_pct=0.0,
        slippage_pct=0.0,
        cash_earn_mode="none",
    )
    prices = {
        "AAA": {
            date(2020, 1, 6): 200.0,
            date(2020, 1, 7): 200.0,
            date(2020, 1, 8): 200.0,
        }
    }
    provider = SyntheticMarketDataProvider(prices)
    result = run_simulation(trades, cfg, market_data=provider)
    by_d = {s.timestamp: s for s in result.daily_mtm_timeseries}
    # cost 10k, source MTM = 10k * 1.25 = 12500; cash 90000 → equity 102500
    assert by_d[date(2020, 1, 8)].equity == 102500.0


def test_ticker_map_identity_default():
    provider = SyntheticMarketDataProvider({"AAPL": {date(2020, 1, 6): 1.0}})
    assert provider.resolve_ticker("AAPL", "NASDAQ") == "AAPL"
    assert provider.resolve_ticker("AAPL", "NASDAQ", {"NASDAQ:AAPL": "AAPL.US"}) == "AAPL.US"
    assert provider.resolve_ticker("AAPL", "", {"AAPL": "AAPL.US"}) == "AAPL.US"
