# Multi-Symbol Portfolio Backtester (MSBT) — Phase 3

Event-based portfolio simulation: upload multiple trade CSVs, normalize, and simulate a shared-cash portfolio with overlaps. Does **not** sum trade returns.

**Phase 1:** supplied Return % for closed-trade outcomes; event equity curve.  
**Phase 2:** optional `MarketDataProvider` (Yahoo via yfinance + disk cache, or synthetic fixtures) for daily MTM equity, open-position valuation, risk metrics (vol/Sharpe/Sortino/max DD), and optional benchmark.  
**Phase 3:** parameter-grid scenarios, SQLite run history, and TradingView ingestion hooks (interfaces only — no live webhook).

## Return % convention

Source files store percent points (e.g. 11.95 means +11.95%).  
Internally the engine converts to decimal: 11.95 -> 0.1195.  
Both Entry and Exit rows must agree on Return %.

## Filename convention

```
{strategy}_{exchange}_{symbol}_{YYYY-MM-DD}.csv
```

Example: TripleStrategy_AMEX_IVV_2026-09-05.csv

## Install

```bash
cd MSBT
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# or: pip install -r requirements.txt && pip install -e .
```

## Run tests

```bash
source .venv/bin/activate
pytest -q
```

Tests use **synthetic** market data only (no live Yahoo in CI).

## Run Streamlit UI

```bash
source .venv/bin/activate
streamlit run app.py
```

Cloud-safe entrypoint: `app.py` inserts `src/` on `sys.path` and calls `main()`.  
Toggle **Enable Phase 2 market data** for Yahoo daily MTM / risk / benchmark. Cache writes to `data/market_cache/` (or `/tmp/msbt_market_cache` if not writable).

After a single simulation, **Save this run** stores it in `data/runs.sqlite` (UTC timestamps; gitignored). The sidebar lists past runs (open, edit name/tags/notes, delete, compare 2+).

**Parameter grid:** in Configuration, pick buy % values and entry-priority modes (optional fee/slippage/max-symbol axes), then **Run parameter grid**. Each cell is an independent `run_simulation` on the same normalized trades; the base trade list is not mutated. Comparison table includes params, final equity, return %, CAGR when the equity span allows it, max DD only when daily MTM exists (otherwise NA), accepted, and rejected. Download CSV from the UI. If market data is enabled, one provider object is reused across cells so Yahoo hits the disk cache.

## Engine API

```python
from msbt import SimulationConfig, run_simulation
from msbt.importers import import_trade_files
from msbt.validation import validate_trades
from msbt.market_data import SyntheticMarketDataProvider, YahooFinanceProvider

trades, issues = import_trade_files(["fixtures/TripleStrategy_AMEX_SAMPLE_2026-09-05.csv"])
trades, report = validate_trades(trades, prior_issues=issues, files_uploaded=1)
config = SimulationConfig(
    initial_capital=100_000,
    buy_pct_of_equity=0.05,
    max_pct_per_symbol=0.20,
    allow_leverage=False,
    open_valuation_mode="market",       # market | source | cost_basis
    yahoo_price_adjustment="adjusted",  # adjusted | unadjusted
    risk_free_rate=0.0,
    sortino_target=0.0,
    discrepancy_threshold_pct=0.02,
)
# Phase 1 behavior:
result = run_simulation(trades=[t for t in trades if t.is_valid_for_sim], config=config, market_data=None)
# Phase 2:
# provider = YahooFinanceProvider()  # or SyntheticMarketDataProvider(...)
# result = run_simulation(..., market_data=provider)
print(result.performance_metrics.final_equity)
print(result.daily_mtm_timeseries[:3])
print(result.risk_metrics)
```


## Phase 3 — grid, history, TradingView

```python
from msbt.analytics.grid import run_parameter_grid
from msbt.analytics.history import save_run, list_runs, get_run, compare_runs
from msbt.ingestion.tradingview import import_symbol_list, map_strategy_names

rows = run_parameter_grid(
    trades,
    config,
    {
        "buy_pct_of_equity": [0.05, 0.10, 0.15, 0.20],
        "entry_priority": ["highest_avg_trade_return", "highest_win_rate"],
    },
    market_data=None,  # or a shared YahooFinanceProvider / synthetic provider
)
run_id = save_run(result, name="baseline", tags="demo", notes="5% buy")
print(list_runs())
print(compare_runs([run_id]))

# Future hook only — not a live TradingView webhook:
print(import_symbol_list("NASDAQ:AAPL, NYSE:IBM"))
print(map_strategy_names("Triple", {"triple": "TripleStrategy"}))
```

History DB: `data/runs.sqlite` (or `/tmp/msbt_runs.sqlite` if the project data dir is not writable).

## Locked economics

- `equity_for_sizing` is frozen after exits and before entries on timestamp T
- `gross_exit_value = cost_basis * (1 + return_pct)` — Yahoo never rewrites closed P&L
- Fees/slippage are cash costs; they do not rewrite `return_pct`
- Same-timestamp: exits before entries
- Look-ahead ban on priority metrics (`sell_date < decision_timestamp`); future market prices do not affect selection
- Cash rounded to 2 decimals after each movement
- `allow_leverage=False`
- Open positions: Phase 1 at cost basis; Phase 2 `market` / `source` / `cost_basis`
- Daily MTM: last available Close on/before date (no interpolation/forward-fill fabrication); gaps reported

Note: Average trade return is not equal to portfolio return (sizing, overlap, cash constraints, rejects).

## Layout

```
MSBT/
  src/msbt/
    importers/ validation/ models/ simulation/
    market_data/   # Phase 2 providers + cache
    analytics/     # export, risk, benchmark, grid, history
    ingestion/     # TradingView hooks (interfaces only)
    streamlit_app/
  tests/
  fixtures/
  data/market_cache/
  data/runs.sqlite     # created at runtime; gitignored
  app.py
  pyproject.toml
  requirements.txt
```

## Caveats

- **Acceptance Example 2 prompt text** claims GOOG is partially filled for 41800 after two 31350 fills at buy 30%. That is arithmetically inconsistent. The engine follows locked sizing rules (see Phase 1 tests).
- Return % in CSVs is percent points; conversion to decimal is documented in code and README.
