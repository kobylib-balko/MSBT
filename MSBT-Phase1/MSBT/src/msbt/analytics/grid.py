"""Parameter-grid scenarios (Phase 3).

Each cell is an independent ``run_simulation`` over the same normalized trades.
The base trade dataset is not mutated. A shared ``market_data`` provider is
reused across cells so Yahoo (if enabled) hits the disk cache instead of
re-downloading per scenario.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from itertools import product
from typing import Any, Mapping, Optional, Sequence

from msbt.models.config import SimulationConfig
from msbt.models.results import SimulationResult
from msbt.simulation.engine import run_simulation

# Axes the grid may vary. Other SimulationConfig fields stay at base values.
GRID_FIELDS = frozenset(
    {
        "buy_pct_of_equity",
        "entry_priority",
        "max_pct_per_symbol",
        "entry_fee_pct",
        "exit_fee_pct",
        "slippage_pct",
        "initial_capital",
        "allow_partial_fills",
        "allow_multiple_positions_same_symbol",
        "open_valuation_mode",
        "min_history_trades",
        "risk_free_rate",
        "sortino_target",
    }
)

COMPARISON_COLUMNS = (
    "scenario",
    "params",
    "buy_pct_of_equity",
    "entry_priority",
    "final_equity",
    "return_pct",
    "cagr",
    "max_drawdown",
    "accepted",
    "rejected",
)


def cagr(
    start_val: float,
    end_val: float,
    start_d: date,
    end_d: date,
) -> Optional[float]:
    """Compound annual growth rate from two dated equity points, or None."""
    if start_val <= 0 or end_val <= 0:
        return None
    days = (end_d - start_d).days
    if days <= 0:
        return None
    years = days / 365.25
    if years <= 0:
        return None
    return (end_val / start_val) ** (1.0 / years) - 1.0


def normalize_trades(trades: Sequence[Any]) -> list[Any]:
    """Eligible trades only, in input order. Does not mutate ``trades``."""
    return [t for t in trades if getattr(t, "is_valid_for_sim", True)]


def expand_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of non-empty axes, in insertion order.

    Empty sequences are skipped. Raises if no combinations remain.
    """
    axes: list[tuple[str, list[Any]]] = []
    for key, values in grid.items():
        if key not in GRID_FIELDS:
            raise ValueError(f"Unsupported grid axis: {key}")
        vals = list(values)
        if not vals:
            continue
        axes.append((key, vals))
    if not axes:
        raise ValueError("grid has no values to vary")
    combos: list[dict[str, Any]] = []
    keys = [k for k, _ in axes]
    for combo in product(*[v for _, v in axes]):
        params: dict[str, Any] = {}
        for k, v in zip(keys, combo):
            params[k] = _coerce_param(k, v)
        combos.append(params)
    return combos


def _coerce_param(key: str, value: Any) -> Any:
    if hasattr(value, "value") and key == "entry_priority":
        return value.value
    return value


def config_variant(base: SimulationConfig, params: Mapping[str, Any]) -> SimulationConfig:
    """Independent config copy. Does not mutate ``base`` or its ticker_map."""
    kwargs = dict(params)
    kwargs["ticker_map"] = dict(base.ticker_map)
    return replace(base, **kwargs)


def _series_cagr(snapshots: Sequence[Any]) -> Optional[float]:
    if not snapshots or len(snapshots) < 2:
        return None
    first, last = snapshots[0], snapshots[-1]
    return cagr(first.equity, last.equity, first.timestamp, last.timestamp)


def extract_run_metrics(result: SimulationResult) -> dict[str, Any]:
    """Compact comparison metrics for one simulation.

    CAGR is computed from daily MTM when present, otherwise from the event
    equity curve (None if the span is not positive). Max drawdown comes from
    risk metrics only when a daily MTM series exists; otherwise NA (None).
    """
    pm = result.performance_metrics
    daily = result.daily_mtm_timeseries
    if daily:
        run_cagr = _series_cagr(daily)
        max_dd = None
        rm = result.risk_metrics
        if rm is not None and getattr(rm, "max_drawdown", None) is not None:
            max_dd = rm.max_drawdown
        elif pm.max_drawdown is not None:
            max_dd = pm.max_drawdown
    else:
        run_cagr = _series_cagr(result.portfolio_timeseries)
        max_dd = None

    cfg = result.simulation_config or {}
    return {
        "simulation_id": result.simulation_id,
        "buy_pct_of_equity": cfg.get("buy_pct_of_equity"),
        "entry_priority": cfg.get("entry_priority"),
        "max_pct_per_symbol": cfg.get("max_pct_per_symbol"),
        "entry_fee_pct": cfg.get("entry_fee_pct"),
        "exit_fee_pct": cfg.get("exit_fee_pct"),
        "slippage_pct": cfg.get("slippage_pct"),
        "initial_capital": pm.initial_capital,
        "final_equity": pm.final_equity,
        "return_pct": pm.total_return_pct,
        "cagr": run_cagr,
        "max_drawdown": max_dd,
        "accepted": pm.accepted,
        "rejected": pm.rejected,
        "partial": pm.partial,
        "candidates": pm.candidates,
        "has_daily_mtm": bool(daily),
    }


def scenario_row(
    index: int,
    params: Mapping[str, Any],
    result: SimulationResult,
) -> dict[str, Any]:
    metrics = extract_run_metrics(result)
    row: dict[str, Any] = {
        "scenario": index,
        "params": dict(params),
        **metrics,
    }
    # Varied axes win over config echo so the table matches the requested cell.
    for k, v in params.items():
        row[k] = v
    return row


def run_parameter_grid(
    trades: Sequence[Any],
    base_config: SimulationConfig,
    grid: Mapping[str, Sequence[Any]],
    market_data: Any = None,
) -> list[dict[str, Any]]:
    """Run independent simulations over a parameter grid.

    Parameters
    ----------
    trades:
        Normalized (or raw) trades. Filtered once for simulation eligibility.
        The input sequence and trade objects are not modified.
    base_config:
        Settings shared by every cell. Grid axes override copies of this.
    grid:
        Mapping of SimulationConfig field → values (cartesian product).
        At minimum, vary ``buy_pct_of_equity`` and/or ``entry_priority``.
    market_data:
        Optional provider reused for every cell (None keeps Phase 1 economics).

    Returns
    -------
    list[dict]
        One JSON-friendly comparison row per scenario. ``cagr`` / ``max_drawdown``
        are None when unavailable (max drawdown is NA without daily MTM).
    """
    normalized = normalize_trades(trades)
    if not normalized:
        raise ValueError("no valid trades to simulate")
    combos = expand_grid(grid)

    rows: list[dict[str, Any]] = []
    for i, params in enumerate(combos, start=1):
        variant = config_variant(base_config, params)
        result = run_simulation(normalized, variant, market_data=market_data)
        rows.append(scenario_row(i, params, result))
    return rows
