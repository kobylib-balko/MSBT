"""Phase 4 — Robustness analytics.

Uses only already-normalized / uploaded trades (and an optional Phase 2
``market_data`` provider if the caller already enabled one). No new external
rate or Yahoo requirements.

Split rule (documented for callers / UI)
---------------------------------------
Train = trades with ``buy_date < cutoff``.
Test  = trades with ``buy_date >= cutoff``.
Sell dates are ignored for membership (a Train buy may exit after the cutoff).

Robust score (grid selection on Train only)
------------------------------------------
Prefer ``CAGR / |max_drawdown|`` when both CAGR and a non-zero max DD are
available (daily MTM). Otherwise ``total_return_pct - |max_dd|`` when DD
exists, else ``total_return_pct``. Higher is better. Ties break by higher
return, then higher CAGR, then lower scenario index.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Any, Mapping, Optional, Sequence

from dateutil.relativedelta import relativedelta

from msbt.analytics.grid import (
    config_variant,
    extract_run_metrics,
    normalize_trades,
    run_parameter_grid,
)
from msbt.models.config import SimulationConfig
from msbt.simulation.engine import run_simulation

# ---------------------------------------------------------------------------
# Dates / splits
# ---------------------------------------------------------------------------


def buy_dates(trades: Sequence[Any]) -> list[date]:
    """Sorted unique buy dates from eligible trades."""
    dates = sorted(
        {
            t.buy_date
            for t in normalize_trades(trades)
            if getattr(t, "buy_date", None) is not None
        }
    )
    return dates


def buy_date_span(trades: Sequence[Any]) -> Optional[tuple[date, date]]:
    """(min_buy, max_buy) or None if empty."""
    ds = buy_dates(trades)
    if not ds:
        return None
    return ds[0], ds[-1]


def median_buy_date(trades: Sequence[Any]) -> Optional[date]:
    """Median of unique buy dates (lower-middle for even counts)."""
    ds = buy_dates(trades)
    if not ds:
        return None
    return ds[(len(ds) - 1) // 2]


def cutoff_at_fraction(trades: Sequence[Any], fraction: float = 0.70) -> Optional[date]:
    """Cutoff so ~``fraction`` of unique buy dates fall strictly before it.

    Uses the chronological list of unique buy dates. For fraction=0.70, the
    cutoff is the date at index ``floor(n * 0.70)`` (Train gets the earlier
    ~70% of distinct buy days).
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    ds = buy_dates(trades)
    if not ds:
        return None
    if len(ds) == 1:
        return ds[0]
    idx = int(len(ds) * fraction)
    idx = min(max(idx, 1), len(ds) - 1)
    return ds[idx]


def default_cutoff(
    trades: Sequence[Any],
    mode: str = "median",
) -> Optional[date]:
    """Default Train/Test cutoff.

    ``mode``:
      - ``median`` — median of unique buy dates
      - ``70_30`` / ``time_70_30`` — ~70% of unique buy dates in Train
    """
    m = (mode or "median").strip().lower()
    if m in ("70_30", "70/30", "time_70_30", "time"):
        return cutoff_at_fraction(trades, 0.70)
    return median_buy_date(trades)


def split_by_buy_date(
    trades: Sequence[Any],
    cutoff: date,
) -> tuple[list[Any], list[Any]]:
    """Split eligible trades by buy_date vs cutoff.

    Train: buy_date < cutoff.
    Test:  buy_date >= cutoff.
    Input order preserved within each side. Does not mutate ``trades``.
    """
    normalized = normalize_trades(trades)
    train = [t for t in normalized if t.buy_date < cutoff]
    test = [t for t in normalized if t.buy_date >= cutoff]
    return train, test


# ---------------------------------------------------------------------------
# Metrics / scoring
# ---------------------------------------------------------------------------


def metrics_from_result(result: Any) -> dict[str, Any]:
    """Compact metrics including Sharpe when risk/performance expose it."""
    m = extract_run_metrics(result)
    sharpe = None
    sortino = None
    rm = getattr(result, "risk_metrics", None)
    pm = result.performance_metrics
    if rm is not None:
        sharpe = getattr(rm, "sharpe", None)
        sortino = getattr(rm, "sortino", None)
    if sharpe is None:
        sharpe = getattr(pm, "sharpe", None)
    if sortino is None:
        sortino = getattr(pm, "sortino", None)
    m["sharpe"] = sharpe
    m["sortino"] = sortino
    if m.get("cagr") is None and getattr(pm, "cagr", None) is not None:
        m["cagr"] = pm.cagr
    return m


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and x == x and abs(x) != float("inf")


def robust_score(metrics: Mapping[str, Any]) -> float:
    """Scalar score for ranking Train grid cells (higher is better).

    Prefer CAGR / |max_drawdown| when both are available and |DD| > 1e-12.
    Else total_return with DD penalty: return_pct - |max_dd|.
    Else total_return_pct alone.
    """
    ret = metrics.get("return_pct")
    cagr = metrics.get("cagr")
    dd = metrics.get("max_drawdown")
    abs_dd = abs(dd) if _finite(dd) else None

    if _finite(cagr) and abs_dd is not None and abs_dd > 1e-12:
        return float(cagr) / abs_dd
    base = float(ret) if _finite(ret) else float("-inf")
    if abs_dd is not None:
        return base - abs_dd
    return base


def select_best_grid_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pick the best Train grid cell by :func:`robust_score` (deterministic)."""
    if not rows:
        raise ValueError("no grid rows to select from")
    ranked = sorted(
        enumerate(rows),
        key=lambda iv: (
            robust_score(iv[1]),
            iv[1].get("return_pct") if _finite(iv[1].get("return_pct")) else float("-inf"),
            iv[1].get("cagr") if _finite(iv[1].get("cagr")) else float("-inf"),
            -iv[0],
        ),
        reverse=True,
    )
    best = dict(ranked[0][1])
    best["robust_score"] = robust_score(best)
    return best


def degradation(train_m: Mapping[str, Any], test_m: Mapping[str, Any]) -> dict[str, Any]:
    """Test vs Train deltas / ratios.

    - return / CAGR / Sharpe: Test − Train
    - max_drawdown: Test/Train using absolute values when both non-zero
    """
    out: dict[str, Any] = {}
    for key in ("return_pct", "cagr", "sharpe"):
        a, b = train_m.get(key), test_m.get(key)
        out[f"delta_{key}"] = (b - a) if _finite(a) and _finite(b) else None
    td, dd = train_m.get("max_drawdown"), test_m.get("max_drawdown")
    if _finite(td) and _finite(dd) and abs(td) > 1e-12:
        out["dd_ratio_test_over_train"] = abs(dd) / abs(td)
    else:
        out["dd_ratio_test_over_train"] = None
    return out


def _run_metrics(
    trades: Sequence[Any],
    config: SimulationConfig,
    market_data: Any = None,
) -> tuple[Any, dict[str, Any]]:
    result = run_simulation(list(trades), config, market_data=market_data)
    return result, metrics_from_result(result)


def _params_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(row.get("params") or {})
    for k in ("buy_pct_of_equity", "entry_priority", "max_pct_per_symbol"):
        if k not in params and row.get(k) is not None:
            params[k] = row[k]
    return params


# ---------------------------------------------------------------------------
# 1. Train vs Test
# ---------------------------------------------------------------------------


def run_train_vs_test(
    trades: Sequence[Any],
    config: SimulationConfig,
    cutoff: Optional[date] = None,
    market_data: Any = None,
    cutoff_mode: str = "median",
) -> dict[str, Any]:
    """Run the same base config on Train and Test slices side-by-side."""
    cutoff = cutoff or default_cutoff(trades, mode=cutoff_mode)
    if cutoff is None:
        raise ValueError("no trades with buy dates to split")
    train, test = split_by_buy_date(trades, cutoff)
    if not train:
        raise ValueError(f"Train set empty for cutoff {cutoff.isoformat()}")
    if not test:
        raise ValueError(f"Test set empty for cutoff {cutoff.isoformat()}")

    _, train_m = _run_metrics(train, config, market_data)
    _, test_m = _run_metrics(test, config, market_data)
    return {
        "cutoff": cutoff.isoformat(),
        "split_rule": "Train: buy_date < cutoff; Test: buy_date >= cutoff",
        "n_train": len(train),
        "n_test": len(test),
        "train": train_m,
        "test": test_m,
        "degradation": degradation(train_m, test_m),
        "side_by_side": [
            {"slice": "Train", "n_trades": len(train), **train_m},
            {"slice": "Test", "n_trades": len(test), **test_m},
        ],
    }


# ---------------------------------------------------------------------------
# 2. Locked OOS after grid
# ---------------------------------------------------------------------------


def default_small_grid(base: SimulationConfig) -> dict[str, list[Any]]:
    """Small configurable default grid for OOS / walk-forward."""
    buy = sorted({round(base.buy_pct_of_equity, 6), 0.05, 0.10})
    buy = [b for b in buy if 0 < b <= 1]
    return {
        "buy_pct_of_equity": buy,
        "entry_priority": [base.entry_priority],
    }


def run_locked_oos(
    trades: Sequence[Any],
    base_config: SimulationConfig,
    grid: Optional[Mapping[str, Sequence[Any]]] = None,
    cutoff: Optional[date] = None,
    market_data: Any = None,
    cutoff_mode: str = "median",
) -> dict[str, Any]:
    """Optimize on Train only; evaluate the winning params once on Test.

    Selection never sees Test metrics.
    """
    cutoff = cutoff or default_cutoff(trades, mode=cutoff_mode)
    if cutoff is None:
        raise ValueError("no trades with buy dates to split")
    train, test = split_by_buy_date(trades, cutoff)
    if not train:
        raise ValueError(f"Train set empty for cutoff {cutoff.isoformat()}")
    if not test:
        raise ValueError(f"Test set empty for cutoff {cutoff.isoformat()}")

    grid = dict(grid) if grid else default_small_grid(base_config)
    train_rows = run_parameter_grid(train, base_config, grid, market_data=market_data)
    best = select_best_grid_row(train_rows)
    params = _params_from_row(best)

    locked = config_variant(base_config, params)
    _, train_locked = _run_metrics(train, locked, market_data)
    _, test_locked = _run_metrics(test, locked, market_data)

    return {
        "cutoff": cutoff.isoformat(),
        "split_rule": "Train: buy_date < cutoff; Test: buy_date >= cutoff",
        "n_train": len(train),
        "n_test": len(test),
        "grid_axes": {k: list(v) for k, v in grid.items()},
        "selection_score": (
            "CAGR/|max_dd| if DD else return - |dd| else return (Train only)"
        ),
        "best_params": params,
        "best_train_row": best,
        "train_grid": train_rows,
        "train_metrics": train_locked,
        "test_metrics": test_locked,
        "degradation": degradation(train_locked, test_locked),
        "selection_used_test": False,
    }


# ---------------------------------------------------------------------------
# 3. Walk-forward
# ---------------------------------------------------------------------------


def _add_years(d: date, years: float) -> date:
    """Add years (fractional via months)."""
    whole = int(years)
    frac = years - whole
    out = d + relativedelta(years=whole)
    if abs(frac) > 1e-12:
        months = int(round(frac * 12))
        out = out + relativedelta(months=months)
    return out


def walk_forward_windows(
    min_buy: date,
    max_buy: date,
    train_years: float = 4.0,
    test_years: float = 1.0,
    step_years: float = 1.0,
    mode: str = "expanding",
) -> list[dict[str, date]]:
    """Build walk-forward windows from the buy-date span.

    Each window: ``train_start <= buy < train_end`` (train),
                 ``train_end <= buy < test_end`` (test).

    ``mode``:
      - ``rolling`` — fixed-length train of ``train_years`` ending at train_end
      - ``expanding`` — train from ``min_buy`` to train_end (grows)
    """
    if train_years <= 0 or test_years <= 0 or step_years <= 0:
        raise ValueError("train_years, test_years, and step_years must be positive")
    mode_l = (mode or "expanding").strip().lower()
    if mode_l not in ("expanding", "rolling"):
        raise ValueError("mode must be 'expanding' or 'rolling'")
    if min_buy > max_buy:
        return []

    windows: list[dict[str, date]] = []
    train_end = _add_years(min_buy, train_years)
    safety = 0
    while train_end <= max_buy and safety < 500:
        safety += 1
        test_end = _add_years(train_end, test_years)
        if mode_l == "rolling":
            train_start = _add_years(train_end, -train_years)
            if train_start < min_buy:
                train_start = min_buy
        else:
            train_start = min_buy
        if train_start < train_end and train_end <= max_buy:
            windows.append(
                {
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": train_end,
                    "test_end": test_end,
                }
            )
        next_end = _add_years(train_end, step_years)
        if next_end <= train_end:
            break
        train_end = next_end
    return windows


def _slice_buys(
    trades: Sequence[Any],
    start: date,
    end: date,
) -> list[Any]:
    """Trades with start <= buy_date < end."""
    return [t for t in normalize_trades(trades) if start <= t.buy_date < end]


def run_walk_forward(
    trades: Sequence[Any],
    base_config: SimulationConfig,
    train_years: float = 4.0,
    test_years: float = 1.0,
    step_years: float = 1.0,
    mode: str = "expanding",
    grid: Optional[Mapping[str, Sequence[Any]]] = None,
    market_data: Any = None,
) -> dict[str, Any]:
    """Walk-forward: small-grid optimize on each train slice; lock on test."""
    span = buy_date_span(trades)
    if span is None:
        raise ValueError("no trades with buy dates")
    min_buy, max_buy = span
    windows = walk_forward_windows(
        min_buy, max_buy, train_years, test_years, step_years, mode=mode
    )
    grid = dict(grid) if grid else default_small_grid(base_config)

    rows: list[dict[str, Any]] = []
    test_returns: list[float] = []
    for i, w in enumerate(windows, start=1):
        train = _slice_buys(trades, w["train_start"], w["train_end"])
        test = _slice_buys(trades, w["test_start"], w["test_end"])
        if not train or not test:
            rows.append(
                {
                    "window": i,
                    "train_start": w["train_start"].isoformat(),
                    "train_end": w["train_end"].isoformat(),
                    "test_start": w["test_start"].isoformat(),
                    "test_end": w["test_end"].isoformat(),
                    "n_train": len(train),
                    "n_test": len(test),
                    "skipped": True,
                    "reason": "empty train or test slice",
                }
            )
            continue
        train_rows = run_parameter_grid(
            train, base_config, grid, market_data=market_data
        )
        best = select_best_grid_row(train_rows)
        params = _params_from_row(best)
        locked = config_variant(base_config, params)
        _, train_m = _run_metrics(train, locked, market_data)
        _, test_m = _run_metrics(test, locked, market_data)
        if _finite(test_m.get("return_pct")):
            test_returns.append(float(test_m["return_pct"]))
        rows.append(
            {
                "window": i,
                "train_start": w["train_start"].isoformat(),
                "train_end": w["train_end"].isoformat(),
                "test_start": w["test_start"].isoformat(),
                "test_end": w["test_end"].isoformat(),
                "n_train": len(train),
                "n_test": len(test),
                "skipped": False,
                "best_params": params,
                "train_return_pct": train_m.get("return_pct"),
                "train_cagr": train_m.get("cagr"),
                "train_max_drawdown": train_m.get("max_drawdown"),
                "test_return_pct": test_m.get("return_pct"),
                "test_cagr": test_m.get("cagr"),
                "test_max_drawdown": test_m.get("max_drawdown"),
                "test_sharpe": test_m.get("sharpe"),
                "degradation": degradation(train_m, test_m),
            }
        )

    avg_ret = sum(test_returns) / len(test_returns) if test_returns else None
    compound = None
    if test_returns:
        acc = 1.0
        for r in test_returns:
            acc *= 1.0 + r
        compound = acc - 1.0

    return {
        "mode": mode,
        "train_years": train_years,
        "test_years": test_years,
        "step_years": step_years,
        "min_buy": min_buy.isoformat(),
        "max_buy": max_buy.isoformat(),
        "n_windows": len(windows),
        "n_evaluated": sum(1 for r in rows if not r.get("skipped")),
        "grid_axes": {k: list(v) for k, v in grid.items()},
        "windows": rows,
        "oos_summary": {
            "avg_test_return_pct": avg_ret,
            "compound_test_return_pct": compound,
            "n_test_windows": len(test_returns),
            "note": (
                "Average is arithmetic mean of per-window Test total returns. "
                "Compound multiplies (1+r) across windows (stitched OOS returns, "
                "not a continuous equity curve)."
            ),
        },
    }


# ---------------------------------------------------------------------------
# 4. Stress pack
# ---------------------------------------------------------------------------


def symbol_net_pnl_contribution(
    trades: Sequence[Any],
    config: SimulationConfig,
    market_data: Any = None,
) -> dict[str, float]:
    """Sum executed closed-trade net PnL by symbol (deterministic)."""
    result = run_simulation(
        list(normalize_trades(trades)), config, market_data=market_data
    )
    contrib: dict[str, float] = {}
    for tr in result.trade_results:
        if tr.rejection_reason:
            continue
        if tr.net_pnl is None:
            continue
        contrib[tr.symbol] = contrib.get(tr.symbol, 0.0) + float(tr.net_pnl)
    return contrib


def top_symbol_by_net_pnl(
    trades: Sequence[Any],
    config: SimulationConfig,
    market_data: Any = None,
) -> Optional[str]:
    """Symbol with highest net PnL; lexicographically first on ties."""
    contrib = symbol_net_pnl_contribution(trades, config, market_data)
    if not contrib:
        return None
    best_val = max(contrib.values())
    return sorted(s for s, v in contrib.items() if abs(v - best_val) < 1e-9)[0]


def drop_symbol(trades: Sequence[Any], symbol: str) -> list[Any]:
    """Rebuild trade list without ``symbol`` (exact match)."""
    return [t for t in normalize_trades(trades) if t.symbol != symbol]


def _stress_variant_configs(
    base: SimulationConfig,
) -> list[tuple[str, SimulationConfig]]:
    variants: list[tuple[str, SimulationConfig]] = [
        ("baseline", base),
        (
            "costs_x3",
            replace(
                base,
                entry_fee_pct=base.entry_fee_pct * 3.0,
                exit_fee_pct=base.exit_fee_pct * 3.0,
                slippage_pct=base.slippage_pct * 3.0,
                ticker_map=dict(base.ticker_map),
            ),
        ),
        (
            "partial_fills_off",
            replace(
                base,
                allow_partial_fills=False,
                ticker_map=dict(base.ticker_map),
            ),
        ),
    ]
    if base.cash_earn_mode != "none":
        variants.append(
            (
                "cash_earn_none",
                replace(
                    base,
                    cash_earn_mode="none",
                    ticker_map=dict(base.ticker_map),
                ),
            )
        )
    return variants


def run_stress_pack(
    trades: Sequence[Any],
    base_config: SimulationConfig,
    *,
    cutoff: Optional[date] = None,
    use_test_set: bool = True,
    market_data: Any = None,
    drop_symbol_from: str = "train",
    cutoff_mode: str = "median",
) -> dict[str, Any]:
    """One-click stress variants vs baseline on a fixed trade set / params.

    Parameters
    ----------
    use_test_set:
        If True and a cutoff can be formed, evaluate stresses on the Test
        slice; symbol-to-drop is chosen from Train (or full sample per
        ``drop_symbol_from``).
    drop_symbol_from:
        ``train`` | ``full`` — which sample ranks symbols by net PnL.
    """
    normalized = normalize_trades(trades)
    eval_trades = list(normalized)
    rank_trades = list(normalized)
    cutoff_iso = None
    if use_test_set:
        c = cutoff or default_cutoff(trades, mode=cutoff_mode)
        if c is not None:
            train, test = split_by_buy_date(trades, c)
            if train and test:
                cutoff_iso = c.isoformat()
                eval_trades = test
                rank_trades = train if drop_symbol_from == "train" else list(normalized)

    contrib = symbol_net_pnl_contribution(rank_trades, base_config, market_data)
    top_sym = None
    if contrib:
        best_val = max(contrib.values())
        top_sym = sorted(s for s, v in contrib.items() if abs(v - best_val) < 1e-9)[0]

    rows: list[dict[str, Any]] = []
    baseline_m: Optional[dict[str, Any]] = None
    for name, cfg in _stress_variant_configs(base_config):
        _, m = _run_metrics(eval_trades, cfg, market_data)
        row: dict[str, Any] = {"variant": name, "dropped_symbol": None, **m}
        if name == "baseline":
            baseline_m = m
        rows.append(row)

    if top_sym is not None:
        dropped = drop_symbol(eval_trades, top_sym)
        if dropped:
            _, m = _run_metrics(dropped, base_config, market_data)
            rows.append(
                {
                    "variant": f"drop_top_symbol:{top_sym}",
                    "dropped_symbol": top_sym,
                    **m,
                }
            )

    for row in rows:
        if baseline_m is None or row["variant"] == "baseline":
            row["delta_return_pct"] = 0.0 if row["variant"] == "baseline" else None
            row["delta_cagr"] = 0.0 if row["variant"] == "baseline" else None
            row["worse_return"] = False
            continue
        br = baseline_m.get("return_pct")
        rr = row.get("return_pct")
        row["delta_return_pct"] = (rr - br) if _finite(br) and _finite(rr) else None
        bc, rc = baseline_m.get("cagr"), row.get("cagr")
        row["delta_cagr"] = (rc - bc) if _finite(bc) and _finite(rc) else None
        row["worse_return"] = bool(
            _finite(row.get("delta_return_pct")) and row["delta_return_pct"] < 0
        )

    return {
        "cutoff": cutoff_iso,
        "eval_set": "test" if cutoff_iso and use_test_set else "full",
        "n_eval_trades": len(eval_trades),
        "drop_symbol_from": drop_symbol_from,
        "top_symbol": top_sym,
        "symbol_pnl": dict(sorted(contrib.items())) if contrib else {},
        "rows": rows,
        "note": (
            "Variants share baseline params except the named stress. "
            "drop_top_symbol removes the highest net-PnL contributor "
            f"(ranked on {drop_symbol_from}) from the eval trade list."
        ),
    }
