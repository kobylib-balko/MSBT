"""Optional benchmark series (Yahoo ticker or uploaded Date/Value)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Sequence

from msbt.analytics.risk import RiskMetrics, compute_risk_metrics


@dataclass
class BenchmarkMetrics:
    ticker_or_name: str
    cumulative_return: Optional[float] = None
    cagr: Optional[float] = None
    ann_volatility: Optional[float] = None
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    max_drawdown: Optional[float] = None
    excess_return_vs_portfolio: Optional[float] = None
    aligned_dates: list[date] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    unavailable_reasons: dict[str, str] = field(default_factory=dict)
    risk: Optional[RiskMetrics] = None

    def to_dict(self) -> dict:
        return {
            "Benchmark": self.ticker_or_name,
            "Cumulative return": self.cumulative_return,
            "CAGR": self.cagr,
            "Ann. volatility": self.ann_volatility,
            "Sharpe": self.sharpe,
            "Sortino": self.sortino,
            "Max drawdown": self.max_drawdown,
            "Excess vs portfolio": self.excess_return_vs_portfolio,
            "Unavailable": self.unavailable_reasons or None,
        }


def align_series(
    dates_a: Sequence[date],
    values_a: Sequence[float],
    dates_b: Sequence[date],
    values_b: Sequence[float],
) -> tuple[list[date], list[float], list[float]]:
    """Inner-join on dates."""
    map_a = dict(zip(dates_a, values_a))
    map_b = dict(zip(dates_b, values_b))
    common = sorted(set(map_a) & set(map_b))
    return common, [map_a[d] for d in common], [map_b[d] for d in common]


def _cagr(start_val: float, end_val: float, start_d: date, end_d: date) -> Optional[float]:
    if start_val <= 0 or end_val <= 0:
        return None
    days = (end_d - start_d).days
    if days <= 0:
        return None
    years = days / 365.25
    if years <= 0:
        return None
    return (end_val / start_val) ** (1.0 / years) - 1.0


def compute_benchmark_metrics(
    dates: Sequence[date],
    values: Sequence[float],
    *,
    name: str,
    portfolio_dates: Optional[Sequence[date]] = None,
    portfolio_equity: Optional[Sequence[float]] = None,
    risk_free_rate: float = 0.0,
    sortino_target: float = 0.0,
) -> BenchmarkMetrics:
    bm = BenchmarkMetrics(ticker_or_name=name)
    if len(dates) < 2 or len(values) < 2:
        bm.unavailable_reasons["all"] = "insufficient benchmark points"
        return bm

    bm.aligned_dates = list(dates)
    bm.values = list(values)
    bm.cumulative_return = (values[-1] / values[0]) - 1.0 if values[0] else None
    bm.cagr = _cagr(values[0], values[-1], dates[0], dates[-1])

    risk = compute_risk_metrics(
        values,
        dates=dates,
        risk_free_rate=risk_free_rate,
        sortino_target=sortino_target,
    )
    bm.risk = risk
    bm.ann_volatility = risk.ann_volatility
    bm.sharpe = risk.sharpe
    bm.sortino = risk.sortino
    bm.max_drawdown = risk.max_drawdown
    bm.unavailable_reasons.update(risk.unavailable_reasons)

    if portfolio_dates is not None and portfolio_equity is not None:
        common, pe, be = align_series(portfolio_dates, portfolio_equity, dates, values)
        if len(common) >= 2 and pe[0] and be[0]:
            port_ret = pe[-1] / pe[0] - 1.0
            bench_ret = be[-1] / be[0] - 1.0
            bm.excess_return_vs_portfolio = port_ret - bench_ret
            bm.aligned_dates = common
        else:
            bm.unavailable_reasons["excess"] = "insufficient overlapping dates"

    return bm
