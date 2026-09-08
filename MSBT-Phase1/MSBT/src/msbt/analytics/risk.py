"""Risk metrics from daily MTM equity returns only (Phase 2)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Sequence


TRADING_DAYS_PER_YEAR = 252


@dataclass
class RiskMetrics:
    daily_returns: list[float] = field(default_factory=list)
    ann_volatility: Optional[float] = None
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_drawdown_duration_days: Optional[int] = None
    unavailable_reasons: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "Ann. volatility": self.ann_volatility,
            "Sharpe": self.sharpe,
            "Sortino": self.sortino,
            "Max drawdown": self.max_drawdown,
            "Max DD duration (days)": self.max_drawdown_duration_days,
            "Unavailable": self.unavailable_reasons or None,
            "N daily returns": len(self.daily_returns),
        }


def daily_returns_from_equity(equity: Sequence[float]) -> list[float]:
    """Simple returns r_t = E_t / E_{t-1} - 1. Skips non-positive prior equity."""
    out: list[float] = []
    for i in range(1, len(equity)):
        prev, cur = equity[i - 1], equity[i]
        if prev is None or cur is None or prev <= 0:
            continue
        out.append((cur / prev) - 1.0)
    return out


def compute_drawdown(
    equity: Sequence[float],
    dates: Optional[Sequence[date]] = None,
) -> tuple[Optional[float], Optional[int]]:
    """Max drawdown (fraction) and duration in calendar/index days from peak to trough recovery.

    Duration: number of steps from peak to recovery (or to end if not recovered).
    Max DD duration is the longest such underwater period.
    """
    if len(equity) < 2:
        return None, None
    peak = equity[0]
    peak_i = 0
    max_dd: Optional[float] = 0.0
    max_dur = 0
    in_dd_start: Optional[int] = None

    for i, e in enumerate(equity):
        if e > peak:
            # recovered / new peak
            if in_dd_start is not None:
                dur = i - in_dd_start
                if dates is not None:
                    dur = (dates[i] - dates[in_dd_start]).days
                max_dur = max(max_dur, dur)
                in_dd_start = None
            peak = e
            peak_i = i
        else:
            if peak > 0:
                dd = (e - peak) / peak  # negative or zero
                if max_dd is None or dd < max_dd:
                    max_dd = dd
            if in_dd_start is None and e < peak:
                in_dd_start = peak_i

    if in_dd_start is not None:
        end_i = len(equity) - 1
        dur = end_i - in_dd_start
        if dates is not None:
            dur = (dates[end_i] - dates[in_dd_start]).days
        max_dur = max(max_dur, dur)

    return max_dd, max_dur if max_dur > 0 else (0 if max_dd is not None else None)


def compute_risk_metrics(
    equity: Sequence[float],
    *,
    dates: Optional[Sequence[date]] = None,
    risk_free_rate: float = 0.0,
    sortino_target: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> RiskMetrics:
    """Compute vol/Sharpe/Sortino/max DD from a daily equity series.

    risk_free_rate and sortino_target are annualized rates expressed as decimals;
    daily rf = risk_free_rate / periods_per_year.
    """
    m = RiskMetrics()
    if len(equity) < 2:
        m.unavailable_reasons["all"] = "insufficient equity points (need >= 2)"
        return m

    rets = daily_returns_from_equity(equity)
    m.daily_returns = rets
    if len(rets) < 1:
        m.unavailable_reasons["all"] = "no valid daily returns"
        return m

    mean_r = sum(rets) / len(rets)
    var = sum((r - mean_r) ** 2 for r in rets) / len(rets)  # population; deterministic
    std = math.sqrt(var)
    ann_vol = std * math.sqrt(periods_per_year)
    m.ann_volatility = ann_vol

    daily_rf = risk_free_rate / periods_per_year
    if std == 0.0 or ann_vol == 0.0:
        m.sharpe = None
        m.unavailable_reasons["sharpe"] = "undefined (volatility is zero)"
    else:
        # Annualized Sharpe: (mean_daily - daily_rf) / std_daily * sqrt(N)
        m.sharpe = ((mean_r - daily_rf) / std) * math.sqrt(periods_per_year)

    # Sortino: downside deviation vs target
    daily_target = sortino_target / periods_per_year
    downside = [(min(0.0, r - daily_target)) ** 2 for r in rets]
    down_var = sum(downside) / len(rets)
    down_std = math.sqrt(down_var)
    if down_std == 0.0:
        if mean_r > daily_target:
            m.sortino = None
            m.unavailable_reasons["sortino"] = "undefined (no downside deviation)"
        else:
            m.sortino = None
            m.unavailable_reasons["sortino"] = "undefined (zero downside deviation)"
    else:
        m.sortino = ((mean_r - daily_target) / down_std) * math.sqrt(periods_per_year)

    dd, dd_dur = compute_drawdown(equity, dates)
    m.max_drawdown = dd
    m.max_drawdown_duration_days = dd_dur
    return m
