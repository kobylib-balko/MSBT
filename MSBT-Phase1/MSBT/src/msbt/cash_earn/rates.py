"""Synthetic risk-free rates and total-return index for cash earn.

Index construction (locked):
  I starts at 100 on the calendar day before the first earn day.
  I_t = I_{t-1} * (1 + rate_t) ** (1/365)
  where rate_t is the annualized rate for day t (decimal).

Cash earn application (locked):
  After exits AND entries on simulation day T (EOD cash after entries),
  cash *= I_T / I_{prev}, where prev is the previous simulation day
  (or the day before the first sim day). Intervening calendar days are
  compounded via the index so weekends earn when rates are daily.

No look-ahead: rate for day t uses only rates with date <= t (last known
on-or-before). Future rates never affect prior days.

Production: supply a CSV via SimulationConfig.cash_earn_rates_path or
place rates under data/synthetic_rf_rates.csv with columns:
  date, annualized_rate
(annualized_rate as decimal, e.g. 0.05 for 5%).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import pandas as pd

from msbt.simulation.accounting import money

PathLike = Union[str, Path]


# Small embedded series spanning common test / sample windows (annualized decimals).
_EMBEDDED_RATES: list[tuple[str, float]] = [
    ("2019-01-01", 0.024),
    ("2019-07-01", 0.022),
    ("2020-01-01", 0.015),
    ("2020-04-01", 0.001),
    ("2020-07-01", 0.001),
    ("2020-10-01", 0.001),
    ("2021-01-01", 0.001),
    ("2021-07-01", 0.0005),
    ("2022-01-01", 0.005),
    ("2022-07-01", 0.020),
    ("2023-01-01", 0.045),
    ("2023-07-01", 0.053),
    ("2024-01-01", 0.053),
    ("2024-07-01", 0.052),
    ("2025-01-01", 0.043),
    ("2025-07-01", 0.042),
    ("2026-01-01", 0.040),
]


def default_embedded_rates() -> dict[date, float]:
    return {date.fromisoformat(d): r for d, r in _EMBEDDED_RATES}


def load_rates_csv(path: PathLike) -> dict[date, float]:
    """Load date,annualized_rate CSV. Rate may be decimal or percent points if > 1."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = {c.strip().lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("as_of") or list(df.columns)[0]
    rate_col = (
        cols.get("annualized_rate")
        or cols.get("rate")
        or cols.get("rf")
        or list(df.columns)[1]
    )
    out: dict[date, float] = {}
    for _, row in df.iterrows():
        raw = str(row[date_col]).strip()[:10]
        d = date.fromisoformat(raw)
        rate = float(row[rate_col])
        if rate > 1.0:  # treat as percent points
            rate = rate / 100.0
        out[d] = rate
    return out


def build_total_return_index(
    rates: Mapping[date, float],
    start: date,
    end: date,
    *,
    initial_level: float = 100.0,
) -> dict[date, float]:
    """Build daily index levels from start-1 through end (inclusive).

    Uses last-known rate on-or-before each day (no look-ahead).
    Days before any known rate use rate 0.0.
    """
    if end < start:
        return {}
    sorted_rate_dates = sorted(rates.keys())
    last_rate = 0.0
    rate_i = 0
    # Seed day before start
    seed = start - timedelta(days=1)
    index: dict[date, float] = {seed: initial_level}
    prev_level = initial_level
    d = start
    while d <= end:
        while rate_i < len(sorted_rate_dates) and sorted_rate_dates[rate_i] <= d:
            last_rate = rates[sorted_rate_dates[rate_i]]
            rate_i += 1
        prev_level = prev_level * ((1.0 + last_rate) ** (1.0 / 365.0))
        index[d] = prev_level
        d += timedelta(days=1)
    return index


def apply_cash_earn_ratio(cash: float, ratio: float) -> tuple[float, float]:
    """Return (new_cash, interest_pnl) after applying index ratio to cash."""
    if cash <= 0 or ratio is None or ratio == 1.0:
        return money(cash), 0.0
    before = money(cash)
    after = money(before * ratio)
    return after, money(after - before)


class SyntheticRateProvider:
    """Provides RF rates and a total-return index for cash earn."""

    def __init__(
        self,
        rates: Optional[Mapping[date, float]] = None,
        *,
        rates_path: Optional[PathLike] = None,
    ) -> None:
        if rates is not None:
            self.rates = dict(rates)
            self.source = "provided"
        elif rates_path is not None:
            self.rates = load_rates_csv(rates_path)
            self.source = str(rates_path)
        else:
            self.rates = default_embedded_rates()
            self.source = "embedded"

    @classmethod
    def from_config_path_or_default(
        cls,
        rates_path: Optional[str] = None,
        search_roots: Optional[Sequence[PathLike]] = None,
    ) -> "SyntheticRateProvider":
        if rates_path:
            p = Path(rates_path)
            if p.is_file():
                return cls(rates_path=p)
        roots = list(search_roots or [])
        for root in roots:
            candidate = Path(root) / "synthetic_rf_rates.csv"
            if candidate.is_file():
                return cls(rates_path=candidate)
        return cls()

    def index_for_range(self, start: date, end: date) -> dict[date, float]:
        return build_total_return_index(self.rates, start, end)

    def ratio(self, index: Mapping[date, float], day: date, prev: date) -> float:
        """I_day / I_prev. Returns 1.0 if either level missing."""
        a = index.get(day)
        b = index.get(prev)
        if a is None or b is None or b == 0:
            return 1.0
        return a / b
