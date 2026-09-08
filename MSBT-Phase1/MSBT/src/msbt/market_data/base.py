"""MarketDataProvider abstraction (Phase 2).

Providers must not interpolate or forward-fill missing prices.
Simulation engine receives a provider — never calls Yahoo directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class OHLCVBar:
    """One trading-day OHLCV observation."""

    bar_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class PriceSeriesResult:
    """Result of a get_daily_ohlcv request, including gap reporting."""

    yahoo_ticker: str
    start: date
    end: date
    adjustment_policy: str
    bars: list[OHLCVBar]
    missing_dates: list[date] = field(default_factory=list)
    trading_dates_requested: list[date] = field(default_factory=list)
    provider_name: str = ""
    download_timestamp: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    def close_on(self, d: date) -> Optional[float]:
        """Exact Close on date d, or None if missing (no fabrication)."""
        for b in self.bars:
            if b.bar_date == d:
                return b.close
        return None

    def last_close_on_or_before(self, d: date) -> Optional[tuple[date, float]]:
        """Last available Close on or before d. Returns (bar_date, close) or None."""
        best: Optional[tuple[date, float]] = None
        for b in self.bars:
            if b.bar_date <= d:
                if best is None or b.bar_date > best[0]:
                    best = (b.bar_date, b.close)
        return best

    def closes_by_date(self) -> dict[date, float]:
        return {b.bar_date: b.close for b in self.bars}


class MarketDataProvider(ABC):
    """Interface for daily OHLCV retrieval."""

    @abstractmethod
    def get_daily_ohlcv(
        self,
        yahoo_ticker: str,
        start: date,
        end: date,
        *,
        adjustment_policy: str = "adjusted",
    ) -> PriceSeriesResult:
        """Return daily OHLCV for [start, end] inclusive.

        Must report missing dates/gaps. Must not interpolate or forward-fill.
        """

    def resolve_ticker(
        self,
        symbol: str,
        exchange: str = "",
        ticker_map: Optional[dict[str, str]] = None,
    ) -> str:
        """Map symbol (+ optional exchange) → yahoo ticker. Default: identity."""
        if ticker_map:
            key = f"{exchange}:{symbol}" if exchange else symbol
            if key in ticker_map:
                return ticker_map[key]
            if symbol in ticker_map:
                return ticker_map[symbol]
        return symbol
