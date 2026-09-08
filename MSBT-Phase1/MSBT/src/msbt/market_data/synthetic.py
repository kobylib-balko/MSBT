"""Deterministic synthetic market data for tests (no live Yahoo)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from msbt.market_data.base import MarketDataProvider, OHLCVBar, PriceSeriesResult


def _daterange_weekdays(start: date, end: date) -> list[date]:
    out: list[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            out.append(d)
        d += timedelta(days=1)
    return out


class SyntheticMarketDataProvider(MarketDataProvider):
    """In-memory deterministic OHLCV keyed by yahoo_ticker.

    prices: dict[ticker, dict[date, close]] (OHLC derived from close if only close given)
    or dict[ticker, dict[date, OHLCVBar]]
    """

    def __init__(
        self,
        prices: Optional[dict[str, dict[date, float | OHLCVBar]]] = None,
        *,
        weekday_calendar: bool = True,
    ) -> None:
        self._raw = prices or {}
        self.weekday_calendar = weekday_calendar
        self.fetch_count: dict[str, int] = {}  # for cache-reuse style assertions

    def set_close(self, ticker: str, d: date, close: float) -> None:
        self._raw.setdefault(ticker, {})[d] = close

    def get_daily_ohlcv(
        self,
        yahoo_ticker: str,
        start: date,
        end: date,
        *,
        adjustment_policy: str = "adjusted",
    ) -> PriceSeriesResult:
        self.fetch_count[yahoo_ticker] = self.fetch_count.get(yahoo_ticker, 0) + 1
        series = self._raw.get(yahoo_ticker, {})
        if self.weekday_calendar:
            requested = _daterange_weekdays(start, end)
        else:
            requested = []
            d = start
            while d <= end:
                requested.append(d)
                d += timedelta(days=1)

        bars: list[OHLCVBar] = []
        missing: list[date] = []
        for d in requested:
            if d not in series:
                missing.append(d)
                continue
            val = series[d]
            if isinstance(val, OHLCVBar):
                bars.append(val)
            else:
                c = float(val)
                bars.append(
                    OHLCVBar(bar_date=d, open=c, high=c, low=c, close=c, volume=0.0)
                )

        # Also include any provided bars in range even if not weekday (when calendar off)
        for d, val in sorted(series.items()):
            if start <= d <= end and all(b.bar_date != d for b in bars):
                if isinstance(val, OHLCVBar):
                    bars.append(val)
                else:
                    c = float(val)
                    bars.append(
                        OHLCVBar(bar_date=d, open=c, high=c, low=c, close=c, volume=0.0)
                    )
        bars.sort(key=lambda b: b.bar_date)

        return PriceSeriesResult(
            yahoo_ticker=yahoo_ticker,
            start=start,
            end=end,
            adjustment_policy=adjustment_policy,
            bars=bars,
            missing_dates=missing,
            trading_dates_requested=requested,
            provider_name="synthetic",
            download_timestamp=datetime.now(timezone.utc).isoformat(),
        )
