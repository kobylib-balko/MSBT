"""Yahoo Finance MarketDataProvider via yfinance + disk cache."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from msbt.market_data.base import MarketDataProvider, OHLCVBar, PriceSeriesResult
from msbt.market_data.cache import DiskOHLCVCache


def default_cache_root() -> Path:
    """Prefer project data/market_cache; on Cloud fall back to /tmp."""
    # Project-relative from this file: .../MSBT/data/market_cache
    here = Path(__file__).resolve()
    project = here.parents[3]  # src/msbt/market_data -> MSBT
    candidate = project / "data" / "market_cache"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return candidate
    except OSError:
        tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "msbt_market_cache"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp


class YahooFinanceProvider(MarketDataProvider):
    """Fetches daily OHLCV from Yahoo via yfinance; caches to disk.

    adjustment_policy:
      - adjusted: use Auto Adjust / Adj Close semantics (yfinance auto_adjust=True)
      - unadjusted: raw Close (auto_adjust=False)
    """

    def __init__(
        self,
        cache_root: Optional[Path | str] = None,
        *,
        ticker_map: Optional[dict[str, str]] = None,
    ) -> None:
        self.cache = DiskOHLCVCache(cache_root or default_cache_root())
        self.ticker_map = ticker_map or {}

    def resolve_ticker(
        self,
        symbol: str,
        exchange: str = "",
        ticker_map: Optional[dict[str, str]] = None,
    ) -> str:
        return super().resolve_ticker(
            symbol, exchange, ticker_map if ticker_map is not None else self.ticker_map
        )

    def get_daily_ohlcv(
        self,
        yahoo_ticker: str,
        start: date,
        end: date,
        *,
        adjustment_policy: str = "adjusted",
    ) -> PriceSeriesResult:
        if adjustment_policy not in ("adjusted", "unadjusted"):
            raise ValueError(
                f"yahoo_price_adjustment must be 'adjusted' or 'unadjusted', got {adjustment_policy!r}"
            )

        cached = self.cache.load(yahoo_ticker, adjustment_policy, start, end)
        # Determine if we need a network fetch: any weekday in range missing from cache
        need_fetch = self._needs_fetch(cached, start, end)

        notes: list[str] = []
        download_ts: Optional[str] = None
        if need_fetch:
            fetched = self._download(yahoo_ticker, start, end, adjustment_policy)
            download_ts = datetime.now(timezone.utc).isoformat()
            if fetched:
                self.cache.store(yahoo_ticker, adjustment_policy, fetched)
                for b in fetched:
                    if start <= b.bar_date <= end:
                        cached[b.bar_date] = b
                notes.append(f"downloaded {len(fetched)} bars from yfinance")
            else:
                notes.append("yfinance returned no bars")
        else:
            notes.append("served from disk cache")

        bars = [cached[d] for d in sorted(cached.keys()) if start <= d <= end]
        # Report gaps relative to weekdays that appeared in neither cache nor download
        # We only know true trading calendar from what Yahoo returned historically;
        # report weekdays in range with no bar as missing (honest gap list).
        missing = self._weekday_gaps(bars, start, end)

        return PriceSeriesResult(
            yahoo_ticker=yahoo_ticker,
            start=start,
            end=end,
            adjustment_policy=adjustment_policy,
            bars=bars,
            missing_dates=missing,
            trading_dates_requested=[],
            provider_name="yahoo_finance",
            download_timestamp=download_ts,
            notes=notes,
        )

    def _needs_fetch(self, cached: dict[date, OHLCVBar], start: date, end: date) -> bool:
        if not cached:
            return True
        # If span of cache doesn't cover request endpoints (with small slack), refetch
        keys = sorted(cached.keys())
        if keys[0] > start or keys[-1] < end:
            return True
        return False

    def _download(
        self,
        yahoo_ticker: str,
        start: date,
        end: date,
        adjustment_policy: str,
    ) -> list[OHLCVBar]:
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance is required for YahooFinanceProvider. "
                "Install with: pip install yfinance"
            ) from e

        auto_adjust = adjustment_policy == "adjusted"
        # yfinance end is exclusive; add one day
        end_excl = end + timedelta(days=1)
        ticker = yf.Ticker(yahoo_ticker)
        df = ticker.history(
            start=start.isoformat(),
            end=end_excl.isoformat(),
            auto_adjust=auto_adjust,
            actions=False,
        )
        bars: list[OHLCVBar] = []
        if df is None or df.empty:
            return bars
        for idx, row in df.iterrows():
            # idx may be Timestamp with tz
            if hasattr(idx, "date"):
                d = idx.date()
            else:
                d = date.fromisoformat(str(idx)[:10])
            bars.append(
                OHLCVBar(
                    bar_date=d,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row.get("Volume", 0.0) or 0.0),
                )
            )
        return bars

    @staticmethod
    def _weekday_gaps(bars: list[OHLCVBar], start: date, end: date) -> list[date]:
        have = {b.bar_date for b in bars}
        missing: list[date] = []
        d = start
        while d <= end:
            if d.weekday() < 5 and d not in have:
                missing.append(d)
            d += timedelta(days=1)
        return missing
