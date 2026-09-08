"""Disk cache keyed by (yahoo_ticker, date, adjustment_policy)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from msbt.market_data.base import OHLCVBar


def _safe_ticker(ticker: str) -> str:
    return ticker.replace("/", "_").replace("\\", "_").replace(":", "_")


class DiskOHLCVCache:
    """Per-ticker JSON/CSV-ish cache under a root directory.

    Layout: {root}/{ticker}/{adjustment_policy}.jsonl
    Each line: {"date": "YYYY-MM-DD", "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, yahoo_ticker: str, adjustment_policy: str) -> Path:
        d = self.root / _safe_ticker(yahoo_ticker)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{adjustment_policy}.jsonl"

    def load(
        self,
        yahoo_ticker: str,
        adjustment_policy: str,
        start: date,
        end: date,
    ) -> dict[date, OHLCVBar]:
        path = self._path(yahoo_ticker, adjustment_policy)
        out: dict[date, OHLCVBar] = {}
        if not path.exists():
            return out
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                d = date.fromisoformat(row["date"])
                if start <= d <= end:
                    out[d] = OHLCVBar(
                        bar_date=d,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row.get("volume", 0.0)),
                    )
        return out

    def load_all(
        self, yahoo_ticker: str, adjustment_policy: str
    ) -> dict[date, OHLCVBar]:
        path = self._path(yahoo_ticker, adjustment_policy)
        out: dict[date, OHLCVBar] = {}
        if not path.exists():
            return out
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                d = date.fromisoformat(row["date"])
                out[d] = OHLCVBar(
                    bar_date=d,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0.0)),
                )
        return out

    def store(
        self,
        yahoo_ticker: str,
        adjustment_policy: str,
        bars: Iterable[OHLCVBar],
    ) -> None:
        """Merge bars into cache; no duplicate dates (newer overwrite)."""
        existing = self.load_all(yahoo_ticker, adjustment_policy)
        for b in bars:
            existing[b.bar_date] = b
        path = self._path(yahoo_ticker, adjustment_policy)
        meta_path = path.with_suffix(".meta.json")
        with path.open("w", encoding="utf-8") as f:
            for d in sorted(existing.keys()):
                b = existing[d]
                f.write(
                    json.dumps(
                        {
                            "date": b.bar_date.isoformat(),
                            "open": b.open,
                            "high": b.high,
                            "low": b.low,
                            "close": b.close,
                            "volume": b.volume,
                        }
                    )
                    + "\n"
                )
        meta_path.write_text(
            json.dumps(
                {
                    "yahoo_ticker": yahoo_ticker,
                    "adjustment_policy": adjustment_policy,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "n_bars": len(existing),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def has_range(
        self,
        yahoo_ticker: str,
        adjustment_policy: str,
        start: date,
        end: date,
        expected_dates: Optional[set[date]] = None,
    ) -> bool:
        """True if every date in expected_dates (or all weekdays in range) is cached."""
        cached = self.load(yahoo_ticker, adjustment_policy, start, end)
        if expected_dates is not None:
            return expected_dates.issubset(cached.keys())
        return bool(cached)
