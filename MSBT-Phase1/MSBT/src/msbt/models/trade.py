"""Internal Trade model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


class TradeStatus(str, Enum):
    COMPLETED = "completed"
    OPEN = "open"
    INVALID = "invalid"


@dataclass
class Trade:
    trade_id: str
    source_trade_number: int
    symbol: str
    strategy: str
    exchange: str
    buy_date: date
    sell_date: Optional[date]  # None if open
    buy_price: float
    sell_price: Optional[float]
    return_pct: float  # decimal form: 0.1195 for 11.95%
    duration_bars: Optional[int]
    source_file: str
    status: TradeStatus = TradeStatus.COMPLETED
    # Reference-only fields from source
    source_return_pct_raw: Optional[float] = None  # as percent points from file
    source_net_pnl_usd: Optional[float] = None
    entry_signal: Optional[str] = None
    exit_signal: Optional[str] = None
    validation_errors: list[str] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.status == TradeStatus.OPEN

    @property
    def is_completed(self) -> bool:
        return self.status == TradeStatus.COMPLETED

    @property
    def is_valid_for_sim(self) -> bool:
        return self.status in (TradeStatus.COMPLETED, TradeStatus.OPEN)
