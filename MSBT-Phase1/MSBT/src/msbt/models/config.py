"""Simulation configuration (Phase 1 locked economics + Phase 2 market/risk)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Optional


class EntryPriority(str, Enum):
    HIGHEST_RETURN_PER_BAR = "highest_return_per_bar"
    HIGHEST_WIN_RATE = "highest_win_rate"
    HIGHEST_AVG_TRADE_RETURN = "highest_avg_trade_return"
    HISTORICAL_AVG_RETURN = "historical_avg_return"


PRIORITY_ALIASES = {
    EntryPriority.HISTORICAL_AVG_RETURN: EntryPriority.HIGHEST_AVG_TRADE_RETURN,
}

VALID_OPEN_VALUATION_MODES = frozenset({"cost_basis", "source", "market"})
VALID_YAHOO_ADJUSTMENT = frozenset({"adjusted", "unadjusted"})


@dataclass
class SimulationConfig:
    initial_capital: float = 100_000.0
    buy_pct_of_equity: float = 0.05
    max_pct_per_symbol: float = 0.20
    entry_fee_pct: float = 0.0
    exit_fee_pct: float = 0.0
    slippage_pct: float = 0.0
    allow_multiple_positions_same_symbol: bool = True
    allow_partial_fills: bool = True
    allow_leverage: bool = False
    timezone: str = "America/New_York"
    entry_priority: str = EntryPriority.HIGHEST_AVG_TRADE_RETURN.value
    min_history_trades: int = 1
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    max_open_positions: Optional[int] = None
    min_allocation_cash: float = 0.0
    fractional_shares: bool = True
    open_valuation_mode: str = "cost_basis"
    risk_free_rate: float = 0.0
    sortino_target: float = 0.0
    discrepancy_threshold_pct: float = 0.02
    yahoo_price_adjustment: str = "adjusted"
    valuation_date: Optional[date] = None
    ticker_map: dict[str, str] = field(default_factory=dict)
    benchmark_ticker: Optional[str] = None

    def __post_init__(self) -> None:
        if self.buy_pct_of_equity <= 0 or self.buy_pct_of_equity > 1:
            raise ValueError("buy_pct_of_equity must be in (0, 1]")
        if self.max_pct_per_symbol <= 0 or self.max_pct_per_symbol > 1:
            raise ValueError("max_pct_per_symbol must be in (0, 1]")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.allow_leverage:
            raise ValueError("Phase 1/2: allow_leverage must be False")
        if self.open_valuation_mode not in VALID_OPEN_VALUATION_MODES:
            raise ValueError(
                f"open_valuation_mode must be one of {sorted(VALID_OPEN_VALUATION_MODES)}"
            )
        if self.yahoo_price_adjustment not in VALID_YAHOO_ADJUSTMENT:
            raise ValueError(
                f"yahoo_price_adjustment must be one of {sorted(VALID_YAHOO_ADJUSTMENT)}"
            )

    def resolved_priority(self) -> EntryPriority:
        try:
            p = EntryPriority(self.entry_priority)
        except ValueError as e:
            raise ValueError(f"Unknown entry_priority: {self.entry_priority}") from e
        return PRIORITY_ALIASES.get(p, p)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("start_date", "end_date", "valuation_date"):
            if d.get(k) is not None:
                d[k] = d[k].isoformat()
        return d
