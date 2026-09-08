"""Entry priority scoring with strict look-ahead ban.

HARD LOOK-AHEAD BAN: Never use a trade's own eventual return_pct (or any
future data) to decide whether to accept it. Metrics use only trades with
sell_date < decision_timestamp (strict).
"""

from __future__ import annotations

from datetime import date
from statistics import mean
from typing import Optional

from msbt.models.config import EntryPriority, SimulationConfig
from msbt.models.trade import Trade, TradeStatus


def _closed_before(
    history: list[Trade],
    symbol: str,
    decision_ts: date,
) -> list[Trade]:
    """Trades of `symbol` with sell_date strictly before decision_ts."""
    out = []
    for t in history:
        if t.symbol != symbol:
            continue
        if t.status != TradeStatus.COMPLETED:
            continue
        if t.sell_date is None:
            continue
        # Strict look-ahead ban: sell_date < decision_timestamp
        if t.sell_date < decision_ts:
            out.append(t)
    return out


def score_symbol(
    symbol: str,
    decision_ts: date,
    history: list[Trade],
    config: SimulationConfig,
) -> Optional[float]:
    """Return priority score for symbol at decision time, or None if warm-up/null."""
    closed = _closed_before(history, symbol, decision_ts)
    if len(closed) < config.min_history_trades:
        return None

    priority = config.resolved_priority()

    if priority == EntryPriority.HIGHEST_AVG_TRADE_RETURN:
        return mean(t.return_pct for t in closed)

    if priority == EntryPriority.HIGHEST_WIN_RATE:
        wins = sum(1 for t in closed if t.return_pct > 0)
        return wins / len(closed)

    if priority == EntryPriority.HIGHEST_RETURN_PER_BAR:
        ratios = []
        for t in closed:
            if t.duration_bars is None or t.duration_bars <= 0:
                continue
            ratios.append(t.return_pct / t.duration_bars)
        if not ratios:
            return None
        return mean(ratios)

    return None


def rank_entries(
    candidates: list[Trade],
    decision_ts: date,
    history: list[Trade],
    config: SimulationConfig,
) -> list[Trade]:
    """Sort entry candidates by priority (desc), then tie-breakers.

    Tie-breakers (in order): higher closed-trade count → lexicographic symbol → trade_id.
    Null scores rank after all valid scores.
    """

    def sort_key(t: Trade):
        sc = score_symbol(t.symbol, decision_ts, history, config)
        closed_count = len(_closed_before(history, t.symbol, decision_ts))
        # Higher score first: negate. Nulls last: use a flag.
        has_score = 0 if sc is None else 1
        score_val = sc if sc is not None else 0.0
        return (
            has_score,  # 1 before 0
            score_val,  # higher better
            closed_count,  # higher better
            # For ascending sort we negate the above via reverse; use tuple carefully
        )

    # Sort with custom: has_score desc, score desc, closed_count desc, symbol asc, trade_id asc
    decorated = []
    for t in candidates:
        sc = score_symbol(t.symbol, decision_ts, history, config)
        closed_count = len(_closed_before(history, t.symbol, decision_ts))
        decorated.append(
            (
                0 if sc is not None else 1,  # valid scores first
                -(sc if sc is not None else 0.0),
                -closed_count,
                t.symbol,
                t.trade_id,
                t,
            )
        )
    decorated.sort()
    return [d[-1] for d in decorated]
