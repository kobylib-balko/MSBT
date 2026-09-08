"""Phase 1 event-based portfolio simulation engine.

Authoritative event order at timestamp T:
  1. Exits with sell_date == T (trade_id ascending)
  2. Entries with buy_date == T (priority sorted)
  3. Snapshot end-of-event portfolio state

Sizing equity rule (locked): equity_for_sizing is portfolio equity immediately
AFTER all exits at T and BEFORE any new entries. Opening a position does not
change total equity (cash → invested cost basis).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from msbt.models.config import SimulationConfig
from msbt.models.results import (
    ACCEPTED,
    PARTIAL_INSUFFICIENT,
    REJECTED_BELOW_MIN,
    REJECTED_DATE_FILTER,
    REJECTED_INSUFFICIENT,
    REJECTED_MAX_OPEN,
    REJECTED_MAX_SYMBOL,
    REJECTED_SAME_SYMBOL,
    EquitySnapshot,
    PerformanceSummary,
    SimulationResult,
    TradeResult,
)
from msbt.models.trade import Trade, TradeStatus
from msbt.simulation.accounting import compute_entry, compute_exit, money
from msbt.simulation.priority import rank_entries


@dataclass
class OpenPosition:
    trade: Trade
    cost_basis: float
    qty: float
    entry_fee: float
    entry_slippage: float
    allocated_capital: float
    status_label: str  # Accepted or Partial...


@dataclass
class PortfolioState:
    cash: float
    opens: dict[str, OpenPosition] = field(default_factory=dict)  # trade_id -> pos

    def invested_cost_basis(self) -> float:
        return money(sum(p.cost_basis for p in self.opens.values()))

    def equity(self) -> float:
        # Phase 1 primary: open positions at cost basis
        return money(self.cash + self.invested_cost_basis())

    def open_notional_for_symbol(self, symbol: str) -> float:
        return money(
            sum(p.cost_basis for p in self.opens.values() if p.trade.symbol == symbol)
        )

    def symbol_has_open(self, symbol: str) -> bool:
        return any(p.trade.symbol == symbol for p in self.opens.values())


def _date_filtered(trade: Trade, config: SimulationConfig) -> bool:
    """True if trade should be rejected due to date range filter."""
    if config.start_date and trade.buy_date < config.start_date:
        return True
    if config.end_date and trade.buy_date > config.end_date:
        return True
    return False


def _make_rejected(
    trade: Trade,
    reason: str,
) -> TradeResult:
    return TradeResult(
        trade_id=trade.trade_id,
        symbol=trade.symbol,
        strategy=trade.strategy,
        exchange=trade.exchange,
        source_file=trade.source_file,
        source_trade_number=trade.source_trade_number,
        buy_date=trade.buy_date,
        sell_date=trade.sell_date,
        gross_return=trade.return_pct if trade.is_completed else None,
        net_return=None,
        allocated_capital=0.0,
        qty=0.0,
        cost_basis=0.0,
        gross_pnl=None,
        net_pnl=None,
        entry_fee=0.0,
        exit_fee=0.0,
        entry_slippage=0.0,
        exit_slippage=0.0,
        status=reason,
        rejection_reason=reason,
        duration_bars=trade.duration_bars,
        is_open=trade.is_open,
        buy_price=trade.buy_price,
        sell_price=trade.sell_price,
    )


def run_simulation(
    trades: list[Trade],
    config: SimulationConfig,
    market_data: Any = None,  # Phase 2 only; ignored in Phase 1
) -> SimulationResult:
    """Run Phase 1 event-based capital simulation.

    market_data is accepted for API compatibility; Phase 1 does not use it.
    """
    _ = market_data  # Phase 1: unused

    sim_id = str(uuid.uuid4())
    eligible = [t for t in trades if t.is_valid_for_sim]
    # History for priority: all completed trades (even if not executed by us);
    # look-ahead ban uses sell_date < decision_ts against source closed trades.
    # Spec: metrics from already closed trades of that symbol before decision time.
    # We use the *source* completed trades' sell dates / returns for priority scoring
    # (historical performance of the strategy on that symbol), not only accepted ones.
    priority_history = [t for t in eligible if t.is_completed]

    state = PortfolioState(cash=money(config.initial_capital))
    trade_results: list[TradeResult] = []
    rejected: list[TradeResult] = []
    snapshots: list[EquitySnapshot] = []
    # Track which trade_ids we've already processed as entries
    pending_by_buy: dict[date, list[Trade]] = {}
    exits_by_sell: dict[date, list[str]] = {}  # sell_date -> open position trade_ids scheduled

    # Pre-reject date-filtered
    to_simulate: list[Trade] = []
    for t in eligible:
        if _date_filtered(t, config):
            r = _make_rejected(t, REJECTED_DATE_FILTER)
            trade_results.append(r)
            rejected.append(r)
        else:
            to_simulate.append(t)

    for t in to_simulate:
        pending_by_buy.setdefault(t.buy_date, []).append(t)

    # Event timestamps
    event_dates: set[date] = set()
    for t in to_simulate:
        event_dates.add(t.buy_date)
        if t.sell_date is not None:
            event_dates.add(t.sell_date)

    # Also need exit events for positions we open — handled dynamically via opens
    # We'll iterate sorted unique dates that appear; when we open, we register exit.

    initial = money(config.initial_capital)
    # Snapshot at start if we want — only on event dates per Phase 1

    # Map trade_id -> Trade for exit lookup
    trade_map = {t.trade_id: t for t in to_simulate}

    # Collect all dates that could be events; recompute as we go for exits of accepted
    all_dates = sorted(event_dates)

    # Use a pointer approach: process known dates; when new exit dates appear they're
    # already in event_dates from sell_date of candidates. Only accepted positions exit.
    # All candidate sell dates are already in event_dates.

    realized_pnl_total = 0.0
    closed_executed: list[TradeResult] = []

    for T in all_dates:
        # ---- 1. EXITS (trade_id ascending) ----
        exit_ids = sorted(
            tid for tid, pos in state.opens.items() if pos.trade.sell_date == T
        )
        for tid in exit_ids:
            pos = state.opens.pop(tid)
            t = pos.trade
            if t.is_open or t.sell_date is None:
                # Shouldn't exit open positions in Phase 1 event loop via sell_date
                continue
            ex = compute_exit(
                cost_basis=pos.cost_basis,
                return_pct=t.return_pct,
                exit_fee_pct=config.exit_fee_pct,
                slippage_pct=config.slippage_pct,
                entry_fee=pos.entry_fee,
                entry_slippage=pos.entry_slippage,
            )
            state.cash = money(state.cash + ex.cash_inflow)
            realized_pnl_total = money(realized_pnl_total + ex.net_pnl)
            result = TradeResult(
                trade_id=t.trade_id,
                symbol=t.symbol,
                strategy=t.strategy,
                exchange=t.exchange,
                source_file=t.source_file,
                source_trade_number=t.source_trade_number,
                buy_date=t.buy_date,
                sell_date=t.sell_date,
                gross_return=t.return_pct,
                net_return=ex.net_return,
                allocated_capital=pos.allocated_capital,
                qty=pos.qty,
                cost_basis=pos.cost_basis,
                gross_pnl=ex.gross_pnl,
                net_pnl=ex.net_pnl,
                entry_fee=pos.entry_fee,
                exit_fee=ex.exit_fee,
                entry_slippage=pos.entry_slippage,
                exit_slippage=ex.exit_slippage,
                status=pos.status_label,
                rejection_reason=None,
                duration_bars=t.duration_bars,
                is_open=False,
                buy_price=t.buy_price,
                sell_price=t.sell_price,
            )
            trade_results.append(result)
            closed_executed.append(result)

        # ---- equity_for_sizing frozen after exits, before entries ----
        equity_for_sizing = state.equity()

        # ---- 2. ENTRIES (priority sorted) ----
        candidates = pending_by_buy.get(T, [])
        ranked = rank_entries(candidates, T, priority_history, config)

        for t in ranked:
            # Same-symbol policy
            if (
                not config.allow_multiple_positions_same_symbol
                and state.symbol_has_open(t.symbol)
            ):
                r = _make_rejected(t, REJECTED_SAME_SYMBOL)
                trade_results.append(r)
                rejected.append(r)
                continue

            # Max open positions
            if (
                config.max_open_positions is not None
                and len(state.opens) >= config.max_open_positions
            ):
                r = _make_rejected(t, REJECTED_MAX_OPEN)
                trade_results.append(r)
                rejected.append(r)
                continue

            target = money(equity_for_sizing * config.buy_pct_of_equity)
            remaining_symbol_cap = money(
                max(
                    0.0,
                    money(config.max_pct_per_symbol * equity_for_sizing)
                    - state.open_notional_for_symbol(t.symbol),
                )
            )

            # Available cash for allocation (no leverage): cover capital + entry costs
            cost_factor = 1.0 + config.entry_fee_pct + config.slippage_pct
            if cost_factor <= 0:
                cost_factor = 1.0
            max_by_cash = money(state.cash / cost_factor) if cost_factor else state.cash

            allowed = money(min(target, max_by_cash, remaining_symbol_cap))

            if allowed <= 0:
                # Prefer symbol-cap reason when that constraint binds
                if remaining_symbol_cap <= 0:
                    reason = REJECTED_MAX_SYMBOL
                else:
                    reason = REJECTED_INSUFFICIENT
                r = _make_rejected(t, reason)
                trade_results.append(r)
                rejected.append(r)
                continue

            if allowed < target:
                if not config.allow_partial_fills:
                    if remaining_symbol_cap < target and remaining_symbol_cap <= max_by_cash:
                        reason = REJECTED_MAX_SYMBOL
                    else:
                        reason = REJECTED_INSUFFICIENT
                    r = _make_rejected(t, reason)
                    trade_results.append(r)
                    rejected.append(r)
                    continue
                if allowed < config.min_allocation_cash:
                    r = _make_rejected(t, REJECTED_BELOW_MIN)
                    trade_results.append(r)
                    rejected.append(r)
                    continue
                status_label = PARTIAL_INSUFFICIENT
            else:
                if allowed < config.min_allocation_cash:
                    r = _make_rejected(t, REJECTED_BELOW_MIN)
                    trade_results.append(r)
                    rejected.append(r)
                    continue
                status_label = ACCEPTED

            entry = compute_entry(
                allocated_capital=allowed,
                buy_price=t.buy_price,
                entry_fee_pct=config.entry_fee_pct,
                slippage_pct=config.slippage_pct,
                fractional_shares=config.fractional_shares,
            )
            if entry.cash_outflow > state.cash + 1e-9:
                # Safety: never allocate more than available cash
                r = _make_rejected(t, REJECTED_INSUFFICIENT)
                trade_results.append(r)
                rejected.append(r)
                continue

            state.cash = money(state.cash - entry.cash_outflow)
            state.opens[t.trade_id] = OpenPosition(
                trade=t,
                cost_basis=entry.allocated_capital,
                qty=entry.qty,
                entry_fee=entry.entry_fee,
                entry_slippage=entry.entry_slippage,
                allocated_capital=entry.allocated_capital,
                status_label=status_label,
            )

            # Open trades: record as still open (no exit yet)
            if t.is_open:
                trade_results.append(
                    TradeResult(
                        trade_id=t.trade_id,
                        symbol=t.symbol,
                        strategy=t.strategy,
                        exchange=t.exchange,
                        source_file=t.source_file,
                        source_trade_number=t.source_trade_number,
                        buy_date=t.buy_date,
                        sell_date=None,
                        gross_return=None,
                        net_return=None,
                        allocated_capital=entry.allocated_capital,
                        qty=entry.qty,
                        cost_basis=entry.allocated_capital,
                        gross_pnl=None,
                        net_pnl=None,
                        entry_fee=entry.entry_fee,
                        exit_fee=0.0,
                        entry_slippage=entry.entry_slippage,
                        exit_slippage=0.0,
                        status=status_label,
                        rejection_reason=None,
                        duration_bars=t.duration_bars,
                        is_open=True,
                        buy_price=t.buy_price,
                        sell_price=None,
                    )
                )

        # ---- 3. SNAPSHOT ----
        eq = state.equity()
        invested = state.invested_cost_basis()
        snapshots.append(
            EquitySnapshot(
                timestamp=T,
                equity=eq,
                cash=state.cash,
                invested_cost_basis=invested,
                open_positions=len(state.opens),
                exposure_pct=(invested / eq) if eq else 0.0,
                cumulative_return=(eq - initial) / initial if initial else 0.0,
            )
        )

    # Any remaining open positions already recorded if they were open-status;
    # completed trades still open in sim (shouldn't happen if sell_date in events)
    # Record accepted opens that are completed-but-not-yet-exited? All sell dates in events.
    # Positions still open at end: completed trades shouldn't remain; open-status may.
    for tid, pos in list(state.opens.items()):
        t = pos.trade
        if not t.is_open:
            # Still open because sell is after last event? shouldn't happen
            # Record as open at end with cost basis
            trade_results.append(
                TradeResult(
                    trade_id=t.trade_id,
                    symbol=t.symbol,
                    strategy=t.strategy,
                    exchange=t.exchange,
                    source_file=t.source_file,
                    source_trade_number=t.source_trade_number,
                    buy_date=t.buy_date,
                    sell_date=t.sell_date,
                    gross_return=t.return_pct,
                    net_return=None,
                    allocated_capital=pos.allocated_capital,
                    qty=pos.qty,
                    cost_basis=pos.cost_basis,
                    gross_pnl=None,
                    net_pnl=None,
                    entry_fee=pos.entry_fee,
                    exit_fee=0.0,
                    entry_slippage=pos.entry_slippage,
                    exit_slippage=0.0,
                    status=pos.status_label,
                    rejection_reason=None,
                    duration_bars=t.duration_bars,
                    is_open=True,
                    buy_price=t.buy_price,
                    sell_price=t.sell_price,
                )
            )

    final_equity = state.equity()
    perf = _build_performance(
        config=config,
        final_equity=final_equity,
        realized_pnl=realized_pnl_total,
        trade_results=trade_results,
        closed_executed=closed_executed,
        state=state,
    )
    symbol_metrics = _breakdown_by(trade_results, key="symbol")
    source_metrics = _breakdown_by(trade_results, key="source_file")

    return SimulationResult(
        portfolio_timeseries=snapshots,
        trade_results=trade_results,
        performance_metrics=perf,
        source_metrics=source_metrics,
        symbol_metrics=symbol_metrics,
        rejected_trades=rejected,
        simulation_config=config.to_dict(),
        validation_report=[],
        simulation_id=sim_id,
    )


def _build_performance(
    config: SimulationConfig,
    final_equity: float,
    realized_pnl: float,
    trade_results: list[TradeResult],
    closed_executed: list[TradeResult],
    state: PortfolioState,
) -> PerformanceSummary:
    initial = money(config.initial_capital)
    accepted = sum(1 for r in trade_results if r.status == ACCEPTED)
    partial = sum(1 for r in trade_results if r.status == PARTIAL_INSUFFICIENT)
    rej = [r for r in trade_results if r.rejection_reason]
    by_reason: dict[str, int] = {}
    for r in rej:
        by_reason[r.rejection_reason or ""] = by_reason.get(r.rejection_reason or "", 0) + 1

    nets = [r.net_pnl for r in closed_executed if r.net_pnl is not None]
    wins = [n for n in nets if n > 0]
    losses = [n for n in nets if n < 0]
    win_rate = (len(wins) / len(nets)) if nets else None
    avg_win = (sum(wins) / len(wins)) if wins else None
    avg_loss = (sum(losses) / len(losses)) if losses else None
    gross_wins = sum(wins) if wins else 0.0
    gross_losses = abs(sum(losses)) if losses else 0.0
    if gross_losses > 0:
        profit_factor = gross_wins / gross_losses
    elif gross_wins > 0:
        profit_factor = float("inf")
    else:
        profit_factor = None

    bars = [r.duration_bars for r in closed_executed if r.duration_bars is not None]
    avg_bars = (sum(bars) / len(bars)) if bars else None
    med_bars = None
    if bars:
        sb = sorted(bars)
        mid = len(sb) // 2
        med_bars = float(sb[mid]) if len(sb) % 2 else (sb[mid - 1] + sb[mid]) / 2.0

    # Phase 1 cost-basis: unrealized = 0; equity including opens = cash + cost bases
    equity_incl = final_equity
    # Realized-only: cash only would understate; spec says report realized-only excluding opens.
    # Interpreting: equity if opens marked at cost equals cash+cost; realized-only view =
    # initial + realized_pnl (cash that would exist if opens returned cost basis).
    equity_realized_only = money(state.cash + state.invested_cost_basis())
    # Actually realized-only typically means: don't count open positions' value beyond
    # returning capital — which at cost basis is the same. Separate: cash + 0 unrealized.
    # We'll set equity_realized_only = cash + cost bases (same) and unrealized=0.

    return PerformanceSummary(
        initial_capital=initial,
        final_equity=final_equity,
        absolute_pnl=money(final_equity - initial),
        total_return_pct=(final_equity - initial) / initial if initial else 0.0,
        realized_pnl=money(realized_pnl),
        unrealized_pnl=0.0,
        equity_including_opens=equity_incl,
        equity_realized_only=equity_realized_only,
        candidates=len(trade_results),
        accepted=accepted,
        partial=partial,
        rejected=len(rej),
        rejected_by_reason=by_reason,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        avg_holding_bars=avg_bars,
        median_holding_bars=med_bars,
    )


def _breakdown_by(results: list[TradeResult], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[TradeResult]] = {}
    for r in results:
        k = getattr(r, key) if key != "source_file" else r.source_file
        if key == "symbol":
            k = r.symbol
        groups.setdefault(str(k), []).append(r)

    rows = []
    for name, items in sorted(groups.items()):
        closed = [r for r in items if r.net_pnl is not None and not r.is_open]
        nets = [r.net_pnl for r in closed if r.net_pnl is not None]
        wins = sum(1 for n in nets if n > 0)
        bars = [r.duration_bars for r in closed if r.duration_bars is not None]
        rows.append(
            {
                key: name,
                "trades": len(items),
                "closed": len(closed),
                "win_rate": (wins / len(nets)) if nets else None,
                "total_net_pnl": money(sum(nets)) if nets else 0.0,
                "avg_return": (
                    sum(r.gross_return for r in closed if r.gross_return is not None)
                    / len(closed)
                    if closed
                    else None
                ),
                "avg_holding_bars": (sum(bars) / len(bars)) if bars else None,
            }
        )
    return rows
