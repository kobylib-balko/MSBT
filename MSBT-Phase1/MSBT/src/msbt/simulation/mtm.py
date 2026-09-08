"""Daily mark-to-market equity curve (Phase 2).

Closed-trade economics remain Return-% based (Phase 1 locked).
Market prices never rewrite closed trade P&L or affect selection.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional

from msbt.analytics.benchmark import compute_benchmark_metrics
from msbt.analytics.risk import compute_risk_metrics
from msbt.models.config import SimulationConfig
from msbt.models.results import (
    ACCEPTED,
    PARTIAL_INSUFFICIENT,
    EquitySnapshot,
    SimulationResult,
    TradeResult,
)
from msbt.models.trade import Trade
from msbt.simulation.accounting import money


def _weekdays(start: date, end: date) -> list[date]:
    out: list[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _position_value_simple(
    *,
    cost_basis: float,
    qty: float,
    return_pct: float,
    mode: str,
    close: Optional[float],
) -> tuple[float, Optional[float], Optional[float], str]:
    if mode == "cost_basis":
        return cost_basis, None, None, "cost_basis"
    if mode == "source":
        val = money(cost_basis * (1.0 + return_pct))
        return val, None, return_pct, "source_return"
    if close is None:
        return cost_basis, None, None, "missing_price"
    mv = money(qty * close)
    mret = (mv / cost_basis - 1.0) if cost_basis else None
    return mv, close, mret, "market"


class _Pos:
    __slots__ = (
        "trade_id", "symbol", "exchange", "cost_basis", "qty",
        "buy_date", "sell_date", "return_pct", "source_net_pnl",
    )

    def __init__(self, r: TradeResult, t: Optional[Trade]):
        self.trade_id = r.trade_id
        self.symbol = r.symbol
        self.exchange = r.exchange
        self.cost_basis = r.cost_basis
        self.qty = r.qty
        self.buy_date = r.buy_date
        self.sell_date = r.sell_date if not r.is_open else None
        self.return_pct = (t.return_pct if t else (r.gross_return or 0.0)) or 0.0
        self.source_net_pnl = t.source_net_pnl_usd if t else None


def build_daily_mtm(
    *,
    trades: list[Trade],
    config: SimulationConfig,
    market_data: Any,
    event_result: SimulationResult,
) -> tuple[
    list[EquitySnapshot],
    Any,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    Optional[Any],
]:
    """Build daily MTM series from accepted fills; mark opens to Close.

    Closed trades keep Phase-1 Return-% economics untouched.
    """
    accepted = [
        r
        for r in event_result.trade_results
        if r.status in (ACCEPTED, PARTIAL_INSUFFICIENT) and r.allocated_capital > 0
    ]
    trade_map = {t.trade_id: t for t in trades}

    if not accepted and not event_result.portfolio_timeseries:
        return [], None, [], [], [], None

    start = min(
        (r.buy_date for r in accepted),
        default=event_result.portfolio_timeseries[0].timestamp
        if event_result.portfolio_timeseries
        else date.today(),
    )
    last_event = (
        event_result.portfolio_timeseries[-1].timestamp
        if event_result.portfolio_timeseries
        else start
    )
    valuation_end = config.valuation_date or last_event
    if valuation_end < start:
        valuation_end = last_event

    symbols_needed: dict[str, tuple[str, str]] = {}
    for r in accepted:
        t = trade_map.get(r.trade_id)
        exch = t.exchange if t else r.exchange
        yahoo = market_data.resolve_ticker(r.symbol, exch, config.ticker_map)
        symbols_needed[r.symbol] = (exch, yahoo)

    adj = config.yahoo_price_adjustment
    series_by_symbol: dict[str, Any] = {}
    md_report: list[dict[str, Any]] = []
    for sym, (exch, yahoo) in symbols_needed.items():
        try:
            res = market_data.get_daily_ohlcv(
                yahoo, start, valuation_end, adjustment_policy=adj
            )
        except Exception as e:  # noqa: BLE001
            md_report.append(
                {"symbol": sym, "yahoo_ticker": yahoo, "error": str(e), "missing_dates": []}
            )
            continue
        series_by_symbol[sym] = res
        md_report.append(
            {
                "symbol": sym,
                "yahoo_ticker": yahoo,
                "provider": res.provider_name,
                "adjustment_policy": res.adjustment_policy,
                "requested_start": start.isoformat(),
                "requested_end": valuation_end.isoformat(),
                "bars_received": len(res.bars),
                "missing_dates": [d.isoformat() for d in res.missing_dates[:50]],
                "missing_count": len(res.missing_dates),
                "notes": res.notes,
                "download_timestamp": res.download_timestamp,
            }
        )

    calendar = _weekdays(start, valuation_end)
    for s in event_result.portfolio_timeseries:
        if start <= s.timestamp <= valuation_end:
            calendar.append(s.timestamp)
    calendar = sorted(set(calendar))

    positions_by_id = {r.trade_id: _Pos(r, trade_map.get(r.trade_id)) for r in accepted}
    cash_events = {s.timestamp: s.cash for s in event_result.portfolio_timeseries}
    initial = money(config.initial_capital)
    mode = config.open_valuation_mode
    if mode not in ("market", "source", "cost_basis"):
        mode = "market"

    daily: list[EquitySnapshot] = []
    affected: list[str] = []

    def cash_on(d: date) -> float:
        prior = [ed for ed in cash_events if ed <= d]
        if not prior:
            return initial
        return cash_events[max(prior)]

    for d in calendar:
        active = [
            p
            for p in positions_by_id.values()
            if p.buy_date <= d and (p.sell_date is None or p.sell_date > d)
        ]
        cash = cash_on(d)
        invested = money(sum(p.cost_basis for p in active))
        mkt_sum = 0.0
        unreal = 0.0
        missing: list[str] = []

        for p in active:
            close = None
            ser = series_by_symbol.get(p.symbol)
            if ser is not None:
                found = ser.last_close_on_or_before(d)
                if found:
                    close = found[1]
                elif mode == "market":
                    missing.append(p.symbol)
                    affected.append(d.isoformat())
            elif mode == "market":
                missing.append(p.symbol)
                affected.append(d.isoformat())

            val, _, _, _note = _position_value_simple(
                cost_basis=p.cost_basis,
                qty=p.qty,
                return_pct=p.return_pct,
                mode=mode,
                close=close,
            )
            if mode == "market" and close is None:
                val = p.cost_basis
            mkt_sum += val
            unreal += val - p.cost_basis

        mkt_sum = money(mkt_sum)
        unreal = money(unreal)
        if mode == "cost_basis":
            equity = money(cash + invested)
            mkt_sum = invested
            unreal = 0.0
        else:
            equity = money(cash + mkt_sum)

        daily.append(
            EquitySnapshot(
                timestamp=d,
                equity=equity,
                cash=cash,
                invested_cost_basis=invested,
                open_positions=len(active),
                exposure_pct=(invested / equity) if equity else 0.0,
                cumulative_return=(equity - initial) / initial if initial else 0.0,
                market_value_opens=mkt_sum,
                unrealized_pnl=unreal,
                missing_price_symbols=tuple(sorted(set(missing))),
                is_daily_mtm=True,
            )
        )

    open_vals_final: list[dict[str, Any]] = []
    final_d = valuation_end
    for p in positions_by_id.values():
        if p.sell_date is not None and p.sell_date <= final_d:
            continue
        if p.buy_date > final_d:
            continue
        close = None
        close_d = None
        ser = series_by_symbol.get(p.symbol)
        if ser is not None:
            found = ser.last_close_on_or_before(final_d)
            if found:
                close_d, close = found
        _val, _mp, _mr, note = _position_value_simple(
            cost_basis=p.cost_basis,
            qty=p.qty,
            return_pct=p.return_pct,
            mode=mode if mode != "cost_basis" else "market",
            close=close,
        )
        market_ret = None
        if close is not None and p.cost_basis:
            market_ret = (p.qty * close) / p.cost_basis - 1.0
        source_ret = p.return_pct
        disc = False
        disc_detail = None
        if market_ret is not None and source_ret is not None:
            if abs(market_ret - source_ret) > config.discrepancy_threshold_pct:
                disc = True
                disc_detail = (
                    f"|market_return {market_ret:.6f} - source_return {source_ret:.6f}| "
                    f"> {config.discrepancy_threshold_pct}"
                )
        open_vals_final.append(
            {
                "trade_id": p.trade_id,
                "symbol": p.symbol,
                "qty": p.qty,
                "cost_basis": p.cost_basis,
                "market_price": close,
                "price_date": close_d.isoformat() if close_d else None,
                "market_value": money(p.qty * close) if close is not None else None,
                "unrealized_pnl": money(p.qty * close - p.cost_basis) if close is not None else None,
                "market_return": market_ret,
                "source_return_ref": source_ret,
                "source_net_pnl_ref": p.source_net_pnl,
                "discrepancy_flag": disc,
                "discrepancy_detail": disc_detail,
                "valuation_mode": mode,
                "notes": note,
            }
        )

    risk = None
    if daily:
        risk = compute_risk_metrics(
            [s.equity for s in daily],
            dates=[s.timestamp for s in daily],
            risk_free_rate=config.risk_free_rate,
            sortino_target=config.sortino_target,
        )

    benchmark = None
    if config.benchmark_ticker and daily:
        try:
            bres = market_data.get_daily_ohlcv(
                config.benchmark_ticker,
                start,
                valuation_end,
                adjustment_policy=adj,
            )
            bdates = [b.bar_date for b in bres.bars]
            bvals = [b.close for b in bres.bars]
            benchmark = compute_benchmark_metrics(
                bdates,
                bvals,
                name=config.benchmark_ticker,
                portfolio_dates=[s.timestamp for s in daily],
                portfolio_equity=[s.equity for s in daily],
                risk_free_rate=config.risk_free_rate,
                sortino_target=config.sortino_target,
            )
            md_report.append(
                {
                    "symbol": "__benchmark__",
                    "yahoo_ticker": config.benchmark_ticker,
                    "bars_received": len(bres.bars),
                    "missing_count": len(bres.missing_dates),
                }
            )
        except Exception as e:  # noqa: BLE001
            md_report.append(
                {
                    "symbol": "__benchmark__",
                    "yahoo_ticker": config.benchmark_ticker,
                    "error": str(e),
                }
            )

    return daily, risk, md_report, open_vals_final, sorted(set(affected)), benchmark
