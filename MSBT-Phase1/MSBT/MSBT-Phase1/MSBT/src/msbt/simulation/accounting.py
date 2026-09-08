"""Closed-trade economics (LOCKED).

gross_exit_value = cost_basis * (1 + return_pct)
Fees/slippage are cash costs; they do NOT rewrite return_pct.
Round monetary cash balances to 2 decimal places after each cash movement.
"""

from __future__ import annotations

from dataclasses import dataclass


def money(x: float) -> float:
    """Round monetary amount to 2 decimal places (bankers? use round half even via round)."""
    return round(float(x), 2)


@dataclass
class EntryEconomics:
    allocated_capital: float
    entry_fee: float
    entry_slippage: float
    cash_outflow: float
    qty: float


@dataclass
class ExitEconomics:
    gross_exit_value: float
    gross_pnl: float
    exit_fee: float
    exit_slippage: float
    cash_inflow: float
    net_pnl: float
    net_return: float


def compute_entry(
    allocated_capital: float,
    buy_price: float,
    entry_fee_pct: float,
    slippage_pct: float,
    *,
    fractional_shares: bool = True,
) -> EntryEconomics:
    allocated = money(allocated_capital)
    entry_fee = money(allocated * entry_fee_pct)
    entry_slippage = money(allocated * slippage_pct)
    cash_outflow = money(allocated + entry_fee + entry_slippage)
    if buy_price and buy_price > 0:
        qty = allocated / buy_price
        if not fractional_shares:
            qty = float(int(qty))
            allocated = money(qty * buy_price)
            entry_fee = money(allocated * entry_fee_pct)
            entry_slippage = money(allocated * slippage_pct)
            cash_outflow = money(allocated + entry_fee + entry_slippage)
    else:
        qty = 0.0
    return EntryEconomics(
        allocated_capital=allocated,
        entry_fee=entry_fee,
        entry_slippage=entry_slippage,
        cash_outflow=cash_outflow,
        qty=qty,
    )


def compute_exit(
    cost_basis: float,
    return_pct: float,
    exit_fee_pct: float,
    slippage_pct: float,
    entry_fee: float,
    entry_slippage: float,
) -> ExitEconomics:
    """Authoritative gross outcome from return_pct; fees as cash costs."""
    gross_exit_value = money(cost_basis * (1.0 + return_pct))
    gross_pnl = money(gross_exit_value - cost_basis)
    exit_fee = money(gross_exit_value * exit_fee_pct)
    exit_slippage = money(gross_exit_value * slippage_pct)
    cash_inflow = money(gross_exit_value - exit_fee - exit_slippage)
    net_pnl = money(
        (gross_exit_value - cost_basis)
        - (entry_fee + exit_fee + entry_slippage + exit_slippage)
    )
    net_return = (net_pnl / cost_basis) if cost_basis else 0.0
    return ExitEconomics(
        gross_exit_value=gross_exit_value,
        gross_pnl=gross_pnl,
        exit_fee=exit_fee,
        exit_slippage=exit_slippage,
        cash_inflow=cash_inflow,
        net_pnl=net_pnl,
        net_return=net_return,
    )
