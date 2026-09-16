"""Phase 2.5 cash earn (synthetic RF total-return index)."""

from msbt.cash_earn.rates import (
    SyntheticRateProvider,
    apply_cash_earn_ratio,
    build_total_return_index,
    default_embedded_rates,
    load_rates_csv,
)

__all__ = [
    "SyntheticRateProvider",
    "apply_cash_earn_ratio",
    "build_total_return_index",
    "default_embedded_rates",
    "load_rates_csv",
]
