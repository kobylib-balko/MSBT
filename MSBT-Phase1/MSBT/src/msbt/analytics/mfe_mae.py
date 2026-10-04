"""MFE / MAE excursion analytics for executed closed trades.

Sign convention:
  MFE (favorable) is typically ≥ 0 (peak favorable move as decimal return).
  MAE (adverse) is typically ≤ 0 (worst adverse move as decimal return).

Suggested stop uses the more adverse side of winners' MAE:
  with MAE ≤ 0, percentile(winners_mae, 0.25) is deeper drawdown than the median.
Suggested target uses median (50th pct) of winners' MFE.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from statistics import mean, median
from typing import Any, Optional, Sequence

from msbt.analytics.stats import _closed_executed
from msbt.models.results import TradeResult


@dataclass
class MfeMaeStats:
    n_with_data: int
    n_closed_executed: int
    median_mfe: Optional[float]
    mean_mfe: Optional[float]
    p75_mfe: Optional[float]
    p90_mfe: Optional[float]
    median_mae: Optional[float]
    mean_mae: Optional[float]
    p25_mae: Optional[float]
    median_giveback: Optional[float]
    median_mfe_mae_ratio: Optional[float]
    winners_median_mfe: Optional[float]
    winners_median_mae: Optional[float]
    winners_n: int
    losers_median_mfe: Optional[float]
    losers_median_mae: Optional[float]
    losers_n: int
    suggested_stop_pct: Optional[float]
    suggested_target_pct: Optional[float]
    recommendations: list[str]
    scope_note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pctile(values: Sequence[float], q: float) -> Optional[float]:
    """Percentile of ``values`` at quantile ``q`` in [0, 1] (linear interpolation)."""
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)
    if n == 1:
        return float(xs[0])
    q = max(0.0, min(1.0, float(q)))
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


def _safe_median(values: Sequence[float]) -> Optional[float]:
    return float(median(values)) if values else None


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    return float(mean(values)) if values else None


def _return_for_giveback(r: TradeResult) -> Optional[float]:
    if r.gross_return is not None:
        return r.gross_return
    return r.net_return


def _build_recommendations(
    *,
    n_with_data: int,
    median_giveback: Optional[float],
    winners_median_mae: Optional[float],
    losers_median_mae: Optional[float],
    suggested_stop_pct: Optional[float],
    suggested_target_pct: Optional[float],
    median_mfe_mae_ratio: Optional[float],
) -> list[str]:
    recs: list[str] = []
    if n_with_data == 0:
        recs.append(
            "Favorable/Adverse excursion columns were missing or empty — "
            "no MFE/MAE stats or stop/target suggestions for this run."
        )
        return recs

    if median_giveback is not None and median_giveback > 0.02:
        pct = median_giveback * 100.0
        recs.append(
            f"Winners often give back ~{pct:.1f}% from peak — consider a target "
            "or trail near Train MFE quantiles."
        )

    if (
        winners_median_mae is not None
        and losers_median_mae is not None
        and winners_median_mae > losers_median_mae
        and suggested_stop_pct is not None
    ):
        # winners' MAE is shallower (less negative) than losers'
        stop_pp = suggested_stop_pct * 100.0
        recs.append(
            f"Winners' adverse excursion is shallower than losers'; a stop near "
            f"{stop_pp:.1f}% (winners' MAE 25th pct) may cut deep losers — "
            "fit on this sample; validate OOS / Phase 4."
        )

    if median_mfe_mae_ratio is not None and median_mfe_mae_ratio >= 2.0:
        recs.append(
            f"Median MFE/|MAE| ≈ {median_mfe_mae_ratio:.1f} — edge shows "
            "favorable asymmetry on this sample."
        )

    if suggested_target_pct is not None and n_with_data > 0:
        tgt_pp = suggested_target_pct * 100.0
        if not any("give back" in r.lower() for r in recs):
            recs.append(
                f"Median winners' MFE is ~{tgt_pp:.1f}% — a research target band "
                "near that level is a starting point only."
            )

    recs.append(
        "These stats describe executed closed trades in this run; do not use a "
        "trade's own MFE/MAE to accept/reject that same trade (look-ahead). "
        "Suggestions are research aids — apply by changing strategy exits and "
        "re-exporting CSVs."
    )

    # Keep 3–6 when we have data (look-ahead always present)
    if len(recs) < 3 and suggested_stop_pct is not None:
        recs.insert(
            0,
            f"Descriptive stop band ~{suggested_stop_pct * 100:.1f}% "
            "(winners' MAE 25th pct); not auto-applied.",
        )
    return recs[:6]


def compute_mfe_mae_stats(
    trade_results: Sequence[TradeResult],
    *,
    train_cutoff: Optional[date] = None,
) -> MfeMaeStats:
    """Compute MFE/MAE stats on executed closed trades (optionally train-only)."""
    closed = _closed_executed(trade_results)
    if train_cutoff is not None:
        closed = [r for r in closed if r.buy_date is not None and r.buy_date <= train_cutoff]
        scope_note = (
            f"Train sample only (buy_date ≤ {train_cutoff.isoformat()}); "
            "not locked OOS."
        )
    else:
        scope_note = "Full sample for this run (not locked OOS)."

    n_closed = len(closed)
    with_data = [
        r
        for r in closed
        if r.mfe_pct is not None or r.mae_pct is not None
    ]
    n_with = len(with_data)

    if n_with == 0:
        return MfeMaeStats(
            n_with_data=0,
            n_closed_executed=n_closed,
            median_mfe=None,
            mean_mfe=None,
            p75_mfe=None,
            p90_mfe=None,
            median_mae=None,
            mean_mae=None,
            p25_mae=None,
            median_giveback=None,
            median_mfe_mae_ratio=None,
            winners_median_mfe=None,
            winners_median_mae=None,
            winners_n=0,
            losers_median_mfe=None,
            losers_median_mae=None,
            losers_n=0,
            suggested_stop_pct=None,
            suggested_target_pct=None,
            recommendations=_build_recommendations(
                n_with_data=0,
                median_giveback=None,
                winners_median_mae=None,
                losers_median_mae=None,
                suggested_stop_pct=None,
                suggested_target_pct=None,
                median_mfe_mae_ratio=None,
            ),
            scope_note=scope_note,
        )

    mfes = [r.mfe_pct for r in with_data if r.mfe_pct is not None]
    maes = [r.mae_pct for r in with_data if r.mae_pct is not None]

    givebacks: list[float] = []
    ratios: list[float] = []
    for r in with_data:
        ret = _return_for_giveback(r)
        if r.mfe_pct is not None and ret is not None:
            givebacks.append(r.mfe_pct - ret)
        if (
            r.mfe_pct is not None
            and r.mae_pct is not None
            and abs(r.mae_pct) > 1e-15
        ):
            ratios.append(abs(r.mfe_pct) / abs(r.mae_pct))

    winners = [r for r in with_data if (r.net_pnl or 0) > 0]
    losers = [r for r in with_data if (r.net_pnl or 0) < 0]
    w_mfe = [r.mfe_pct for r in winners if r.mfe_pct is not None]
    w_mae = [r.mae_pct for r in winners if r.mae_pct is not None]
    l_mfe = [r.mfe_pct for r in losers if r.mfe_pct is not None]
    l_mae = [r.mae_pct for r in losers if r.mae_pct is not None]

    # MAE ≤ 0: 25th pct is more adverse (deeper). Target = median winners' MFE.
    suggested_stop = _pctile(w_mae, 0.25) if w_mae else None
    suggested_target = _pctile(w_mfe, 0.50) if w_mfe else None

    median_giveback = _safe_median(givebacks)
    median_ratio = _safe_median(ratios)
    winners_median_mae = _safe_median(w_mae)
    losers_median_mae = _safe_median(l_mae)

    recs = _build_recommendations(
        n_with_data=n_with,
        median_giveback=median_giveback,
        winners_median_mae=winners_median_mae,
        losers_median_mae=losers_median_mae,
        suggested_stop_pct=suggested_stop,
        suggested_target_pct=suggested_target,
        median_mfe_mae_ratio=median_ratio,
    )

    return MfeMaeStats(
        n_with_data=n_with,
        n_closed_executed=n_closed,
        median_mfe=_safe_median(mfes),
        mean_mfe=_safe_mean(mfes),
        p75_mfe=_pctile(mfes, 0.75),
        p90_mfe=_pctile(mfes, 0.90),
        median_mae=_safe_median(maes),
        mean_mae=_safe_mean(maes),
        p25_mae=_pctile(maes, 0.25),
        median_giveback=median_giveback,
        median_mfe_mae_ratio=median_ratio,
        winners_median_mfe=_safe_median(w_mfe),
        winners_median_mae=winners_median_mae,
        winners_n=len(winners),
        losers_median_mfe=_safe_median(l_mfe),
        losers_median_mae=losers_median_mae,
        losers_n=len(losers),
        suggested_stop_pct=suggested_stop,
        suggested_target_pct=suggested_target,
        recommendations=recs,
        scope_note=scope_note,
    )
