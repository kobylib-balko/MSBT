"""Pre-simulation validation. Never silently drop or fix data."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from msbt.models.trade import Trade, TradeStatus


@dataclass
class ValidationReport:
    files_uploaded: int = 0
    rows_imported: int = 0  # paired trade candidates
    valid_trades: int = 0
    invalid_trades: int = 0
    open_trades: int = 0
    completed_trades: int = 0
    issues: list[dict] = field(default_factory=list)
    counts_by_severity: dict[str, int] = field(default_factory=dict)

    def to_rows(self) -> list[dict]:
        return list(self.issues)


def validate_trades(
    trades: list[Trade],
    *,
    prior_issues: Optional[list[dict]] = None,
    files_uploaded: int = 0,
) -> tuple[list[Trade], ValidationReport]:
    """Validate trades; mark duplicates as invalid. Returns (all trades, report)."""
    report = ValidationReport(files_uploaded=files_uploaded)
    if prior_issues:
        report.issues.extend(prior_issues)

    report.rows_imported = len(trades)

    # Duplicate detection: same symbol + buy_date + sell_date + return_pct + source_file
    seen: dict[tuple, str] = {}
    for t in trades:
        if t.status == TradeStatus.INVALID:
            continue
        key = (
            t.symbol,
            t.buy_date,
            t.sell_date,
            round(t.return_pct, 12),
            t.source_file,
            t.source_trade_number,
        )
        # Completely duplicated trades (same identity)
        dup_key = (t.symbol, t.buy_date, t.sell_date, round(t.return_pct, 12), t.source_file)
        if dup_key in seen and seen[dup_key] != t.trade_id:
            # Only flag if same source trade content appears twice with different trade numbers
            # Spec: completely duplicated trades
            pass
        seen[key] = t.trade_id

    # Second pass: exact content duplicates across different trade_ids in same file
    content_counts: Counter = Counter()
    content_first: dict[tuple, str] = {}
    for t in trades:
        if t.status == TradeStatus.INVALID:
            continue
        ck = (t.symbol, t.buy_date, t.sell_date, round(t.return_pct, 12), t.source_file)
        content_counts[ck] += 1
        if ck not in content_first:
            content_first[ck] = t.trade_id

    for t in trades:
        if t.status == TradeStatus.INVALID:
            continue
        ck = (t.symbol, t.buy_date, t.sell_date, round(t.return_pct, 12), t.source_file)
        if content_counts[ck] > 1 and t.trade_id != content_first[ck]:
            t.status = TradeStatus.INVALID
            msg = "Completely duplicated trade (same symbol/buy/sell/return/source)"
            t.validation_errors.append(msg)
            report.issues.append(
                {
                    "source_file": t.source_file,
                    "trade_number": t.source_trade_number,
                    "severity": "error",
                    "message": msg,
                }
            )

    for t in trades:
        if t.status == TradeStatus.INVALID:
            report.invalid_trades += 1
        elif t.status == TradeStatus.OPEN:
            report.open_trades += 1
            report.valid_trades += 1
        elif t.status == TradeStatus.COMPLETED:
            report.completed_trades += 1
            report.valid_trades += 1

    sev = Counter(i.get("severity", "info") for i in report.issues)
    report.counts_by_severity = dict(sev)
    return trades, report
