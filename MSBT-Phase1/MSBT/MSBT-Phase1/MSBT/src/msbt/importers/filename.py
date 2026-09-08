"""Filename convention: {strategy}_{exchange}_{symbol}_{YYYY-MM-DD}[.csv|.xlsx|.xls]

Example: TripleStrategy_NYSE_ORCL_2026-09-07
     or: TripleStrategy_NYSE_ORCL_2026-09-07.csv
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Extension is optional — exports are often named without .csv in the stem,
# while uploads may still include .csv / .xlsx / .xls.
FILENAME_RE = re.compile(
    r"^(?P<strategy>[^_]+)_(?P<exchange>[^_]+)_(?P<symbol>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})"
    r"(?:\.(?P<ext>csv|xlsx|xls))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedFilename:
    strategy: str
    exchange: str
    symbol: str
    file_date: str
    original: str


def try_parse_trade_filename(filename: str) -> Optional[ParsedFilename]:
    name = Path(filename).name
    m = FILENAME_RE.match(name)
    if not m:
        return None
    return ParsedFilename(
        strategy=m.group("strategy"),
        exchange=m.group("exchange"),
        symbol=m.group("symbol").upper(),
        file_date=m.group("date"),
        original=name,
    )


def parse_trade_filename(filename: str) -> ParsedFilename:
    parsed = try_parse_trade_filename(filename)
    if parsed is None:
        name = Path(filename).name
        raise ValueError(
            f"Filename '{name}' does not match required pattern "
            f"{{strategy}}_{{exchange}}_{{symbol}}_{{YYYY-MM-DD}} "
            f"(optional .csv/.xlsx/.xls). "
            f"Example: TripleStrategy_NYSE_ORCL_2026-09-07.csv"
        )
    return parsed


def resolve_file_meta(
    filename: str,
    *,
    symbol: Optional[str] = None,
    strategy: Optional[str] = None,
    exchange: Optional[str] = None,
) -> ParsedFilename:
    """Parse required filename pattern; optional overrides only refine fields."""
    parsed = parse_trade_filename(filename)
    return ParsedFilename(
        strategy=(strategy or parsed.strategy).strip() or parsed.strategy,
        exchange=(exchange or parsed.exchange).strip() or parsed.exchange,
        symbol=(symbol or parsed.symbol).strip().upper() or parsed.symbol,
        file_date=parsed.file_date,
        original=parsed.original,
    )
