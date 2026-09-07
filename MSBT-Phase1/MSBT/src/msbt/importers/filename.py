"""Filename convention: {strategy}_{exchange}_{symbol}_{YYYY-MM-DD}.csv"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


FILENAME_RE = re.compile(
    r"^(?P<strategy>[^_]+)_(?P<exchange>[^_]+)_(?P<symbol>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})\.(csv|xlsx|xls)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedFilename:
    strategy: str
    exchange: str
    symbol: str
    file_date: str
    original: str


def parse_trade_filename(filename: str) -> ParsedFilename:
    """Parse and require strategy/exchange/symbol/date from filename.

    Raises ValueError if the name does not match the convention.
    Does not guess the symbol from sheet contents.
    """
    name = Path(filename).name
    m = FILENAME_RE.match(name)
    if not m:
        raise ValueError(
            f"Filename '{name}' does not match required pattern "
            f"{{strategy}}_{{exchange}}_{{symbol}}_{{YYYY-MM-DD}}.csv"
        )
    return ParsedFilename(
        strategy=m.group("strategy"),
        exchange=m.group("exchange"),
        symbol=m.group("symbol").upper(),
        file_date=m.group("date"),
        original=name,
    )
