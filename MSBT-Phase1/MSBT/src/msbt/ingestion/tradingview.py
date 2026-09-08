"""TradingView ingestion hooks (interfaces only — Phase 3).

These functions parse local symbol lists and exported CSV/JSON signals and map
strategy names onto internal tags. They are **not** wired to a live TradingView
webhook or broker in v1. A future adapter can call them from an ingestion route
without changing the simulation engine.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Mapping, Sequence, Union

Payload = Union[str, bytes, Mapping[str, Any], Sequence[Any]]


def import_symbol_list(source: str | Sequence[str]) -> list[str]:
    """Import a TradingView symbol / watchlist export into ticker strings.

    Accepts a sequence of symbols or text split on newlines, commas, or
    whitespace. ``EXCHANGE:SYMBOL`` tokens keep the symbol after the last colon.
    Duplicates are dropped (first-seen order). Does not contact TradingView.
    """
    if isinstance(source, str):
        tokens = _split_symbols(source)
    else:
        tokens = []
        for item in source:
            tokens.extend(_split_symbols(str(item)))

    seen: set[str] = set()
    out: list[str] = []
    for raw in tokens:
        symbol = _normalize_symbol(raw)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def receive_exported_signals(payload: Payload) -> list[dict[str, Any]]:
    """Normalize exported CSV or JSON signals into record dicts.

    Intended for files a user exports (or a future webhook body saved to disk).
    Recognizes common alert fields: symbol/ticker, strategy/strategy_name,
    time/timestamp, action/side, price. Not a live webhook receiver.
    """
    data = _decode_payload(payload)
    if isinstance(data, list):
        return [_normalize_signal(item) for item in data if item is not None]
    if isinstance(data, Mapping):
        if "signals" in data and isinstance(data["signals"], list):
            return [_normalize_signal(item) for item in data["signals"]]
        return [_normalize_signal(data)]
    if isinstance(data, str):
        text = data.strip()
        if not text:
            return []
        if text[0] in "[{":
            parsed = json.loads(text)
            return receive_exported_signals(parsed)
        return _signals_from_csv(text)
    raise TypeError(f"unsupported signal payload type: {type(payload).__name__}")


def map_strategy_names(
    names: str | Sequence[str],
    mapping: Mapping[str, str] | None = None,
) -> str | list[str]:
    """Map external TradingView strategy name(s) to internal tags.

    Lookup is case-insensitive. Unknown names pass through unchanged.
    A single string returns a string; a sequence returns a list.
    """
    table = {str(k).strip().lower(): str(v) for k, v in (mapping or {}).items()}

    def _one(name: str) -> str:
        key = name.strip().lower()
        if key in table:
            return table[key]
        return name.strip()

    if isinstance(names, str):
        return _one(names)
    return [_one(str(n)) for n in names]


def _split_symbols(text: str) -> list[str]:
    cleaned = text.replace(";", ",").replace("\t", ",")
    parts: list[str] = []
    for chunk in cleaned.split(","):
        parts.extend(chunk.split())
    return parts


def _normalize_symbol(token: str) -> str:
    token = token.strip().strip("\"'")
    if not token or token.startswith("#"):
        return ""
    if ":" in token:
        token = token.split(":")[-1]
    return token.strip().upper()


def _decode_payload(payload: Payload) -> Any:
    if isinstance(payload, bytes):
        return payload.decode("utf-8-sig")
    return payload


def _signals_from_csv(text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return []
    return [_normalize_signal(row) for row in reader]


def _first(record: Mapping[str, Any], *keys: str) -> Any:
    lowered = {str(k).strip().lower(): v for k, v in record.items()}
    for key in keys:
        if key in lowered and lowered[key] not in (None, ""):
            return lowered[key]
    return None


def _normalize_signal(item: Any) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {"raw": item}
    symbol = _first(item, "symbol", "ticker", "instrument")
    strategy = _first(item, "strategy", "strategy_name", "comment", "alert")
    when = _first(item, "time", "timestamp", "datetime", "date")
    action = _first(item, "action", "side", "order", "signal")
    price = _first(item, "price", "close", "value")
    out: dict[str, Any] = {
        "symbol": _normalize_symbol(str(symbol)) if symbol else None,
        "strategy": None if strategy is None else str(strategy).strip(),
        "time": None if when is None else str(when).strip(),
        "action": None if action is None else str(action).strip(),
        "price": _maybe_float(price),
    }
    extra = {
        str(k): v
        for k, v in item.items()
        if str(k).strip().lower()
        not in {
            "symbol",
            "ticker",
            "instrument",
            "strategy",
            "strategy_name",
            "comment",
            "alert",
            "time",
            "timestamp",
            "datetime",
            "date",
            "action",
            "side",
            "order",
            "signal",
            "price",
            "close",
            "value",
        }
    }
    if extra:
        out["extra"] = extra
    return out


def _maybe_float(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return value
