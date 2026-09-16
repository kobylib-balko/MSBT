"""Import and normalize trade CSV/Excel files into Trade objects.

Return % convention (documented):
  File values are percent points (e.g. 11.95 means +11.95%).
  Internally we store decimal returns: 11.95 → 0.1195.
  Both Entry and Exit rows must agree within 1e-9 (on the raw percent points).
"""

from __future__ import annotations

from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Optional, Union

import pandas as pd

from msbt.importers.filename import resolve_file_meta
from msbt.models.trade import Trade, TradeStatus

REQUIRED_COLUMNS = [
    "Trade number",
    "Type",
    "Date and time",
    "Signal",
    "Price USD",
    "Return %",
]

OPTIONAL_COLUMNS = [
    "Size (qty)",
    "Size (value)",
    "Net PnL USD",
    "Commission USD",
    "Favorable excursion USD",
    "Favorable excursion %",
    "Adverse excursion USD",
    "Adverse excursion %",
    "Cumulative PnL USD",
    "Cumulative PnL %",
    "Duration (bars)",
]

RETURN_EPS = 1e-9
PathLike = Union[str, Path]


def _normalize_header(h: str) -> str:
    return " ".join(str(h).replace("\ufeff", "").strip().split())


def _read_dataframe(source: Union[PathLike, BinaryIO, bytes], filename: str) -> pd.DataFrame:
    name = Path(filename).name.lower()
    is_excel = name.endswith(".xlsx") or name.endswith(".xls")
    # Default to CSV when extension is missing (pattern allows bare ..._YYYY-MM-DD)
    if isinstance(source, (str, Path)):
        path = Path(source)
        if is_excel or path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        return pd.read_csv(path, encoding="utf-8-sig")
    if isinstance(source, bytes):
        bio = BytesIO(source)
    else:
        bio = source  # type: ignore[assignment]
    if is_excel:
        return pd.read_excel(bio)
    return pd.read_csv(bio, encoding="utf-8-sig")


def _parse_date(val) -> Optional[date]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() == "open" or s in ("—", "-", "–"):
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    # ISO first
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        pass
    # Common alt formats
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s[:10], fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unparseable date: {val!r}")


def _parse_price(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s in ("—", "-", "–") or s.lower() == "open":
        return None
    return float(s.replace(",", ""))


def _parse_return_raw(val) -> Optional[float]:
    """Return percent-points as float (11.95), or None if missing."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().replace("%", "")
    if not s:
        return None
    return float(s)


def percent_points_to_decimal(pct_points: float) -> float:
    """11.95 (percent points) → 0.1195 (decimal)."""
    return pct_points / 100.0


def import_trade_file(
    source: Union[PathLike, BinaryIO, bytes],
    filename: Optional[str] = None,
    *,
    trade_id_prefix: Optional[str] = None,
    symbol: Optional[str] = None,
    strategy: Optional[str] = None,
    exchange: Optional[str] = None,
) -> tuple[list[Trade], list[dict]]:
    """Import one file. Returns (trades including invalids, validation rows).

    Filename pattern is optional. If it does not match
    ``{strategy}_{exchange}_{symbol}_{YYYY-MM-DD}.ext``, pass ``symbol``
    (and optionally strategy/exchange) explicitly.
    """
    if filename is None:
        if isinstance(source, (str, Path)):
            filename = Path(source).name
        else:
            raise ValueError("filename is required when source is not a path")

    validation_rows: list[dict] = []
    trades: list[Trade] = []

    try:
        meta = resolve_file_meta(
            filename, symbol=symbol, strategy=strategy, exchange=exchange
        )
    except ValueError as e:
        validation_rows.append(
            {
                "source_file": Path(filename).name,
                "trade_number": None,
                "severity": "error",
                "message": str(e),
            }
        )
        return [], validation_rows

    try:
        df = _read_dataframe(source, filename)
    except Exception as e:
        validation_rows.append(
            {
                "source_file": meta.original,
                "trade_number": None,
                "severity": "error",
                "message": f"Failed to read file: {e}",
            }
        )
        return [], validation_rows

    # Normalize headers
    df = df.copy()
    df.columns = [_normalize_header(c) for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        validation_rows.append(
            {
                "source_file": meta.original,
                "trade_number": None,
                "severity": "error",
                "message": f"Missing required columns: {missing}",
            }
        )
        return [], validation_rows

    # Drop completely empty rows
    df = df.dropna(how="all")
    if df.empty:
        validation_rows.append(
            {
                "source_file": meta.original,
                "trade_number": None,
                "severity": "error",
                "message": "File has no data rows",
            }
        )
        return [], validation_rows

    prefix = trade_id_prefix or f"{meta.symbol}"
    grouped = df.groupby("Trade number", dropna=False)

    for trade_num, group in grouped:
        errors: list[str] = []
        if pd.isna(trade_num):
            errors.append("Missing Trade number")
            validation_rows.append(
                {
                    "source_file": meta.original,
                    "trade_number": None,
                    "severity": "error",
                    "message": "Row(s) with missing Trade number",
                }
            )
            continue

        try:
            trade_num_int = int(trade_num)
        except (TypeError, ValueError):
            errors.append(f"Invalid Trade number: {trade_num!r}")
            trade_num_int = -1

        types = group["Type"].astype(str).str.strip().str.lower()
        entry_rows = group[types == "entry long"]
        exit_rows = group[types == "exit long"]

        if len(entry_rows) != 1 or len(exit_rows) != 1:
            msg = (
                f"Trade {trade_num_int}: expected exactly one Entry long and one Exit long; "
                f"got {len(entry_rows)} entry, {len(exit_rows)} exit"
            )
            errors.append(msg)
            validation_rows.append(
                {
                    "source_file": meta.original,
                    "trade_number": trade_num_int,
                    "severity": "error",
                    "message": msg,
                }
            )
            tid = f"{prefix}-{trade_num_int}"
            trades.append(
                Trade(
                    trade_id=tid,
                    source_trade_number=trade_num_int,
                    symbol=meta.symbol,
                    strategy=meta.strategy,
                    exchange=meta.exchange,
                    buy_date=date.min,
                    sell_date=None,
                    buy_price=0.0,
                    sell_price=None,
                    return_pct=0.0,
                    duration_bars=None,
                    source_file=meta.original,
                    status=TradeStatus.INVALID,
                    validation_errors=errors,
                )
            )
            continue

        entry = entry_rows.iloc[0]
        exit_ = exit_rows.iloc[0]

        buy_date = None
        sell_date = None
        is_open = False
        try:
            buy_date = _parse_date(entry["Date and time"])
            if buy_date is None:
                errors.append("Missing/invalid buy date on Entry")
        except ValueError as e:
            errors.append(str(e))

        exit_date_raw = exit_["Date and time"]
        exit_signal = str(exit_.get("Signal", "")).strip().lower()
        if str(exit_date_raw).strip().lower() == "open" or exit_signal == "open":
            is_open = True
            sell_date = None
        else:
            try:
                sell_date = _parse_date(exit_date_raw)
                if sell_date is None:
                    errors.append("Missing/invalid sell date on Exit")
            except ValueError as e:
                errors.append(str(e))

        try:
            buy_price = _parse_price(entry["Price USD"])
            if buy_price is None or buy_price <= 0:
                errors.append("Missing/invalid buy price")
                buy_price = 0.0
        except (TypeError, ValueError) as e:
            errors.append(f"Invalid buy price: {e}")
            buy_price = 0.0

        sell_price: Optional[float] = None
        if not is_open:
            try:
                sell_price = _parse_price(exit_["Price USD"])
            except (TypeError, ValueError) as e:
                errors.append(f"Invalid sell price: {e}")

        entry_ret = _parse_return_raw(entry["Return %"])
        exit_ret = _parse_return_raw(exit_["Return %"])
        if entry_ret is None or exit_ret is None:
            errors.append("Missing Return % on Entry and/or Exit")
            ret_raw = entry_ret if entry_ret is not None else exit_ret
        else:
            if abs(entry_ret - exit_ret) > RETURN_EPS:
                errors.append(
                    f"Entry/Exit Return % mismatch: {entry_ret} vs {exit_ret}"
                )
            ret_raw = entry_ret

        if ret_raw is None:
            return_pct = 0.0
        else:
            return_pct = percent_points_to_decimal(ret_raw)

        if buy_date and sell_date and sell_date < buy_date:
            errors.append(f"Sell date {sell_date} < buy date {buy_date}")

        duration_bars = None
        if "Duration (bars)" in group.columns:
            for row in (entry, exit_):
                v = row.get("Duration (bars)")
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    try:
                        duration_bars = int(float(v))
                        break
                    except (TypeError, ValueError):
                        errors.append(f"Invalid Duration (bars): {v!r}")

        net_pnl = None
        if "Net PnL USD" in entry.index:
            try:
                v = entry["Net PnL USD"]
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    net_pnl = float(v)
            except (TypeError, ValueError):
                pass

        entry_signal = None
        exit_signal = None
        try:
            es = entry.get("Signal")
            if es is not None and not (isinstance(es, float) and pd.isna(es)):
                entry_signal = str(es).strip() or None
        except Exception:
            pass
        try:
            xs = exit_.get("Signal")
            if xs is not None and not (isinstance(xs, float) and pd.isna(xs)):
                exit_signal = str(xs).strip() or None
        except Exception:
            pass

        tid = f"{prefix}-{trade_num_int}"
        if errors:
            status = TradeStatus.INVALID
            for err in errors:
                validation_rows.append(
                    {
                        "source_file": meta.original,
                        "trade_number": trade_num_int,
                        "severity": "error",
                        "message": err,
                    }
                )
        elif is_open:
            status = TradeStatus.OPEN
            validation_rows.append(
                {
                    "source_file": meta.original,
                    "trade_number": trade_num_int,
                    "severity": "info",
                    "message": "Open position (exit date=Open)",
                }
            )
        else:
            status = TradeStatus.COMPLETED

        if buy_date is None:
            buy_date = date.min
            if status != TradeStatus.INVALID:
                status = TradeStatus.INVALID

        trades.append(
            Trade(
                trade_id=tid,
                source_trade_number=trade_num_int,
                symbol=meta.symbol,
                strategy=meta.strategy,
                exchange=meta.exchange,
                buy_date=buy_date,
                sell_date=sell_date,
                buy_price=buy_price,
                sell_price=sell_price,
                return_pct=return_pct,
                duration_bars=duration_bars,
                source_file=meta.original,
                status=status,
                source_return_pct_raw=ret_raw,
                source_net_pnl_usd=net_pnl,
                entry_signal=entry_signal,
                exit_signal=exit_signal,
                validation_errors=errors,
            )
        )

    return trades, validation_rows


def import_trade_files(
    paths: list[PathLike],
) -> tuple[list[Trade], list[dict]]:
    all_trades: list[Trade] = []
    all_validation: list[dict] = []
    for p in paths:
        trades, rows = import_trade_file(p)
        all_trades.extend(trades)
        all_validation.extend(rows)
    return all_trades, all_validation
