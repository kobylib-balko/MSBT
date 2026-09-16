from datetime import date

from msbt.models.trade import Trade, TradeStatus


def make_trade(
    tid: str,
    symbol: str,
    buy: str,
    sell: str | None,
    ret: float,
    *,
    strategy: str = "Test",
    exchange: str = "NASDAQ",
    buy_price: float = 100.0,
    sell_price: float | None = 110.0,
    duration_bars: int | None = 10,
    status: TradeStatus | None = None,
    source_file: str = "Test_NASDAQ_SYM_2020-01-01.csv",
    source_trade_number: int = 1,
    entry_signal: str | None = "Long1",
    exit_signal: str | None = "Exitlong1",
) -> Trade:
    is_open = sell is None
    if status is None:
        status = TradeStatus.OPEN if is_open else TradeStatus.COMPLETED
    return Trade(
        trade_id=tid,
        source_trade_number=source_trade_number,
        symbol=symbol,
        strategy=strategy,
        exchange=exchange,
        buy_date=date.fromisoformat(buy),
        sell_date=date.fromisoformat(sell) if sell else None,
        buy_price=buy_price,
        sell_price=sell_price if not is_open else None,
        return_pct=ret,
        duration_bars=duration_bars,
        source_file=source_file.replace("SYM", symbol),
        status=status,
        entry_signal=entry_signal,
        exit_signal=None if is_open else exit_signal,
    )
