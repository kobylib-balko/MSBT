import pytest

from msbt.models.config import SimulationConfig
from tests.helpers import make_trade


@pytest.fixture
def acceptance1_trades():
    return [
        make_trade("T1", "AAPL", "2020-01-10", "2020-01-20", 0.10, source_trade_number=1),
        make_trade("T2", "MSFT", "2020-01-15", "2020-01-25", 0.20, source_trade_number=2),
    ]


@pytest.fixture
def acceptance1_config():
    return SimulationConfig(
        initial_capital=100_000,
        buy_pct_of_equity=0.20,
        max_pct_per_symbol=0.40,
        entry_fee_pct=0.0,
        exit_fee_pct=0.0,
        slippage_pct=0.0,
        allow_partial_fills=True,
        allow_multiple_positions_same_symbol=True,
        allow_leverage=False,
    )
