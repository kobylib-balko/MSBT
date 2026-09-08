"""Multi-Symbol Portfolio Backtester (MSBT) — Phase 2."""

from msbt.models.config import SimulationConfig
from msbt.models.trade import Trade
from msbt.simulation.engine import run_simulation

__version__ = "0.2.0"
__all__ = ["SimulationConfig", "Trade", "run_simulation", "__version__"]
