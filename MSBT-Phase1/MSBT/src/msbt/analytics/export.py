"""Export simulation outputs to Excel/CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from msbt.models.results import SimulationResult

PathLike = Union[str, Path]


def _summary_df(result: SimulationResult) -> pd.DataFrame:
    d = result.performance_metrics.to_dict()
    # Flatten rejected_by_reason
    rows = []
    for k, v in d.items():
        if k == "Rejected by reason" and isinstance(v, dict):
            for rk, rv in v.items():
                rows.append({"Metric": f"Rejected: {rk}", "Value": rv})
        else:
            rows.append({"Metric": k, "Value": v})
    return pd.DataFrame(rows)


def export_results_excel(result: SimulationResult, path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger = pd.DataFrame([r.to_dict() for r in result.trade_results])
    equity = pd.DataFrame([s.to_dict() for s in result.portfolio_timeseries])
    summary = _summary_df(result)
    symbol = pd.DataFrame(result.symbol_metrics)
    source = pd.DataFrame(result.source_metrics)
    validation = pd.DataFrame(result.validation_report)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        ledger.to_excel(writer, sheet_name="Ledger", index=False)
        equity.to_excel(writer, sheet_name="EquityCurve", index=False)
        symbol.to_excel(writer, sheet_name="BySymbol", index=False)
        source.to_excel(writer, sheet_name="BySource", index=False)
        if not validation.empty:
            validation.to_excel(writer, sheet_name="Validation", index=False)
    return path


def export_results_csv_dir(result: SimulationResult, directory: PathLike) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.to_dict() for r in result.trade_results]).to_csv(
        directory / "ledger.csv", index=False
    )
    pd.DataFrame([s.to_dict() for s in result.portfolio_timeseries]).to_csv(
        directory / "equity_curve.csv", index=False
    )
    _summary_df(result).to_csv(directory / "summary.csv", index=False)
    pd.DataFrame(result.symbol_metrics).to_csv(directory / "by_symbol.csv", index=False)
    pd.DataFrame(result.source_metrics).to_csv(directory / "by_source.csv", index=False)
    if result.validation_report:
        pd.DataFrame(result.validation_report).to_csv(
            directory / "validation.csv", index=False
        )
    return directory
