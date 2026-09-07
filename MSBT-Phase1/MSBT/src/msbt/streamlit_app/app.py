"""Minimal Streamlit UI: upload → validate → configure → run → results → export."""

from __future__ import annotations

import io
import tempfile
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

from msbt.analytics.export import export_results_excel
from msbt.importers.csv_importer import import_trade_file
from msbt.models.config import EntryPriority, SimulationConfig
from msbt.simulation.engine import run_simulation
from msbt.validation.validate import validate_trades


st.set_page_config(page_title="MSBT Phase 1", layout="wide")
st.title("Multi-Symbol Portfolio Backtester — Phase 1")
st.caption(
    "Event-based capital simulation using Return %. "
    "Average trade return ≠ portfolio return (sizing, overlap, cash, rejects)."
)

with st.expander("Expected file names", expanded=True):
    st.markdown(
        """
Files must be named:

`{strategy}_{exchange}_{symbol}_{YYYY-MM-DD}`  

Examples:
- `TripleStrategy_NYSE_ORCL_2026-09-07`
- `TripleStrategy_NYSE_ORCL_2026-09-07.csv`

Symbol is taken from the filename. Extension `.csv` / `.xlsx` / `.xls` is optional.
"""
    )

uploaded = st.file_uploader(
    "Upload trade CSV/Excel files",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
)

if not uploaded:
    st.info("Upload one or more trade files using the naming pattern above.")
    st.stop()

all_trades = []
all_issues = []
file_count = 0

try:
    for uf in uploaded:
        file_count += 1
        data = uf.getvalue()  # safe across Streamlit reruns (do not use .read())
        trades, issues = import_trade_file(data, filename=uf.name)
        all_trades.extend(trades)
        all_issues.extend(issues)

    all_trades, report = validate_trades(
        all_trades, prior_issues=all_issues, files_uploaded=file_count
    )
except Exception:
    st.error("Upload/import crashed. Details:")
    st.code(traceback.format_exc())
    st.stop()

st.subheader("Validation")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Files", report.files_uploaded)
c2.metric("Trades imported", report.rows_imported)
c3.metric("Valid", report.valid_trades)
c4.metric("Invalid", report.invalid_trades)

if report.issues:
    issues_df = pd.DataFrame(report.issues)
    if "trade_number" in issues_df.columns:
        issues_df["trade_number"] = issues_df["trade_number"].astype("string")
    st.dataframe(issues_df, width="stretch")
    buf = io.BytesIO()
    issues_df.to_csv(buf, index=False)
    st.download_button(
        "Download validation report (CSV)",
        buf.getvalue(),
        file_name="validation_report.csv",
        mime="text/csv",
    )
elif report.valid_trades == 0:
    st.error("No valid trades imported. Check filename pattern and CSV columns.")
else:
    st.success("No validation issues.")

st.subheader("Configuration")
with st.form("config"):
    col1, col2, col3 = st.columns(3)
    initial_capital = col1.number_input("Initial capital", value=100000.0, min_value=1.0)
    buy_pct = col2.number_input("Buy % of equity", value=5.0, min_value=0.01, max_value=100.0) / 100.0
    max_sym = col3.number_input("Max % per symbol", value=20.0, min_value=0.01, max_value=100.0) / 100.0
    col4, col5, col6 = st.columns(3)
    entry_fee = col4.number_input("Entry fee %", value=0.0, min_value=0.0) / 100.0
    exit_fee = col5.number_input("Exit fee %", value=0.0, min_value=0.0) / 100.0
    slip = col6.number_input("Slippage %", value=0.0, min_value=0.0) / 100.0
    col7, col8, col9 = st.columns(3)
    multi = col7.checkbox("Allow multiple positions same symbol", value=True)
    partial = col8.checkbox("Allow partial fills", value=True)
    priority = col9.selectbox(
        "Entry priority",
        options=[e.value for e in EntryPriority],
        index=2,
    )
    run = st.form_submit_button("Run simulation", type="primary")

if run:
    try:
        config = SimulationConfig(
            initial_capital=float(initial_capital),
            buy_pct_of_equity=float(buy_pct),
            max_pct_per_symbol=float(max_sym),
            entry_fee_pct=float(entry_fee),
            exit_fee_pct=float(exit_fee),
            slippage_pct=float(slip),
            allow_multiple_positions_same_symbol=bool(multi),
            allow_partial_fills=bool(partial),
            allow_leverage=False,
            entry_priority=str(priority),
        )
        valid = [t for t in all_trades if t.is_valid_for_sim]
        if not valid:
            st.error("Nothing to simulate — no valid trades.")
            st.stop()
        result = run_simulation(trades=valid, config=config, market_data=None)
        result.validation_report = report.to_rows()

        st.subheader("Results")
        pm = result.performance_metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final equity", f"{pm.final_equity:,.2f}")
        m2.metric("Total return", f"{pm.total_return_pct * 100:.2f}%")
        m3.metric("Accepted", pm.accepted)
        m4.metric("Rejected", pm.rejected)
        st.info(pm.note)

        st.write("Trade ledger")
        ledger = pd.DataFrame([r.to_dict() for r in result.trade_results])
        st.dataframe(ledger, width="stretch")
        st.write("Event equity curve")
        curve = pd.DataFrame([s.to_dict() for s in result.portfolio_timeseries])
        st.dataframe(curve, width="stretch")
        if not curve.empty and "equity" in curve.columns:
            chart = curve.copy()
            chart["timestamp"] = chart["timestamp"].astype(str)
            st.line_chart(chart.set_index("timestamp")["equity"])

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            export_results_excel(result, tmp.name)
            xdata = Path(tmp.name).read_bytes()
        st.download_button(
            "Export Excel",
            xdata,
            file_name="msbt_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        st.error("Simulation crashed. Details:")
        st.code(traceback.format_exc())
