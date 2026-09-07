"""Minimal Streamlit UI: upload → validate → configure → run → results → export."""

from __future__ import annotations

import io
import tempfile
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

uploaded = st.file_uploader(
    "Upload trade CSV/Excel files",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
)

all_trades = []
all_issues = []
file_count = 0

if uploaded:
    for uf in uploaded:
        file_count += 1
        data = uf.read()
        trades, issues = import_trade_file(data, filename=uf.name)
        all_trades.extend(trades)
        all_issues.extend(issues)

    all_trades, report = validate_trades(
        all_trades, prior_issues=all_issues, files_uploaded=file_count
    )

    st.subheader("Validation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files", report.files_uploaded)
    c2.metric("Trades imported", report.rows_imported)
    c3.metric("Valid", report.valid_trades)
    c4.metric("Invalid", report.invalid_trades)

    if report.issues:
        st.dataframe(pd.DataFrame(report.issues), use_container_width=True)
        buf = io.BytesIO()
        pd.DataFrame(report.issues).to_csv(buf, index=False)
        st.download_button(
            "Download validation report (CSV)",
            buf.getvalue(),
            file_name="validation_report.csv",
            mime="text/csv",
        )

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
        run = st.form_submit_button("Run simulation")

    if run:
        config = SimulationConfig(
            initial_capital=initial_capital,
            buy_pct_of_equity=buy_pct,
            max_pct_per_symbol=max_sym,
            entry_fee_pct=entry_fee,
            exit_fee_pct=exit_fee,
            slippage_pct=slip,
            allow_multiple_positions_same_symbol=multi,
            allow_partial_fills=partial,
            allow_leverage=False,
            entry_priority=priority,
        )
        valid = [t for t in all_trades if t.is_valid_for_sim]
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
        st.dataframe(
            pd.DataFrame([r.to_dict() for r in result.trade_results]),
            use_container_width=True,
        )
        st.write("Event equity curve")
        st.dataframe(
            pd.DataFrame([s.to_dict() for s in result.portfolio_timeseries]),
            use_container_width=True,
        )

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            export_results_excel(result, tmp.name)
            data = Path(tmp.name).read_bytes()
        st.download_button(
            "Export Excel",
            data,
            file_name="msbt_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
else:
    st.info(
        "Upload files named `{strategy}_{exchange}_{symbol}_{YYYY-MM-DD}.csv`. "
        "Return % in files is percent points (11.95 → 0.1195 in math)."
    )
