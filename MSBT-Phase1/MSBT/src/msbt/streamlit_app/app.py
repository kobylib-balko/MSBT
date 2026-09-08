"""Streamlit UI: Phase 1 event sim + optional Phase 2 daily MTM / risk."""

from __future__ import annotations

import io
import tempfile
import traceback
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from msbt.analytics.export import export_results_excel
from msbt.importers.csv_importer import import_trade_file
from msbt.models.config import EntryPriority, SimulationConfig
from msbt.simulation.engine import run_simulation
from msbt.validation.validate import validate_trades


def _df(data):
    try:
        return st.dataframe(data, use_container_width=True)
    except TypeError:
        return st.dataframe(data)


def main() -> None:
    st.set_page_config(page_title="MSBT Phase 2", layout="wide")
    st.title("Multi-Symbol Portfolio Backtester — Phase 2")
    st.caption(
        "Event-based capital simulation using Return %. "
        "Optional daily MTM via market data (Yahoo). "
        "Average trade return != portfolio return (sizing, overlap, cash, rejects)."
    )

    with st.expander("Expected file names", expanded=False):
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
        return

    all_trades = []
    all_issues = []
    file_count = 0

    try:
        for uf in uploaded:
            file_count += 1
            data = uf.getvalue()
            st.write(f"Processing `{uf.name}` ({len(data)} bytes)…")
            trades, issues = import_trade_file(data, filename=uf.name)
            all_trades.extend(trades)
            all_issues.extend(issues)

        all_trades, report = validate_trades(
            all_trades, prior_issues=all_issues, files_uploaded=file_count
        )
    except Exception:
        st.error("Upload/import crashed. Details:")
        st.code(traceback.format_exc())
        return

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
        _df(issues_df)
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
        buy_pct = (
            col2.number_input("Buy % of equity", value=5.0, min_value=0.01, max_value=100.0)
            / 100.0
        )
        max_sym = (
            col3.number_input("Max % per symbol", value=20.0, min_value=0.01, max_value=100.0)
            / 100.0
        )
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

        st.markdown("#### Phase 2 — Daily MTM / risk")
        enable_p2 = st.checkbox("Enable Phase 2 market data (Yahoo / daily MTM)", value=False)
        p2c1, p2c2, p2c3 = st.columns(3)
        open_mode = p2c1.selectbox(
            "Open valuation mode",
            options=["market", "source", "cost_basis"],
            index=0,
        )
        yahoo_adj = p2c2.selectbox(
            "Yahoo price adjustment",
            options=["adjusted", "unadjusted"],
            index=0,
        )
        disc_thr = p2c3.number_input(
            "Discrepancy threshold %", value=2.0, min_value=0.0
        ) / 100.0
        p2c4, p2c5, p2c6 = st.columns(3)
        rf = p2c4.number_input("Risk-free rate (annual)", value=0.0)
        sortino_t = p2c5.number_input("Sortino target (annual)", value=0.0)
        bench = p2c6.text_input("Benchmark Yahoo ticker (optional)", value="")
        val_date_str = st.text_input(
            "Valuation date YYYY-MM-DD (optional, default=last event)", value=""
        )

        run = st.form_submit_button("Run simulation")

    if not run:
        return

    try:
        valuation_date = None
        if val_date_str.strip():
            valuation_date = date.fromisoformat(val_date_str.strip())

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
            open_valuation_mode=str(open_mode) if enable_p2 else "cost_basis",
            risk_free_rate=float(rf),
            sortino_target=float(sortino_t),
            discrepancy_threshold_pct=float(disc_thr),
            yahoo_price_adjustment=str(yahoo_adj),
            valuation_date=valuation_date,
            benchmark_ticker=bench.strip() or None,
        )
        valid = [t for t in all_trades if t.is_valid_for_sim]
        if not valid:
            st.error("Nothing to simulate — no valid trades.")
            return

        market_data = None
        if enable_p2:
            from msbt.market_data import YahooFinanceProvider

            market_data = YahooFinanceProvider()
            st.info("Phase 2 enabled: fetching market data (cached under data/market_cache or /tmp).")

        result = run_simulation(trades=valid, config=config, market_data=market_data)
        result.validation_report = report.to_rows()

        st.subheader("Results")
        pm = result.performance_metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final equity", f"{pm.final_equity:,.2f}")
        m2.metric("Total return", f"{pm.total_return_pct * 100:.2f}%")
        m3.metric("Accepted", pm.accepted)
        m4.metric("Rejected", pm.rejected)
        st.info(pm.note)

        if enable_p2 and result.risk_metrics is not None:
            st.subheader("Risk metrics (daily MTM)")
            r1, r2, r3, r4, r5 = st.columns(5)
            rm = result.risk_metrics
            r1.metric(
                "Ann. vol",
                f"{rm.ann_volatility * 100:.2f}%" if rm.ann_volatility is not None else "n/a",
            )
            r2.metric("Sharpe", f"{rm.sharpe:.3f}" if rm.sharpe is not None else "n/a")
            r3.metric("Sortino", f"{rm.sortino:.3f}" if rm.sortino is not None else "n/a")
            r4.metric(
                "Max DD",
                f"{rm.max_drawdown * 100:.2f}%" if rm.max_drawdown is not None else "n/a",
            )
            r5.metric(
                "DD duration (days)",
                f"{rm.max_drawdown_duration_days}"
                if rm.max_drawdown_duration_days is not None
                else "n/a",
            )
            if rm.unavailable_reasons:
                st.warning(f"Unavailable metrics: {rm.unavailable_reasons}")

        if enable_p2 and result.open_valuations:
            st.subheader("Open position valuations")
            ov_df = pd.DataFrame(result.open_valuations)
            _df(ov_df)
            flagged = [o for o in result.open_valuations if o.get("discrepancy_flag")]
            if flagged:
                st.warning(f"{len(flagged)} open position(s) exceed discrepancy threshold.")

        if enable_p2 and result.benchmark_metrics is not None:
            st.subheader("Benchmark")
            _df(pd.DataFrame([result.benchmark_metrics.to_dict()]))

        if enable_p2 and result.market_data_report:
            with st.expander("Market data report / gaps"):
                _df(pd.DataFrame(result.market_data_report))
                if result.affected_dates:
                    st.write("Affected valuation dates:", ", ".join(result.affected_dates[:40]))

        st.write("Trade ledger")
        _df(pd.DataFrame([r.to_dict() for r in result.trade_results]))

        st.write("Event equity curve")
        curve = pd.DataFrame([s.to_dict() for s in result.portfolio_timeseries])
        _df(curve)
        if not curve.empty:
            chart = curve.copy()
            date_col = "Date" if "Date" in chart.columns else "timestamp"
            if date_col in chart.columns:
                chart[date_col] = chart[date_col].astype(str)
                eq_col = "Equity" if "Equity" in chart.columns else "equity"
                if eq_col in chart.columns:
                    st.line_chart(chart.set_index(date_col)[eq_col])

        if enable_p2 and result.daily_mtm_timeseries:
            st.write("Daily MTM equity curve")
            mtm = pd.DataFrame([s.to_dict() for s in result.daily_mtm_timeseries])
            _df(mtm)
            if not mtm.empty and "Date" in mtm.columns and "Equity" in mtm.columns:
                mtm_chart = mtm.copy()
                mtm_chart["Date"] = mtm_chart["Date"].astype(str)
                st.line_chart(mtm_chart.set_index("Date")["Equity"])

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


if __name__ == "__main__":
    main()
