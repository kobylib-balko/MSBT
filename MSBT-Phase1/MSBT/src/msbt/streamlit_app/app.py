"""Streamlit UI: Phase 1 event sim, Phase 2 MTM, Phase 3 grid + history."""

from __future__ import annotations

import hashlib
import io
import tempfile
import traceback
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from msbt.analytics.export import export_results_excel
from msbt.analytics.grid import run_parameter_grid
from msbt.analytics.history import (
    compare_runs,
    delete_run,
    get_run,
    list_runs,
    save_run,
    update_run_meta,
)
from msbt.importers.csv_importer import import_trade_file
from msbt.models.config import EntryPriority, SimulationConfig
from msbt.simulation.engine import run_simulation
from msbt.validation.validate import validate_trades


def _df(data):
    try:
        return st.dataframe(data, use_container_width=True)
    except TypeError:
        return st.dataframe(data)


def _parse_pct_list(text: str) -> list[float]:
    out: list[float] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part) / 100.0)
    return out


def _file_sig(uploaded) -> str:
    h = hashlib.sha256()
    for uf in uploaded:
        h.update(uf.name.encode("utf-8", errors="replace"))
        h.update(b"\0")
        h.update(str(getattr(uf, "size", len(uf.getvalue()))).encode())
    return h.hexdigest()


def _source_hash(uploaded) -> str:
    h = hashlib.sha256()
    for uf in uploaded:
        h.update(uf.name.encode("utf-8", errors="replace"))
        h.update(b"\0")
        h.update(uf.getvalue())
    return h.hexdigest()


def _render_history_sidebar() -> None:
    st.sidebar.header("Past runs")
    try:
        runs = list_runs()
    except Exception as exc:
        st.sidebar.warning(f"History unavailable: {exc}")
        return

    if not runs:
        st.sidebar.caption("No saved runs yet. Save a simulation or grid after it finishes.")
        return

    labels = {}
    for rec in runs:
        title = rec.get("name") or rec["id"][:8]
        ret = rec.get("return_pct")
        ret_s = f"{ret * 100:.2f}%" if isinstance(ret, (int, float)) else "n/a"
        labels[rec["id"]] = f"{title} · {ret_s} · {rec['created_at']}"

    ids = list(labels)
    selected = st.sidebar.multiselect(
        "Select runs",
        options=ids,
        format_func=lambda i: labels[i],
    )
    existing = set(ids)
    selected = [i for i in selected if i in existing]

    c1, c2 = st.sidebar.columns(2)
    if c1.button("Open", disabled=len(selected) != 1):
        st.session_state["opened_run_id"] = selected[0]
    if c2.button("Compare", disabled=len(selected) < 2):
        st.session_state["compare_ids"] = list(selected)

    if st.sidebar.button("Delete selected", disabled=not selected):
        for run_id in selected:
            delete_run(run_id)
        if st.session_state.get("opened_run_id") in selected:
            st.session_state.pop("opened_run_id", None)
        st.session_state["flash"] = f"Deleted {len(selected)} run(s)."
        st.rerun()

    if st.sidebar.button("Close viewer"):
        st.session_state.pop("opened_run_id", None)
        st.session_state.pop("compare_ids", None)

    if len(selected) == 1:
        try:
            rec = get_run(selected[0])
        except KeyError:
            return
        with st.sidebar.form("edit_run_meta"):
            name = st.text_input("Name", value=rec.get("name") or "")
            tags = st.text_input("Tags", value=", ".join(rec.get("tags") or []))
            notes = st.text_area("Notes", value=rec.get("notes") or "")
            if st.form_submit_button("Update notes"):
                update_run_meta(selected[0], name, tags, notes)
                st.success("Updated.")


def _metric_or_na(value, pct: bool = False) -> str:
    if value is None:
        return "NA"
    if pct:
        return f"{value * 100:.2f}%"
    if isinstance(value, float):
        return f"{value:,.4f}"
    return str(value)


def _render_history_panels() -> None:
    opened = st.session_state.get("opened_run_id")
    compare_ids = st.session_state.get("compare_ids")

    if compare_ids:
        st.subheader("Compare saved runs")
        try:
            table = compare_runs(compare_ids)
        except KeyError as exc:
            st.error(str(exc))
        else:
            df = pd.DataFrame(table)
            for col in ("cagr", "max_drawdown"):
                if col in df.columns:
                    df[col] = df[col].apply(lambda v: "NA" if v is None else v)
            _df(df)
            st.download_button(
                "Download comparison CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="msbt_compare_runs.csv",
                mime="text/csv",
                key="dl_compare_runs",
            )

    if opened:
        st.subheader("Opened run")
        try:
            rec = get_run(opened)
        except KeyError:
            st.warning("That run is no longer in history.")
            return
        st.markdown(
            f"**{rec.get('name') or rec['id']}**  \n"
            f"`{rec['id']}` · {rec['created_at']} UTC · {rec.get('kind')}"
        )
        if rec.get("notes"):
            st.caption(rec["notes"])
        summary = rec.get("summary") or {}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final equity", _metric_or_na(summary.get("final_equity")))
        m2.metric("Return", _metric_or_na(summary.get("return_pct"), pct=True))
        m3.metric("CAGR", _metric_or_na(summary.get("cagr"), pct=True))
        m4.metric("Max DD", _metric_or_na(summary.get("max_drawdown"), pct=True))
        with st.expander("Config JSON", expanded=False):
            st.json(rec.get("config") or {})
        equity = rec.get("equity") or {}
        event = equity.get("event") or []
        if event:
            curve = pd.DataFrame(event)
            st.write("Event equity curve")
            _df(curve)
            if "Date" in curve.columns and "Equity" in curve.columns:
                chart = curve.copy()
                chart["Date"] = chart["Date"].astype(str)
                st.line_chart(chart.set_index("Date")["Equity"])
        grid = rec.get("grid_comparison")
        if grid:
            st.write("Grid comparison")
            gdf = pd.DataFrame(grid)
            _df(gdf)


def _render_single_results(result) -> None:
    pm = result.performance_metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final equity", f"{pm.final_equity:,.2f}")
    m2.metric("Total return", f"{pm.total_return_pct * 100:.2f}%")
    m3.metric("Accepted", pm.accepted)
    m4.metric("Rejected", pm.rejected)
    st.info(pm.note)

    if result.risk_metrics is not None:
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

    if result.open_valuations:
        st.subheader("Open position valuations")
        ov_df = pd.DataFrame(result.open_valuations)
        _df(ov_df)
        flagged = [o for o in result.open_valuations if o.get("discrepancy_flag")]
        if flagged:
            st.warning(f"{len(flagged)} open position(s) exceed discrepancy threshold.")

    if result.benchmark_metrics is not None:
        st.subheader("Benchmark")
        _df(pd.DataFrame([result.benchmark_metrics.to_dict()]))

    if result.market_data_report:
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

    if result.daily_mtm_timeseries:
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

    st.subheader("Save this run")
    with st.form("save_single_run"):
        name = st.text_input("Run name", value=f"sim {result.simulation_id[:8]}")
        tags = st.text_input("Tags (comma-separated)", value="")
        notes = st.text_area("Notes", value="")
        save = st.form_submit_button("Save this run")
    if save:
        try:
            rid = save_run(
                result,
                name=name,
                tags=tags,
                notes=notes,
                source_hash=st.session_state.get("source_hash"),
            )
            st.session_state["flash"] = f"Saved run `{rid}`."
            st.rerun()
        except Exception:
            st.error("Could not save run.")
            st.code(traceback.format_exc())


def _grid_display(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    prefer = [
        "scenario",
        "buy_pct_of_equity",
        "entry_priority",
        "max_pct_per_symbol",
        "entry_fee_pct",
        "exit_fee_pct",
        "slippage_pct",
        "final_equity",
        "return_pct",
        "cagr",
        "max_drawdown",
        "accepted",
        "rejected",
        "partial",
        "simulation_id",
    ]
    cols = [c for c in prefer if c in df.columns]
    extra = [c for c in df.columns if c not in cols and c != "params"]
    show = df[cols + extra].copy()
    for col in ("cagr", "max_drawdown"):
        if col in show.columns:
            show[col] = show[col].apply(lambda v: "NA" if v is None or (isinstance(v, float) and pd.isna(v)) else v)
    return show


def _render_grid_results(rows: list[dict], base_config: dict) -> None:
    st.subheader("Parameter grid")
    st.caption(
        "Each row is an independent simulation. Max DD is NA without daily MTM. "
        "CAGR uses the equity span when dates differ."
    )
    show = _grid_display(rows)
    _df(show)
    st.download_button(
        "Download grid CSV",
        show.to_csv(index=False).encode("utf-8"),
        file_name="msbt_parameter_grid.csv",
        mime="text/csv",
        key="dl_grid_csv",
    )
    with st.form("save_grid_run"):
        name = st.text_input("Grid name", value="parameter grid")
        tags = st.text_input("Tags", value="grid")
        notes = st.text_area("Notes", value="")
        save = st.form_submit_button("Save this grid")
    if save:
        try:
            rid = save_run(
                name=name,
                tags=tags,
                notes=notes,
                grid_comparison=rows,
                config=base_config,
                source_hash=st.session_state.get("source_hash"),
            )
            st.session_state["flash"] = f"Saved grid `{rid}`."
            st.rerun()
        except Exception:
            st.error("Could not save grid.")
            st.code(traceback.format_exc())


def _flash(message: str) -> None:
    st.session_state["flash"] = message


def main() -> None:
    st.set_page_config(page_title="MSBT Phase 3", layout="wide")
    st.title("Multi-Symbol Portfolio Backtester — Phase 3")
    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)

    st.caption(
        "Event-based capital simulation using Return %. "
        "Optional daily MTM via market data (Yahoo). "
        "Parameter grid and saved-run history reuse the same trades without mutating them. "
        "Average trade return != portfolio return (sizing, overlap, cash, rejects)."
    )

    _render_history_sidebar()
    _render_history_panels()

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

    sig = _file_sig(uploaded)
    if st.session_state.get("file_sig") != sig:
        st.session_state["file_sig"] = sig
        st.session_state.pop("last_single", None)
        st.session_state.pop("last_grid", None)
        st.session_state.pop("last_grid_config", None)
    st.session_state["source_hash"] = _source_hash(uploaded)

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
    priority_values = [e.value for e in EntryPriority]
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
            options=priority_values,
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

        st.markdown("#### Phase 3 — Parameter grid")
        st.caption(
            "Independent sims over the selected axes. Defaults stay small so a demo is fast. "
            "Optional axes are ignored when left blank."
        )
        g1, g2 = st.columns(2)
        buy_pct_choices = g1.multiselect(
            "Grid buy % of equity",
            options=[5, 10, 15, 20],
            default=[5, 10],
        )
        priority_choices = g2.multiselect(
            "Grid entry priority modes",
            options=priority_values,
            default=[EntryPriority.HIGHEST_AVG_TRADE_RETURN.value],
        )
        with st.expander("Optional grid axes (comma-separated percent points)", expanded=False):
            extra_max_sym = st.text_input("Max % per symbol values", value="")
            extra_entry_fee = st.text_input("Entry fee % values", value="")
            extra_exit_fee = st.text_input("Exit fee % values", value="")
            extra_slip = st.text_input("Slippage % values", value="")

        run = st.form_submit_button("Run simulation")
        run_grid = st.form_submit_button("Run parameter grid")

    if run or run_grid:
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
                st.info(
                    "Phase 2 enabled: one market-data provider is reused "
                    "(cache under data/market_cache or /tmp)."
                )

            if run:
                result = run_simulation(trades=valid, config=config, market_data=market_data)
                result.validation_report = report.to_rows()
                st.session_state["last_single"] = result

            if run_grid:
                grid: dict = {}
                if buy_pct_choices:
                    grid["buy_pct_of_equity"] = [float(v) / 100.0 for v in buy_pct_choices]
                if priority_choices:
                    grid["entry_priority"] = list(priority_choices)
                if extra_max_sym.strip():
                    grid["max_pct_per_symbol"] = _parse_pct_list(extra_max_sym)
                if extra_entry_fee.strip():
                    grid["entry_fee_pct"] = _parse_pct_list(extra_entry_fee)
                if extra_exit_fee.strip():
                    grid["exit_fee_pct"] = _parse_pct_list(extra_exit_fee)
                if extra_slip.strip():
                    grid["slippage_pct"] = _parse_pct_list(extra_slip)
                n = 1
                for vals in grid.values():
                    n *= max(len(vals), 1)
                if n > 24:
                    st.warning(f"Grid has {n} scenarios; this may take a while.")
                rows = run_parameter_grid(
                    valid, config, grid, market_data=market_data
                )
                st.session_state["last_grid"] = rows
                st.session_state["last_grid_config"] = config.to_dict()
        except Exception:
            st.error("Simulation crashed. Details:")
            st.code(traceback.format_exc())

    last = st.session_state.get("last_single")
    if last is not None:
        st.subheader("Results")
        _render_single_results(last)

    grid_rows = st.session_state.get("last_grid")
    if grid_rows:
        _render_grid_results(grid_rows, st.session_state.get("last_grid_config") or {})


if __name__ == "__main__":
    main()
