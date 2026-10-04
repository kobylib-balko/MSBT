"""Streamlit UI: Phase 1–4 + UX backlog + Phase 2.5 cash earn."""

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
from msbt.analytics.robustness import (
    buy_date_span,
    default_cutoff,
    run_locked_oos,
    run_stress_pack,
    run_train_vs_test,
    run_walk_forward,
)
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

BENCHMARK_PRESETS = [
    ("S&P 500 (^GSPC)", "^GSPC"),
    ("Nasdaq (^IXIC)", "^IXIC"),
    ("BND", "BND"),
    ("ACWI", "ACWI"),
    ("SCHD", "SCHD"),
    ("Custom…", "__custom__"),
]


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


def _metric_or_na(value, pct: bool = False) -> str:
    if value is None:
        return "NA"
    if pct:
        return f"{value * 100:.2f}%"
    if isinstance(value, float):
        if abs(value) == float("inf"):
            return "∞"
        return f"{value:,.4f}"
    return str(value)


def _fmt_money(v) -> str:
    if v is None:
        return "n/a"
    return f"{v:,.2f}"


def _event_curve_cash_changed(snapshots) -> pd.DataFrame:
    """Only first day + days where cash changed; Cumulative return in %."""
    rows = []
    prev_cash = None
    for i, s in enumerate(snapshots):
        d = s.to_dict()
        if i == 0 or prev_cash is None or abs((s.cash or 0) - prev_cash) > 1e-9:
            d["Cumulative return"] = (s.cumulative_return or 0.0) * 100.0
            if "Exposure %" in d and d["Exposure %"] is not None:
                d["Exposure %"] = d["Exposure %"] * 100.0
            rows.append(d)
        prev_cash = s.cash
    return pd.DataFrame(rows)


def _highlight_better(row_strategy, row_bench, higher_better: set[str], lower_better: set[str]):
    """Return CSS styles for strategy vs benchmark side-by-side table."""
    styles = []
    for col in row_strategy.index:
        styles.append("")
    return styles


def _benchmark_comparison_table(result) -> pd.DataFrame | None:
    """Strategy vs benchmark side-by-side; metrics as display-ready values."""
    bm = result.benchmark_metrics
    pm = result.performance_metrics
    rm = result.risk_metrics
    if bm is None:
        return None

    def _pct(v):
        return None if v is None else v * 100.0

    strategy = {
        "Total Return (%)": _pct(pm.total_return_pct),
        "CAGR (%)": _pct(pm.cagr),
        "Sharpe": rm.sharpe if rm else pm.sharpe,
        "Sortino": rm.sortino if rm else pm.sortino,
        "Max drawdown (%)": _pct(rm.max_drawdown if rm else pm.max_drawdown),
        "Ann. vol (%)": _pct(rm.ann_volatility if rm else pm.ann_volatility),
    }
    bench = {
        "Total Return (%)": _pct(bm.cumulative_return),
        "CAGR (%)": _pct(bm.cagr),
        "Sharpe": bm.sharpe,
        "Sortino": bm.sortino,
        "Max drawdown (%)": _pct(bm.max_drawdown),
        "Ann. vol (%)": _pct(bm.ann_volatility),
    }
    df = pd.DataFrame(
        {
            "Metric": list(strategy.keys()),
            "Strategy": list(strategy.values()),
            f"Benchmark ({bm.ticker_or_name})": list(bench.values()),
        }
    )
    return df


def _style_benchmark_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    higher = {"Total Return (%)", "CAGR (%)", "Sharpe", "Sortino"}
    lower = {"Max drawdown (%)", "Ann. vol (%)"}  # max DD is more negative = worse; lower abs better
    # For max drawdown: values are negative fractions*100; "better" = higher (less negative).
    # For ann vol: lower is better.

    def _cell(s_val, b_val, metric):
        if s_val is None or b_val is None or (isinstance(s_val, float) and pd.isna(s_val)):
            return ""
        if metric in higher:
            if s_val > b_val:
                return "background-color: #c6efce"
            if b_val > s_val:
                return "background-color: #c6efce"  # applied per-column below
        if metric == "Ann. vol (%)":
            if s_val < b_val:
                return "background-color: #c6efce"
        if metric == "Max drawdown (%)":
            # less negative (higher) is better
            if s_val > b_val:
                return "background-color: #c6efce"
        return ""

    def style_strategy(col):
        styles = []
        for i, v in enumerate(col):
            metric = df.loc[i, "Metric"]
            b = df.iloc[i, 2]
            if metric in higher and v is not None and b is not None and not pd.isna(v) and not pd.isna(b):
                styles.append("background-color: #c6efce" if v > b else "")
            elif metric == "Ann. vol (%)" and v is not None and b is not None and not pd.isna(v) and not pd.isna(b):
                styles.append("background-color: #c6efce" if v < b else "")
            elif metric == "Max drawdown (%)" and v is not None and b is not None and not pd.isna(v) and not pd.isna(b):
                styles.append("background-color: #c6efce" if v > b else "")
            else:
                styles.append("")
        return styles

    def style_bench(col):
        styles = []
        for i, v in enumerate(col):
            metric = df.loc[i, "Metric"]
            s = df.iloc[i, 1]
            if metric in higher and v is not None and s is not None and not pd.isna(v) and not pd.isna(s):
                styles.append("background-color: #c6efce" if v > s else "")
            elif metric == "Ann. vol (%)" and v is not None and s is not None and not pd.isna(v) and not pd.isna(s):
                styles.append("background-color: #c6efce" if v < s else "")
            elif metric == "Max drawdown (%)" and v is not None and s is not None and not pd.isna(v) and not pd.isna(s):
                styles.append("background-color: #c6efce" if v > s else "")
            else:
                styles.append("")
        return styles

    styler = df.style.format(
        {
            "Strategy": lambda v: "n/a" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.4f}",
            df.columns[2]: lambda v: "n/a" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.4f}",
        }
    )
    styler = styler.apply(style_strategy, subset=["Strategy"])
    styler = styler.apply(style_bench, subset=[df.columns[2]])
    return styler


def _ledger_df(result) -> pd.DataFrame:
    """Hide rejected; Gross/Net return as %; Signal column."""
    rows = []
    for r in result.trade_results:
        if r.rejection_reason:
            continue
        d = r.to_dict()
        if d.get("Gross return") is not None:
            d["Gross return"] = d["Gross return"] * 100.0
        if d.get("Net return") is not None:
            d["Net return"] = d["Net return"] * 100.0
        if d.get("Market return") is not None:
            d["Market return"] = d["Market return"] * 100.0
        rows.append(d)
    return pd.DataFrame(rows)


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
    pct_cols = [
        "buy_pct_of_equity",
        "max_pct_per_symbol",
        "entry_fee_pct",
        "exit_fee_pct",
        "slippage_pct",
        "return_pct",
        "cagr",
        "max_drawdown",
    ]
    for col in pct_cols:
        if col in show.columns:
            show[col] = show[col].apply(
                lambda v: None
                if v is None or (isinstance(v, float) and pd.isna(v))
                else (v * 100.0 if isinstance(v, (int, float)) else v)
            )
    return show


def _style_grid_best(df: pd.DataFrame):
    """Highlight best value per numeric column."""
    higher_better = {"final_equity", "return_pct", "cagr", "accepted"}
    lower_better = {"max_drawdown", "rejected", "entry_fee_pct", "exit_fee_pct", "slippage_pct"}
    # max_drawdown displayed as % (negative): higher (less negative) is better

    def apply_col(col: pd.Series):
        name = col.name
        styles = [""] * len(col)
        numeric = pd.to_numeric(col, errors="coerce")
        if numeric.notna().sum() == 0:
            return styles
        if name in higher_better or name == "max_drawdown":
            best = numeric.max()
            for i, v in enumerate(numeric):
                if pd.notna(v) and abs(v - best) < 1e-12:
                    styles[i] = "background-color: #c6efce"
        elif name in lower_better:
            best = numeric.min()
            for i, v in enumerate(numeric):
                if pd.notna(v) and abs(v - best) < 1e-12:
                    styles[i] = "background-color: #c6efce"
        return styles

    styler = df.style
    for c in df.columns:
        if c in ("scenario", "entry_priority", "simulation_id", "params"):
            continue
        styler = styler.apply(apply_col, subset=[c])
    return styler


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


def _render_trading_portfolio_stats(result) -> None:
    ts = result.trading_stats
    ps = result.portfolio_stats
    if ts is not None:
        st.subheader("Trading stats")
        c = st.columns(6)
        c[0].metric("# Trades", ts.number_of_trades)
        c[1].metric("Executed", ts.executed)
        c[2].metric("Rejected", ts.rejected)
        c[3].metric("Winning", ts.winning)
        c[4].metric("Losing", ts.losing)
        c[5].metric("Win rate", _metric_or_na(ts.win_rate, pct=True))
        c2 = st.columns(6)
        c2[0].metric("Avg win", _fmt_money(ts.avg_win))
        c2[1].metric("Avg loss", _fmt_money(ts.avg_loss))
        c2[2].metric("Largest win", _fmt_money(ts.largest_win))
        c2[3].metric("Largest loss", _fmt_money(ts.largest_loss))
        pf = ts.profit_factor
        c2[4].metric(
            "Profit factor",
            "∞" if pf == float("inf") else _metric_or_na(pf),
        )
        c2[5].metric("Avg ret / bar", _metric_or_na(ts.avg_return_per_bar, pct=True))
        c3 = st.columns(4)
        c3[0].metric("Avg hold (bars)", _metric_or_na(ts.avg_holding_bars))
        c3[1].metric("Median hold (bars)", _metric_or_na(ts.median_holding_bars))
        c3[2].metric("Max consec wins", ts.max_consecutive_wins)
        c3[3].metric("Max consec losses", ts.max_consecutive_losses)
        st.caption(
            f"Invested bars = {ts.invested_bars} ({ts.invested_bars_definition}). "
            "Avg return per bar = portfolio total return / invested bars."
        )

    if ps is not None and ps.n_days:
        st.subheader("Portfolio stats")
        p = st.columns(5)
        p[0].metric("Avg positions / day", _metric_or_na(ps.avg_positions_per_day))
        p[1].metric("Max exposure", _metric_or_na(ps.max_exposure_pct, pct=True))
        p[2].metric("Avg exposure", _metric_or_na(ps.avg_exposure_pct, pct=True))
        p[3].metric("% time invested", _metric_or_na(ps.pct_time_invested, pct=True))
        p[4].metric("% time in cash", _metric_or_na(ps.pct_time_in_cash, pct=True))
        st.caption(f"From {ps.source} equity curve ({ps.n_days} days).")



def _ensure_mfe_mae_stats(result):
    """Use attached stats, or recompute from trade_results when missing."""
    stats = getattr(result, "mfe_mae_stats", None)
    if stats is not None:
        return stats
    trs = getattr(result, "trade_results", None) or []
    if not trs:
        return None
    from msbt.analytics.mfe_mae import compute_mfe_mae_stats

    stats = compute_mfe_mae_stats(trs)
    try:
        result.mfe_mae_stats = stats
    except Exception:
        pass
    return stats


def _render_mfe_mae(result) -> None:
    stats = _ensure_mfe_mae_stats(result)
    if stats is None:
        return
    st.subheader("MFE / MAE")
    c = st.columns(5)
    c[0].metric("N with data", stats.n_with_data)
    c[1].metric("Median MFE", _metric_or_na(stats.median_mfe, pct=True))
    c[2].metric("Median MAE", _metric_or_na(stats.median_mae, pct=True))
    c[3].metric("Median giveback", _metric_or_na(stats.median_giveback, pct=True))
    c[4].metric("Median MFE/|MAE|", _metric_or_na(stats.median_mfe_mae_ratio))
    c2 = st.columns(4)
    c2[0].metric("Winners median MFE", _metric_or_na(stats.winners_median_mfe, pct=True))
    c2[1].metric("Winners median MAE", _metric_or_na(stats.winners_median_mae, pct=True))
    c2[2].metric("Losers median MFE", _metric_or_na(stats.losers_median_mfe, pct=True))
    c2[3].metric("Losers median MAE", _metric_or_na(stats.losers_median_mae, pct=True))
    st.caption(
        f"Winners n={stats.winners_n}, losers n={stats.losers_n} "
        f"(of {stats.n_closed_executed} executed closed). {stats.scope_note}"
    )
    if stats.suggested_stop_pct is not None or stats.suggested_target_pct is not None:
        s = st.columns(2)
        s[0].metric("Suggested stop %", _metric_or_na(stats.suggested_stop_pct, pct=True))
        s[1].metric("Suggested target %", _metric_or_na(stats.suggested_target_pct, pct=True))
    if stats.recommendations:
        st.info("\n".join(f"• {r}" for r in stats.recommendations))
    st.caption(
        "MFE/MAE suggestions are descriptive only — not applied automatically to this simulation."
    )


def _render_single_results(result) -> None:
    pm = result.performance_metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Final equity", f"{pm.final_equity:,.2f}")
    m2.metric("Total return", f"{pm.total_return_pct * 100:.2f}%")
    m3.metric("CAGR", _metric_or_na(pm.cagr, pct=True))
    m4.metric("Accepted", pm.accepted)
    m5.metric("Rejected", pm.rejected)
    if pm.cash_interest_pnl:
        st.caption(f"Cash interest P&L: {pm.cash_interest_pnl:,.2f}")
    if result.cash_earn_notes:
        st.caption(result.cash_earn_notes)
    st.info(pm.note)

    _render_trading_portfolio_stats(result)
    _render_mfe_mae(result)

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

    bdf = _benchmark_comparison_table(result)
    if bdf is not None:
        st.subheader("Benchmark (strategy vs benchmark)")
        try:
            st.dataframe(_style_benchmark_df(bdf), use_container_width=True)
        except Exception:
            _df(bdf)

    if result.market_data_report:
        with st.expander("Market data report / gaps"):
            _df(pd.DataFrame(result.market_data_report))
            if result.affected_dates:
                st.write("Affected valuation dates:", ", ".join(result.affected_dates[:40]))

    st.write("Trade ledger (executed only)")
    ledger = _ledger_df(result)
    _df(ledger)

    st.write("Event equity curve (first day + cash-change days)")
    curve = _event_curve_cash_changed(result.portfolio_timeseries)
    _df(curve)
    if not curve.empty and "Date" in curve.columns and "Equity" in curve.columns:
        chart = curve.copy()
        chart["Date"] = chart["Date"].astype(str)
        st.line_chart(chart.set_index("Date")["Equity"])

    # Single daily MTM chart (no duplicate table+chart pair of the same curve twice)
    if result.daily_mtm_timeseries:
        st.write("Daily MTM equity curve")
        mtm = pd.DataFrame([s.to_dict() for s in result.daily_mtm_timeseries])
        if not mtm.empty and "Cumulative return" in mtm.columns:
            mtm = mtm.copy()
            mtm["Cumulative return"] = mtm["Cumulative return"] * 100.0
        if not mtm.empty and "Exposure %" in mtm.columns:
            mtm["Exposure %"] = mtm["Exposure %"] * 100.0
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


def _render_grid_results(rows: list[dict], base_config: dict) -> None:
    st.subheader("Parameter grid")
    st.caption(
        "Each row is an independent simulation. Max DD is NA without daily MTM. "
        "CAGR uses the equity span when dates differ. Percent params shown as %."
    )
    show = _grid_display(rows)
    try:
        st.dataframe(_style_grid_best(show), use_container_width=True)
    except Exception:
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



def _phase4_metrics_table(rows: list[dict], pct_keys: list[str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    pct_keys = pct_keys or [
        "return_pct",
        "cagr",
        "max_drawdown",
        "buy_pct_of_equity",
        "delta_return_pct",
        "delta_cagr",
        "train_return_pct",
        "test_return_pct",
        "train_cagr",
        "test_cagr",
        "train_max_drawdown",
        "test_max_drawdown",
        "avg_test_return_pct",
        "compound_test_return_pct",
    ]
    show = df.copy()
    for col in pct_keys:
        if col in show.columns:
            show[col] = show[col].apply(
                lambda v: None
                if v is None or (isinstance(v, float) and pd.isna(v))
                else (v * 100.0 if isinstance(v, (int, float)) else v)
            )
    return show


def _style_stress(df: pd.DataFrame):
    """Green/red on delta_return_pct vs baseline."""

    def color_delta(col: pd.Series):
        styles = []
        for v in col:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                styles.append("")
            elif v < 0:
                styles.append("background-color: #ffc7ce; color: #9c0006")
            elif v > 0:
                styles.append("background-color: #c6efce; color: #006100")
            else:
                styles.append("")
        return styles

    styler = df.style
    if "delta_return_pct" in df.columns:
        styler = styler.apply(color_delta, subset=["delta_return_pct"])
    if "delta_cagr" in df.columns:
        styler = styler.apply(color_delta, subset=["delta_cagr"])
    return styler


def _resolve_phase4_cutoff(valid_trades, auto_70_30: bool, cutoff_str: str):
    if auto_70_30:
        return default_cutoff(valid_trades, mode="70_30"), "auto 70/30 by unique buy dates"
    if cutoff_str.strip():
        return date.fromisoformat(cutoff_str.strip()), "user cutoff"
    return default_cutoff(valid_trades, mode="median"), "median buy date"


def _render_phase4_section(valid_trades, report) -> None:
    """Phase 4 — Robustness controls and results (uses active baseline config)."""
    st.markdown("---")
    st.subheader("Phase 4 — Robustness")
    st.caption(
        "Split rule: **Train** = buy_date < cutoff; **Test** = buy_date ≥ cutoff "
        "(sell date ignored for membership). "
        "Uses the baseline config from the last **Run simulation** / **Run parameter grid**. "
        "Locked OOS / walk-forward select params on Train only (robust score: "
        "CAGR/|max DD| when DD exists, else return − |DD|, else return)."
    )

    span = buy_date_span(valid_trades)
    if span:
        st.caption(f"Buy-date span in uploaded data: **{span[0]}** → **{span[1]}** ({len(valid_trades)} valid trades).")

    cfg: SimulationConfig | None = st.session_state.get("active_config")
    if cfg is None:
        st.info(
            "Run a simulation (or parameter grid) once above to lock the baseline "
            "config for Phase 4. Buttons stay disabled until then."
        )

    c1, c2, c3 = st.columns(3)
    auto_70 = c1.checkbox("Auto 70/30 cutoff (by unique buy dates)", value=False, key="p4_auto70")
    med = default_cutoff(valid_trades, mode="median")
    default_cut = med.isoformat() if med else ""
    cutoff_str = c2.text_input(
        "Cutoff date YYYY-MM-DD",
        value=default_cut,
        disabled=auto_70,
        key="p4_cutoff",
    )
    drop_from = c3.selectbox(
        "Stress: rank drop-symbol on",
        options=["train", "full"],
        index=0,
        key="p4_drop_from",
    )

    st.markdown("##### Small grid (Locked OOS / Walk-forward)")
    g1, g2, g3 = st.columns(3)
    p4_buys = g1.multiselect(
        "Buy % of equity",
        options=[5, 10, 15, 20],
        default=[5, 10],
        key="p4_buys",
    )
    p4_prios = g2.multiselect(
        "Entry priority",
        options=[e.value for e in EntryPriority],
        default=[EntryPriority.HIGHEST_AVG_TRADE_RETURN.value],
        key="p4_prios",
    )
    p4_max = g3.text_input("Optional max % / symbol (comma % pts)", value="", key="p4_max")

    st.markdown("##### Walk-forward windows")
    w1, w2, w3, w4 = st.columns(4)
    wf_train = w1.number_input("Train years", value=4.0, min_value=0.5, step=0.5, key="p4_wf_train")
    wf_test = w2.number_input("Test years", value=1.0, min_value=0.25, step=0.25, key="p4_wf_test")
    wf_step = w3.number_input("Step years", value=1.0, min_value=0.25, step=0.25, key="p4_wf_step")
    wf_mode = w4.selectbox("Mode", options=["expanding", "rolling"], index=0, key="p4_wf_mode")

    b1, b2, b3, b4 = st.columns(4)
    disabled = cfg is None or not valid_trades
    run_tt = b1.button("Run Train vs Test", disabled=disabled, key="p4_btn_tt")
    run_oos = b2.button("Run Locked OOS (grid+test)", disabled=disabled, key="p4_btn_oos")
    run_wf = b3.button("Run Walk-forward", disabled=disabled, key="p4_btn_wf")
    run_st = b4.button("Run Stress pack", disabled=disabled, key="p4_btn_st")

    if cfg is None:
        return

    market_data = st.session_state.get("active_market_data")
    # Rebuild provider if Phase 2 was enabled but object not kept (session refresh)
    if market_data is None and st.session_state.get("active_enable_p2"):
        try:
            from msbt.market_data import YahooFinanceProvider

            market_data = YahooFinanceProvider()
            st.session_state["active_market_data"] = market_data
        except Exception:
            market_data = None

    def _p4_grid() -> dict:
        grid: dict = {}
        if p4_buys:
            grid["buy_pct_of_equity"] = [float(v) / 100.0 for v in p4_buys]
        if p4_prios:
            grid["entry_priority"] = list(p4_prios)
        if p4_max.strip():
            grid["max_pct_per_symbol"] = _parse_pct_list(p4_max)
        if not grid:
            grid = {"buy_pct_of_equity": [cfg.buy_pct_of_equity]}
        return grid

    try:
        if run_tt:
            cutoff, how = _resolve_phase4_cutoff(valid_trades, auto_70, cutoff_str)
            out = run_train_vs_test(
                valid_trades, cfg, cutoff=cutoff, market_data=market_data
            )
            st.session_state["last_phase4"] = {"kind": "train_test", "data": out, "how": how}

        if run_oos:
            cutoff, how = _resolve_phase4_cutoff(valid_trades, auto_70, cutoff_str)
            out = run_locked_oos(
                valid_trades,
                cfg,
                grid=_p4_grid(),
                cutoff=cutoff,
                market_data=market_data,
            )
            st.session_state["last_phase4"] = {"kind": "locked_oos", "data": out, "how": how}

        if run_wf:
            out = run_walk_forward(
                valid_trades,
                cfg,
                train_years=float(wf_train),
                test_years=float(wf_test),
                step_years=float(wf_step),
                mode=str(wf_mode),
                grid=_p4_grid(),
                market_data=market_data,
            )
            st.session_state["last_phase4"] = {"kind": "walk_forward", "data": out, "how": None}

        if run_st:
            cutoff, how = _resolve_phase4_cutoff(valid_trades, auto_70, cutoff_str)
            out = run_stress_pack(
                valid_trades,
                cfg,
                cutoff=cutoff,
                use_test_set=True,
                drop_symbol_from=str(drop_from),
                market_data=market_data,
            )
            st.session_state["last_phase4"] = {"kind": "stress", "data": out, "how": how}
    except Exception:
        st.error("Phase 4 run failed. Details:")
        st.code(traceback.format_exc())
        return

    payload = st.session_state.get("last_phase4")
    if not payload:
        return

    kind = payload["kind"]
    data = payload["data"]
    how = payload.get("how")

    if kind == "train_test":
        st.markdown("#### Train vs Test")
        st.caption(
            f"Cutoff **{data['cutoff']}** ({how}). {data['split_rule']} "
            f"n_train={data['n_train']}, n_test={data['n_test']}."
        )
        show = _phase4_metrics_table(data["side_by_side"])
        prefer = [
            "slice",
            "n_trades",
            "final_equity",
            "return_pct",
            "cagr",
            "sharpe",
            "max_drawdown",
            "accepted",
            "rejected",
        ]
        cols = [c for c in prefer if c in show.columns]
        _df(show[cols])
        deg = data.get("degradation") or {}
        st.caption(
            "Interpretation: compare Train vs Test on the **same** config. "
            f"Δ return (Test−Train)={_metric_or_na(deg.get('delta_return_pct'), pct=True)}; "
            f"Δ CAGR={_metric_or_na(deg.get('delta_cagr'), pct=True)}; "
            f"Δ Sharpe={_metric_or_na(deg.get('delta_sharpe'))}; "
            f"DD ratio |Test|/|Train|={_metric_or_na(deg.get('dd_ratio_test_over_train'))}."
        )

    elif kind == "locked_oos":
        st.markdown("#### Locked OOS (grid on Train → once on Test)")
        st.caption(
            f"Cutoff **{data['cutoff']}** ({how}). Selection on Train only "
            f"({data['selection_score']}). Best params: `{data['best_params']}`."
        )
        rows = [
            {"slice": "Train (locked)", **data["train_metrics"]},
            {"slice": "Test (locked)", **data["test_metrics"]},
        ]
        show = _phase4_metrics_table(rows)
        prefer = [
            "slice",
            "final_equity",
            "return_pct",
            "cagr",
            "sharpe",
            "max_drawdown",
            "accepted",
            "rejected",
            "buy_pct_of_equity",
            "entry_priority",
        ]
        cols = [c for c in prefer if c in show.columns]
        _df(show[cols])
        deg = data.get("degradation") or {}
        st.caption(
            "Interpretation: params were **not** re-fit on Test. "
            f"Δ return={_metric_or_na(deg.get('delta_return_pct'), pct=True)}; "
            f"Δ CAGR={_metric_or_na(deg.get('delta_cagr'), pct=True)}; "
            f"Δ Sharpe={_metric_or_na(deg.get('delta_sharpe'))}; "
            f"DD ratio={_metric_or_na(deg.get('dd_ratio_test_over_train'))}. "
            "Large negative Δ return / higher DD ratio ⇒ overfitting risk."
        )
        with st.expander("Train grid cells", expanded=False):
            _df(_phase4_metrics_table(data.get("train_grid") or []))

    elif kind == "walk_forward":
        st.markdown("#### Walk-forward")
        st.caption(
            f"Mode **{data['mode']}**, train={data['train_years']}y / "
            f"test={data['test_years']}y / step={data['step_years']}y. "
            f"Span {data['min_buy']} → {data['max_buy']}. "
            f"Windows={data['n_windows']} (evaluated {data['n_evaluated']})."
        )
        win_rows = []
        for w in data.get("windows") or []:
            flat = {k: v for k, v in w.items() if k != "degradation" and k != "best_params"}
            bp = w.get("best_params") or {}
            flat["best_buy_pct"] = bp.get("buy_pct_of_equity")
            flat["best_priority"] = bp.get("entry_priority")
            win_rows.append(flat)
        show = _phase4_metrics_table(win_rows)
        _df(show)
        summ = data.get("oos_summary") or {}
        st.caption(
            f"Stitched OOS: avg Test return={_metric_or_na(summ.get('avg_test_return_pct'), pct=True)}; "
            f"compound={_metric_or_na(summ.get('compound_test_return_pct'), pct=True)}. "
            f"{summ.get('note', '')}"
        )

    elif kind == "stress":
        st.markdown("#### Stress pack")
        st.caption(
            f"Eval set=**{data['eval_set']}** (cutoff {data.get('cutoff')}; {how}). "
            f"Top symbol by net PnL on **{data['drop_symbol_from']}**: `{data.get('top_symbol')}`. "
            f"{data.get('note', '')}"
        )
        show = _phase4_metrics_table(data.get("rows") or [])
        prefer = [
            "variant",
            "dropped_symbol",
            "final_equity",
            "return_pct",
            "cagr",
            "sharpe",
            "max_drawdown",
            "delta_return_pct",
            "delta_cagr",
            "accepted",
            "rejected",
        ]
        cols = [c for c in prefer if c in show.columns]
        view = show[cols]
        try:
            st.dataframe(_style_stress(view), use_container_width=True)
        except Exception:
            _df(view)
        st.caption(
            "Interpretation: red Δ return = stress underperforms baseline on the eval set; "
            "green = surprisingly better. costs×3 and partial-fills-off usually hurt; "
            "dropping the top contributor tests concentration risk."
        )



def main() -> None:
    st.set_page_config(page_title="MSBT 0.5", layout="wide")
    st.title("Multi-Symbol Portfolio Backtester — 0.5")
    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)

    st.caption(
        "Event-based capital simulation using Return %. "
        "Optional daily MTM via market data (Yahoo). "
        "Phase 2.5 cash earn on uninvested cash (EOD after entries). "
        "Phase 4 robustness: Train/Test, locked OOS, walk-forward, stress pack. "
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
        st.session_state.pop("last_phase4", None)
        st.session_state.pop("active_config", None)
        st.session_state.pop("active_enable_p2", None)
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
        # Defaults: 0.05% entry/exit fee, 0.01% slippage (stored as decimals)
        entry_fee = col4.number_input("Entry fee %", value=0.05, min_value=0.0) / 100.0
        exit_fee = col5.number_input("Exit fee %", value=0.05, min_value=0.0) / 100.0
        slip = col6.number_input("Slippage %", value=0.01, min_value=0.0) / 100.0
        col7, col8, col9 = st.columns(3)
        multi = col7.checkbox("Allow multiple positions same symbol", value=True)
        partial = col8.checkbox("Allow partial fills", value=True)
        priority = col9.selectbox(
            "Entry priority",
            options=priority_values,
            index=2,
        )

        st.markdown("#### Phase 2 — Daily MTM / risk")
        enable_p2 = st.checkbox(
            "Enable Phase 2 market data (Yahoo / daily MTM)", value=True
        )
        p2c1, p2c2, p2c3 = st.columns(3)
        open_mode = p2c1.selectbox(
            "Open valuation mode",
            options=["source", "market", "cost_basis"],
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
        p2c4, p2c5 = st.columns(2)
        rf = p2c4.number_input("Risk-free rate (annual)", value=0.0)
        sortino_t = p2c5.number_input("Sortino target (annual)", value=0.0)

        bench_labels = [b[0] for b in BENCHMARK_PRESETS]
        bench_choice = st.selectbox("Benchmark", options=bench_labels, index=0)
        custom_bench = st.text_input("Custom benchmark ticker", value="")
        val_date_str = st.text_input(
            "Valuation date YYYY-MM-DD (optional, default=last event)", value=""
        )

        st.markdown("#### Phase 2.5 — Cash earn")
        cash_mode = st.selectbox(
            "Cash earn mode",
            options=["synthetic_rf", "none", "symbol"],
            index=0,
            help="EOD cash after entries: cash *= I_t/I_{t-1}. Symbol mode is stubbed.",
        )
        cash_symbol = st.text_input("Cash earn symbol (if mode=symbol)", value="")

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

            preset_ticker = dict(BENCHMARK_PRESETS).get(bench_choice, "^GSPC")
            if preset_ticker == "__custom__":
                benchmark_ticker = custom_bench.strip() or None
            else:
                benchmark_ticker = preset_ticker

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
                benchmark_ticker=benchmark_ticker if enable_p2 else None,
                cash_earn_mode=str(cash_mode),
                cash_earn_symbol=cash_symbol.strip() or None,
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

            st.session_state["active_config"] = config
            st.session_state["active_enable_p2"] = bool(enable_p2)
            st.session_state["active_market_data"] = market_data

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

    # Phase 4 uses baseline config from last form submit
    valid_for_p4 = [t for t in all_trades if t.is_valid_for_sim]
    if valid_for_p4:
        _render_phase4_section(valid_for_p4, report)

    last = st.session_state.get("last_single")
    if last is not None:
        st.subheader("Results")
        _render_single_results(last)

    grid_rows = st.session_state.get("last_grid")
    if grid_rows:
        _render_grid_results(grid_rows, st.session_state.get("last_grid_config") or {})


if __name__ == "__main__":
    main()
