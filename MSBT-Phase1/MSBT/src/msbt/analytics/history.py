"""Persist and compare completed backtests (Phase 3).

SQLite file defaults to ``data/runs.sqlite`` (gitignored). Timestamps are stored
as UTC ISO-8601. Display code may render that ISO string as-is.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from msbt.analytics.grid import extract_run_metrics
from msbt.models.config import SimulationConfig
from msbt.models.results import SimulationResult

PathLike = Union[str, Path]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    tags_json TEXT NOT NULL DEFAULT '[]',
    notes TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT 'single',
    config_json TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    equity_json TEXT NOT NULL,
    ledger_counts_json TEXT NOT NULL,
    grid_json TEXT,
    engine_version TEXT,
    source_hash TEXT
);
"""

COMPARE_COLUMNS = (
    "id",
    "name",
    "created_at",
    "tags",
    "notes",
    "kind",
    "buy_pct_of_equity",
    "entry_priority",
    "max_pct_per_symbol",
    "initial_capital",
    "final_equity",
    "return_pct",
    "cagr",
    "max_drawdown",
    "accepted",
    "rejected",
    "partial",
)


def default_runs_db() -> Path:
    """Prefer project ``data/runs.sqlite``; fall back to /tmp if not writable."""
    here = Path(__file__).resolve()
    project = here.parents[3]  # src/msbt/analytics -> MSBT
    candidate = project / "data" / "runs.sqlite"
    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        probe = candidate.parent / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return candidate
    except OSError:
        tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "msbt_runs.sqlite"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        return tmp


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, separators=(",", ":"))


def _loads(raw: Optional[str], fallback: Any) -> Any:
    if raw is None or raw == "":
        return fallback
    return json.loads(raw)


def _normalize_tags(tags: str | Sequence[str] | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        parts = [p.strip() for p in tags.split(",")]
        return [p for p in parts if p]
    return [str(t).strip() for t in tags if str(t).strip()]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _connect(db_path: PathLike) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def _resolve_db(db_path: Optional[PathLike]) -> Path:
    return Path(db_path) if db_path is not None else default_runs_db()


def _config_dict(config: SimulationConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, SimulationConfig):
        return config.to_dict()
    return dict(config)


def _equity_payload(result: SimulationResult) -> dict[str, Any]:
    return {
        "event": [s.to_dict() for s in result.portfolio_timeseries],
        "daily_mtm": [s.to_dict() for s in result.daily_mtm_timeseries],
    }


def _ledger_counts(result: SimulationResult) -> dict[str, Any]:
    pm = result.performance_metrics
    return {
        "candidates": pm.candidates,
        "accepted": pm.accepted,
        "partial": pm.partial,
        "rejected": pm.rejected,
        "rejected_by_reason": pm.rejected_by_reason,
        "trade_count": len(result.trade_results),
        "open": sum(1 for t in result.trade_results if t.is_open),
    }


def _engine_version() -> str:
    try:
        from msbt import __version__

        return str(__version__)
    except Exception:
        return ""


def _strip_grid_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Drop non-JSON values so grid comparison can be stored."""
    out: list[dict[str, Any]] = []
    for row in rows:
        clean: dict[str, Any] = {}
        for k, v in row.items():
            if k == "result":
                continue
            try:
                json.dumps(v, default=_json_default)
            except TypeError:
                continue
            clean[k] = v
        out.append(clean)
    return out


def save_run(
    result: SimulationResult | None = None,
    *,
    name: str = "",
    tags: str | Sequence[str] | None = None,
    notes: str = "",
    grid_comparison: Optional[Sequence[Mapping[str, Any]]] = None,
    config: SimulationConfig | Mapping[str, Any] | None = None,
    source_hash: Optional[str] = None,
    db_path: Optional[PathLike] = None,
) -> str:
    """Persist a completed single backtest or a parameter-grid comparison.

    Returns the run id (the simulation id for a single result, else a new id).
    """
    if result is None and not grid_comparison:
        raise ValueError("save_run requires a simulation result and/or grid_comparison")

    tag_list = _normalize_tags(tags)
    kind = "grid" if grid_comparison else "single"
    run_id = result.simulation_id if result is not None and result.simulation_id else str(uuid.uuid4())

    if result is not None:
        cfg = result.simulation_config or _config_dict(config)
        metrics = extract_run_metrics(result)
        summary = {
            **metrics,
            "performance": result.performance_metrics.to_dict(),
            "kind": kind,
            "name": name,
        }
        equity = _equity_payload(result)
        ledger = _ledger_counts(result)
    else:
        cfg = _config_dict(config)
        grid_rows = _strip_grid_rows(grid_comparison or [])
        first = grid_rows[0] if grid_rows else {}
        summary = {
            "kind": "grid",
            "n_scenarios": len(grid_rows),
            "simulation_id": run_id,
            "buy_pct_of_equity": cfg.get("buy_pct_of_equity", first.get("buy_pct_of_equity")),
            "entry_priority": cfg.get("entry_priority", first.get("entry_priority")),
            "max_pct_per_symbol": cfg.get("max_pct_per_symbol"),
            "initial_capital": first.get("initial_capital"),
            "final_equity": first.get("final_equity"),
            "return_pct": first.get("return_pct"),
            "cagr": first.get("cagr"),
            "max_drawdown": first.get("max_drawdown"),
            "accepted": first.get("accepted"),
            "rejected": first.get("rejected"),
            "partial": first.get("partial"),
            "name": name,
        }
        equity = {"event": [], "daily_mtm": []}
        ledger = {
            "candidates": None,
            "accepted": first.get("accepted"),
            "partial": first.get("partial"),
            "rejected": first.get("rejected"),
            "rejected_by_reason": {},
            "trade_count": None,
            "open": None,
            "n_scenarios": len(grid_rows),
        }

    grid_json = _dumps(_strip_grid_rows(grid_comparison)) if grid_comparison else None
    created = _utc_now()
    path = _resolve_db(db_path)

    with _connect(path) as conn:
        existing = conn.execute("SELECT created_at FROM runs WHERE id = ?", (run_id,)).fetchone()
        if existing is not None:
            created = existing["created_at"]
        conn.execute(
            """
            INSERT INTO runs (
                id, created_at, name, tags_json, notes, kind,
                config_json, summary_json, equity_json, ledger_counts_json,
                grid_json, engine_version, source_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                tags_json = excluded.tags_json,
                notes = excluded.notes,
                kind = excluded.kind,
                config_json = excluded.config_json,
                summary_json = excluded.summary_json,
                equity_json = excluded.equity_json,
                ledger_counts_json = excluded.ledger_counts_json,
                grid_json = excluded.grid_json,
                engine_version = excluded.engine_version,
                source_hash = excluded.source_hash
            """,
            (
                run_id,
                created,
                name or "",
                _dumps(tag_list),
                notes or "",
                kind,
                _dumps(cfg),
                _dumps(summary),
                _dumps(equity),
                _dumps(ledger),
                grid_json,
                _engine_version(),
                source_hash,
            ),
        )
    return run_id


def _row_to_summary(row: sqlite3.Row) -> dict[str, Any]:
    summary = _loads(row["summary_json"], {})
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "name": row["name"],
        "tags": _loads(row["tags_json"], []),
        "notes": row["notes"],
        "kind": row["kind"],
        "engine_version": row["engine_version"],
        "final_equity": summary.get("final_equity"),
        "return_pct": summary.get("return_pct"),
        "cagr": summary.get("cagr"),
        "max_drawdown": summary.get("max_drawdown"),
        "accepted": summary.get("accepted"),
        "rejected": summary.get("rejected"),
    }


def list_runs(db_path: Optional[PathLike] = None) -> list[dict[str, Any]]:
    """List saved runs (newest first), without equity curves or grid payloads."""
    path = _resolve_db(db_path)
    if not path.exists():
        return []
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC, id ASC"
        ).fetchall()
    return [_row_to_summary(r) for r in rows]


def get_run(run_id: str, db_path: Optional[PathLike] = None) -> dict[str, Any]:
    """Load one run including config, summary, equity, and optional grid."""
    path = _resolve_db(db_path)
    with _connect(path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise KeyError(f"run not found: {run_id}")
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "name": row["name"],
        "tags": _loads(row["tags_json"], []),
        "notes": row["notes"],
        "kind": row["kind"],
        "config": _loads(row["config_json"], {}),
        "summary": _loads(row["summary_json"], {}),
        "equity": _loads(row["equity_json"], {"event": [], "daily_mtm": []}),
        "ledger_counts": _loads(row["ledger_counts_json"], {}),
        "grid_comparison": _loads(row["grid_json"], None),
        "engine_version": row["engine_version"],
        "source_hash": row["source_hash"],
    }


def update_run_meta(
    run_id: str,
    name: str,
    tags: str | Sequence[str] | None,
    notes: str,
    db_path: Optional[PathLike] = None,
) -> None:
    """Edit user-facing name, tags, and notes. Does not change simulation outputs."""
    path = _resolve_db(db_path)
    tag_list = _normalize_tags(tags)
    with _connect(path) as conn:
        cur = conn.execute(
            "UPDATE runs SET name = ?, tags_json = ?, notes = ? WHERE id = ?",
            (name or "", _dumps(tag_list), notes or "", run_id),
        )
        if cur.rowcount == 0:
            raise KeyError(f"run not found: {run_id}")


def delete_run(run_id: str, db_path: Optional[PathLike] = None) -> None:
    """Delete a saved run. Raises KeyError if the id is unknown."""
    path = _resolve_db(db_path)
    with _connect(path) as conn:
        cur = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        if cur.rowcount == 0:
            raise KeyError(f"run not found: {run_id}")


def compare_runs(
    ids: Sequence[str],
    db_path: Optional[PathLike] = None,
) -> list[dict[str, Any]]:
    """Aligned summary table for the given run ids (request order preserved)."""
    if not ids:
        return []
    table: list[dict[str, Any]] = []
    for run_id in ids:
        rec = get_run(run_id, db_path=db_path)
        summary = rec.get("summary") or {}
        config = rec.get("config") or {}
        row = {
            "id": rec["id"],
            "name": rec["name"],
            "created_at": rec["created_at"],
            "tags": ", ".join(rec.get("tags") or []),
            "notes": rec["notes"],
            "kind": rec["kind"],
            "buy_pct_of_equity": summary.get("buy_pct_of_equity", config.get("buy_pct_of_equity")),
            "entry_priority": summary.get("entry_priority", config.get("entry_priority")),
            "max_pct_per_symbol": summary.get(
                "max_pct_per_symbol", config.get("max_pct_per_symbol")
            ),
            "initial_capital": summary.get("initial_capital", config.get("initial_capital")),
            "final_equity": summary.get("final_equity"),
            "return_pct": summary.get("return_pct"),
            "cagr": summary.get("cagr"),
            "max_drawdown": summary.get("max_drawdown"),
            "accepted": summary.get("accepted"),
            "rejected": summary.get("rejected"),
            "partial": summary.get("partial"),
        }
        table.append(row)
    return table
