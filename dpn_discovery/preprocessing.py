"""
Step 1 — Event Log Preprocessing.

Parses a raw event log and extracts:
  • Activity set  Σ
  • Decision-variable set  V
  • Structured ``EventLog`` with per-event data bindings

Supported formats:
  • **XES** (.xes, .xes.gz) — loaded via pm4py
  • **CSV** (.csv)           — loaded via pm4py
  • **JSON** (.json)         — legacy internal format

All columns *not* listed in ``XES_STANDARD_COLUMNS`` are treated
as data variables (decision attributes).

Reference: §4 Step 1 of the specification.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pm4py

from dpn_discovery.models import DataPayload, Event, EventLog, Trace

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# XES standard columns — metadata, NOT data variables
# ═══════════════════════════════════════════════════════════════════════════

XES_STANDARD_COLUMNS: frozenset[str] = frozenset({
    # Core XES attributes
    "concept:name",
    "case:concept:name",
    "time:timestamp",
    "lifecycle:transition",
    "org:resource",
    "org:group",
    "org:role",
    # Common XES extensions
    "identity:id",
    "cost:total",
    "cost:currency",
    # Case-level attributes (prefixed with case:)
    "case:variant",
    "case:variant-index",
    "case:creator",
    # Renamed standard columns
    "case_id",
    "activity",
    "timestamp",
    # Other common metadata
    "@@index",
    "@@case_index",
    "@@event_index",
    "@@startevent_timestamp",
    "@@classifier",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_event_log(path: str | Path) -> EventLog:
    """Load an event log from a file and return a preprocessed ``EventLog``.

    Supported extensions:
      • ``.xes`` / ``.xes.gz``  — XES format (loaded via pm4py)
      • ``.csv``                — CSV format  (loaded via pm4py)
      • ``.json``               — legacy JSON format

    All columns not in ``XES_STANDARD_COLUMNS`` are treated as data
    variables.
    """
    filepath = Path(path)
    suffix = filepath.suffixes  # e.g. ['.xes'] or ['.xes', '.gz']
    ext = suffix[0].lower() if suffix else ""

    if ext == ".json":
        return _load_json(filepath)
    elif ext == ".xes":
        return _load_xes(filepath)
    elif ext == ".csv":
        return _load_csv(filepath)
    else:
        raise ValueError(
            f"Unsupported event log format '{ext}'. "
            "Expected .xes, .xes.gz, .csv, or .json."
        )


def parse_event_log(raw: dict[str, Any]) -> EventLog:
    """Parse a raw JSON dict into a fully populated ``EventLog``.

    Dynamically extracts activity labels (Σ) and variable names (V)
    from the log — no "magic" strings (§6 constraint 4).
    """
    traces: list[Trace] = []
    activities: set[str] = set()
    variables: set[str] = set()

    for raw_trace in raw["traces"]:
        events: list[Event] = []
        for raw_event in raw_trace["events"]:
            activity: str = raw_event["activity"]
            payload: DataPayload = _normalise_payload(raw_event.get("payload", {}))

            activities.add(activity)
            variables.update(payload.keys())

            events.append(Event(activity=activity, payload=payload))
        traces.append(Trace(events=events))

    return EventLog(traces=traces, activities=activities, variables=variables)


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

def _load_json(filepath: Path) -> EventLog:
    """Load from the legacy JSON format."""
    raw: dict[str, Any] = json.loads(filepath.read_text(encoding="utf-8"))
    return parse_event_log(raw)


def _load_xes(filepath: Path) -> EventLog:
    """Load a ``.xes`` or ``.xes.gz`` file via pm4py."""
    df: pd.DataFrame = pm4py.read_xes(str(filepath))
    return _dataframe_to_event_log(df)


def _load_csv(filepath: Path) -> EventLog:
    """Load a ``.csv`` file via pandas + pm4py.

    The CSV must contain at least columns mappable to
    ``case:concept:name``, ``concept:name``, and
    ``time:timestamp``.  If those columns already exist we skip
    ``pm4py.format_dataframe`` to avoid pm4py corrupting values
    (e.g. activity names like 'T1' being parsed as timestamps).
    """
    df = pd.read_csv(str(filepath))

    xes_required = {"case:concept:name", "concept:name", "time:timestamp"}
    if not xes_required.issubset(set(df.columns)):
        # Columns need renaming — let pm4py handle it.
        df = pm4py.format_dataframe(df)

    return _dataframe_to_event_log(df)


# ---------------------------------------------------------------------------
# DataFrame → EventLog conversion
# ---------------------------------------------------------------------------

def _dataframe_to_event_log(df: pd.DataFrame) -> EventLog:
    """Convert a pm4py-style ``DataFrame`` into an ``EventLog``.

    Every column whose name is **not** in ``XES_STANDARD_COLUMNS``
    is treated as a data variable (decision attribute).
    """
    # Determine which columns are data variables.
    all_columns = set(df.columns)
    data_columns = sorted(all_columns - XES_STANDARD_COLUMNS)

    # Also exclude any remaining ``case:`` prefixed columns that
    # were not explicitly listed (they are case-level metadata).
    data_columns = [
        c for c in data_columns
        if not c.startswith("case:") and not c.startswith("@@")
    ]

    logger.debug("Data variable columns detected: %s", data_columns)

    # Identify the case-id and activity columns.
    case_col = "case:concept:name"
    activity_col = "concept:name"

    if case_col not in df.columns:
        raise KeyError(
            f"Column '{case_col}' not found. "
            "Ensure the event log follows the XES naming convention."
        )
    if activity_col not in df.columns:
        raise KeyError(
            f"Column '{activity_col}' not found. "
            "Ensure the event log follows the XES naming convention."
        )

    traces: list[Trace] = []
    activities: set[str] = set()
    variables: set[str] = set(data_columns)

    for _case_id, case_df in df.groupby(case_col, sort=False):
        # Sort events within each case by timestamp if available.
        if "time:timestamp" in case_df.columns:
            case_df = case_df.sort_values("time:timestamp")

        events: list[Event] = []
        for _, row in case_df.iterrows():
            activity = str(row[activity_col])
            activities.add(activity)

            payload: DataPayload = {}
            for col in data_columns:
                val = row[col]
                # Skip NaN / None values.
                if pd.isna(val):
                    continue
                payload[col] = _normalise_value(val)

            events.append(Event(activity=activity, payload=payload))

        traces.append(Trace(events=events))

    logger.info(
        "Loaded event log: %d traces, Σ = %s, V = %s",
        len(traces),
        activities,
        variables,
    )
    return EventLog(traces=traces, activities=activities, variables=variables)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_payload(raw_payload: dict[str, Any]) -> DataPayload:
    """Ensure payload values are ``int | float | str``."""
    result: DataPayload = {}
    for key, value in raw_payload.items():
        match value:
            case int() | float() | str():
                result[key] = value
            case _:
                result[key] = str(value)
    return result


def _normalise_value(value: Any) -> int | float | str:
    """Coerce a single cell value to int, float, or str."""
    if isinstance(value, (int, bool)):
        return int(value)
    if isinstance(value, float):
        # Promote clean floats to int (e.g. 1.0 → 1).
        if value == int(value):
            return int(value)
        return value
    return str(value)
