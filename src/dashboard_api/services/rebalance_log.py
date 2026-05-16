"""Persistent log of executed rebalance events.

The dashboard uses this log to show the date of the last
rebalance and to raise a "rebalance due" banner whenever more
than 30 days have elapsed since the most recent entry.

Persists as JSON at ``data/rebalance_log.json`` — a list of
``{"ts_utc": "<iso8601>", "strategy_name": "<key>"}`` entries
sorted oldest first.

Functions
---------
read_log
    Load and return the parsed log entries.
last_rebalance
    Return the most-recent entry, or None when the log is empty.
mark_done
    Append a new entry stamped at the current UTC instant.
"""

import json
import pathlib
from datetime import datetime, timezone
from typing import List, Optional


default_log_path = pathlib.Path('data/rebalance_log.json')
rebalance_due_days = 30


def read_log(path: Optional[pathlib.Path] = None):
    """Return the parsed list of rebalance-log entries.

    Args:
        path: Optional override path. Defaults to
            ``default_log_path``.

    Returns:
        List of dicts (oldest first). Empty list when the file
        does not exist.
    """
    p = path or default_log_path
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(
            f'rebalance log at {p} is not a JSON list.')
    return data


def last_rebalance(path: Optional[pathlib.Path] = None):
    """Return the most recent log entry, or None when empty."""
    entries = read_log(path)
    if not entries:
        return None
    return entries[-1]


def mark_done(
        strategy_name: str,
        path: Optional[pathlib.Path] = None,
        now: Optional[datetime] = None):
    """Append a new entry stamped at the current UTC instant.

    Args:
        strategy_name: Registry key of the strategy that was
            just executed.
        path: Optional override path. Defaults to
            ``default_log_path``.
        now: Optional clock injection for tests. Defaults to
            ``datetime.now(timezone.utc)``.

    Returns:
        The newly-appended entry dict.
    """
    p = path or default_log_path
    entries = read_log(p)
    ts = (now or datetime.now(timezone.utc))
    entry = {
        'ts_utc': ts.isoformat(),
        'strategy_name': strategy_name,
    }
    entries.append(entry)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        json.dump(entries, f, indent=2)
    return entry


def days_since(
        entry,
        now: Optional[datetime] = None) -> Optional[int]:
    """Calendar days between `entry['ts_utc']` and `now`.

    Args:
        entry: Log entry dict (or None).
        now: Optional clock injection.

    Returns:
        Integer day count, or None when `entry` is None.
    """
    if entry is None:
        return None
    ts = datetime.fromisoformat(entry['ts_utc'])
    ref = now or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return int((ref - ts).days)
