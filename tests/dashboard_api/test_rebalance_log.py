"""Tests for the rebalance_log read/append helper."""

from datetime import datetime, timezone

import pytest

from src.dashboard_api.services import rebalance_log


def test_read_log_returns_empty_when_file_absent(tmp_path):
    path = tmp_path / 'rebalance_log.json'
    assert rebalance_log.read_log(path) == []


def test_mark_done_writes_iso_timestamp(tmp_path):
    path = tmp_path / 'rebalance_log.json'
    now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    entry = rebalance_log.mark_done(
        strategy_name='mvo', path=path, now=now)
    assert entry == {
        'ts_utc': '2026-05-15T12:00:00+00:00',
        'strategy_name': 'mvo'}
    assert path.exists()


def test_round_trip_read_after_write(tmp_path):
    path = tmp_path / 'rebalance_log.json'
    now1 = datetime(2026, 1, 15, tzinfo=timezone.utc)
    now2 = datetime(2026, 2, 15, tzinfo=timezone.utc)
    rebalance_log.mark_done('mvo', path=path, now=now1)
    rebalance_log.mark_done('equal_weight', path=path, now=now2)
    entries = rebalance_log.read_log(path)
    assert len(entries) == 2
    assert entries[0]['strategy_name'] == 'mvo'
    assert entries[1]['strategy_name'] == 'equal_weight'


def test_last_rebalance_returns_newest_entry(tmp_path):
    path = tmp_path / 'rebalance_log.json'
    rebalance_log.mark_done(
        'a', path=path,
        now=datetime(2025, 1, 1, tzinfo=timezone.utc))
    rebalance_log.mark_done(
        'b', path=path,
        now=datetime(2025, 6, 1, tzinfo=timezone.utc))
    last = rebalance_log.last_rebalance(path)
    assert last['strategy_name'] == 'b'


def test_days_since_handles_naive_and_aware_timestamps():
    entry = {
        'ts_utc': '2026-01-01T00:00:00',
        'strategy_name': 'x'}
    now = datetime(2026, 1, 31, tzinfo=timezone.utc)
    assert rebalance_log.days_since(entry, now) == 30


def test_days_since_returns_none_for_missing_entry():
    assert rebalance_log.days_since(None) is None


def test_read_log_raises_when_file_is_not_a_list(tmp_path):
    path = tmp_path / 'rebalance_log.json'
    path.write_text('{"not": "a list"}')
    with pytest.raises(RuntimeError, match='not a JSON list'):
        rebalance_log.read_log(path)
