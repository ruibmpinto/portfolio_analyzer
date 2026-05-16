"""Tests for CandidateTicker JSON round-tripping."""

import json
from types import MappingProxyType

import pytest

from src.screening.candidate import (
    CandidateTicker, load_candidates_from_json)


def test_load_candidates_from_json_roundtrips(tmp_path):
    """Loader reconstructs CandidateTicker fields verbatim."""
    rows = [
        {
            'ticker': 'AAPL', 'exchange': 'NASDAQ',
            'currency': 'USD', 'sector': 'Technology',
            'market_cap': 3.0e12, 'composite_score': 0.87,
            'metrics': {'momentum': 0.72, 'sharpe': 1.4},
        },
        {
            'ticker': 'NESN.SW', 'exchange': 'SIX',
            'currency': 'CHF', 'sector': 'Consumer Defensive',
            'market_cap': 2.5e11, 'composite_score': 0.61,
            'metrics': {'momentum': 0.35, 'sharpe': 0.9},
        },
    ]
    path = tmp_path / 'screener.json'
    path.write_text(json.dumps(rows))
    candidates = load_candidates_from_json(str(path))
    assert [c.ticker for c in candidates] == ['AAPL', 'NESN.SW']
    assert candidates[0].composite_score == pytest.approx(0.87)
    assert candidates[1].currency == 'CHF'
    assert isinstance(candidates[0].metrics, MappingProxyType)
    assert candidates[0].metrics['momentum'] == pytest.approx(0.72)


def test_load_candidates_raises_on_missing_field(tmp_path):
    """Loader raises KeyError listing the missing field."""
    rows = [{
        'ticker': 'AAPL', 'exchange': 'NASDAQ',
        # currency missing
        'sector': 'Technology', 'market_cap': 3e12,
        'composite_score': 0.5, 'metrics': {}}]
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(rows))
    with pytest.raises(KeyError, match='currency'):
        load_candidates_from_json(str(path))


def test_load_candidates_empty_file(tmp_path):
    """Empty list yields an empty list."""
    path = tmp_path / 'empty.json'
    path.write_text('[]')
    assert load_candidates_from_json(str(path)) == []
