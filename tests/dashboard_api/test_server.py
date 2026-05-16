"""End-to-end smoke test for the FastAPI dashboard sidecar.

Boots a ``TestClient`` against ``create_app`` with a minimal
``DashboardContext`` that returns deterministic data without
hitting yfinance, the screener cache, or the Monte Carlo
engine. Exercises every lightweight route (health, meta,
refresh, rebalance-done). The heavy data routes (overview,
holdings, sells, strategies) require live providers and are
covered by the manual end-to-end check documented in the plan
file; here we assert only that they exist on the router.
"""

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.dashboard_api.cache import SnapshotCache
from src.dashboard_api.context import DashboardContext
from src.dashboard_api.server import create_app


@pytest.fixture
def temp_log_path(tmp_path, monkeypatch):
    """Redirect rebalance_log + screener metadata into tmp_path."""
    log_path = tmp_path / 'rebalance_log.json'
    meta_path = tmp_path / 'metadata.json'
    meta_path.write_text(json.dumps({
        'NASDAQ': {
            'count': 1,
            'updated': '2026-01-01T00:00:00',
            'source': 'test'},
        'NYSE': {
            'count': 1,
            'updated': '2026-02-01T00:00:00',
            'source': 'test'}}))
    monkeypatch.setattr(
        'src.dashboard_api.config.rebalance_log_path', log_path)
    monkeypatch.setattr(
        'src.dashboard_api.config.screener_metadata_path',
        meta_path)
    monkeypatch.setattr(
        'src.dashboard_api.routes.meta.config.rebalance_log_path',
        log_path)
    monkeypatch.setattr(
        ('src.dashboard_api.routes.meta.config'
         '.screener_metadata_path'),
        meta_path)
    return log_path


@pytest.fixture
def client(temp_log_path):
    """TestClient with a context that never builds an analyzer."""
    ctx = DashboardContext(
        analyzer=_StubAnalyzer(),
        data_provider=_StubProvider(),
        cache=SnapshotCache(ttl_seconds=60.0),
        categories={},
        base_currency='CHF')
    app = create_app(ctx=ctx)
    return TestClient(app)


def test_health_returns_ok(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_meta_includes_screener_and_rebalance_keys(client):
    response = client.get('/api/meta')
    assert response.status_code == 200
    payload = response.json()
    assert payload['base_currency'] == 'CHF'
    assert 'screener' in payload
    assert 'last_rebalance' in payload
    assert 'rebalance_due' in payload
    # No log yet → due
    assert payload['rebalance_due'] is True
    assert payload['days_since_rebalance'] is None
    assert payload['screener']['latest_updated_utc'] == (
        '2026-02-01T00:00:00')


def test_rebalance_done_appends_and_meta_reflects_it(client):
    post = client.post(
        '/api/rebalance-done',
        json={'strategy_name': 'mvo'})
    assert post.status_code == 200
    entry = post.json()['entry']
    assert entry['strategy_name'] == 'mvo'
    # The meta cache was invalidated, so a fresh GET sees the entry
    meta = client.get('/api/meta').json()
    assert meta['last_rebalance']['strategy_name'] == 'mvo'
    assert meta['days_since_rebalance'] == 0
    assert meta['rebalance_due'] is False


def test_refresh_returns_invalidated_status(client):
    response = client.post('/api/refresh')
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'invalidated'
    assert 'ts_utc' in body


def test_router_registers_every_planned_route(client):
    """Catalogue check: every documented route is mounted."""
    app = client.app
    paths = {route.path for route in app.routes}
    expected = {
        '/api/health',
        '/api/meta',
        '/api/refresh',
        '/api/rebalance-done',
        '/api/overview',
        '/api/holdings',
        '/api/sells',
        '/api/strategies',
    }
    missing = expected - paths
    assert missing == set(), (
        f'router is missing these planned paths: {sorted(missing)}')


# -------------------------------------------------------------------------
# Stubs
# -------------------------------------------------------------------------


class _StubProvider:
    """Returns nothing; data routes are not exercised here."""

    def get_price_history(
            self, ticker, start_date=None, end_date=None):
        import pandas as pd
        return pd.Series(dtype=float)


class _StubAnalyzer:
    """Minimal stand-in for PortfolioAnalyzer used by /api/meta only."""

    class _StubPortfolio:
        transactions = []

    portfolio = _StubPortfolio()
