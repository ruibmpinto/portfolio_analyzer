"""Tests for the GET /api/rules endpoint.

The route is a pure function of the venue-tax + cost-model defaults —
no portfolio state, no market data, no FX cache. Tests hit the router
directly (no TestClient needed) to keep the surface area small.
"""

import pytest

from src.dashboard_api.routes.rules import _build_payload


def test_payload_has_top_level_shape():
    """The payload advertises formula, examples, and operational notes."""
    p = _build_payload()
    assert set(p) == {'formula', 'examples', 'operational_notes'}


def test_formula_expression_matches_documented_math():
    """The exposed formula string names all three input components."""
    f = _build_payload()['formula']
    assert (f['expression']
            == 'breakeven_bps = f_s_bps + f_b_bps + 2 * half_spread_bps')
    # Every component listed has a name and a meaning string
    for c in f['components']:
        assert set(c) == {'name', 'meaning'}
        assert c['meaning']


def test_all_four_markets_present():
    """One example per market covered by venue_taxes (US, UK, DE, FR)."""
    examples = _build_payload()['examples']
    tickers = {ex['ticker'] for ex in examples}
    assert tickers == {'AAPL', 'RR.L', 'VOW3.DE', 'TTE.PA'}


def test_every_example_carries_the_bps_fields():
    """No example is missing a breakeven, trigger, or per-side fee bps."""
    required = {
        'ticker', 'market_label', 'currency', 'shares', 'price_native',
        'f_s_bps', 'f_b_bps', 'half_spread_bps',
        'breakeven_bps', 'suggested_trigger_bps'}
    for ex in _build_payload()['examples']:
        assert required.issubset(ex.keys()), ex


def test_uk_breakeven_dominated_by_stamp_duty():
    """RR.L breakeven must exceed the 50 bps UK stamp-duty floor."""
    uk = next(ex for ex in _build_payload()['examples']
              if ex['ticker'] == 'RR.L')
    assert uk['breakeven_bps'] >= 50.0


def test_us_example_has_no_venue_tax_contribution():
    """AAPL sees only broker fees + spread; no tax leg."""
    us = next(ex for ex in _build_payload()['examples']
              if ex['ticker'] == 'AAPL')
    # f_b_bps for a US buy has zero venue tax; f_s carries SEC + FINRA.
    # Total breakeven should be well under UK's 50 bps floor.
    assert us['breakeven_bps'] < 20.0


def test_suggested_trigger_is_twice_breakeven():
    """Identity: 2 * breakeven, for every example."""
    for ex in _build_payload()['examples']:
        assert ex['suggested_trigger_bps'] == pytest.approx(
            2.0 * ex['breakeven_bps'])


def test_operational_notes_nonempty_strings():
    """Notes list contains at least one non-empty string per bullet."""
    notes = _build_payload()['operational_notes']
    assert len(notes) >= 3
    for note in notes:
        assert isinstance(note, str) and note.strip()


def test_route_registered_on_app():
    """/api/rules is reachable through the FastAPI router."""
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    from src.dashboard_api.server import create_app
    from src.dashboard_api.context import DashboardContext
    # Ctx with a stub analyzer avoids the yfinance path; the rules
    # route doesn't touch ctx anyway.
    ctx = DashboardContext.__new__(DashboardContext)
    with patch.object(
            DashboardContext, '__init__', return_value=None):
        app = create_app(ctx=ctx)
    with TestClient(app) as client:
        response = client.get('/api/rules')
        assert response.status_code == 200
        assert 'examples' in response.json()
