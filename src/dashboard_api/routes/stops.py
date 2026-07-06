"""Stops endpoint: per-holding breakeven gap and suggested stop levels.

Consumes the analyzer's holdings snapshot (shares, currency, price)
and produces per-position stop-loss / rebuy price levels derived
from ``breakeven_gap``. The Stops dashboard page renders one row
per open position.

Routes
------
GET /api/stops
"""

from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Request

from src.modelling.trading.breakeven import breakeven_gap


router = APIRouter()


# Basis-point denominator; hoisted for clarity in the price
# derivations below.
_bps_per_unit = 10_000.0


@router.get('/api/stops')
def get_stops(request: Request):
    """Return per-holding breakeven + stop-loss levels."""
    ctx = request.app.state.ctx
    return ctx.cache.get_or_compute(
        'stops', lambda: _build_payload(ctx))


def _build_payload(ctx) -> Dict:
    """Iterate the holdings snapshot into per-position stop rows."""
    analyzer = ctx.analyzer
    snapshot = analyzer.get_holdings_snapshot(
        categories=ctx.categories or None)
    price_panel = analyzer.get_price_panel()
    today = datetime.now()

    rows = []
    for _, snap in snapshot.iterrows():
        ticker = snap['ticker']
        currency = snap['currency']
        rows.append(_row_from_primitives(
            ticker=ticker,
            shares=float(snap['shares']),
            currency=currency,
            weight_pct=float(snap['weight_pct']),
            price_native=_latest_native_price(price_panel, ticker),
            fx_rate_to_chf=analyzer._native_to_base(
                1.0, currency, today)))

    # Sort largest-position-first: users care most about the stops
    # they'd set on their biggest holdings.
    rows.sort(key=lambda r: r['weight_pct'], reverse=True)
    return {'rows': rows}


def _row_from_primitives(
        ticker: str, shares: float, currency: str,
        weight_pct: float, price_native: float,
        fx_rate_to_chf: float) -> Dict:
    """Build a single Stops row from position primitives.

    Isolated from ``_build_payload`` so tests can exercise the math
    without spinning up an analyzer + price panel + FX cache.

    Args:
        ticker: Yahoo-style ticker; drives venue tax and spread lookup.
        shares: Positive share count in the position.
        currency: ISO-4217 currency of the trade.
        weight_pct: Position weight in the portfolio (0-100), used
            only for row sorting by the caller.
        price_native: Latest native-currency close.
        fx_rate_to_chf: Native-to-CHF rate at the valuation date.

    Returns:
        Dict with the position's fee breakdown (``f_s_bps``,
        ``f_b_bps``, ``half_spread_bps``, ``breakeven_bps``,
        ``suggested_trigger_bps``) plus derived native prices
        (``suggested_trigger_price_native`` at
        ``current * (1 - trigger_bps/1e4)`` and
        ``suggested_rebuy_price_native`` at
        ``trigger * (1 - breakeven_bps/1e4)``, compounded not
        approximated).
    """
    gap = breakeven_gap(
        shares=shares, price_native=price_native,
        currency=currency, ticker=ticker,
        fx_rate_to_chf=fx_rate_to_chf)
    # Stop trigger sits below current by suggested_trigger_bps
    trigger_price = price_native * (
        1.0 - gap['suggested_trigger_bps'] / _bps_per_unit)
    # Rebuy target sits below the trigger by breakeven_bps
    # (compounded — the fees on the rebuy scale with the rebuy
    # value, not the original current price).
    rebuy_price = trigger_price * (
        1.0 - gap['breakeven_bps'] / _bps_per_unit)
    return {
        'ticker': ticker,
        'shares': shares,
        'currency': currency,
        'weight_pct': weight_pct,
        'current_price_native': price_native,
        'f_s_bps': gap['f_s_bps'],
        'f_b_bps': gap['f_b_bps'],
        'half_spread_bps': gap['half_spread_bps'],
        'breakeven_bps': gap['breakeven_bps'],
        'suggested_trigger_bps': gap['suggested_trigger_bps'],
        'suggested_trigger_price_native': trigger_price,
        'suggested_rebuy_price_native': rebuy_price,
    }


def _latest_native_price(price_panel, ticker: str) -> float:
    """Most-recent non-NaN native close for ``ticker``.

    Raises rather than returning NaN so a broken price panel
    surfaces at request time instead of poisoning downstream
    breakeven math with silently-zero prices.
    """
    if ticker not in price_panel.columns:
        raise RuntimeError(
            f'No price history for {ticker!r} in the analyzer '
            f'panel. Cannot compute breakeven.')
    series = price_panel[ticker].dropna()
    if series.empty:
        raise RuntimeError(
            f'Price panel for {ticker!r} is empty after dropna. '
            f'Cannot compute breakeven.')
    return float(series.iloc[-1])
