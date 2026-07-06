"""Rules endpoint: breakeven-gap formula plus per-market examples.

Purely didactic — the payload teaches the sell-and-rebuy breakeven
math on a fixed set of illustrative tickers (one per market). It does
not read from the analyzer, so it has no dependency on the portfolio
state, market data, or FX cache.

Routes
------
GET /api/rules
"""

from typing import Dict, List

from fastapi import APIRouter

from src.modelling.trading.breakeven import breakeven_gap


router = APIRouter()


@router.get('/api/rules')
def get_rules():
    """Return the breakeven formula and four illustrative examples."""
    return _build_payload()


def _build_payload() -> Dict:
    """Assemble the static Rules payload.

    No caller state is consumed; the payload is a pure function of the
    hard-coded example list and the venue-tax / cost-model defaults.
    """
    return {
        'formula': _formula(),
        'examples': _example_rows(),
        'operational_notes': _operational_notes(),
    }


def _formula() -> Dict:
    """Human-readable description of the breakeven-gap math."""
    return {
        'summary': (
            'The breakeven gap is the minimum price drop from a sell '
            'to a rebuy at which the round-trip fees + spread do not '
            'eat the saved capital. Below the gap, the sell-and-rebuy '
            'loses money; above it, the trade earns more shares for '
            'the same cash.'),
        'expression': (
            'breakeven_bps = f_s_bps + f_b_bps + 2 * half_spread_bps'),
        'components': [
            {
                'name': 'f_s_bps',
                'meaning': (
                    'Sell-leg fees as bps of trade value. Includes '
                    'commission, exchange + clearing, US regulatory '
                    '(SEC + FINRA when applicable), FX conversion, '
                    'and any sell-side venue tax.'),
            },
            {
                'name': 'f_b_bps',
                'meaning': (
                    'Buy-leg fees as bps of trade value. Same '
                    'components as f_s_bps, plus any buy-side venue '
                    'tax (UK stamp duty, FR/IT FTT, Swiss stamp).'),
            },
            {
                'name': 'half_spread_bps',
                'meaning': (
                    'Rough half-spread for a large-cap name on the '
                    'venue. Charged twice (once on the sell, once on '
                    'the rebuy) to model market friction.'),
            },
            {
                'name': 'breakeven_bps',
                'meaning': (
                    'Total round-trip cost in bps of trade value. '
                    'Rebuy price must be at least this much below the '
                    'sell price for the trade to net zero on fees + '
                    'spread.'),
            },
            {
                'name': 'suggested_trigger_bps',
                'meaning': (
                    'Recommended stop-loss distance from the current '
                    'price, set at 2 * breakeven so a noise-triggered '
                    'stop still leaves a one-times-breakeven margin '
                    'if the trade completes.'),
            },
        ],
    }


def _example_rows() -> List[Dict]:
    """Four illustrative rows — one per market covered in ``venue_taxes``.

    Prices and share counts are pedagogical round numbers. The bps
    fields are FX-invariant so ``fx_rate_to_chf=1.0`` produces the
    correct venue-specific breakeven regardless of the trade currency.
    """
    specs = [
        {
            'ticker': 'AAPL', 'market_label': 'United States (NYSE/NASDAQ)',
            'currency': 'USD', 'shares': 100.0, 'price_native': 200.0,
        },
        {
            'ticker': 'RR.L', 'market_label': 'United Kingdom (LSE)',
            'currency': 'GBP', 'shares': 500.0, 'price_native': 10.0,
        },
        {
            'ticker': 'VOW3.DE', 'market_label': 'Germany (Xetra)',
            'currency': 'EUR', 'shares': 50.0, 'price_native': 100.0,
        },
        {
            'ticker': 'TTE.PA', 'market_label': 'France (Euronext Paris)',
            'currency': 'EUR', 'shares': 100.0, 'price_native': 60.0,
        },
    ]
    rows = []
    for spec in specs:
        gap = breakeven_gap(
            shares=spec['shares'],
            price_native=spec['price_native'],
            currency=spec['currency'],
            ticker=spec['ticker'],
            fx_rate_to_chf=1.0)
        # Merge the example metadata with the computed bps fields
        rows.append({**spec, **gap})
    return rows


def _operational_notes() -> List[str]:
    """IBKR-configuration guidance for making the strategy work."""
    return [
        'Disable IBKR Auto-FX so a sell-then-buy round-trip does not '
        'accidentally convert CHF twice.',
        'Pre-fund the trade currency before the sell so the rebuy '
        'does not trigger a fresh FX conversion.',
        'Set stops on a close basis, not intraday touches, to cut '
        'false triggers 5-10x in choppy markets.',
    ]
