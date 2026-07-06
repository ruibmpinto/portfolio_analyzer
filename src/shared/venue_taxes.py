"""Per-venue transaction taxes and typical half-spreads.

Broker-agnostic reference data consumed by the forward fee model
(``src/modelling/monte_carlo/cost_model.py``) and the breakeven-gap
calculator (``src/analysis/breakeven.py``). Keyed by yfinance-style
ticker suffix (e.g. ``.SW``, ``.L``, ``.PA``). The empty-string key
covers US and any other suffix-less market.

Numbers reflect statutory taxes (UK stamp duty, FR/IT FTT, Swiss
federal stamp) plus rough half-spread estimates for large-cap
names. The tax component is stable across brokers; commission
belongs in cost_model.py.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class VenueTax:
    """Per-venue tax and spread reference data, in basis points.

    Attributes:
        buy_bps: Transaction tax charged to the buy leg only.
            Non-zero for UK stamp duty, FR/IT FTT, and Swiss
            federal stamp.
        sell_bps: Transaction tax charged to the sell leg only.
            Non-zero only for venues that tax both sides
            (currently just SIX / Switzerland).
        half_spread_bps: Rough half-spread estimate for
            large-cap names on the venue; used for breakeven
            overlays, not for Monte Carlo simulation.
        is_us: True when the venue is a US exchange
            (NYSE/NASDAQ/AMEX). Gates US-only regulatory fees
            (SEC Section 31, FINRA TAF/CAT) inside the cost
            model. Yahoo tickers for US listings carry no
            suffix, so this flag is set on the empty-string
            entry only.
    """

    buy_bps: float
    sell_bps: float
    half_spread_bps: float
    is_us: bool = False


def venue_table() -> Dict[str, VenueTax]:
    """Return the ticker-suffix -> VenueTax lookup table.

    Rebuilt on every call to avoid module-level state. Cheap:
    the table has O(10) entries and callers hit it a handful
    of times per request.
    """
    return {
        # SIX Swiss Exchange - Swiss federal stamp 0.075% each side
        # for Swiss securities (0.15% total; PwC Switzerland).
        '.SW': VenueTax(7.5, 7.5, 3.0),
        # London Stock Exchange - UK Stamp Duty Reserve Tax 0.5%
        # on buys since 1986 (gov.uk / HMRC).
        '.L':  VenueTax(50.0, 0.0, 3.0),
        # Euronext Paris - FR FTT 0.4% on buys of French issuers
        # with market cap > EUR 1B (since 1 April 2025; PwC France).
        '.PA': VenueTax(40.0, 0.0, 2.0),
        # Borsa Italiana - IT FTT 0.2% on buys of Italian issuers
        # with market cap > EUR 500M (since 2026; PwC Italy).
        '.MI': VenueTax(20.0, 0.0, 3.0),
        # Xetra (Deutsche Boerse electronic)
        '.DE': VenueTax(0.0, 0.0, 2.0),
        # Frankfurt floor
        '.F':  VenueTax(0.0, 0.0, 2.0),
        # Euronext Amsterdam
        '.AS': VenueTax(0.0, 0.0, 2.0),
        # Euronext Lisbon
        '.LS': VenueTax(0.0, 0.0, 5.0),
        # BME Madrid
        '.MC': VenueTax(0.0, 0.0, 3.0),
        # US and any other suffix-less market (no tax, ~2 bps spread).
        # is_us=True gates SEC + FINRA regulatory fees on sells.
        '':    VenueTax(0.0, 0.0, 2.0, is_us=True),
    }


def lookup_venue(ticker: str) -> VenueTax:
    """Return the VenueTax for ``ticker``.

    Uses longest-suffix match. The empty-string key is always
    last (``str.endswith('')`` is True for every string), so it
    only fires when no real suffix matches — covering US and
    other unsuffixed markets without a separate fallback branch.

    Args:
        ticker: yfinance-style ticker symbol (e.g. ``'AAPL'``,
            ``'UBSG.SW'``, ``'RR.L'``).

    Returns:
        Matching VenueTax.
    """
    table = venue_table()
    # Longest suffix first so overlapping keys resolve
    # deterministically (e.g. ".SW" beats ".W" if ever added).
    # Empty string sorts last and catches everything else.
    for suffix in sorted(table, key=len, reverse=True):
        if ticker.endswith(suffix):
            return table[suffix]
