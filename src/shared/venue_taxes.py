"""Per-venue schedule for IBKR-Tiered fees and statutory taxes.

One frozen dataclass ``VenueSchedule`` carries every constant needed
to price a trade on a given (venue-suffix, trade-currency) pair:
statutory tax on each side, IBKR Tiered commission (min + cap +
either per-share or per-value), broker per-share extras, a flag for
US regulatory fees, and a market-microstructure half-spread proxy.

The table is the single source of truth. Adding a new venue is one
row; adding a new fee schedule change is one field edit. No branches
elsewhere on venue identity — the cost model just consumes the row.

Two small out-of-band lists live alongside the table:
- UCITS ETFs on LSE are stamp-exempt (Irish/Luxembourg domicile);
  their row is inherited from ``.L`` with ``tax_buy_bps`` zeroed.
- Certain mutual funds are IBKR no-transaction-fee (NTF);
  ``is_ntf_mutual_fund`` returns True for those and the cost model
  short-circuits to zero fee.
"""

from dataclasses import dataclass, replace
from typing import Dict, Tuple


@dataclass(frozen=True)
class VenueSchedule:
    """Per-(venue, currency) fee constants.

    All amounts in the trade's native currency; all rates in bps of
    trade value (except ``commission_per_share_native`` which is in
    native currency per share).

    Attributes:
        tax_buy_bps: Statutory tax charged to the buy leg.
        tax_sell_bps: Statutory tax charged to the sell leg.
        commission_bps: IBKR Tiered commission as bps of trade value.
        commission_per_share_native: IBKR Tiered commission per
            share (US uses this; non-US uses ``commission_bps``).
        commission_min_native: Minimum per-order commission.
        commission_cap_pct: Fraction of trade value at which
            commission is capped.
        broker_extras_per_share_native: Per-share broker fees on
            top of commission (US NSCC/DTC/CAT; zero elsewhere).
        charges_us_regulatory: True where IBKR passes through
            SEC Section 31 + FINRA TAF on the sell leg (US only).
        half_spread_bps: Rough half-spread heuristic for large-cap
            names on the venue.
    """
    tax_buy_bps: float
    tax_sell_bps: float
    commission_bps: float
    commission_per_share_native: float
    commission_min_native: float
    commission_cap_pct: float
    broker_extras_per_share_native: float
    charges_us_regulatory: bool
    half_spread_bps: float


# UCITS ETFs on LSE - stamp-exempt (Irish/Luxembourg domicile). UK
# Stamp Duty Reserve Tax applies only to UK-incorporated equities;
# UCITS ETFs listed on LSE do not qualify. Extend when a new UCITS
# ETF appears in the transaction log.
_uk_stamp_exempt_lse = frozenset({
    'VUAA.L', 'VUSA.L', 'VWRL.L', 'CSPX.L', 'EQQQ.L'})


# IBKR no-transaction-fee mutual funds (observed in real
# statements). Extend as new NTF funds arrive.
_ntf_mutual_fund_isins = frozenset({
    'IE00BFZP7W55', 'IE00BDDRH524'})


def schedule_table() -> Dict[Tuple[str, str], VenueSchedule]:
    """Return the ``(yahoo_suffix, trade_currency) -> VenueSchedule`` table.

    Rebuilt on every call to avoid module-level state. Cheap: the
    table has O(10) entries and callers hit it a handful of times
    per request.

    Numbers reflect IBKR Tiered Retail (top tier) plus statutory
    taxes documented at gov.uk (UK) and the PwC Worldwide Tax
    Summaries (CH, FR, IT). Half-spreads are venue heuristics for
    large-cap names.

    Notes on specific rows:
      - ``('.SW', 'CHF')`` sets ``tax_buy_bps=tax_sell_bps=0``
        because IBKR-UK (the broker the model targets) is not a
        Swiss securities dealer, and the observed ``Comm/Fee`` on
        SIX rows carries no stamp component. If a Swiss-dealer
        broker's schedule is added later, that row is 7.5/7.5.
      - Multiple ``.L`` rows exist (GBP/EUR/USD) because LSE
        accepts three currency denominations with different
        IBKR minima; the row is picked by trade currency.
    """
    return {
        # US (NYSE / NASDAQ / AMEX) - per-share commission, sell-
        # side SEC + FINRA regulatory, per-share NSCC/DTC/CAT.
        ('',    'USD'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_bps=0.0,
            commission_per_share_native=0.0035,
            commission_min_native=0.35,
            commission_cap_pct=0.01,
            broker_extras_per_share_native=0.0034,
            charges_us_regulatory=True,
            half_spread_bps=2.0),
        # LSE - three currency denominations. 50 bps UK stamp on
        # buys by default; UCITS ETFs override via lookup_schedule.
        ('.L',  'GBP'): VenueSchedule(
            tax_buy_bps=50.0, tax_sell_bps=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.00,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            half_spread_bps=3.0),
        ('.L',  'EUR'): VenueSchedule(
            tax_buy_bps=50.0, tax_sell_bps=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.25,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            half_spread_bps=3.0),
        ('.L',  'USD'): VenueSchedule(
            tax_buy_bps=50.0, tax_sell_bps=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.70,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            half_spread_bps=3.0),
        # SIX Swiss Exchange - stamp intentionally 0 via IBKR-UK;
        # observed all-in commission is ~7 bps on SIX equities.
        ('.SW', 'CHF'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_bps=7.0,
            commission_per_share_native=0.0,
            commission_min_native=1.50,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            half_spread_bps=3.0),
        # Xetra (Deutsche Boerse electronic)
        ('.DE', 'EUR'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.25,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            half_spread_bps=2.0),
        # Frankfurt floor
        ('.F',  'EUR'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.25,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            half_spread_bps=2.0),
        # Euronext Paris - FR FTT 40 bps on buys of large caps.
        ('.PA', 'EUR'): VenueSchedule(
            tax_buy_bps=40.0, tax_sell_bps=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.25,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            half_spread_bps=2.0),
        # Euronext Amsterdam
        ('.AS', 'EUR'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.25,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            half_spread_bps=2.0),
        # Euronext Lisbon
        ('.LS', 'EUR'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.25,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            half_spread_bps=5.0),
        # BME Madrid
        ('.MC', 'EUR'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.25,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            half_spread_bps=3.0),
        # Borsa Italiana - IT FTT 20 bps on buys.
        ('.MI', 'EUR'): VenueSchedule(
            tax_buy_bps=20.0, tax_sell_bps=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.25,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            half_spread_bps=3.0),
    }


def is_ntf_mutual_fund(ticker: str) -> bool:
    """True when the ticker is an IBKR no-transaction-fee fund.

    NTF funds carry zero broker fee at IBKR. The cost model uses
    this to short-circuit before consulting the schedule table.
    """
    return ticker in _ntf_mutual_fund_isins


def lookup_schedule(
        ticker: str, currency: str) -> VenueSchedule:
    """Return the VenueSchedule for a ``(ticker, currency)`` pair.

    Uses longest-suffix match on ticker plus exact-match on
    currency. The empty-string suffix catches US names (no Yahoo
    suffix) and only pairs with ``USD``. UCITS ETFs on LSE inherit
    the ``.L`` row but with UK stamp zeroed.

    Args:
        ticker: Yahoo-style ticker (e.g. ``'AAPL'``, ``'RR.L'``,
            ``'VUAA.L'``, ``'UBSG.SW'``).
        currency: ISO-4217 trade currency.

    Returns:
        Matching ``VenueSchedule``.

    Raises:
        RuntimeError: When no row matches the given pair. Callers
            treat this as a data-mapping error (unmodelled venue
            or currency); no silent default.
    """
    table = schedule_table()
    # Longest suffix first so overlapping keys resolve
    # deterministically. Empty-string key sorts last.
    for (suffix, ccy) in sorted(
            table.keys(), key=lambda k: -len(k[0])):
        if ticker.endswith(suffix) and ccy == currency:
            base = table[(suffix, ccy)]
            # UCITS ETF stamp exemption inline; zero the buy-side
            # stamp on Irish/Luxembourg-domiciled LSE ETFs.
            if suffix == '.L' and ticker in _uk_stamp_exempt_lse:
                return replace(
                    base, tax_buy_bps=0.0, tax_sell_bps=0.0)
            return base
    raise RuntimeError(
        f'No IBKR schedule modelled for ticker {ticker!r} in '
        f'currency {currency!r}. Extend schedule_table.')
