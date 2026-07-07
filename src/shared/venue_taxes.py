"""Per-broker, per-venue schedule for retail-broker fees and taxes.

One frozen dataclass ``VenueSchedule`` carries every constant needed
to price a trade on a given (broker, venue-suffix, trade-currency)
tuple: statutory tax on each side, the broker's commission (flat +
bps + per-share, with min + cap), broker per-share extras, a flag
for US regulatory fees, an optional cross-currency commission
denomination, and a market-microstructure half-spread proxy.

The table is the single source of truth. Adding a new venue is one
row; adding a new broker is one nested dict. No branches elsewhere
on broker/venue identity - the cost model just consumes the row.

Two small out-of-band lists live alongside the table:
- UCITS ETFs on LSE are stamp-exempt (Irish/Luxembourg domicile);
  their row is inherited from ``.L`` with ``tax_buy_bps`` zeroed.
- Certain mutual funds are IBKR no-transaction-fee (NTF);
  ``is_ntf_mutual_fund`` returns True for those and the cost model
  short-circuits to zero fee.
"""

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class VenueSchedule:
    """Per-(broker, venue, currency) fee constants.

    All amounts in the row's native currency; all rates in bps of
    trade value (except ``commission_per_share_native`` which is in
    native currency per share).

    Attributes:
        tax_buy_bps: Statutory tax charged to the buy leg.
        tax_sell_bps: Statutory tax charged to the sell leg.
        commission_flat_native: Flat per-order fee. Zero for IBKR
            (which uses bps/per-share); nonzero for Degiro.
        commission_bps: Commission as bps of trade value (IBKR
            non-US).
        commission_per_share_native: Commission per share (IBKR US
            uses this).
        commission_min_native: Minimum per-order commission.
        commission_cap_pct: Fraction of trade value at which
            commission is capped. Set to ``1.0`` when no cap.
        broker_extras_per_share_native: Per-share broker fees on
            top of commission (US NSCC/DTC/CAT; zero elsewhere).
        charges_us_regulatory: True where the broker passes through
            SEC Section 31 + FINRA TAF on the sell leg.
        commission_currency: When set, the commission (flat + bps +
            per-share + min + cap) is charged in this ISO-4217
            currency, NOT the trade currency. Degiro bills US/LSE/
            Euronext trades in EUR regardless of trade currency;
            IBKR bills in trade currency (leave ``None``).
        half_spread_bps: Rough half-spread heuristic for large-cap
            names on the venue.
    """
    tax_buy_bps: float
    tax_sell_bps: float
    commission_flat_native: float
    commission_bps: float
    commission_per_share_native: float
    commission_min_native: float
    commission_cap_pct: float
    broker_extras_per_share_native: float
    charges_us_regulatory: bool
    commission_currency: Optional[str]
    half_spread_bps: float


# Known ETF tickers. Consumed by ``lookup_schedule`` for two
# effects that both key on "is this ticker an ETF":
#   1. LSE UCITS ETFs are stamp-exempt (Irish/Luxembourg domicile
#      escapes UK Stamp Duty Reserve Tax, which only applies to
#      UK-incorporated equities).
#   2. Degiro prices ETFs on "other exchanges" at EUR 2.00 tracker
#      fee + EUR 1.00 handling = EUR 3.00 flat, vs EUR 3.90 +
#      EUR 1.00 for stocks. Confirmed against real trades:
#      XAIX.DE 2.81 CHF, VUAA.L 2.83 CHF, both ~= EUR 3 * EUR/CHF.
# Extend when a new ETF appears in the transaction log.
_known_etfs = frozenset({
    'VUAA.L', 'VUSA.L', 'VWRL.L', 'CSPX.L', 'EQQQ.L',
    'VUSA.AS', 'IUSC.SW', 'XAIX.DE'})


# IBKR no-transaction-fee mutual funds (observed in real
# statements). Extend as new NTF funds arrive.
_ntf_mutual_fund_isins = frozenset({
    'IE00BFZP7W55', 'IE00BDDRH524'})


def _ibkr_schedule() -> Dict[Tuple[str, str], VenueSchedule]:
    """IBKR Tiered Retail schedule keyed by (yahoo suffix, currency).

    Numbers reflect IBKR Tiered Retail top-tier commissions plus
    statutory taxes documented at gov.uk (UK) and PwC Worldwide Tax
    Summaries (FR, IT). Half-spreads are venue heuristics for
    large-cap names.

    Notes:
      - ``('.SW', 'CHF')`` sets ``tax_buy_bps=tax_sell_bps=0``
        because IBKR-UK is not a Swiss securities dealer; observed
        Comm/Fee on SIX rows carries no stamp component.
      - Multiple ``.L`` rows exist (GBP/EUR/USD) because LSE
        accepts three currency denominations with different IBKR
        minima; the row is picked by trade currency.
    """
    return {
        ('',    'USD'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_flat_native=0.0,
            commission_bps=0.0,
            commission_per_share_native=0.0035,
            commission_min_native=0.35,
            commission_cap_pct=0.01,
            broker_extras_per_share_native=0.0034,
            charges_us_regulatory=True,
            commission_currency=None,
            half_spread_bps=2.0),
        ('.L',  'GBP'): VenueSchedule(
            tax_buy_bps=50.0, tax_sell_bps=0.0,
            commission_flat_native=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.00,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency=None,
            half_spread_bps=3.0),
        ('.L',  'EUR'): VenueSchedule(
            tax_buy_bps=50.0, tax_sell_bps=0.0,
            commission_flat_native=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.25,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency=None,
            half_spread_bps=3.0),
        ('.L',  'USD'): VenueSchedule(
            tax_buy_bps=50.0, tax_sell_bps=0.0,
            commission_flat_native=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.70,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency=None,
            half_spread_bps=3.0),
        ('.SW', 'CHF'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_flat_native=0.0,
            commission_bps=7.0,
            commission_per_share_native=0.0,
            commission_min_native=1.50,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency=None,
            half_spread_bps=3.0),
        ('.DE', 'EUR'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_flat_native=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.25,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency=None,
            half_spread_bps=2.0),
        ('.F',  'EUR'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_flat_native=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.25,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency=None,
            half_spread_bps=2.0),
        ('.PA', 'EUR'): VenueSchedule(
            tax_buy_bps=40.0, tax_sell_bps=0.0,
            commission_flat_native=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.25,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency=None,
            half_spread_bps=2.0),
        ('.AS', 'EUR'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_flat_native=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.25,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency=None,
            half_spread_bps=2.0),
        ('.LS', 'EUR'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_flat_native=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.25,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency=None,
            half_spread_bps=5.0),
        ('.MC', 'EUR'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_flat_native=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.25,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency=None,
            half_spread_bps=3.0),
        ('.MI', 'EUR'): VenueSchedule(
            tax_buy_bps=20.0, tax_sell_bps=0.0,
            commission_flat_native=0.0,
            commission_bps=5.0,
            commission_per_share_native=0.0,
            commission_min_native=1.25,
            commission_cap_pct=0.005,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency=None,
            half_spread_bps=3.0),
    }


def _degiro_schedule() -> Dict[Tuple[str, str], VenueSchedule]:
    """Degiro Swiss-resident tariff (2026-01-01).

    Flat per-order fee = exchange fee + EUR 1.00 handling fee
    (CHF 1.00 handling for SIX). Handling fee bundles the third-
    party costs including US SEC/FINRA passthrough, so
    ``charges_us_regulatory`` is False everywhere for this broker.
    All commissions for non-CHF venues are billed in EUR regardless
    of trade currency (``commission_currency='EUR'``); SIX is
    billed in CHF (``None`` == same as trade currency).

    Statutory venue taxes (UK stamp, FR/IT FTT) still apply and
    are passed through to the client - identical to IBKR. Swiss
    federal stamp is not levied (Degiro is not a Swiss dealer).

    Half-spread heuristics match the IBKR rows since they are a
    market cost, not a broker cost.

    Assumptions / v1 scope:
      - ETF-vs-stock fee distinction (EUR 2.00 vs EUR 3.90) is
        applied per-ticker via ``_known_etfs`` in
        ``lookup_schedule``, not per-row.
      - Exchange Connectivity Fee (EUR 2.50/year/exchange) is a
        periodic charge, out of scope for per-trade
        ``fees_breakdown``.
    """
    return {
        ('',    'USD'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_flat_native=2.00,
            commission_bps=0.0,
            commission_per_share_native=0.0,
            commission_min_native=0.0,
            commission_cap_pct=1.0,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency='EUR',
            half_spread_bps=2.0),
        ('.L',  'GBP'): VenueSchedule(
            tax_buy_bps=50.0, tax_sell_bps=0.0,
            commission_flat_native=4.90,
            commission_bps=0.0,
            commission_per_share_native=0.0,
            commission_min_native=0.0,
            commission_cap_pct=1.0,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency='EUR',
            half_spread_bps=3.0),
        ('.L',  'USD'): VenueSchedule(
            tax_buy_bps=50.0, tax_sell_bps=0.0,
            commission_flat_native=4.90,
            commission_bps=0.0,
            commission_per_share_native=0.0,
            commission_min_native=0.0,
            commission_cap_pct=1.0,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency='EUR',
            half_spread_bps=3.0),
        ('.SW', 'CHF'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_flat_native=6.00,
            commission_bps=0.0,
            commission_per_share_native=0.0,
            commission_min_native=0.0,
            commission_cap_pct=1.0,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency=None,
            half_spread_bps=3.0),
        ('.DE', 'EUR'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_flat_native=4.90,
            commission_bps=0.0,
            commission_per_share_native=0.0,
            commission_min_native=0.0,
            commission_cap_pct=1.0,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency=None,
            half_spread_bps=2.0),
        ('.F',  'EUR'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_flat_native=6.00,
            commission_bps=0.0,
            commission_per_share_native=0.0,
            commission_min_native=0.0,
            commission_cap_pct=1.0,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency=None,
            half_spread_bps=2.0),
        ('.PA', 'EUR'): VenueSchedule(
            tax_buy_bps=40.0, tax_sell_bps=0.0,
            commission_flat_native=4.90,
            commission_bps=0.0,
            commission_per_share_native=0.0,
            commission_min_native=0.0,
            commission_cap_pct=1.0,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency=None,
            half_spread_bps=2.0),
        ('.AS', 'EUR'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_flat_native=4.90,
            commission_bps=0.0,
            commission_per_share_native=0.0,
            commission_min_native=0.0,
            commission_cap_pct=1.0,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency=None,
            half_spread_bps=2.0),
        ('.LS', 'EUR'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_flat_native=4.90,
            commission_bps=0.0,
            commission_per_share_native=0.0,
            commission_min_native=0.0,
            commission_cap_pct=1.0,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency=None,
            half_spread_bps=5.0),
        ('.MC', 'EUR'): VenueSchedule(
            tax_buy_bps=0.0, tax_sell_bps=0.0,
            commission_flat_native=4.90,
            commission_bps=0.0,
            commission_per_share_native=0.0,
            commission_min_native=0.0,
            commission_cap_pct=1.0,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency=None,
            half_spread_bps=3.0),
        ('.MI', 'EUR'): VenueSchedule(
            tax_buy_bps=20.0, tax_sell_bps=0.0,
            commission_flat_native=4.90,
            commission_bps=0.0,
            commission_per_share_native=0.0,
            commission_min_native=0.0,
            commission_cap_pct=1.0,
            broker_extras_per_share_native=0.0,
            charges_us_regulatory=False,
            commission_currency=None,
            half_spread_bps=3.0),
    }


def schedule_table(
        broker: str = 'ibkr'
) -> Dict[Tuple[str, str], VenueSchedule]:
    """Return the ``(yahoo_suffix, currency) -> VenueSchedule`` table
    for the requested broker.

    Args:
        broker: ``'ibkr'`` or ``'degiro'``. Case-sensitive; unknown
            values raise ``ValueError``.
    """
    if broker == 'ibkr':
        return _ibkr_schedule()
    if broker == 'degiro':
        return _degiro_schedule()
    raise ValueError(
        f"Unknown broker {broker!r}; expected 'ibkr' or 'degiro'.")


def is_ntf_mutual_fund(ticker: str) -> bool:
    """True when the ticker is an IBKR no-transaction-fee fund.

    NTF funds carry zero broker fee at IBKR. The cost model uses
    this to short-circuit before consulting the schedule table.
    Degiro does not offer NTF mutual funds at parity, so this flag
    is IBKR-specific by construction; the caller decides whether
    to consult it.
    """
    return ticker in _ntf_mutual_fund_isins


def lookup_schedule(
        ticker: str,
        currency: str,
        broker: str = 'ibkr') -> VenueSchedule:
    """Return the VenueSchedule for a ``(broker, ticker, currency)`` triple.

    Uses longest-suffix match on ticker plus exact-match on currency
    within the broker's table. Empty-string suffix catches US names
    (no Yahoo suffix) and only pairs with ``USD``. UCITS ETFs on
    LSE inherit the ``.L`` row but with UK stamp zeroed.

    Args:
        ticker: Yahoo-style ticker (e.g. ``'AAPL'``, ``'RR.L'``,
            ``'VUAA.L'``, ``'UBSG.SW'``).
        currency: ISO-4217 trade currency.
        broker: ``'ibkr'`` or ``'degiro'``. Selects which broker's
            table to consult.

    Returns:
        Matching ``VenueSchedule``.

    Raises:
        RuntimeError: When no row matches the given triple. Callers
            treat this as a data-mapping error (unmodelled venue,
            currency, or broker); no silent default.
        ValueError: When ``broker`` is unknown.
    """
    table = schedule_table(broker)
    for (suffix, ccy) in sorted(
            table.keys(), key=lambda k: -len(k[0])):
        if ticker.endswith(suffix) and ccy == currency:
            base = table[(suffix, ccy)]
            if ticker in _known_etfs:
                if suffix == '.L':
                    base = replace(
                        base, tax_buy_bps=0.0, tax_sell_bps=0.0)
                if broker == 'degiro':
                    base = replace(
                        base, commission_flat_native=3.00)
            return base
    raise RuntimeError(
        f'No {broker} schedule modelled for ticker {ticker!r} in '
        f'currency {currency!r}. Extend schedule_table.')
