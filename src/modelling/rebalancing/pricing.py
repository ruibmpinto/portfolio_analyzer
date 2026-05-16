"""CHF price lookup for tickers not present in current holdings.

Used by Rebalancer._generate_actions when a strategy proposes a
target weight on a ticker that the analyzer's holdings_df does
not contain. The helper converts a native-currency quote from
the data_provider to CHF via a single FX pair lookup, supporting
'CHF', 'GBp' (London pence), and any major currency for which
yfinance exposes a <CURRENCY>CHF=X pair.

Loud failures only: native or FX lookup misses raise
RuntimeError. No silent fallbacks.
"""

from typing import Protocol


pence_per_pound = 100.0


class _PriceProvider(Protocol):
    """Minimal interface of DataProvider used here."""

    def get_current_price(self, ticker: str) -> float: ...


def fetch_chf_price(
    data_provider: _PriceProvider,
    ticker: str,
    currency: str) -> float:
    """Return the latest price for `ticker` in CHF.

    Args:
        data_provider: A DataProvider with ``get_current_price``.
        ticker: Native exchange symbol.
        currency: ISO-4217 listing currency, or 'GBp' for London
            pence.

    Returns:
        Price in CHF, strictly positive.

    Raises:
        RuntimeError: If the native quote or the FX pair quote
            cannot be fetched.
    """
    native_price = _fetch_native(data_provider, ticker)
    if currency == 'CHF':
        return native_price
    if currency == 'GBp':
        fx = _fetch_native(data_provider, 'GBPCHF=X')
        return (native_price / pence_per_pound) * fx
    pair = f'{currency}CHF=X'
    fx = _fetch_native(data_provider, pair)
    return native_price * fx


def _fetch_native(
        data_provider: _PriceProvider,
        ticker: str) -> float:
    try:
        return float(data_provider.get_current_price(ticker))
    except Exception as e:
        raise RuntimeError(
            f'fetch_chf_price: data_provider.get_current_price'
            f'({ticker!r}) failed: {type(e).__name__}: {e}'
        ) from e
