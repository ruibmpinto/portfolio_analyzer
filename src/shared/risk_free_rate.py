"""
Live risk-free rate, OECD long-term Switzerland (FRED) with
offline fallback.

Provides the single source of truth for the annual risk-free
rate used by analyzer metrics, rebalancer strategies, and the
Monte Carlo engine. Returns a DECIMAL fraction (e.g. 0.004 for
0.4% per annum), NOT a percentage. Caches the fetched value
for the lifetime of the process; falls back to a sentinel rate
(0.005 = 0.5% per annum) on any fetch failure with a warning.
"""

import warnings

import pandas_datareader.data as pdr


fred_series_id_default = 'IRLTLT01CHM156N'
fallback_rate_decimal = 0.005
percent_per_unit = 100.0

_cache = {}


def get_live_risk_free_rate(
    series_id: str = fred_series_id_default,
    use_cache: bool = True) -> float:
    """
    Fetch the latest annual risk-free rate from FRED.

    Args:
        series_id: FRED series ID. Default is the OECD long-term
            interest rate for Switzerland (monthly, percent per
            annum on FRED's side; this function converts to a
            decimal fraction before returning).
        use_cache: Cache the fetched value for the lifetime of
            the process. FRED data is monthly, so re-fetching
            during a single run is wasted work.

    Returns:
        Annual rate as a decimal fraction (e.g. 0.004 means
        0.4%). Falls back to ``fallback_rate_decimal`` (0.005 =
        0.5%) on any fetch failure with a warning. The fallback
        is NOT cached so a transient failure does not poison the
        rest of the process.
    """
    if use_cache and series_id in _cache:
        return _cache[series_id]

    fetched_ok = False
    try:
        raw_pct = float(
            pdr.DataReader(series_id, 'fred').iloc[-1, 0])
        rate = raw_pct / percent_per_unit
        fetched_ok = True
    except (IOError, OSError, ValueError, KeyError,
            IndexError, ConnectionError, RuntimeError) as e:
        warnings.warn(
            f'FRED fetch for {series_id!r} failed '
            f'({type(e).__name__}: {e}); using fallback '
            f'{fallback_rate_decimal:.4f} '
            f'({fallback_rate_decimal * percent_per_unit:.2f}%).')
        rate = fallback_rate_decimal

    if use_cache and fetched_ok:
        _cache[series_id] = rate
    return rate
