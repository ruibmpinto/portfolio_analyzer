"""
Resumable per-exchange screener cache fetcher.

For each requested ticker, fetches 3 years of weekly OHLCV +
the snapshot fundamentals dict, builds a Listing, and appends
the in-memory batch as a new shard parquet under
`data/screener_cache/<exchange>/` via ExchangeUniverse.save().
The shard directory itself acts as the checkpoint: when
resume=True (default), tickers already present in any existing
shard are skipped on the next call.

Designed to survive rate-limit interruptions: every
`save_every` fetched tickers the pending batch is flushed as
a new shard and the in-memory buffer is reset, and a finally
block flushes whatever remains if the loop is cut short
(e.g. KeyboardInterrupt or unhandled exception). Existing
shards are never modified.

NYSE and NASDAQ requests dispatch to a defeatbeta-backed bulk
path that collapses 6-12 hours of yfinance throttling to a few
minutes by reading the parquet snapshots already cached on
disk by the defeatbeta_api package. All other exchanges keep
the per-ticker yfinance path unchanged.
"""

import time
import warnings
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.screening.listing import (
    Listing, fundamental_field_names, weekly_series_length)
from src.screening.universe import ExchangeUniverse
from src.shared.data_provider import DefeatBetaProvider


default_save_every = 50
default_rate_limit_sleep = 0.5

us_exchanges = ('NYSE', 'NASDAQ')
defeatbeta_history_weeks = 160


# yfinance Ticker.info field name -> Listing fundamental field
# Mapping is explicit; missing yfinance keys map to NaN.
_yfinance_to_fundamental = {
    'pe_ratio_ttm': 'trailingPE',
    'pe_ratio_forward': 'forwardPE',
    'pb_ratio': 'priceToBook',
    'ps_ratio': 'priceToSalesTrailing12Months',
    'dividend_yield_ttm': 'dividendYield',
    'eps_ttm': 'trailingEps',
    'revenue_growth_yoy': 'revenueGrowth',
    'earnings_growth_yoy': 'earningsGrowth',
    'profit_margin': 'profitMargins',
    'operating_margin': 'operatingMargins',
    'roe': 'returnOnEquity',
    'roa': 'returnOnAssets',
    'debt_to_equity': 'debtToEquity',
    'free_cash_flow': 'freeCashflow',
    'beta_yf': 'beta',
    'shares_outstanding': 'sharesOutstanding',
    'short_ratio': 'shortRatio',
}


def refresh_exchange(
    exchange: str,
    tickers: Optional[List[str]] = None,
    resume: bool = True,
    rate_limit_sleep: float = default_rate_limit_sleep,
    save_every: int = default_save_every,
    defeatbeta_provider: Optional[DefeatBetaProvider] = None
) -> None:
    """
    Refresh the on-disk shard cache for one exchange.

    NYSE and NASDAQ go through the bulk defeatbeta path
    (collapses 6-12h of yfinance throttling to minutes). Every
    other exchange uses the per-ticker yfinance path.

    Args:
        exchange: Exchange code (e.g. 'NASDAQ', 'SIX').
        tickers: List of ticker symbols to fetch. Required;
            no on-disk universe loader is wired up yet.
        resume: When True, skip tickers already present in any
            existing shard for this exchange.
        rate_limit_sleep: Seconds to sleep between per-ticker
            yfinance calls. Ignored for US exchanges.
        save_every: Flush the in-memory batch as a new shard
            every N successfully-fetched tickers. The flushed
            batch is reset after each shard write so subsequent
            shards contain only the new fetches.
        defeatbeta_provider: Optional injected provider for
            tests. Default constructs DefeatBetaProvider() when
            the dispatch lands on the US-bulk path.

    Side effects:
        Appends new shard parquet files under
        `data/screener_cache/<exchange>/`. Existing shards
        are never touched.
    """
    if exchange in us_exchanges:
        _refresh_via_defeatbeta(
            exchange, tickers, resume, defeatbeta_provider)
    else:
        _refresh_via_yfinance(
            exchange, tickers, resume, rate_limit_sleep,
            save_every)


def _refresh_via_yfinance(
    exchange: str,
    tickers: Optional[List[str]],
    resume: bool,
    rate_limit_sleep: float,
    save_every: int = default_save_every) -> None:
    """
    Per-ticker yfinance refresh path (non-US exchanges).

    Args:
        exchange: Exchange code (non-US).
        tickers: Ticker symbols to fetch.
        resume: Skip tickers already cached when True.
        rate_limit_sleep: Seconds to sleep between calls.
        save_every: Flush every N successful fetches.
    """
    if tickers is None:
        raise RuntimeError(
            'refresh_exchange: tickers must be provided '
            '(no on-disk ticker list loader configured).')

    already_done: set = set()
    if resume:
        try:
            already_done = {
                l.ticker
                for l in ExchangeUniverse.load(exchange)}
        except FileNotFoundError:
            pass

    pending: List[Listing] = []
    to_fetch = [t for t in tickers if t not in already_done]

    try:
        for i, ticker in enumerate(to_fetch):
            try:
                listing = _fetch_one_listing(ticker, exchange)
            except Exception as e:
                warnings.warn(
                    f'cache_refresh: skipping {ticker!r} '
                    f'after fetch error '
                    f'({type(e).__name__}: {e}).')
                continue
            pending.append(listing)
            if rate_limit_sleep > 0:
                time.sleep(rate_limit_sleep)
            if (i + 1) % save_every == 0 and pending:
                _persist(exchange, pending)
                pending = []
    finally:
        # Flush any remaining listings on completion or
        # interrupt; existing shards stay untouched
        if pending:
            _persist(exchange, pending)


def _refresh_via_defeatbeta(
    exchange: str,
    tickers: Optional[List[str]],
    resume: bool,
    provider: Optional[DefeatBetaProvider],
    save_every: int = default_save_every) -> None:
    """
    Bulk defeatbeta refresh path (NYSE / NASDAQ).

    Args:
        exchange: 'NYSE' or 'NASDAQ'.
        tickers: Ticker symbols to fetch.
        resume: Skip tickers already cached when True.
        provider: Pre-built DefeatBetaProvider, or None to
            construct one. Tests inject a stub via this arg.
        save_every: Flush every N successful fetches.
    """
    if tickers is None:
        raise RuntimeError(
            'refresh_exchange: tickers must be provided '
            '(no on-disk ticker list loader configured).')

    if provider is None:
        provider = DefeatBetaProvider()

    already_done: set = set()
    if resume:
        try:
            already_done = {
                l.ticker
                for l in ExchangeUniverse.load(exchange)}
        except FileNotFoundError:
            pass

    pending: List[Listing] = []
    to_fetch = [t for t in tickers if t not in already_done]

    try:
        for i, ticker in enumerate(to_fetch):
            try:
                listing = _build_listing_via_defeatbeta(
                    ticker, exchange, provider)
            except Exception as e:
                warnings.warn(
                    f'cache_refresh: skipping {ticker!r} '
                    f'after defeatbeta fetch error '
                    f'({type(e).__name__}: {e}).')
                continue
            pending.append(listing)
            if (i + 1) % save_every == 0 and pending:
                _persist(exchange, pending)
                pending = []
    finally:
        if pending:
            _persist(exchange, pending)


def _persist(
    exchange: str, new_listings: List[Listing]) -> None:
    """Append the new batch as a new shard parquet."""
    ExchangeUniverse(exchange, new_listings).save()


def _fetch_one_listing(
    ticker: str, exchange: str) -> Listing:
    """
    Build one Listing via two yfinance calls.

    Args:
        ticker: Native exchange symbol.
        exchange: Exchange code; stamped onto the Listing.

    Returns:
        Listing with 156 weekly bars (NaN-padded if yfinance
        returned fewer rows) and 30 fundamentals: the 17
        yfinance-derived fields plus the 13 Phase-8 defeatbeta
        fields NaN-filled (defeatbeta enrichment is a separate
        path).

    Raises:
        Any yfinance exception propagates so refresh_exchange
        can decide whether to skip or abort.
    """
    yf_ticker = yf.Ticker(ticker)
    history = yf_ticker.history(period='3y', interval='1wk')
    info = yf_ticker.info

    closes = _to_padded_array(history.get('Close'))
    volumes = _to_padded_array(history.get('Volume'))
    dividends = _to_padded_array(history.get('Dividends'))
    dates = _padded_dates(history.index)

    fundamentals = {}
    for our_name, yf_name in _yfinance_to_fundamental.items():
        raw = info.get(yf_name)
        fundamentals[our_name] = (
            float(raw) if raw is not None else float('nan'))
    # Phase 8: defeatbeta-only fields are not populated by the
    # yfinance path; NaN-fill so the Listing contract holds.
    for name in fundamental_field_names:
        fundamentals.setdefault(name, float('nan'))

    market_cap = info.get('marketCap')
    market_cap_value = (
        float(market_cap) if market_cap is not None
        else float('nan'))
    currency = info.get('currency') or 'UNKNOWN'
    sector = info.get('sector') or 'Unknown'

    return Listing(
        ticker=ticker, exchange=exchange,
        currency=currency, sector=sector,
        market_cap=market_cap_value,
        closes_weekly=closes, closes_weekly_dates=dates,
        volumes_weekly=volumes, dividends_weekly=dividends,
        fundamentals=fundamentals,
        refreshed_at=datetime.now())


def _build_listing_via_defeatbeta(
    ticker: str,
    exchange: str,
    provider: DefeatBetaProvider) -> Listing:
    """
    Build one Listing via defeatbeta bulk parquet snapshots.

    Args:
        ticker: US-listed ticker symbol.
        exchange: 'NYSE' or 'NASDAQ'; stamped onto the Listing.
        provider: A live DefeatBetaProvider.

    Returns:
        Listing with 156 weekly bars (resampled from defeatbeta
        daily prices over the last `defeatbeta_history_weeks`)
        and the 30-key fundamentals envelope. Currency is
        always 'USD' since defeatbeta only covers US listings.

    Notes:
        Reaches into ``provider._ticker(ticker)`` to read the
        raw price DataFrame for per-day volume and the
        ``info()`` table for sector. This is a deliberate
        coupling to the provider's per-symbol cache: the
        public ``DataProvider`` ABC has no concept of "raw
        OHLCV frame" or "company profile", so the alternative
        would be N more parquet round-trips per ticker.

        ``market_cap`` is sourced from
        ``fundamentals['market_cap_chf']`` — name retained
        for cross-currency uniformity in the Listing schema,
        but the value here is USD because defeatbeta only
        covers US-listed equities. Consumers reading
        ``listing.market_cap`` get the correct USD numeric
        value; the field name is a Phase 8 naming quirk
        documented in the spec.

    Raises:
        Any defeatbeta-side exception propagates so
        `_refresh_via_defeatbeta` can warn-and-skip.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(
        weeks=defeatbeta_history_weeks)

    # Fundamentals (30-key envelope; missing fields surface as
    # NaN courtesy of provider._safe_last / _safe_dcf).
    fundamentals = provider.get_fundamental_data(ticker)

    # Daily close prices, resampled to weekly closes.
    prices = provider.get_price_history(
        ticker, start_date=start_date, end_date=end_date)
    weekly_close = prices.resample('W').last().dropna()
    closes = _to_padded_array(weekly_close)

    # Volumes come from the underlying defeatbeta price()
    # frame; resample as weekly sum (matches yfinance's weekly
    # volume convention).
    raw = provider._ticker(ticker).price().copy()
    raw['report_date'] = pd.to_datetime(raw['report_date'])
    raw = raw.set_index('report_date').sort_index()
    weekly_volume = raw['volume'].resample(
        'W').sum().dropna()
    volumes = _to_padded_array(weekly_volume)

    # Dividends as weekly sums, reindexed onto the close
    # calendar so length matches.
    div_series = provider.get_dividend_history(
        ticker, start_date=start_date)
    if div_series.empty:
        weekly_div = pd.Series(0.0, index=weekly_close.index)
    else:
        weekly_div = div_series.resample('W').sum().reindex(
            weekly_close.index, fill_value=0.0)
    dividends = _to_padded_array(weekly_div)

    dates = _padded_dates(weekly_close.index)

    sector = _defeatbeta_sector(provider, ticker)

    market_cap_raw = fundamentals['market_cap_chf']
    market_cap = (
        float(market_cap_raw)
        if not pd.isna(market_cap_raw) else float('nan'))

    return Listing(
        ticker=ticker, exchange=exchange,
        currency='USD', sector=sector,
        market_cap=market_cap,
        closes_weekly=closes, closes_weekly_dates=dates,
        volumes_weekly=volumes, dividends_weekly=dividends,
        fundamentals=fundamentals,
        refreshed_at=datetime.now())


def _defeatbeta_sector(
    provider: DefeatBetaProvider, ticker: str) -> str:
    """
    Read the sector label from defeatbeta's info() table.

    Defeatbeta's info() shape varies by ticker (~12 columns);
    return 'Unknown' if the call raises or the column is
    absent. Matches the warn-and-degrade pattern used for
    fundamentals.

    Args:
        provider: A live DefeatBetaProvider.
        ticker: US-listed ticker symbol.

    Returns:
        Sector string, or 'Unknown' on any failure.
    """
    try:
        info_df = provider._ticker(ticker).info()
        if (info_df is None or info_df.empty
                or 'sector' not in info_df.columns):
            return 'Unknown'
        value = info_df.iloc[0]['sector']
        if value is None or pd.isna(value):
            return 'Unknown'
        return str(value)
    except Exception:
        return 'Unknown'


def _to_padded_array(
    series: Optional[pd.Series]) -> np.ndarray:
    """
    Normalise a pandas Series to a length-156 numpy array.

    Pads with NaN at the front when the source is shorter
    than `weekly_series_length`. Truncates to the most recent
    `weekly_series_length` bars when longer.
    """
    if series is None or len(series) == 0:
        return np.full(weekly_series_length, np.nan)
    values = series.to_numpy(dtype=float)
    if len(values) >= weekly_series_length:
        return values[-weekly_series_length:]
    pad = np.full(weekly_series_length - len(values), np.nan)
    return np.concatenate([pad, values])


def _padded_dates(
    index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Normalise a DatetimeIndex to length 156.

    Pads at the front with synthetic weekly dates (W-FRI) when
    the source is shorter; truncates to the most recent 156
    when longer.
    """
    n = len(index)
    if n >= weekly_series_length:
        return index[-weekly_series_length:]
    if n == 0:
        end = pd.Timestamp.utcnow().normalize()
        return pd.date_range(
            end=end, periods=weekly_series_length, freq='W-FRI')
    head_count = weekly_series_length - n
    head_end = index[0] - pd.Timedelta(weeks=1)
    head = pd.date_range(
        end=head_end, periods=head_count, freq='W-FRI')
    return head.append(index)
