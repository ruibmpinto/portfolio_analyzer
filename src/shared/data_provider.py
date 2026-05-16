"""
Abstract data provider interface for market data.

Supports multiple data sources (yfinance, qf-lib) through
a common interface, allowing easy switching between providers.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import warnings

# Optional dependency. Importable absence is non-fatal here so
# unrelated parts of the package keep working; YFinanceProvider
# methods will fail at call time if `has_yfinance` is False.
try:
    import yfinance as yf
    has_yfinance = True
except ImportError:
    yf = None
    has_yfinance = False
    warnings.warn(
        "yfinance not installed. Install with "
        "'pip install yfinance' to use YFinanceProvider.")

# Optional, heavy dependency. Same pattern as yfinance: a flag
# lets QFLibProvider raise a clear error at construction time
# without breaking module import for users who only need
# YFinanceProvider.
try:
    from qf_lib.data_providers.yahoo import YahooDataProvider
    from qf_lib.common.tickers.tickers import Ticker as QFTicker
    has_qf_lib = True
except ImportError:
    YahooDataProvider = None
    QFTicker = None
    has_qf_lib = False
    warnings.warn("qf_lib not installed.")

# Optional dependency.
try:
    from defeatbeta_api.data.ticker import Ticker
    has_defeatbeta = True
except ImportError:
    Ticker = None
    has_defeatbeta = False
    warnings.warn(
        "defeatbeta_api not installed. Install with "
        "'pip install defeatbeta-api' to use DefeatBetaProvider.")

# Module-level constants used by DefeatBetaProvider.is_native to
# distinguish US-listed equities (defeatbeta-supported) from
# non-US suffixed tickers and Yahoo FX pseudo-tickers.
non_us_suffix_marker = '.'
fx_suffix_marker = '=X'


class DataProvider(ABC):
    """
    Abstract base class for market data providers.

    Defines interface for fetching price history,
    fundamental data, and dividend information.
    """

    @abstractmethod
    def get_price_history(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime) -> pd.Series:
        """
        Fetch historical price data for a ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            Series of closing prices indexed by date

        Raises:
            ValueError: If no data available for ticker
        """
        pass

    @abstractmethod
    def get_fundamental_data(self, ticker: str) -> Dict[str, Optional[float]]:
        """
        Fetch fundamental data for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with fundamental metrics:
                - trailing_pe: Trailing P/E ratio
                - forward_pe: Forward P/E ratio
                - price_to_book: P/B ratio
                - trailing_eps: Trailing EPS
                - dividend_yield: Dividend yield
                - dividend_rate: Annual dividend
                - payout_ratio: Dividend payout ratio
                - market_cap: Market capitalization
                - book_value: Book value per share
        """
        pass

    @abstractmethod
    def get_dividend_history(
        self,
        ticker: str,
        start_date: datetime) -> pd.Series:
        """
        Fetch dividend payment history.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for dividend history

        Returns:
            Series of dividend payments indexed by ex-date
        """
        pass

    @abstractmethod
    def get_split_history(self, ticker: str) -> pd.Series:
        """
        Fetch stock-split history.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Series of split ratios indexed by ex-date (e.g.
            `10.0` for a 10-for-1 split). Empty Series when
            the ticker has had no splits or when no data is
            available.
        """
        pass

    @abstractmethod
    def get_dividend_price_factors(self, ticker: str) -> pd.Series:
        """
        Fetch per-event dividend price-deflation factors.

        For each dividend ex-date `t_i` with amount `D_i` and
        prior close `C_prev`, the factor is `1 - D_i / C_prev`
        — the same multiplicative deflation yfinance applies
        to past closes when `auto_adjust=True`. The cumulative
        product over ex-dates strictly after a valuation date
        `t` gives the deflation factor from `t` to today;
        inverting it recovers the split-adjusted-but-not-
        dividend-adjusted past close.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Series of deflation factors (each in (0, 1])
            indexed by ex-date. Empty Series when the ticker
            has paid no dividends or no data is available.
        """
        pass

    @abstractmethod
    def get_current_price(self, ticker: str) -> float:
        """
        Fetch current/latest price for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Latest available price

        Raises:
            ValueError: If no data available for ticker
        """
        pass


class YFinanceProvider(DataProvider):
    """
    Market data provider using yfinance library.

    Lightweight provider with caching for improved
    performance. Suitable for most use cases.
    """

    def __init__(self):
        """Initialize provider with empty cache."""

        self._cache = {}
        self.yf = yf

    def get_price_history(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime) -> pd.Series:
        """
        Fetch historical price data using yfinance.

        Caches results to avoid redundant API calls.
        """
        # Cache key uniquely identifies this (ticker, range) call
        cache_key = (f"{ticker}_prices_{start_date.date()}_{end_date.date()}")

        # Short-circuit if this exact query has been served before
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Build a yfinance Ticker handle (no network call yet)
            stock = self.yf.Ticker(ticker)
            # Fetch OHLCV (Open, High, Low, Close, Volume) daily bars
            # for the requested window
            hist = stock.history(start=start_date, end=end_date)

            if hist.empty:
                raise ValueError(f"No price data for {ticker}")

            # Keep only the Close column; OHLCV not needed here
            prices = hist['Close']

            # Strip timezone so date math stays naive-vs-naive
            if prices.index.tz is not None:
                prices.index = prices.index.tz_localize(None)

            # Memoize for subsequent identical requests
            self._cache[cache_key] = prices
            return prices

        except Exception as e:
            raise ValueError(f"Error fetching {ticker}: {str(e)}")

    def get_fundamental_data(self, ticker: str) -> Dict[str, Optional[float]]:
        """Fetch fundamental data using yfinance."""

        # One fundamentals snapshot per ticker per session
        cache_key = f"{ticker}_fundamentals"
        # Short-circuit if this exact query has been served before
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Build a yfinance Ticker handle
            stock = self.yf.Ticker(ticker)
            # Fetch the (large) info dict from Yahoo
            info = stock.info

            # Pick out only the fields this project consumes
            fundamentals = {
                'trailing_pe': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'price_to_book': info.get('priceToBook'),
                'trailing_eps': info.get('trailingEps'),
                'dividend_yield': info.get('dividendYield'),
                'dividend_rate': info.get('dividendRate'),
                'payout_ratio': info.get('payoutRatio'),
                'market_cap': info.get('marketCap'),
                'book_value': info.get('bookValue'),
                'sector': info.get('sector'),
            }

            self._cache[cache_key] = fundamentals
            return fundamentals

        except Exception as e:
            # Yahoo errors are non-fatal here; warn and return Nones
            warnings.warn(
                f"Error fetching fundamentals for {ticker}: {str(e)}")

            return {k: None for k in [
                'trailing_pe', 'forward_pe',
                'price_to_book', 'trailing_eps',
                'dividend_yield', 'dividend_rate',
                'payout_ratio', 'market_cap', 'book_value',
                'sector'
            ]}

    def get_dividend_history(
            self, 
            ticker: str, 
            start_date: datetime) -> pd.Series:
        """Fetch dividend history using yfinance."""

        # Cache key keyed by ticker + start date
        cache_key = (f"{ticker}_dividends_{start_date.date()}")
        # Short-circuit if this exact query has been served before
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Build a yfinance Ticker handle
            stock = self.yf.Ticker(ticker)
            # Full per-share dividend Series indexed by ex-date
            dividends = stock.dividends

            if dividends.empty:
                return pd.Series(dtype=float)

            # Strip timezone FIRST so the filter compares
            # naive-vs-naive; yfinance returns tz-aware dates.
            if dividends.index.tz is not None:
                dividends.index = dividends.index.tz_localize(None)

            # Drop rows older than the requested start date
            dividends = dividends[dividends.index >= start_date]

            self._cache[cache_key] = dividends
            # Return
            return dividends

        except Exception as e:
            warnings.warn(f"Error fetching dividends for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    def get_split_history(self, ticker: str) -> pd.Series:
        """Fetch stock-split history using yfinance."""

        # One splits snapshot per ticker per session
        cache_key = f"{ticker}_splits"
        # Short-circuit if this exact query has been served before
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Ex-date-indexed Series of split ratios (e.g. 10.0 for 10-for-1).
            # yfinance returns empty when none.
            splits = self.yf.Ticker(ticker).splits

            # Strip timezone for naive-vs-naive date math
            if splits.index.tz is not None:
                splits.index = splits.index.tz_localize(None)

            # Memorize for subsequent identical requests
            self._cache[cache_key] = splits

            return splits

        except Exception as e:
            # Non-fatal: warn and return empty so callers treat
            # the ticker as having no splits.
            warnings.warn(f"Error fetching splits for {ticker}: {str(e)}")

            return pd.Series(dtype=float)

    def get_dividend_price_factors(self, ticker: str) -> pd.Series:
        """Fetch per-event dividend deflation factors using yfinance."""

        # One factors snapshot per ticker per session
        cache_key = f"{ticker}_divfactors"
        # Short-circuit if this exact query has been served before
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Need unadjusted Close plus the per-row Dividends
            # column. auto_adjust=False is the only mode that
            # surfaces both in a single request.
            hist = self.yf.Ticker(ticker).history(
                period='max', auto_adjust=False)

            # Empty history (delisted / no data) -> no factors
            if hist.empty or 'Dividends' not in hist.columns:
                self._cache[cache_key] = pd.Series(dtype=float)
                return self._cache[cache_key]

            # Per-event factor: 1 - D / C_prev, where C_prev
            # is the close on the trading day before the ex-date
            closes = hist['Close']
            divs = hist['Dividends']
            prev_close = closes.shift(1)
            # Only rows with a positive dividend and a defined
            # prior close contribute a factor
            mask = (divs > 0) & (prev_close > 0)
            factors = 1.0 - divs[mask] / prev_close[mask]

            # Strip timezone for naive-vs-naive date math
            if factors.index.tz is not None:
                factors.index = factors.index.tz_localize(None)

            # Memorize for subsequent identical requests
            self._cache[cache_key] = factors

            return factors

        except Exception as e:
            # Non-fatal: warn and return empty so callers treat
            # the ticker as having no dividend adjustments.
            warnings.warn(
                f"Error fetching dividend price factors for "
                f"{ticker}: {str(e)}")

            return pd.Series(dtype=float)

    def get_current_price(self, ticker: str) -> float:
        """Fetch current price using yfinance."""
        try:
            # Build a yfinance Ticker handle
            stock = self.yf.Ticker(ticker)
            # Pull only today's bar (period='1d')
            hist = stock.history(period='1d')

            if hist.empty:
                raise ValueError(f"No current price for {ticker}")

            # Last Close in that 1-day frame is the latest tick
            return float(hist['Close'].iloc[-1])

        except Exception as e:
            raise ValueError(f"Error fetching price for {ticker}: {str(e)}")

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()


class QFLibProvider(DataProvider):
    """
    Market data provider using qf-lib library.

    Advanced quantitative finance library with
    more sophisticated data handling capabilities.
    """

    def __init__(self):
        """Initialize qf-lib data provider.

        Raises:
            ImportError: If qf-lib is not installed.
        """
        if not has_qf_lib:
            raise ImportError(
                "qf-lib not installed. "
                "Install with: pip install qf-lib")
        # qf-lib's Yahoo-backed provider does the I/O
        self._qf_provider = YahooDataProvider()
        # Keep Ticker constructor for wrapping plain strings
        self._Ticker = QFTicker
        self._cache = {}

    def get_price_history(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.Series:
        """Fetch historical price data using qf-lib."""

        # Cache key uniquely identifies this (ticker, range) call
        cache_key = (f"{ticker}_prices_{start_date.date()}_{end_date.date()}")
        # Short-circuit if this exact query has been served before
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Wrap the plain string in qf-lib's Ticker type
            qf_ticker = self._Ticker(ticker)
            # Hit the qf-lib provider for the requested window
            prices = self._qf_provider.get_price(
                qf_ticker,
                start_date=start_date,
                end_date=end_date)

            if prices is None or prices.empty:
                raise ValueError(f"No price data for {ticker}")

            # qf-lib may return a DataFrame; collapse to a Series
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]

            self._cache[cache_key] = prices
            # Return
            return prices

        except Exception as e:
            raise ValueError(f"Error fetching {ticker}: {str(e)}")

    def get_fundamental_data(self, ticker: str) -> Dict[str, Optional[float]]:
        """
        Fetch fundamental data using qf-lib.

        Note: qf-lib focuses on price data,
        so we fall back to yfinance for fundamentals.
        """
        cache_key = f"{ticker}_fundamentals"
        # Short-circuit if this exact query has been served before
        if cache_key in self._cache:
            return self._cache[cache_key]

        # qf-lib doesn't provide fundamentals directly
        # Fall back to the top-level yfinance import
        try:
            # Build a yfinance Ticker handle
            stock = yf.Ticker(ticker)
            # Fetch the (large) info dict from Yahoo
            info = stock.info

            # Pick out only the fields this project consumes
            fundamentals = {
                'trailing_pe': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'price_to_book': info.get('priceToBook'),
                'trailing_eps': info.get('trailingEps'),
                'dividend_yield': info.get('dividendYield'),
                'dividend_rate': info.get('dividendRate'),
                'payout_ratio': info.get('payoutRatio'),
                'market_cap': info.get('marketCap'),
                'book_value': info.get('bookValue'),
                'sector': info.get('sector'),
            }

            self._cache[cache_key] = fundamentals
            return fundamentals

        except Exception as e:
            warnings.warn(
                f"Error fetching fundamentals for "
                f"{ticker}: {str(e)}"
            )
            return {k: None for k in [
                'trailing_pe', 'forward_pe',
                'price_to_book', 'trailing_eps',
                'dividend_yield', 'dividend_rate',
                'payout_ratio', 'market_cap', 'book_value',
                'sector'
            ]}

    def get_dividend_history(
        self,
        ticker: str,
        start_date: datetime) -> pd.Series:
        """
        Fetch dividend history using qf-lib.

        Falls back to yfinance for dividend data.
        """
        cache_key = (f"{ticker}_dividends_{start_date.date()}")
        # Short-circuit if this exact query has been served before
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # qf-lib has no dividend feed; fall back to the
            # top-level yfinance import
            # Build a yfinance Ticker handle
            stock = yf.Ticker(ticker)
            # Full per-share dividend Series indexed by ex-date
            dividends = stock.dividends

            if dividends.empty:
                return pd.Series(dtype=float)

            # Strip timezone FIRST so the filter compares
            # naive-vs-naive; yfinance returns tz-aware dates.
            if dividends.index.tz is not None:
                dividends.index = dividends.index.tz_localize(None)

            # Drop rows older than the requested start date
            dividends = dividends[dividends.index >= start_date]

            self._cache[cache_key] = dividends
            # Return
            return dividends

        except Exception as e:
            warnings.warn(
                f"Error fetching dividends for "
                f"{ticker}: {str(e)}"
            )
            return pd.Series(dtype=float)

    def get_split_history(self, ticker: str) -> pd.Series:
        """Fetch stock-split history (qf-lib has none; uses yfinance)."""

        # One splits snapshot per ticker per session
        cache_key = f"{ticker}_splits"
        # Short-circuit if this exact query has been served before
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Ex-date-indexed Series of split ratios (e.g. 10.0 for 10-for-1).
            # yfinance returns empty when none.
            splits = yf.Ticker(ticker).splits

            # Strip timezone for naive-vs-naive date math
            if splits.index.tz is not None:
                splits.index = splits.index.tz_localize(None)

            # Memorize for subsequent identical requests
            self._cache[cache_key] = splits

            return splits

        except Exception as e:
            # Non-fatal: warn and return empty so callers treat
            # the ticker as having no splits.
            warnings.warn(f"Error fetching splits for {ticker}: {str(e)}")

            return pd.Series(dtype=float)

    def get_dividend_price_factors(self, ticker: str) -> pd.Series:
        """Fetch dividend deflation factors (qf-lib has none; uses yfinance)."""

        # One factors snapshot per ticker per session
        cache_key = f"{ticker}_divfactors"
        # Short-circuit if this exact query has been served before
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # qf-lib has no dividend-adjustment feed; fall back
            # to yfinance. auto_adjust=False is needed to get
            # both unadjusted Close and the Dividends column.
            hist = yf.Ticker(ticker).history(
                period='max', auto_adjust=False)

            # Empty history (delisted / no data) -> no factors
            if hist.empty or 'Dividends' not in hist.columns:
                self._cache[cache_key] = pd.Series(dtype=float)
                return self._cache[cache_key]

            # Per-event factor: 1 - D / C_prev, where C_prev
            # is the close on the trading day before the ex-date
            closes = hist['Close']
            divs = hist['Dividends']
            prev_close = closes.shift(1)
            # Only rows with a positive dividend and a defined
            # prior close contribute a factor
            mask = (divs > 0) & (prev_close > 0)
            factors = 1.0 - divs[mask] / prev_close[mask]

            # Strip timezone for naive-vs-naive date math
            if factors.index.tz is not None:
                factors.index = factors.index.tz_localize(None)

            # Memorize for subsequent identical requests
            self._cache[cache_key] = factors

            return factors

        except Exception as e:
            # Non-fatal: warn and return empty so callers treat
            # the ticker as having no dividend adjustments.
            warnings.warn(
                f"Error fetching dividend price factors for "
                f"{ticker}: {str(e)}")

            return pd.Series(dtype=float)

    def get_current_price(self, ticker: str) -> float:
        """Fetch current price using qf-lib."""

        try:
            # End of the lookup window is "now"
            end_date = datetime.now()
            # Start at midnight today, so the window covers
            # today's session bar
            start_date = datetime(end_date.year, end_date.month, end_date.day)

            # Wrap the plain string in qf-lib's Ticker type
            qf_ticker = self._Ticker(ticker)
            # Pull today's price from qf-lib
            prices = self._qf_provider.get_price(
                qf_ticker,
                start_date=start_date,
                end_date=end_date)

            if prices is None or prices.empty:
                raise ValueError(f"No current price for {ticker}")

            # DataFrame: take last row, first column.
            # Series: take last row.
            if isinstance(prices, pd.DataFrame):
                return float(prices.iloc[-1, 0])
            # Return
            return float(prices.iloc[-1])

        except Exception as e:
            raise ValueError(f"Error fetching price for {ticker}: {str(e)}")

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()


class DefeatBetaProvider(DataProvider):
    """Hugging Face mirror of Yahoo Finance via defeatbeta-api.

    Wraps the ``defeatbeta-api`` package (parquet snapshots of
    Yahoo Finance hosted on Hugging Face). 
    
    Coverage is limited to US-listed equities (NYSE / NASDAQ + ADRs); 
    non-US tickers (e.g. ``NESN.SW``) and Yahoo FX pseudo-tickers
    (``USDCHF=X``) are routed to ``fallback`` when supplied,
    or raise ``RuntimeError`` when ``fallback`` is ``None``.
    """

    def __init__(self, fallback: Optional[DataProvider] = None):
        """Initialize provider with empty cache and optional fallback.

        Args:
            fallback: Provider to delegate calls for non-US
                tickers. ``None`` means non-US tickers raise.

        Raises:
            ImportError: If defeatbeta-api is not installed.
        """
        if not has_defeatbeta:
            raise ImportError("defeatbeta_api not installed.")
        self._cache: Dict[str, 'Ticker'] = {}
        self._fallback = fallback

    def is_native(self, ticker: str) -> bool:
        """Return True iff ``ticker`` is a US-listed plain symbol.

        Excludes dotted exchange suffixes (e.g. ``.SW``, ``.L``)
        and Yahoo FX pseudo-tickers (``=X``).

        Args:
            ticker: Stock ticker symbol.

        Returns:
            True when defeatbeta is expected to cover the symbol.
        """
        return (non_us_suffix_marker not in ticker
                and not ticker.endswith(fx_suffix_marker))

    def _ticker(self, symbol: str) -> 'Ticker':
        """Return a cached defeatbeta ``Ticker`` for ``symbol``.

        ``Ticker`` construction triggers a metadata download
        (~1-2s the first time per symbol); cache instances per
        provider instance to avoid repeated downloads.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            A defeatbeta ``Ticker`` handle.
        """
        if symbol not in self._cache:
            self._cache[symbol] = Ticker(symbol)
        return self._cache[symbol]

    def _delegate_or_raise(
            self, ticker: str, op: str) -> DataProvider:
        """Return the fallback provider or raise if unavailable.

        Args:
            ticker: Stock ticker symbol that triggered routing.
            op: Name of the operation (for error messages).

        Returns:
            The configured fallback provider.

        Raises:
            RuntimeError: If no fallback was configured.
        """
        if self._fallback is None:
            raise RuntimeError(
                f"DefeatBetaProvider: ticker '{ticker}' is not "
                f"US-listed and no fallback provider is "
                f"configured for op '{op}'.")
        return self._fallback

    def get_price_history(
            self,
            ticker: str,
            start_date: datetime,
            end_date: datetime) -> pd.Series:
        """Fetch close-price history via defeatbeta-api.

        Args:
            ticker: Stock ticker symbol.
            start_date: Inclusive start of the window.
            end_date: Inclusive end of the window.

        Returns:
            Series of closing prices indexed by date.

        Raises:
            ValueError: If no data is available for ``ticker``.
            RuntimeError: If ``ticker`` is non-US and no
                fallback is configured.
        """
        if not self.is_native(ticker):
            return self._delegate_or_raise(
                ticker, 'get_price_history').get_price_history(
                    ticker, start_date, end_date)

        df = self._ticker(ticker).price()
        if df is None or df.empty:
            raise ValueError(
                f"No price data for {ticker} from defeatbeta")
        df = df.copy()
        df['report_date'] = pd.to_datetime(df['report_date'])
        df = df.set_index('report_date').sort_index()
        sliced = df.loc[start_date:end_date]
        if sliced.empty:
            raise ValueError(
                f"No price data for {ticker} in window "
                f"[{start_date.date()}, {end_date.date()}]")
        return sliced['close'].astype(float)

    def get_current_price(self, ticker: str) -> float:
        """Fetch the most recent close price via defeatbeta-api.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Latest available close price.

        Raises:
            ValueError: If no data is available for ``ticker``.
            RuntimeError: If ``ticker`` is non-US and no
                fallback is configured.
        """
        if not self.is_native(ticker):
            return self._delegate_or_raise(
                ticker, 'get_current_price').get_current_price(
                    ticker)

        df = self._ticker(ticker).price()
        if df is None or df.empty:
            raise ValueError(
                f"No current price for {ticker} from defeatbeta")
        return float(df.iloc[-1]['close'])

    def get_fundamental_data(
            self, ticker: str) -> Dict[str, Optional[float]]:
        """Fetch a 30-key fundamentals snapshot via defeatbeta.

        Each per-field call is wrapped in ``_safe_last`` (or
        ``_safe_dcf`` for the DCF metric) so that defeatbeta
        coverage gaps surface as ``NaN`` rather than raising
        — this is the documented exception to the loud-failure
        rule because field-level absence is expected.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Dict with the 30 fundamental keys defined by this
            provider; missing fields are ``float('nan')``.

        Raises:
            RuntimeError: If ``ticker`` is non-US and no
                fallback is configured.
        """
        if not self.is_native(ticker):
            return self._delegate_or_raise(
                ticker,
                'get_fundamental_data').get_fundamental_data(
                    ticker)

        t = self._ticker(ticker)
        nan = float('nan')
        return {
            # Original 17 keys (some NaN).
            'pe_ratio_ttm': _safe_last(t.ttm_pe, 'ttm_pe'),
            'pe_ratio_forward': nan,
            'pb_ratio': _safe_last(t.pb_ratio, 'pb_ratio'),
            'ps_ratio': _safe_last(t.ps_ratio, 'ps_ratio'),
            'dividend_yield_ttm': nan,
            'eps_ttm': _safe_last(t.ttm_eps, 'ttm_eps'),
            'revenue_growth_yoy': _safe_last(
                t.quarterly_revenue_yoy_growth,
                'revenue_yoy_growth'),
            'earnings_growth_yoy': _safe_last(
                t.quarterly_eps_yoy_growth, 'eps_yoy_growth'),
            'profit_margin': nan,
            'operating_margin': nan,
            'roe': _safe_last(t.roe, 'roe'),
            'roa': _safe_last(t.roa, 'roa'),
            'debt_to_equity': _safe_last(
                t.debt_to_equity, 'debt_to_equity'),
            'free_cash_flow': _safe_last(t.ttm_fcf, 'ttm_fcf'),
            'beta_yf': nan,
            'shares_outstanding': nan,
            'short_ratio': nan,
            # Phase 8 additions (13 keys).
            'peg_ratio': _safe_last(t.peg_ratio, 'peg_ratio'),
            'roic': _safe_last(t.roic, 'roic'),
            'roce': nan,
            'wacc': _safe_last(t.wacc, 'wacc'),
            'equity_multiplier': _safe_last(
                t.equity_multiplier, 'equity_multiplier'),
            'asset_turnover': _safe_last(
                t.asset_turnover, 'asset_turnover'),
            'enterprise_value': _safe_last(
                t.enterprise_value, 'enterprise_value'),
            'enterprise_to_revenue': _safe_last(
                t.enterprise_to_revenue,
                'enterprise_to_revenue'),
            'enterprise_to_ebitda': _safe_last(
                t.enterprise_to_ebitda,
                'enterprise_to_ebitda'),
            'ttm_revenue': _safe_last(
                t.ttm_revenue, 'ttm_revenue'),
            'ebitda_growth_yoy': _safe_last(
                t.quarterly_ebitda_yoy_growth,
                'ebitda_yoy_growth'),
            'market_cap_chf': _safe_last(
                t.market_capitalization,
                'market_capitalization'),
            'dcf_implied_upside': _safe_dcf(t),
        }

    def get_dividend_history(
            self,
            ticker: str,
            start_date: datetime) -> pd.Series:
        """Fetch dividend payment history via defeatbeta-api.

        Args:
            ticker: Stock ticker symbol.
            start_date: Inclusive start of the window.

        Returns:
            Series of dividend amounts indexed by ex-date.
            Empty Series when the ticker has paid no dividends.

        Raises:
            RuntimeError: If ``ticker`` is non-US and no
                fallback is configured.
        """
        if not self.is_native(ticker):
            return self._delegate_or_raise(
                ticker,
                'get_dividend_history').get_dividend_history(
                    ticker, start_date)

        df = self._ticker(ticker).dividends()
        if df is None or df.empty:
            return pd.Series(dtype=float)
        df = df.copy()
        df['report_date'] = pd.to_datetime(df['report_date'])
        df = df.set_index('report_date').sort_index()
        df = df.loc[start_date:]
        if df.empty:
            return pd.Series(dtype=float)
        return df['amount'].astype(float)

    def get_split_history(self, ticker: str) -> pd.Series:
        """Fetch stock-split history via defeatbeta-api.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Series of split factors indexed by ex-date. Empty
            Series when the ticker has had no splits.

        Raises:
            RuntimeError: If ``ticker`` is non-US and no
                fallback is configured.
        """
        if not self.is_native(ticker):
            return self._delegate_or_raise(
                ticker,
                'get_split_history').get_split_history(ticker)

        df = self._ticker(ticker).splits()
        if df is None or df.empty:
            return pd.Series(dtype=float)
        df = df.copy()
        df['report_date'] = pd.to_datetime(df['report_date'])
        df = df.set_index('report_date').sort_index()
        return df['split_factor'].astype(float)

    def get_dividend_price_factors(
            self, ticker: str) -> pd.Series:
        """Compute per-event dividend deflation factors.

        Same ``1 - D / C_prev`` formula as YFinanceProvider,
        derived from defeatbeta's ``dividends()`` and
        ``price()`` tables.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Series of deflation factors (each in (0, 1])
            indexed by ex-date. Empty Series when the ticker
            has paid no dividends.

        Raises:
            RuntimeError: If ``ticker`` is non-US and no
                fallback is configured.
        """
        if not self.is_native(ticker):
            return self._delegate_or_raise(
                ticker, 'get_dividend_price_factors'
            ).get_dividend_price_factors(ticker)

        t = self._ticker(ticker)
        divs_df = t.dividends()
        if divs_df is None or divs_df.empty:
            return pd.Series(dtype=float)
        prices_df = t.price()
        if prices_df is None or prices_df.empty:
            return pd.Series(dtype=float)

        divs_df = divs_df.copy()
        divs_df['report_date'] = pd.to_datetime(
            divs_df['report_date'])
        divs_df = divs_df.set_index(
            'report_date').sort_index()

        prices_df = prices_df.copy()
        prices_df['report_date'] = pd.to_datetime(
            prices_df['report_date'])
        prices_df = prices_df.set_index(
            'report_date').sort_index()
        closes = prices_df['close'].astype(float)
        # Per-event factor: 1 - D / C_prev. Build a prior-close
        # series, then for each dividend ex-date take the most
        # recent prior close via forward-fill reindex.
        prev_close = closes.shift(1)
        prev_at_div = prev_close.reindex(
            divs_df.index, method='ffill')
        amounts = divs_df['amount'].astype(float)
        mask = (amounts > 0) & (prev_at_div > 0)
        factors = 1.0 - amounts[mask] / prev_at_div[mask]
        return factors

    def clear_cache(self):
        """Clear all cached Ticker instances."""
        self._cache.clear()


def _safe_last(method_callable, column: str) -> float:
    """Call a defeatbeta ratio method and take the last row.

    Returns ``NaN`` when the method raises or the result is
    empty or the column is missing — defeatbeta coverage gaps
    are a known yfinance/defeatbeta variability issue and
    surface as missing field values rather than fatal errors.

    Args:
        method_callable: Bound method on a defeatbeta
            ``Ticker`` instance returning a DataFrame.
        column: Column to read from the last row.

    Returns:
        The last value as ``float``, or ``float('nan')`` on
        any failure.
    """
    try:
        df = method_callable()
        if df is None or df.empty or column not in df.columns:
            return float('nan')
        value = df.iloc[-1][column]
        return float(value)
    except Exception:
        return float('nan')


def _safe_dcf(ticker_obj) -> float:
    """Pull DCF implied upside from ``Ticker.dcf_data()``.

    Returns ``NaN`` when the call fails or the result lacks
    the expected ``implied_upside`` key.

    Args:
        ticker_obj: A defeatbeta ``Ticker`` instance.

    Returns:
        The implied upside as ``float``, or ``float('nan')``.
    """
    try:
        data = ticker_obj.dcf_data()
        if isinstance(data, dict) and 'implied_upside' in data:
            return float(data['implied_upside'])
        return float('nan')
    except Exception:
        return float('nan')
