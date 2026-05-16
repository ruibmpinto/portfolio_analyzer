"""
Main portfolio analyzer class.

Orchestrates all analysis modules to provide comprehensive
portfolio analytics through a simple API.
"""

import pathlib
import threading
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Union

from src.shared.data_provider import (
    DataProvider, YFinanceProvider
)
from src.analysis.core.portfolio import Portfolio
from src.analysis.loaders.csv_loader import CSVLoader
from src.shared.metrics.risk import RiskProfileAnalyzer
from src.shared.metrics.volatility import VolatilityAnalyzer
from src.shared.metrics.ratios import RatioAnalyzer
from src.analysis.plots.portfolio_plotter import PortfolioPlotter

# yfinance is an optional dependency for the package as a
# whole; we only need the dedicated exception classes here so
# we can distinguish rate-limits from other failures. Guarding
# the import keeps the module loadable when yfinance is
# missing (mirrors the pattern in data_provider.py).
try:
    from yfinance.exceptions import YFRateLimitError, YFPricesMissingError
except ImportError:
    class YFRateLimitError(Exception):
        """Stub when yfinance is not installed."""
        pass

    class YFPricesMissingError(Exception):
        """Stub when yfinance is not installed."""
        pass


def _fx_cache_path(pair: str) -> pathlib.Path:
    """Resolve the on-disk cache path for an FX pair."""
    # Anchor at the repo root so the cache is shared regardless
    # of the caller's cwd.
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    return repo_root / 'data' / 'cache' / 'fx' / f'{pair}.pkl'


# Listing-currency fallback for common benchmarks. Used when
# yfinance's fast_info lookup fails (or yfinance is missing).
# Extend on first encounter of a new benchmark ticker.
_benchmark_currency_fallback = {
    'SPY': 'USD',
    'QQQ': 'USD',
    'IWM': 'USD',
    'VTI': 'USD',
    'EWL': 'USD',
    'EZU': 'USD',
    'CHSPI.SW': 'CHF',
    'SMICHA.SW': 'CHF',
    'EUNL.DE': 'EUR',
}


class PortfolioAnalyzer:
    """
    Main portfolio analysis interface.

    Provides comprehensive portfolio analytics including
    returns, risk metrics, ratios, and visualizations.

    Example:
        >>> analyzer = PortfolioAnalyzer(
        ...     'reports/degiro/statements/'
        ...     '20240101_20260514_degiro_statement.csv'
        ... )
        >>> sharpe = analyzer.get_sharpe()
        >>> beta = analyzer.get_beta()
        >>> analyzer.plot_dashboard()
    """

    def __init__(
        self,
        degiro_csv_file_path: Optional[str] = None,
        ibkr_csv_file_path: Optional[str] = None,
        data_provider: Optional[DataProvider] = None,
        base_currency: str = 'CHF'):
        """
        Initialize portfolio analyzer.

        The two CSV paths are peers: pass either one, or both
        to analyse a combined cross-broker portfolio.
        At least one path must be supplied.

        Args:
            degiro_csv_file_path: Path to a Degiro processed
                transaction CSV (canonical 7-column schema).
            ibkr_csv_file_path: Path to an IBKR Activity Statement 
                (multi-section CSV).
            data_provider: Data provider instance (default: YFinanceProvider).
            base_currency: ISO-4217 currency in which all portfolio values, 
                benchmark values, and derived metrics are reported. 
                Defaults to 'CHF'. 
                Native-currency transactions are translated using daily FX 
                history fetched via the data provider.

        Raises:
            ValueError: If neither CSV path is provided.
        """
        if (degiro_csv_file_path is None
                and ibkr_csv_file_path is None):
            raise ValueError(
                "Provide at least one of "
                "degiro_csv_file_path or ibkr_csv_file_path.")
        if data_provider is None:
            data_provider = YFinanceProvider()

        # File locations
        self.degiro_csv_file_path = degiro_csv_file_path
        self.ibkr_csv_file_path = ibkr_csv_file_path
        # Data provider
        self.data_provider = data_provider
        # Reporting currency (everything downstream is in this)
        self.base_currency = base_currency.upper()
        # Max age (days) of an FX cache file before refresh.
        self.fx_cache_max_age_days = 3
        # Fixed FX-cache refresh anchor (covers portfolios from 2024 on).
        self.fx_cache_history_start = datetime(2024, 1, 1)

        # Initialize modules
        self.loader = CSVLoader()
        self.risk = RiskProfileAnalyzer(data_provider)
        self.volatility = VolatilityAnalyzer()
        self.ratios = RatioAnalyzer(data_provider)
        self.plotter = PortfolioPlotter()

        # Combine transactions from whichever sources were
        # supplied. Portfolio re-sorts internally so the merge
        # order here does not matter.
        self.transactions = []
        if degiro_csv_file_path is not None:
            self.transactions += self.loader.load_csv_degiro(
                degiro_csv_file_path)
        if ibkr_csv_file_path is not None:
            self.transactions += self.loader.load_csv_ibkr(
                ibkr_csv_file_path)

        # Add transactions to portfolio
        self.portfolio = Portfolio(self.transactions)

        # Map ticker -> trade currency, built once from the transaction list. 
        # Later transactions overwrite earlier ones for the same ticker.
        self._ticker_currencies: Dict[str, str] = {}
        for t in self.transactions:
            self._ticker_currencies[t.ticker] = t.currency

        # Cached data
        self._market_data = {}
        self._fx_data: Dict[str, pd.Series] = {}
        self._benchmark_currencies: Dict[str, str] = {}
        self._portfolio_values = None
        self._benchmark_data = None
        # Guards _fetch_market_data so concurrent dashboard
        # requests cannot enter the partially-populated state
        # (would otherwise see _market_data non-empty before
        # set_splits has run, leading to wrong holdings counts).
        self._fetch_lock = threading.Lock()

    def get_sharpe(self, risk_free_rate: float = 2.0) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            risk_free_rate: Annual risk-free rate (%)

        Returns:
            Sharpe ratio

        Example:
            >>> sharpe = analyzer.get_sharpe(
            ...     risk_free_rate=2.0
            ... )
        """
        returns = self._get_portfolio_returns()
        # Return
        return self.risk.get_sharpe_ratio(returns, risk_free_rate)

    def get_alpha(
        self,
        benchmark: str = 'SPY',
        risk_free_rate: float = 2.0) -> Optional[float]:
        """
        Calculate Jensen's alpha vs benchmark.

        Args:
            benchmark: Benchmark ticker (default: SPY)
            risk_free_rate: Annual risk-free rate (%)

        Returns:
            Alpha (%) or None

        Example:
            >>> alpha = analyzer.get_alpha(
            ...     benchmark='SPY'
            ... )
        """
        portfolio_returns = self._get_portfolio_returns()
        market_returns = self._get_benchmark_returns(benchmark)
        # Return
        return self.risk.get_alpha(
            portfolio_returns,
            market_returns,
            risk_free_rate)

    def get_beta(self, benchmark: str = 'SPY') -> Optional[float]:
        """
        Calculate portfolio beta vs benchmark.

        Args:
            benchmark: Benchmark ticker (default: SPY)

        Returns:
            Beta coefficient or None
        """
        # Get price data
        self._fetch_market_data()

        holdings = self.portfolio.get_current_holdings()

        # Return
        return self.risk.get_portfolio_beta(
            holdings,
            self._market_data,
            benchmark)

    def get_volatility(
        self, window: Optional[int] = None) -> Union[float, pd.Series]:
        """
        Calculate volatility.

        Args:
            window: Rolling window size. 
                If None, returns annualized volatility. 
                If int, returns rolling volatility series.

        Returns:
            Float if window=None, Series if rolling
        """
        returns = self._get_portfolio_returns()

        if window is None:
            return self.volatility.compute_annualized_volatility(returns)

        return self.volatility.compute_rolling_volatility(returns, window)

    def get_pe_ratio(self) -> Optional[float]:
        """
        Calculate portfolio-weighted P/E ratio.

        Returns:
            Portfolio P/E ratio or None
        """
        self._fetch_market_data()
        holdings = self.portfolio.get_current_holdings()
        # Return
        return self.ratios.get_portfolio_pe_ratio(holdings, self._market_data)

    def get_current_position_values(self) -> Dict[str, float]:
        """
        Per-ticker market value of current holdings, in
        `self.base_currency`.

        Iterates the current holdings and re-uses
        `_calculate_value` per ticker so split, dividend, and
        FX corrections all flow through the standard path.

        Returns:
            Dictionary mapping ticker -> position market value
            in base currency, valued at today.
        """
        self._fetch_market_data()
        holdings = self.portfolio.get_current_holdings()
        today = datetime.now()

        # Per-ticker value in base currency
        values: Dict[str, float] = {}
        for ticker, shares in holdings.items():
            # Reuse the single-position branch of _calculate_value
            # by passing a one-entry holdings dict
            values[ticker] = self._calculate_value(
                {ticker: shares}, today,
                apply_dividend_correction=True)
        # Return
        return values

    def get_holdings_snapshot(
        self,
        categories: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Per-position snapshot DataFrame.

        Aggregates current shares, base-currency values,
        weights, currencies, sectors, and per-share prices into
        a single DataFrame. Closed positions (shares <= 0) are
        dropped. Sorted by `value_chf` descending.

        Args:
            categories: Optional ticker -> category-label map.
                When provided, adds a 'category' column. A held
                ticker missing from this map raises RuntimeError
                (per no-silent-defaults policy).

        Returns:
            DataFrame with columns: ticker, shares, value_chf,
            weight_pct, currency, sector, price_chf [, category].

        Raises:
            RuntimeError: When a held ticker is missing a
                valuation, currency, or category.
        """
        self._fetch_market_data()
        shares_map = self.portfolio.get_current_holdings()
        values_map = self.get_current_position_values()
        total_value = sum(values_map.values())

        # One row per open position; closed positions skipped
        rows = []
        for ticker, shares in shares_map.items():
            
            if shares <= 0:
                continue
            
            if ticker not in values_map:
                raise RuntimeError(
                    f'Ticker {ticker!r} missing valuation.')
            value = values_map[ticker]
            
            if ticker not in self._ticker_currencies:
                raise RuntimeError(
                    f'Ticker {ticker!r} missing currency.')
            currency = self._ticker_currencies[ticker]
            
            # Sector via fundamentals; ETFs / proxies fall back
            fundamentals = self.data_provider.get_fundamental_data(
                ticker)
            
            sector = fundamentals.get('sector')
            if not sector:
                sector = 'Unknown'
            
            weight = 0.0
            if total_value > 0:
                weight = value / total_value * 100.0
            # shares > 0 guaranteed by the loop's skip above
            price_chf = value / shares
            row = {
                'ticker': ticker,
                'shares': shares,
                'value_chf': value,
                'weight_pct': weight,
                'currency': currency,
                'sector': sector,
                'price_chf': price_chf}
            
            if categories is not None:
                if ticker not in categories:
                    raise RuntimeError(
                        f'Ticker {ticker!r} missing category in '
                        f'provided categories map.')
                row['category'] = categories[ticker]
            rows.append(row)

        # Order largest-position-first; reset index 0..N-1
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('value_chf', ascending=False)
            df = df.reset_index(drop=True)
        return df

    def get_price_panel(self) -> pd.DataFrame:
        """
        Wide DataFrame of daily closes for currently-held tickers.

        Each column is a ticker's native-currency daily close series,
        drawn from the analyzer's existing market-data cache.
        Held tickers absent from the cache raise RuntimeError —
        never silently dropped.

        Returns:
            DataFrame with one column per held ticker (in
            `get_current_holdings()` iteration order). Index is
            the union of all included tickers' price-history
            dates (DatetimeIndex). Values are native-currency
            close prices.

        Raises:
            RuntimeError: When at least one held ticker is
                missing from `_market_data`. The message lists
                the missing tickers in sorted order.
        """
        self._fetch_market_data()
        held = self.portfolio.get_current_holdings()

        # Restrict to open positions; collect any held ticker
        # that lacks cached price history so we fail loudly
        columns = {}
        missing = []
        for ticker, shares in held.items():
            if shares <= 0:
                continue
            if ticker not in self._market_data:
                missing.append(ticker)
                continue
            columns[ticker] = self._market_data[ticker]

        # Fail loudly if anything was dropped
        if missing:
            raise RuntimeError(
                f'get_price_panel: held ticker(s) without '
                f'cached price history: {sorted(missing)}.')

        # DataFrame() aligns Series on the union of their indices
        return pd.DataFrame(columns)

    def get_dividend_yield(self) -> Optional[float]:
        """
        Calculate portfolio-weighted dividend yield.

        Returns:
            Dividend yield (decimal) or None
        """
        self._fetch_market_data()
        holdings = self.portfolio.get_current_holdings()
        # Return
        return self.ratios.get_portfolio_dividend_yield(
            holdings, self._market_data)

    def get_sortino_ratio(self, risk_free_rate: float = 2.0) -> float:
        """
        Calculate Sortino ratio.

        Args:
            risk_free_rate: Annual risk-free rate (%)

        Returns:
            Sortino ratio
        """
        returns = self._get_portfolio_returns()
        # Return
        return self.risk.get_sortino_ratio(returns, risk_free_rate)

    def get_information_ratio(self, benchmark: str = 'SPY') -> float:
        """
        Calculate information ratio vs benchmark.

        Args:
            benchmark: Benchmark ticker

        Returns:
            Information ratio
        """
        portfolio_returns = self._get_portfolio_returns()
        benchmark_returns = self._get_benchmark_returns(benchmark)
        # Return
        return self.risk.get_information_ratio(
            portfolio_returns, benchmark_returns)

    def plot_dashboard(
        self,
        benchmark: str = 'SPY',
        save_path: Optional[str] = None,
        is_dividend_reinvested: bool = False):
        """
        Display comprehensive analysis dashboard.

        Shows:
        - Portfolio value vs hold vs benchmark
        - Cumulative returns
        - P/E ratio over time
        - Rolling volatility
        - Risk-return scatter
        - Dividend evolution

        Args:
            benchmark: Benchmark ticker for comparison.
            save_path: If provided, save to this path instead
                of displaying.
            is_dividend_reinvested: If True, the benchmark
                curve also reinvests dividends and withdraws
                withholding tax received by the real portfolio.
                Default False: only buys/sells move the
                benchmark.
        """
        # NAV series for Portfolio Value Over Time
        actual, hold, dates = self._calculate_portfolio_values()
        bench_values = self._get_benchmark_series(
            benchmark, dates, is_dividend_reinvested=is_dividend_reinvested)

        # Daily Modified-Dietz returns.
        # Flow sets: 
        # actual = buy+sell; 
        # hold = buy only (no sells); 
        # benchmark mirrors the actual cash-flow used in _get_benchmark_series.
        actual_flows = self._daily_contributions(dates)
        hold_flows = self._daily_contributions(dates, ops=('buy',))
        actual_r = self._modified_dietz(actual, actual_flows)
        hold_r = self._modified_dietz(hold, hold_flows)
        bench_r = self._modified_dietz(bench_values, actual_flows)

        # Cumulative TWR (%) for plot 2 (Returns vs Benchmark)
        actual_cum = self._cumulative_twr(actual_r)
        hold_cum = self._cumulative_twr(hold_r)
        bench_cum = self._cumulative_twr(bench_r)

        pe_series = self._calculate_pe_over_time(dates)
        # Gross dividends, withholding tax, and fees are
        # displayed as three independent cumulative lines so
        # the user sees income earned vs. tax / fee leakage.
        dividends = self._get_portfolio_dividends(gross=True)
        taxes = self._get_portfolio_dividend_tax()
        fees = self._get_portfolio_fees()

        self.plotter.create_comprehensive_dashboard(
            actual=actual,
            hold=hold,
            benchmark=bench_values,
            actual_returns=actual_r,
            hold_returns=hold_r,
            benchmark_returns=bench_r,
            actual_cum_twr=actual_cum,
            hold_cum_twr=hold_cum,
            benchmark_cum_twr=bench_cum,
            pe_series=pe_series,
            dividends=dividends,
            taxes=taxes,
            fees=fees,
            save_path=save_path,
            base_currency=self.base_currency)

    def get_summary(self, benchmark: str = 'SPY') -> Dict[str, float]:
        """
        Get summary of key metrics.

        Args:
            benchmark: Benchmark ticker

        Returns:
            Dictionary of key metrics
        """
        return {
            'sharpe_ratio': self.get_sharpe(),
            'sortino_ratio': self.get_sortino_ratio(),
            'alpha': self.get_alpha(benchmark),
            'beta': self.get_beta(benchmark),
            'volatility': self.get_volatility(),
            'pe_ratio': self.get_pe_ratio(),
            'dividend_yield': self.get_dividend_yield(),
            'information_ratio': (self.get_information_ratio(benchmark)),
        }

    def _get_portfolio_returns(self) -> pd.Series:
        """Modified-Dietz daily returns for the actual NAV."""

        if self._portfolio_values is None:
            actual, _, _ = self._calculate_portfolio_values()
            self._portfolio_values = actual

        nav = self._portfolio_values

        return self._modified_dietz(nav, self._daily_contributions(nav.index))

    def _get_backcast_returns(
        self,
        values: Optional[Dict[str, float]] = None) -> pd.Series:
        """Backcast daily returns under today's snapshot weights.

        Applies the *current* portfolio composition to the full
        historical price series — `Σ_i w_i × r_{i,t}` where
        `w_i` is today's weight and `r_{i,t}` is asset `i`'s
        return on day `t`. Distinct from `_get_portfolio_returns`,
        which is flow-weighted Modified-Dietz on the actual
        historical share counts.

        Args:
            values: Optional per-ticker base-currency market
                value (output of `get_current_position_values`).
                None (default) fetches a fresh snapshot.

        Returns:
            Daily backcast returns indexed by date.
        """
        self._fetch_market_data()
        if values is None:
            values = self.get_current_position_values()

        # Portfolio total in base currency drives the weights
        total_value = sum(values.values())
        # Per-ticker daily returns, columns keyed by ticker
        prices = pd.DataFrame(self._market_data)
        returns_df = prices.pct_change()

        # Use all trading days (union of all exchanges)
        portfolio_returns = pd.Series(0.0, index=returns_df.index)
        # Per-day total weight that actually contributed
        weight_series = pd.Series(0.0, index=returns_df.index)

        for ticker, value in values.items():
            # Skip names with no fetched price history
            if ticker not in returns_df.columns:
                raise RuntimeError(
                    f'Ticker {ticker!r} missing price history for backcast.')
            # Current-snapshot weight of this position
            w = value / total_value
            asset_ret = returns_df[ticker]
            # Days where this ticker has a quoted return
            valid = asset_ret.notna()
            # Add this position's weighted contribution
            portfolio_returns[valid] += asset_ret[valid] * w
            # Track contributing weight for renormalisation
            weight_series[valid] += w

        # Normalize by the weight that actually contributed
        mask = weight_series > 0
        portfolio_returns[mask] = (
            portfolio_returns[mask] / weight_series[mask])
        # Drop days where no ticker contributed
        portfolio_returns = portfolio_returns[mask]
        # Drop first row (NaN from pct_change)
        portfolio_returns = portfolio_returns.iloc[1:]
        # Return
        return portfolio_returns

    def _modified_dietz(self, nav: pd.Series, flows: pd.Series) -> pd.Series:
        """
        Modified-Dietz daily portfolio returns in `base_currency`, 
        contributions stripped.

        Naive `nav.pct_change()` mistreats every buy as a
        return: NAV jumps when capital is added, and the
        derivative reads that jump as performance. The
        resulting Sharpe / alpha / volatility are inflated
        whenever the portfolio is in an accumulation phase
        (DCA, monthly contributions, lump-sum deposits).

        Modified Dietz removes the contribution effect at the
        daily level:

            r_t = (NAV_t - NAV_{t-1} - C_t) / NAV_{t-1}

        with `C_t` the net deposit-equivalent cash flow on
        day `t` (buy: positive; sell: negative). The flow is
        treated as end-of-day (`w = 0`), so it does NOT enter
        the denominator — it only affects tomorrow's
        starting NAV. Dividends and tax are not external
        contributions and do not appear in `C_t`.

        Returns:
            Series of daily returns (decimal), indexed by
            valuation date. First day and any day with
            NAV_{t-1} == 0 are dropped.
        """
        # Yesterday's NAV is the denominator
        prev = nav.shift(1)
        # Modified-Dietz: subtract today's external flow
        # before dividing by yesterday's NAV
        r = (nav - prev - flows) / prev

        # Drop first day (NaN) and any flat-then-buy day
        # where NAV_{t-1} == 0 produced +/-inf
        return r.replace([float('inf'), float('-inf')], pd.NA).dropna()

    def _cumulative_twr(self, returns: pd.Series) -> pd.Series:
        """Cumulative TWR (%) from a daily return series."""

        # Chain daily returns geometrically, subtract 1, scale to percent. 
        return ((1 + returns).cumprod() - 1) * 100

    def _daily_contributions(
        self,
        dates: pd.DatetimeIndex,
        ops: Tuple[str, ...] = ('buy', 'sell')) -> pd.Series:
        """Net cash flow per day in `base_currency`.

        Only the listed `ops` count. Default ('buy', 'sell')
        fits the actual portfolio and the no-reinvest
        benchmark; pass ('buy',) for the hold strategy.
        """
        # Accumulator: zero on every day, then add as we go
        flows = pd.Series(0.0, index=dates)
        for t in self.portfolio.transactions:
            # Skip operations not requested by the caller
            if t.operation not in ops:
                continue
            # Bucket each transaction at its midnight timestamp
            d = pd.Timestamp(t.date.date())
            # Drop transactions outside the NAV's date range
            if d not in flows.index:
                continue
            # buy: total_cost < 0 -> +flow (deposit)
            # sell: total_cost > 0 -> -flow (withdrawal)
            cash_native = -t.total_cost
            # Translate to base currency at the trade date
            flows.loc[d] += self._native_to_base(cash_native, t.currency, d)

        return flows

    def _get_benchmark_returns(self, benchmark: str) -> pd.Series:
        """Daily returns of `benchmark`, in `base_currency`."""
        start = self.portfolio.get_start_date()
        end = datetime.now()
        prices = self.data_provider.get_price_history(benchmark, start, end)

        # Convert native-currency prices to base currency so
        # returns line up with the CHF-denominated portfolio.
        ccy = self._benchmark_currency(benchmark)
        if ccy != self.base_currency:
            fx = self._fx_data[ccy].reindex(prices.index, method='ffill')
            prices = prices * fx
        # Return
        return prices.pct_change().dropna()

    def _fetch_market_data(self):
        """Fetch price data for every ticker ever held."""
        # Short-circuit if already populated in the cache
        if self._market_data:
            return

        # Serialise the build so concurrent dashboard requests
        # cannot observe a half-populated _market_data
        with self._fetch_lock:
            # Re-check under the lock in case another thread
            # finished while we were waiting
            if self._market_data:
                return

            # Cover every ticker that appears in the transaction
            # history, including positions that have been fully sold.
            tickers = self.portfolio.get_all_tickers()
            # Earliest transaction date bounds the lookup window
            start = self.portfolio.get_start_date()
            # Upper bound is "now" for valuation against today
            end = datetime.now()

            # Build into a local dict; only publish to
            # self._market_data once splits + dividend factors
            # are attached, so readers never see a partial state.
            local_prices: Dict[str, pd.Series] = {}
            splits: Dict[str, pd.Series] = {}
            dividend_factors: Dict[str, pd.Series] = {}
            for ticker in tickers:
                try:
                    # Daily OHLCV bars for this ticker (split-adjusted)
                    local_prices[ticker] = (
                        self.data_provider.get_price_history(
                            ticker, start, end))
                    # Per-ticker split history; aligns share counts
                    # with yfinance's auto-adjusted prices.
                    splits[ticker] = self.data_provider.get_split_history(
                        ticker)
                    # Per-ticker dividend deflation factors; lets
                    # _calculate_value undo yfinance's dividend
                    # adjustment on past closes.
                    dividend_factors[ticker] = (
                        self.data_provider.get_dividend_price_factors(ticker))
                except YFRateLimitError:
                    # Transient: surface so the caller can retry
                    raise
                except YFPricesMissingError:
                    # Delisted / no listing on Yahoo; expected for
                    # some IBKR mutual-fund ISINs. Skip prices, splits,
                    # and dividend factors together.
                    continue
                except Exception as e:
                    warnings.warn(
                        f"Unexpected error fetching market data "
                        f"for {ticker}: {type(e).__name__}: {e}")

            # Yfinance returns UK pence (GBp) for LSE-listed
            # GBP equities. Our trades are stored in actual GBP
            # (divided by 100 at parse time), so rescale the
            # market data the same way for those tickers.
            for ticker, series in local_prices.items():
                if (ticker.endswith('.L')
                        and self._ticker_currencies.get(ticker)
                        == 'GBP'):
                    local_prices[ticker] = series / 100.0

            # Attach splits so holdings get rescaled to post-split
            # equivalents before NAV computation.
            self.portfolio.set_splits(splits)
            # Attach dividend factors so _calculate_value can scale
            # adjusted past closes back to the pre-deflation level.
            self.portfolio.set_dividend_factors(dividend_factors)
            # Publish the price dict last so the early-return
            # guard only fires after the build is complete
            self._market_data = local_prices

        # Pre-load FX series for every non-base currency in use; 
        # conversion happens in _native_to_base.
        self._fetch_fx_data()

    def _fetch_fx_data(self):
        """
        Load daily FX history for every non-base currency
        present in the portfolio.

        Each series is keyed by source currency code and gives
        `base_currency` units per one unit of source currency
        (yfinance convention: `USDCHF=X` returns CHF per USD).
        """
        if self._fx_data:
            return

        currencies = set()
        for t in self.transactions:
            if t.currency != self.base_currency:
                currencies.add(t.currency)
        # Benchmark series may reference currencies that
        # do not appear in our transactions;
        start = self.portfolio.get_start_date()
        end = datetime.now()

        # One yfinance call per currency pair, cached on
        # self._fx_data for reuse by _native_to_base.
        for ccy in currencies:
            # Fetch the daily ccy -> base_currency rate series
            self._fx_data[ccy] = self._load_fx_series(ccy, start, end)

    def _load_fx_series(self, currency: str, start, end) -> pd.Series:
        """
        Load an FX pair backed by an on-disk cache.

        Reads ``data/cache/fx/<pair>.pkl`` when it exists and is
        no older than ``fx_cache_max_age_days``. Otherwise
        fetches the series from the data provider, writes it to
        disk, and returns it. When the network fetch fails and
        a stale cached copy exists, the stale copy is used with
        a warning rather than failing the call.

        Args:
            currency: Source currency code.
            start, end: Date bounds passed to the data provider
                on a fresh fetch. The cached series is returned
                in full; downstream callers slice in
                ``_native_to_base``.

        Returns:
            Daily FX series (base per native) for ``currency``.
            Empty Series when no conversion is needed.

        Raises:
            RuntimeError: When the cache is missing and the
                network fetch also fails (no usable source).
            YFRateLimitError: Re-raised so callers can back off.
        """
        # Same-currency conversion: no series needed
        if currency == self.base_currency:
            return pd.Series(dtype=float)
        # Currency pair in yfinance format
        pair = f'{currency}{self.base_currency}=X'
        cache_path = _fx_cache_path(pair)

        # Cache hit: skip the network entirely
        if cache_path.exists():
            age_days = (
                datetime.now() - datetime.fromtimestamp(
                    cache_path.stat().st_mtime)).days
            # If cache is recent and non-empty,
            # return without hitting the network
            if age_days <= self.fx_cache_max_age_days:
                cached = pd.read_pickle(cache_path)
                if not cached.empty:
                    return cached

        # Cache missing or stale: refresh from self.fx_cache_history_start.
        try:
            series = self.data_provider.get_price_history(
                pair, self.fx_cache_history_start, end)
        except YFRateLimitError:
            warnings.warn(f'yfinance rate-limited on FX {pair}.')
            raise
        except Exception as exc:
            # Network failed; fall back to stale cache if any
            if cache_path.exists():
                warnings.warn(
                    f'FX refresh failed for {pair} '
                    f'({type(exc).__name__}); using stale cache.')
                return pd.read_pickle(cache_path)
            raise RuntimeError(
                f'FX history unavailable for '
                f'{currency}->{self.base_currency}: '
                f'{type(exc).__name__}: {exc}') from exc
        
        # Empty result also counts as a failed refresh
        if series.empty:
            if cache_path.exists():
                warnings.warn(
                    f'FX refresh returned empty for {pair}; '
                    f'using stale cache.')
                return pd.read_pickle(cache_path)
            raise RuntimeError(
                f'FX history unavailable for '
                f'{currency}->{self.base_currency} '
                f'(provider returned empty series).')
        
        # Persist for the next call; cache-write failures are
        # non-fatal since the in-memory series is still valid.
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            series.to_pickle(cache_path)
        except Exception as exc:
            warnings.warn(
                f'FX cache write failed for {pair} '
                f'({type(exc).__name__}); proceeding without '
                f'persisting.')
        # Return
        return series

    def _native_to_base(self, amount: float, currency: str, date) -> float:
        """
        Convert `amount` of `currency` to `self.base_currency`
        using the FX rate quoted on or before `date`.

        Returns the amount unchanged when `currency` is
        already the base currency. Raises if the FX series
        for `currency` is empty or has no quote on or before
        `date` — there is no defensible fallback when prices
        are denominated in a currency we cannot translate.
        """
        # Short-circuit if already populated in the cache
        if currency == self.base_currency:
            return amount

        # Retrieve the cached daily ccy -> base rate series
        series = self._fx_data.get(currency)

        # Empty / missing series means FX history was never
        # loaded for this currency
        if series is None or series.empty:
            raise RuntimeError(
                f"No FX history for "
                f"{currency}->{self.base_currency}; cannot "
                f"value position on {date.date()}.")

        # Restrict the series to quotes on or before `date`
        available = series[series.index <= date]

        # No quote exists on or before this date (e.g. date
        # predates the FX series); valuation is undefined
        if available.empty:
            raise RuntimeError(
                f"FX rate {currency}->{self.base_currency} "
                f"not yet quoted on {date.date()}.")
        # Most recent quote at or before `date` is the rate
        rate = available.iloc[-1]

        # Convert: native amount * (base per native)
        return amount * rate

    def _benchmark_currency(self, benchmark: str) -> str:
        """
        Return the listing currency of a benchmark ticker.

        Cached per ticker so we hit yfinance's `fast_info` at
        most once per benchmark per analyzer.
        """
        # Cache hit: return without touching yfinance
        if benchmark in self._benchmark_currencies:
            return self._benchmark_currencies[benchmark]

        # Prefer yfinance metadata when available
        ccy = None
        # Pick up the yfinance module if the provider exposes it;
        # YFinanceProvider does, QFLibProvider does not
        yf_mod = getattr(self.data_provider, 'yf', None)
        if yf_mod is not None:
            try:
                # fast_info is yfinance's lightweight metadata
                # endpoint — includes the listing currency
                info = yf_mod.Ticker(benchmark).fast_info
                # Some yfinance versions expose `currency` as a
                # dict key, others as an attribute; support both
                ccy = (info.get('currency')
                       if hasattr(info, 'get')
                       else getattr(info, 'currency', None))
            except (AttributeError, KeyError, ValueError) as e:
                # Surface the fallback so it isn't invisible
                warnings.warn(
                    f"fast_info failed for {benchmark} ({type(e).__name__});"
                    f"trying static fallback map.")
                ccy = None
        # Fall back to the static map; 
        # raise if even that cannot resolve the benchmark.
        if not ccy:
            ccy = _benchmark_currency_fallback.get(benchmark)
        if not ccy:
            raise RuntimeError(
                f"Cannot determine currency for benchmark "
                f"{benchmark!r}: yfinance fast_info unavailable "
                f"and no entry in `_benchmark_currency_fallback`. "
                f"Add it explicitly.")
        # Normalise to upper-case so comparisons are stable
        ccy = ccy.upper()
        # Remember the result for subsequent calls
        self._benchmark_currencies[benchmark] = ccy

        # Load FX history for this currency:
        # benchmark's quote currency may not be in any of our transactions.
        if ccy != self.base_currency and ccy not in self._fx_data:
            # Fetch as in _fetch_fx_data driven by the benchmark currency
            self._fx_data[ccy] = self._load_fx_series(
                ccy, 
                self.portfolio.get_start_date(),
                datetime.now())
            
        return ccy

    def _calculate_portfolio_values(self
    ) -> Tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
        """Portfolio NAV over time for both actual and hold.

        Actual NAV layers two corrections on top of the
        split-adjusted `Σ shares × adj_close` formula:
          - per-ticker dividend price recovery (inverts
            yfinance's deflation in `_calculate_value`);
          - per-date cumulative dividend - tax cash, in base
            currency, from the transaction log.
        Together they give the actual cash-position wealth.

        Hold NAV keeps the original `Σ shares × adj_close`
        formula — a DRIP-equivalent counterfactual, which is
        what yfinance's `auto_adjust=True` is designed for.
        """

        self._fetch_market_data()
        start = self.portfolio.get_start_date()
        end = datetime.now()
        date_range = pd.date_range(start=start, end=end, freq='D')

        # Per-date cumulative dividend - tax cash, base currency
        cum_cash = self._accumulated_dividend_cash(date_range)

        actual_values = []
        hold_values = []
        # Reset per call; _calculate_value adds tickers that
        # were held but absent from `_market_data`.
        self._missing_price_tickers: set = set()

        # Portfolio.get_holdings_at_date is split-aware and
        # caches per (date, operations). One pass per day.
        for date in date_range:
            actual = self.portfolio.get_holdings_at_date(date)
            hold = self.portfolio.get_holdings_at_date(
                date, operations=('buy',))

            actual_values.append(
                self._calculate_value(
                    actual, date, apply_dividend_correction=True)
                + float(cum_cash.loc[date]))
            hold_values.append(self._calculate_value(hold, date))

        # Surface once: every ticker that was held but never
        # priced is excluded from NAV.
        if self._missing_price_tickers:
            warnings.warn(
                f"NAV excludes positions with no fetched price "
                f"data: {sorted(self._missing_price_tickers)}")

        return (
            pd.Series(actual_values, index=date_range),
            pd.Series(hold_values, index=date_range),
            date_range)

    def _accumulated_dividend_cash(
        self, dates: pd.DatetimeIndex) -> pd.Series:
        """Per-date cumulative base-currency dividend + tax cash.

        Walks the transaction log once, FX-converts each
        dividend / tax `total_cost` at its event date, then
        cumulates onto `dates` (forward-filled) so each NAV
        date sees the running total received up to that point.

        Returns:
            Series indexed by `dates`, monotonic non-decreasing
            modulo tax debits, expressed in `self.base_currency`.
        """
        # Per-date accumulator: date -> base-ccy cash that day
        daily: Dict[pd.Timestamp, float] = {}
        for t in self.portfolio.transactions:
            # Only dividend receipts and withholding tax move
            # the cash component of NAV
            if t.operation not in ('dividend', 'tax'):
                continue
            # Translate native cash flow to base at event date
            base = self._native_to_base(
                t.total_cost, t.currency, t.date)
            # Bucket by calendar day so multiple events on the
            # same date collapse into one daily entry
            d = pd.Timestamp(t.date.date())
            daily[d] = daily.get(d, 0.0) + base

        # Empty-history short-circuit: zero cash on every date
        if not daily:
            return pd.Series(0.0, index=dates)

        # Build chronological cumulative Series, then reindex
        # onto `dates` with forward-fill so each NAV date carries
        # the running total received up to (and including) it
        s = pd.Series(daily).sort_index().cumsum()
        return s.reindex(dates, method='ffill').fillna(0.0)

    def _calculate_value(
        self,
        holdings: Dict[str, float],
        date,
        apply_dividend_correction: bool = False) -> float:
        """
        Portfolio value at `date`, in `self.base_currency`.

        Sums each position's native-currency value (shares
        times most-recent-close-on-or-before-`date`), then
        converts to `base_currency` using the FX rate quoted
        on the same date.

        Args:
            holdings: Ticker -> share count.
            date: Valuation date.
            apply_dividend_correction: When True, multiply each
                ticker's adjusted close by
                `Portfolio._dividend_price_factor(ticker, date)`
                to recover the split-adjusted-but-not-dividend-
                adjusted past price. Default False so callers
                like `_calculate_pe_at_date` keep the
                untouched-price semantics.
        """
        # Running total across all positions (already in base)
        value = 0.0

        for ticker, shares in holdings.items():
            # Skip closed / phantom positions
            if shares <= 0:
                continue

            # No price history at all is a hard error: silently
            # zero-valuing the position would understate NAV and
            # mask data-pipeline issues.
            if ticker not in self._market_data:
                raise RuntimeError(
                    f"Ticker {ticker!r} has no price history "
                    f"in market_data; cannot value position.")

            # Full price series for this ticker
            prices = self._market_data[ticker]
            # Keep only prices on or before the target date,
            # then drop NaN closes (yfinance occasionally returns
            # a partial row for the current day before the local
            # close has settled).
            available = prices[prices.index <= date].dropna()
            # Pre-listing buy is a data inconsistency; fail
            # loud rather than silently 0-value the position.
            if available.empty:
                raise RuntimeError(
                    f"No {ticker} price on or before "
                    f"{date.date()}; transaction predates "
                    f"yfinance's first quote.")
            # Most recent non-NaN close at or before `date`
            price = available.iloc[-1]
            # Optional: invert yfinance's dividend deflation so
            # `price` represents the share's actual market value
            # on `date` rather than the back-adjusted level.
            if apply_dividend_correction:
                price = price * self.portfolio._dividend_price_factor(
                    ticker, date)
            # Position value in the ticker's trade currency
            native_value = shares * price
            # Currency of this ticker (set by the loader).
            # A ticker without a transaction row is an internal violation.
            if ticker not in self._ticker_currencies:
                raise RuntimeError(
                    f"Ticker {ticker!r} has no transaction in the history; "
                    f"cannot determine trade currency.")
            currency = self._ticker_currencies[ticker]
            # Translate native value to base currency
            value += self._native_to_base(native_value, currency, date)

        return value

    def _get_benchmark_series(
        self, benchmark: str, dates: pd.DatetimeIndex,
        is_dividend_reinvested: bool = False) -> pd.Series:
        """
        Benchmark portfolio value over time, in base currency.

        Mirrors the actual portfolio's cash flows: on each
        transaction date the equivalent base-currency cash
        flow is converted to a quantity of `benchmark` at the
        benchmark's base-currency-equivalent price that day.

        Args:
            benchmark: Benchmark ticker (e.g. 'SPY').
            dates: Date index over which to evaluate the
                benchmark series.
            is_dividend_reinvested: If False (default), only
                'buy' and 'sell' transactions trigger benchmark
                cash flows; dividends and withholding tax do
                not move the benchmark. If True, dividends are
                reinvested into the benchmark and withholding
                tax is withdrawn — useful for a net-total-
                return comparison.

        Returns:
            Series of base-currency benchmark value indexed by
            `dates`. Fetch failures propagate as the caller
            cannot proceed without the benchmark price series.
        """
        prices = self.data_provider.get_price_history(
            benchmark, dates[0], dates[-1])
        # Forward-fill weekends/holidays so every date in
        # `dates` has a quoted price.
        prices_aligned = prices.reindex(dates, method='ffill')

        # Currency of the benchmark's quote (e.g. SPY -> USD).
        # Pre-loads FX history for that currency too.
        benchmark_currency = self._benchmark_currency(benchmark)

        # The opt-in branch reinvests dividends and withdraws withholding tax.
        if is_dividend_reinvested:
            cashflow_ops = ('buy', 'sell', 'dividend', 'tax')
        else:
            cashflow_ops = ('buy', 'sell')

        benchmark_shares = 0.0
        benchmark_values = []

        for date in dates:
            # Transactions executed on this calendar date
            date_transactions = []
            for t in self.portfolio.transactions:
                if t.date.date() == date.date():
                    date_transactions.append(t)

            # prices_aligned reindexed to `dates`
            price_native = prices_aligned.loc[date]

            # Benchmark price expressed in base currency at this date;
            # reused for the share-conversion of
            # each cash flow and for the closing valuation.
            if pd.notna(price_native) and price_native > 0:
                price_base = self._native_to_base(
                    price_native, benchmark_currency, date)
            else:
                price_base = float('nan')

            for trans in date_transactions:
                # Skip non-cash-flow operations
                if trans.operation not in cashflow_ops:
                    continue
                # Sign convention for the benchmark mirror:
                # - buy/sell are *external* deposits / withdrawals
                #   from the portfolio's perspective. A buy means
                #   cash flowed in from outside and was invested,
                #   so the benchmark should also receive cash;
                #   `total_cost` is negative for buys, hence we
                #   negate it.
                # - dividend/tax are *internal* cash events. The
                #   real portfolio's dividend cash already shows
                #   up in `total_cost` with the correct sign
                #   (positive for an income, negative for a tax),
                #   so we use it as-is when reinvesting.
                if trans.operation in ('buy', 'sell'):
                    cash_native = -trans.total_cost
                else:
                    cash_native = trans.total_cost
                cash_base = self._native_to_base(
                    cash_native, trans.currency, date)
                # Convert base cash flow to benchmark shares
                if (cash_base != 0 and pd.notna(price_base)
                        and price_base > 0):
                    benchmark_shares += cash_base / price_base

            # Sanity check: negative share count means inconsistent
            if benchmark_shares < 0:
                raise RuntimeError(
                    f"Benchmark share count went negative on "
                    f"{date.date()}: shares={benchmark_shares}. "
                    f"Cash flows exceed accumulated benchmark "
                    f"holding; check transaction data.")

            # Closing valuation for this date, in base currency
            if pd.notna(price_base):
                value = benchmark_shares * price_base
            elif benchmark_values:
                # No price quote available today; carry forward
                value = benchmark_values[-1]
            else:
                value = 0.0

            benchmark_values.append(value)

        return pd.Series(benchmark_values, index=dates)

    def _calculate_pe_over_time(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Calculate P/E ratio over time."""

        # Accumulator for one P/E value per sampled date
        pe_ratios = []

        # Downsample to every 7th date (weekly) — fundamentals
        # barely move day-to-day, so daily P/E is wasted work
        for date in dates[::7]:
            # Position snapshot as of this date
            holdings = self.portfolio.get_holdings_at_date(date)
            # Portfolio-weighted P/E at that snapshot
            pe = self.ratios.get_portfolio_pe_ratio(
                holdings, self._market_data, date=date)
            pe_ratios.append(pe)

        # Return as a Series indexed by the same weekly dates
        return pd.Series(pe_ratios, index=dates[::7])

    def _get_per_ticker_dividend_history(self) -> Dict[str, pd.Series]:
        """
        Dividend history keyed by ticker, ex-date indexed.

        Iterates over every ticker that has ever
        appeared in the transaction history, not only currently
        held positions: closed positions can still have paid
        dividends during the period.
        """
        # Output map: ticker -> dividend Series (ex-date index)
        dividend_data = {}
        # Earliest transaction date bounds the lookup window
        start = self.portfolio.get_start_date()

        for ticker in self.portfolio.get_all_tickers():
            try:
                divs = (self.data_provider.get_dividend_history(ticker, start))
            except YFRateLimitError:
                raise
            except YFPricesMissingError:
                continue
            except Exception as e:
                warnings.warn(
                    f"Unexpected error fetching dividend "
                    f"history for {ticker}: "
                    f"{type(e).__name__}: {e}")
                continue
            # Skip tickers with no recorded dividends
            if not divs.empty:
                dividend_data[ticker] = divs

        return dividend_data

    def _get_portfolio_dividends(self, gross: bool = False) -> pd.Series:
        """
        Dividend cash flow per event, in `base_currency`.

        Walks the transaction history and translates each
        relevant row to `self.base_currency` at its event-date
        FX rate. Multiple events on the same day are summed.

        Args:
            gross: When False (default), sum both 'dividend'
                receipts and 'tax' withholding rows so a
                dividend and its tax collapse into one net
                entry. When True, sum only 'dividend' rows,
                yielding gross (pre-withholding) dividend
                income; the result is non-negative.

        Returns:
            Series indexed by event date (ascending). Empty
            when the portfolio has no qualifying rows.
        """
        # Select dividend-only vs dividend+tax depending on mode
        if gross:
            ops = ('dividend',)
        else:
            ops = ('dividend', 'tax')
        # Per-day accumulator: date -> dividend cash (base ccy)
        daily: Dict[datetime, float] = {}
        for t in self.transactions:
            # Skip operations outside the selected set
            if t.operation not in ops:
                continue
            # Translate the native-currency cash flow to base
            # at the event-date FX rate
            v = self._native_to_base(t.total_cost, t.currency, t.date)
            # Aggregate same-day events (dividend + its tax)
            daily[t.date] = daily.get(t.date, 0.0) + v

        # Empty-history short-circuit
        if not daily:
            return pd.Series(dtype=float)

        # Chronological Series, indexed by event date
        return pd.Series(daily).sort_index()

    def _get_portfolio_dividend_tax(self) -> pd.Series:
        """
        Withholding-tax cash flow per event, in `base_currency`.

        Walks the transaction history and translates each
        'tax' row to `self.base_currency` at the event-date
        FX rate. Multiple events on the same day are summed.

        Returns:
            Series indexed by event date (ascending), with
            non-positive values (tax `total_cost` is negative).
            Empty when the portfolio has no tax rows.
        """
        # Per-day accumulator: date -> total tax (base ccy)
        daily: Dict[datetime, float] = {}
        for t in self.transactions:
            # Only withholding-tax rows contribute
            if t.operation != 'tax':
                continue
            # Translate the native-currency cash flow to base
            # at the event-date FX rate
            v = self._native_to_base(t.total_cost, t.currency, t.date)
            # Aggregate same-day events
            daily[t.date] = daily.get(t.date, 0.0) + v

        # Empty-history short-circuit
        if not daily:
            return pd.Series(dtype=float)

        # Chronological Series, indexed by event date
        return pd.Series(daily).sort_index()

    def _get_portfolio_fees(self) -> pd.Series:
        """
        Total transaction fees per event, in `base_currency`.

        Walks the transaction history and translates each
        row's `fee + auto_fx_fee` to `self.base_currency` at
        the event-date FX rate. Multiple events on the same
        day are summed. Zero-fee rows are skipped so the
        result stays sparse.

        Returns:
            Series indexed by event date (ascending), with
            non-negative values (fees are stored as
            non-negative magnitudes). Empty when no
            transaction has a non-zero fee.
        """
        # Per-day accumulator: date -> total fees (base ccy)
        daily: Dict[datetime, float] = {}
        for t in self.transactions:
            # Sum of trading and FX fees, both non-negative
            # magnitudes validated at Transaction construction
            native_fee = t.fee + t.auto_fx_fee
            # Skip zero-fee rows so sparse currencies stay absent
            if native_fee == 0.0:
                continue
            # Translate the native-currency fee to base at the
            # event-date FX rate
            v = self._native_to_base(native_fee, t.currency, t.date)
            # Aggregate same-day events
            daily[t.date] = daily.get(t.date, 0.0) + v

        # Empty-history short-circuit
        if not daily:
            return pd.Series(dtype=float)

        # Chronological Series, indexed by event date
        return pd.Series(daily).sort_index()
