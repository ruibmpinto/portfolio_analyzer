"""
Financial ratios analysis module.

Provides fundamental ratio calculations including
P/E, P/B, dividend yield, and other valuation metrics.
"""

import warnings
import pandas as pd
from typing import Dict, Optional
from src.shared.data_provider import DataProvider


class RatioAnalyzer:
    """
    Analyzes financial ratios and valuation metrics.

    Calculates P/E, P/B, dividend yields, and other
    fundamental ratios for individual stocks and portfolios.
    """

    def __init__(self, data_provider: DataProvider):
        """
        Initialize ratio analyzer.

        Args:
            data_provider: DataProvider for fundamental data
        """
        self.data_provider = data_provider
        self._fundamental_cache = {}

    def get_pe_ratio(self, ticker: str) -> Optional[float]:
        """
        Get P/E ratio for a stock.

        Trailing P/E uses reported EPS for TTM, "trailing twelve months".

        Args:
            ticker: Stock ticker symbol

        Returns:
            Trailing P/E ratio or None
        """
        fundamentals = self._get_fundamentals(ticker)
        return fundamentals.get('trailing_pe')

    def get_forward_pe(self, ticker: str) -> Optional[float]:
        """
        Get forward P/E ratio for a stock.

        Forward P/E uses forecasted EPS for NTM, "next twelve months".

        Args:
            ticker: Stock ticker symbol

        Returns:
            Forward P/E ratio or None
        """
        fundamentals = self._get_fundamentals(ticker)
        return fundamentals.get('forward_pe')

    def get_pb_ratio(self, ticker: str) -> Optional[float]:
        """
        Get P/B ratio for a stock.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Price-to-book ratio or None
        """
        fundamentals = self._get_fundamentals(ticker)
        return fundamentals.get('price_to_book')

    def get_dividend_yield(self, ticker: str) -> Optional[float]:
        """
        Get dividend yield for a stock.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dividend yield (as decimal, e.g., 0.025 for 2.5%)
        """
        fundamentals = self._get_fundamentals(ticker)
        return fundamentals.get('dividend_yield')

    def get_portfolio_pe_ratio(
        self,
        holdings: Dict[str, float],
        prices: Dict[str, pd.Series],
        date=None,
    ) -> Optional[float]:
        """
        Calculate portfolio-weighted P/E ratio.

        P/E = Total Market Value / Total Earnings
        where Earnings = Shares * EPS

        Args:
            holdings: Dictionary {ticker: shares}
            prices: Dictionary {ticker: price_series}
            date: Valuation date. None (default) uses each
                ticker's latest close. When supplied, uses the
                most recent close on or before `date` (the
                same convention as `_calculate_value`).

        Returns:
            Portfolio P/E ratio or None

        Example:
            >>> pe = analyzer.get_portfolio_pe_ratio(
            ...     {'AAPL': 100, 'MSFT': 50},
            ...     prices_dict
            ... )
        """
        # Running aggregates over all holdings
        total_market_value = 0.0
        total_earnings = 0.0

        for ticker, shares in holdings.items():
            if shares <= 0:
                continue

            if ticker not in prices:
                continue

            try:
                # Price at `date` if requested, else most recent
                series = prices[ticker]
                if date is None:
                    current_price = series.iloc[-1]
                else:
                    available = series[series.index <= date]
                    if len(available) == 0:
                        warnings.warn(
                            f"P/E skipped {ticker} on "
                            f"{date.date()}: no price quoted "
                            f"at or before this date.")
                        continue
                    current_price = available.iloc[-1]
                # Position market value at that price
                market_value = shares * current_price

                # Trailing earnings-per-share from fundamentals
                fundamentals = self._get_fundamentals(ticker)
                eps = fundamentals.get('trailing_eps')

                # Skip negative-earning names so we don't pollute
                # the aggregate with denominators of the wrong sign
                if eps and eps > 0:
                    # Position-level earnings = shares * EPS
                    earnings = shares * eps
                    # Total market value and earnings for the portfolio
                    total_market_value += market_value
                    total_earnings += earnings

            except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
                warnings.warn(
                    f"P/E agg skipped {ticker}: {type(e).__name__}: {e}")
                continue

        # Portfolio P/E = sum(market value) / sum(earnings)
        if total_earnings > 0:
            return total_market_value / total_earnings

        return None

    def get_portfolio_pb_ratio(
        self,
        holdings: Dict[str, float],
        prices: Dict[str, pd.Series]
    ) -> Optional[float]:
        """
        Calculate portfolio-weighted P/B ratio.

        Args:
            holdings: Dictionary {ticker: shares}
            prices: Dictionary {ticker: price_series}

        Returns:
            Portfolio P/B ratio or None
        """
        # Running aggregates over all holdings
        total_market_value = 0.0
        total_book_value = 0.0

        for ticker, shares in holdings.items():
            if shares <= 0 or ticker not in prices:
                continue

            try:
                # Most recent close = current price
                current_price = prices[ticker].iloc[-1]
                # Position market value at that price
                market_value = shares * current_price

                # Book value per share from fundamentals
                fundamentals = self._get_fundamentals(ticker)
                book_value_per_share = (fundamentals.get('book_value'))

                # Skip negative-book names (impaired equity)
                if book_value_per_share and (book_value_per_share > 0):
                    # Position-level book value = shares * BVPS
                    book_value = shares * book_value_per_share
                    # # Total market value and book value for the portfolio
                    total_market_value += market_value
                    total_book_value += book_value

            except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
                warnings.warn(
                    f"P/B agg skipped {ticker}: {type(e).__name__}: {e}")
                continue

        # Portfolio P/B = sum(market value) / sum(book value)
        if total_book_value > 0:
            return total_market_value / total_book_value

        return None

    def get_portfolio_dividend_yield(
        self,
        holdings: Dict[str, float],
        prices: Dict[str, pd.Series]
    ) -> Optional[float]:
        """
        Calculate portfolio-weighted dividend yield.

        Args:
            holdings: Dictionary {ticker: shares}
            prices: Dictionary {ticker: price_series}

        Returns:
            Portfolio dividend yield (decimal)
        """
        # Running aggregates over all holdings
        total_market_value = 0.0
        total_dividends = 0.0

        for ticker, shares in holdings.items():
            if shares <= 0 or ticker not in prices:
                continue

            try:
                # Most recent close = current price
                current_price = prices[ticker].iloc[-1]
                # Position market value at that price
                market_value = shares * current_price

                # Annual dividend per share (in trade currency)
                fundamentals = self._get_fundamentals(ticker)
                dividend_rate = (fundamentals.get('dividend_rate'))

                # Only count names that actually pay dividends
                if dividend_rate and dividend_rate > 0:
                    # Forward 12-month payout for this position
                    annual_dividends = shares * dividend_rate
                    # Total market value and dividends for the portfolio
                    total_market_value += market_value
                    total_dividends += annual_dividends

            except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
                warnings.warn(
                    f"Dividend yield agg skipped {ticker}: "
                    f"{type(e).__name__}: {e}")
                continue

        # Yield = sum(annual dividends) / sum(market value)
        if total_market_value > 0:
            return total_dividends / total_market_value

        return None

    def get_payout_ratio(
        self, ticker: str
    ) -> Optional[float]:
        """
        Get dividend payout ratio for a stock.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Payout ratio (decimal) or None
        """
        fundamentals = self._get_fundamentals(ticker)
        return fundamentals.get('payout_ratio')

    def get_market_cap(self, ticker: str) -> Optional[float]:
        """
        Get market capitalization for a stock.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Market cap in dollars or None
        """
        fundamentals = self._get_fundamentals(ticker)
        return fundamentals.get('market_cap')

    def get_book_value(self, ticker: str) -> Optional[float]:
        """
        Get book value per share for a stock.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Book value per share or None
        """
        fundamentals = self._get_fundamentals(ticker)
        return fundamentals.get('book_value')

    def get_eps(self, ticker: str) -> Optional[float]:
        """
        Get trailing EPS for a stock.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Trailing EPS or None
        """
        fundamentals = self._get_fundamentals(ticker)
        return fundamentals.get('trailing_eps')

    def _get_fundamentals(
        self, ticker: str
    ) -> Dict[str, Optional[float]]:
        """
        Get fundamental data with caching.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary of fundamental metrics
        """
        # Hit the data provider once per ticker per analyzer
        if ticker not in self._fundamental_cache:
            self._fundamental_cache[ticker] = (
                self.data_provider.get_fundamental_data(ticker))
        # Return
        return self._fundamental_cache[ticker]

    def clear_cache(self):
        """Clear fundamental data cache."""
        self._fundamental_cache.clear()
