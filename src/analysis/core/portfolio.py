"""
Portfolio state management.

Tracks portfolio holdings over time based on transactions.
"""

from datetime import datetime
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from src.analysis.core.transaction import Transaction


class Portfolio:
    """
    Manages portfolio state and holdings tracking.

    Calculates holdings at any point in time based on
    transaction history (buys, sells, dividends).

    Attributes:
        transactions: List of Transaction objects
        holdings: Current holdings {ticker: shares}
    """

    def __init__(
        self,
        transactions: List[Transaction],
        splits: Optional[Dict[str, pd.Series]] = None,
        dividend_factors: Optional[Dict[str, pd.Series]] = None):
        """
        Initialize portfolio with transaction history.

        Args:
            transactions: List of Transaction objects,
                will be sorted by date.
            splits: Optional ticker -> split Series (ex-date
                indexed, ratio-valued, e.g. 10.0 for 10-for-1).
                When supplied, share counts from `get_holdings_at_date`
                are rescaled so they match yfinance's
                `auto_adjust=True` price series.
            dividend_factors: Optional ticker -> per-event
                deflation factor Series (ex-date indexed,
                each value in (0, 1]). When supplied,
                `_dividend_price_factor` inverts the cumulative
                deflation so callers can recover the split-
                adjusted-but-not-dividend-adjusted past price
                from yfinance's `auto_adjust=True` series.
        """
        # Sort transactions by date
        self.transactions = sorted(transactions, key=lambda t: t.date)

        # Default empty; real split data flows through set_splits
        # so the synthetic-split-buy filter runs in one place.
        self.splits: Dict[str, pd.Series] = {}

        # Per-ticker dividend deflation factors; default empty
        # so the no-dividends case stays a no-op.
        self.dividend_factors: Dict[str, pd.Series] = (dividend_factors or {})
        self._holdings_cache = {}

        # Route any constructor-supplied splits through set_splits
        if splits:
            self.set_splits(splits)

    def set_splits(self, splits: Dict[str, pd.Series]) -> None:
        """Attach split data and invalidate the holdings cache.

        Also strips synthetic broker-recorded split-adjustment
        rows: Degiro logs a stock split as a zero-fee "buy" of the 
        post-split additional shares. 
        Applying our own split factor on top of such a
        row would double-count the corporate action, so we drop
        them as soon as we learn which dates are split days.
        """
        # Replace the table; default empty stays a no-op
        self.splits = splits or {}
        self._holdings_cache.clear()
        # No splits known -> nothing to filter; keep all rows
        if not self.splits:
            return
        # Walk transactions, keeping non-synthetic rows
        filtered = []
        for trans in self.transactions:
            # Only zero-fee buys / sells are candidates for the
            # filter. Some brokers (e.g. Degiro) record a 2-for-1
            # split as a paired zero-fee sell-then-buy on the
            # split day, so the sell half must be droppable too.
            if trans.operation not in ('buy', 'sell'):
                filtered.append(trans)
                continue
            # Let through any trade that paid a real fee
            if trans.fee != 0.0 or trans.auto_fx_fee != 0.0:
                filtered.append(trans)
                continue
            # Need split history for this ticker to confirm
            ticker_splits = self.splits.get(trans.ticker)
            # Let through all tickers with no known splits
            if ticker_splits is None or ticker_splits.empty:
                filtered.append(trans)
                continue
            # Compare trade date against every split for the ticker
            trade_ts = pd.Timestamp(trans.date).normalize()
            # is_synthetic flags synthetic split buys
            is_synthetic = False
            for split_date in ticker_splits.index:
                split_ts = pd.Timestamp(split_date).normalize()
                # Allow +/- 2 day slack for weekend / TZ edge cases
                if abs((split_ts - trade_ts).days) <= 2:
                    is_synthetic = True
                    break
            # Let through all transactions not flagged as synthetic split buys, 
            # keep the row unless it matched a known split date
            if not is_synthetic:
                filtered.append(trans)
        self.transactions = filtered

    def set_dividend_factors(
        self, dividend_factors: Dict[str, pd.Series]) -> None:
        """Attach dividend factors and invalidate holdings cache."""

        self.dividend_factors = dividend_factors or {}
        self._holdings_cache.clear()

    def _split_factor(self, ticker: str, trade_date) -> float:
        """Cumulative split ratio applied to a trade made on
        `trade_date` of `ticker`, so post-split-equivalent
        share counts line up with yfinance's adjusted prices.

        yfinance's `auto_adjust=True` divides historical prices
        by every split that has occurred since.
        We mirror that on the share-count side: a pre-split-N buy counts as N
        post-split-equivalent shares.
        """
        # Per-ticker split table; missing means no known splits
        series = self.splits.get(ticker)
        if series is None or series.empty:
            return 1.0

        # Keep only splits strictly AFTER the trade date so an
        # ex-date trade isn't double-counted
        after = series[series.index > pd.Timestamp(trade_date)]

        # Cumulative ratio = product of every subsequent split
        if after.empty:
            return 1.0
        return float(after.prod())

    def _dividend_price_factor(
        self, ticker: str, valuation_date) -> float:
        """Inverse of yfinance's cumulative dividend deflation
        between `valuation_date` and today, for `ticker`.

        yfinance's `auto_adjust=True` multiplies historical
        prices by `prod_(D_i ∈ (t, today]) (1 - D_i/C_prev)`
        — each per-event factor sits in (0, 1] and pulls past
        prices DOWN. We invert the product so callers can scale
        `adj_close(t)` back UP to the split-adjusted-but-not-
        dividend-adjusted level (i.e. the share's market price
        on `t` in its split-adjusted unit).
        """
        # Per-ticker factor table; missing means no known dividends
        series = self.dividend_factors.get(ticker)
        if series is None or series.empty:
            return 1.0

        # Keep only factors strictly AFTER the valuation date
        # so an ex-date valuation isn't double-counted
        after = series[series.index > pd.Timestamp(valuation_date)]

        # Cumulative deflation = product of every subsequent factor;
        # invert to recover the pre-deflation price level
        if after.empty:
            return 1.0
        return 1.0 / float(after.prod())

    def get_holdings_at_date(
        self,
        date: datetime,
        operations: Tuple[str, ...] = ('buy', 'sell')) -> Dict[str, float]:
        """
        Calculate portfolio holdings at a specific date.

        Processes transactions up to (and including) `date`,
        keeping only those whose operation is in `operations`.
        Default ('buy', 'sell') gives the actual portfolio;
        ('buy',) gives the "hold" counterfactual (never sells).

        Share counts are split-adjusted to the post-split equivalent 
        so they pair with yfinance's `auto_adjust=True` price series.

        Args:
            date: Valuation date.
            operations: Transaction operations to include.

        Returns:
            Ticker -> share count (excludes zero holdings).

        Raises:
            RuntimeError: If any ticker resolves to a
                negative share count.
        """
        cache_key = (date.strftime('%Y-%m-%d') + '|' + ','.join(operations))
        # Short-circuit if already populated in the cache
        if cache_key in self._holdings_cache:
            return self._holdings_cache[cache_key]

        holdings: Dict[str, float] = {}

        for trans in self.transactions:
            if trans.date > date:
                break
            if trans.operation not in operations:
                continue
            ticker = trans.ticker
            # Rescale to post-split-equivalent shares
            delta = trans.amount * self._split_factor(ticker, trans.date)
            if trans.operation == 'buy':
                holdings[ticker] = holdings.get(ticker, 0) + delta
            elif trans.operation == 'sell':
                holdings[ticker] = holdings.get(ticker, 0) - delta
            # Dividends and tax don't affect share count

        # Drop fully-closed positions; reject impossible negatives 
        # with tolerance epsilon=1E-5
        filtered_holdings = {}
        epsilon = 1e-5
        for ticker, shares in holdings.items():
            if shares < -epsilon:
                raise RuntimeError(
                    f'Negative share count for {ticker} on '
                    f'{cache_key}: {shares}. Transaction data is '
                    f'inconsistent (missing buy, mis-ordered '
                    f'rows, or unmodeled corporate action).')
            if shares > epsilon:
                filtered_holdings[ticker] = shares

        self._holdings_cache[cache_key] = filtered_holdings
        # Return
        return filtered_holdings

    def get_current_holdings(self) -> Dict[str, float]:
        """
        Calculate current portfolio holdings.

        Processes all transactions to determine current state.

        Returns:
            Dictionary mapping ticker to current share count

        Example:
            >>> current = portfolio.get_current_holdings()
            >>> print(current)
            {'AAPL': 150, 'MSFT': 50, 'GOOGL': 25}
        """
        if not self.transactions:
            return {}

        # Use latest transaction date + 1 day
        latest_date = self.transactions[-1].date
        # Add a day to ensure we capture all transactions
        calc_date = latest_date + timedelta(days=1)

        return self.get_holdings_at_date(calc_date)

    def get_all_tickers(self) -> List[str]:
        """
        Get list of all tickers in transaction history.

        Returns:
            Sorted list of unique ticker symbols

        Example:
            >>> tickers = portfolio.get_all_tickers()
            >>> print(tickers)
            ['AAPL', 'GOOGL', 'MSFT']
        """
        tickers = set(t.ticker for t in self.transactions)
        return sorted(list(tickers))

    def get_start_date(self) -> datetime:
        """
        Get date of first transaction.

        Returns:
            Date of earliest transaction

        Raises:
            ValueError: If no transactions exist
        """
        if not self.transactions:
            raise ValueError("No transactions in portfolio")
        return self.transactions[0].date

    def get_end_date(self) -> datetime:
        """
        Get date of last transaction.

        Returns:
            Date of most recent transaction

        Raises:
            ValueError: If no transactions exist
        """
        if not self.transactions:
            raise ValueError("No transactions in portfolio")
        return self.transactions[-1].date

    def _sum_by_currency(
        self, operation: str, signed: bool = False) -> Dict[str, float]:
        """Per-currency sum of `total_cost` for one operation.

        `signed=False` returns absolute values (used for buy
        gross-invested totals); `signed=True` keeps the
        natural sign (sell proceeds are positive, tax is
        negative, dividend is positive).
        """
        # Accumulator: currency -> running total
        totals: Dict[str, float] = {}
        for t in self.transactions:
            # Skip transactions of the wrong operation type
            if t.operation != operation:
                continue
            # Keep the sign when requested; otherwise magnitude only
            if signed:
                v = t.total_cost
            else:
                v = abs(t.total_cost)
            # Add to the bucket for this trade's currency
            totals[t.currency] = totals.get(t.currency, 0.0) + v
        # Return
        return totals

    def get_total_invested(self) -> Dict[str, float]:
        """Gross invested per currency (sum of buy total_cost)."""

        return self._sum_by_currency('buy')

    def get_total_proceeds(self) -> Dict[str, float]:
        """Sell proceeds per currency (net of fees)."""

        return self._sum_by_currency('sell', signed=True)

    def get_net_invested(self) -> Dict[str, float]:
        """Invested minus proceeds, per currency."""

        invested = self.get_total_invested()
        proceeds = self.get_total_proceeds()

        # Accumulator: currency -> net invested
        net: Dict[str, float] = {}
        # Union of currencies seen in either bucket
        for ccy in set(invested) | set(proceeds):
            # New currency on either side starts at 0
            net[ccy] = invested.get(ccy, 0.0) - proceeds.get(ccy, 0.0)

        return net

    def get_total_dividends(self) -> Dict[str, float]:
        """Gross dividend cash per currency."""

        return self._sum_by_currency('dividend', signed=True)

    def get_total_tax(self) -> Dict[str, float]:
        """Withholding tax per currency (non-positive)."""

        return self._sum_by_currency('tax', signed=True)

    def get_net_dividends(self) -> Dict[str, float]:
        """Dividends + tax per currency (tax already negative)."""

        dividends = self.get_total_dividends()
        tax = self.get_total_tax()

        # Accumulator: currency -> net dividend
        net: Dict[str, float] = {}
        # Union of currencies seen in either bucket
        for ccy in set(dividends) | set(tax):
            # New currency on either side starts at 0
            net[ccy] = dividends.get(ccy, 0.0) + tax.get(ccy, 0.0)

        return net

    def _sum_fee_by_currency(
        self, attr: str) -> Dict[str, float]:
        """Per-currency sum of one Transaction fee attribute.

        `attr` must name a non-negative fee field on Transaction
        (e.g. 'fee' or 'auto_fx_fee'). Every transaction
        contributes; zero-valued rows are skipped to keep the
        result sparse.
        """
        # Accumulator: currency -> running fee total
        totals: Dict[str, float] = {}
        for t in self.transactions:
            v = getattr(t, attr)
            # Skip zero-fee rows so untouched currencies stay
            # absent from the result
            if v == 0.0:
                continue
            # Add to the bucket for this trade's currency
            totals[t.currency] = totals.get(t.currency, 0.0) + v
        # Return
        return totals

    def get_total_trading_fees(self) -> Dict[str, float]:
        """Trading-fee total per currency (non-negative)."""

        return self._sum_fee_by_currency('fee')

    def get_total_fx_fees(self) -> Dict[str, float]:
        """Auto-FX-fee total per currency (non-negative)."""

        return self._sum_fee_by_currency('auto_fx_fee')

    def get_total_fees(self) -> Dict[str, float]:
        """Trading + FX fees per currency (non-negative)."""

        trading = self.get_total_trading_fees()
        fx = self.get_total_fx_fees()

        # Accumulator: currency -> total fees
        total: Dict[str, float] = {}
        # Union of currencies seen in either bucket
        for ccy in set(trading) | set(fx):
            # New currency on either side starts at 0
            total[ccy] = trading.get(ccy, 0.0) + fx.get(ccy, 0.0)

        return total

    def get_transaction_count(self) -> int:
        """Get total number of transactions."""

        return len(self.transactions)

    def get_transaction_count_by_type(self) -> Dict[str, int]:
        """
        Count transactions by type.

        Returns:
            Dictionary with counts for each operation type
        """

        counts = {'buy': 0, 'sell': 0, 'dividend': 0, 'tax': 0}
        for trans in self.transactions:
            if trans.operation in counts:
                counts[trans.operation] += 1
        return counts

    def clear_cache(self):
        """Clear cached holdings calculations."""
        self._holdings_cache.clear()

    def __str__(self) -> str:
        """String representation of portfolio."""
        holdings = self.get_current_holdings()
        return (
            f"Portfolio({len(holdings)} positions, "
            f"{self.get_transaction_count()} transactions)")

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"Portfolio(transactions={len(self.transactions)}, "
            f"holdings={self.get_current_holdings()})")
