"""
Volatility analysis module.

Provides volatility and correlation calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict


class VolatilityAnalyzer:
    """
    Analyzes return volatility and correlations.

    Calculates historical volatility, rolling volatility,
    and correlation matrices.
    """

    def compute_rolling_volatility(
        self,
        returns: pd.Series,
        window: int = 30
    ) -> pd.Series:
        """
        Calculate rolling annualized volatility.

        Uses rolling standard deviation of returns,
        annualized for daily data (sqrt(252)).

        Args:
            returns: Daily return series
            window: Rolling window size (default 30 days)

        Returns:
            Rolling volatility series (annualized %)

        Example:
            >>> analyzer = VolatilityAnalyzer()
            >>> rolling_vol = (
            ...     analyzer.compute_rolling_volatility(
            ...         returns, window=30
            ...     )
            ... )
        """
        # Std dev of returns over a sliding `window`-day window
        rolling_std = returns.rolling(window=window).std()

        # Scale daily-stdev to annual: sqrt(252 trading days);
        # multiply by 100 to express as a percentage
        rolling_vol = rolling_std * np.sqrt(252) * 100
        # Return
        return rolling_vol

    def compute_annualized_volatility(
        self, returns: pd.Series
    ) -> float:
        """
        Calculate annualized volatility.

        Args:
            returns: Daily return series

        Returns:
            Annualized volatility (%)

        Example:
            >>> vol = (
            ...     analyzer.compute_annualized_volatility(
            ...         returns
            ...     )
            ... )
            >>> print(f"Volatility: {vol:.2f}%")
        """
        if len(returns) == 0:
            return 0.0

        # Daily return standard deviation
        std = returns.std()

        # Scale to annual (sqrt-of-time rule, 252 trading days)
        # and convert to percent
        annual_vol = std * np.sqrt(252) * 100
        # Return
        return float(annual_vol)

    def compute_correlation_matrix(
        self, returns_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple assets.

        Args:
            returns_dict: Dictionary {ticker: return_series}

        Returns:
            Correlation matrix DataFrame

        Example:
            >>> returns = {
            ...     'AAPL': aapl_returns,
            ...     'MSFT': msft_returns,
            ...     'GOOGL': googl_returns
            ... }
            >>> corr = (
            ...     analyzer.compute_correlation_matrix(
            ...         returns
            ...     )
            ... )
        """
        # Stack each ticker's return Series as a column
        returns_df = pd.DataFrame(returns_dict)

        # Pairwise Pearson correlation across columns
        corr_matrix = returns_df.corr()
        # Return
        return corr_matrix

    def compute_covariance_matrix(
        self, returns_dict: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Calculate covariance matrix for multiple assets.

        Args:
            returns_dict: Dictionary {ticker: return_series}

        Returns:
            Covariance matrix DataFrame
        """
        # Stack each ticker's return Series as a column
        returns_df = pd.DataFrame(returns_dict)

        # Pairwise covariance across columns
        cov_matrix = returns_df.cov()
        # Return
        return cov_matrix

    def compute_downside_deviation(
        self,
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """
        Calculate downside deviation.

        Only considers returns below threshold.

        Args:
            returns: Return series
            threshold: Return threshold (default 0)

        Returns:
            Annualized downside deviation (%)
        """
        # Keep only the loss days (below threshold)
        downside_returns = returns[returns < threshold]

        if len(downside_returns) == 0:
            return 0.0

        # Std of just the down days
        downside_std = downside_returns.std()
        # Annualise and convert to percent
        annual_downside = downside_std * np.sqrt(252) * 100
        # Return
        return float(annual_downside)

    def compute_value_at_risk(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Historical VaR based on return distribution.

        Args:
            returns: Return series
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR as percentage loss

        Example:
            >>> var_95 = analyzer.compute_value_at_risk(
            ...     returns, confidence=0.95
            ... )
            >>> print(f"95% VaR: {var_95:.2f}%")
        """
        if len(returns) == 0:
            return 0.0

        # Tail probability (e.g. 0.05 for a 95% VaR)
        alpha = 1 - confidence
        # Empirical alpha-quantile of the return distribution
        var = np.percentile(returns, alpha * 100)

        # Flip sign so the answer reads as a positive loss in %
        return float(-var * 100)

    def compute_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum peak-to-trough drawdown.

        Walks the cumulative-return curve, tracks the running
        peak, and returns the worst observed drop from any peak
        to a subsequent trough.

        Args:
            returns: Daily return series

        Returns:
            Maximum drawdown as a percent loss (negative,
            e.g. ``-42.1`` for a 42.1% peak-to-trough decline).
            Zero on an empty series or a monotonically
            non-decreasing curve.

        Example:
            >>> mdd = analyzer.compute_max_drawdown(returns)
            >>> print(f"Max drawdown: {mdd:.2f}%")
        """
        if len(returns) == 0:
            return 0.0

        # Cumulative wealth curve (peaks at every new high)
        cum = (1 + returns).cumprod()
        # Running maximum of the wealth curve up to each date
        running_max = cum.cummax()
        # Per-date drop from the running peak (negative or zero)
        drawdown = (cum - running_max) / running_max
        # Worst drop, converted to percent
        return float(drawdown.min() * 100)

    def compute_conditional_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional VaR (CVaR/Expected Shortfall).

        Average loss beyond VaR threshold.

        Args:
            returns: Return series
            confidence: Confidence level

        Returns:
            CVaR as percentage loss
        """
        if len(returns) == 0:
            return 0.0

        # Same alpha-quantile as for VaR
        alpha = 1 - confidence
        var_threshold = np.percentile(returns, alpha * 100)

        # Slice off the worst-case tail (returns <= VaR)
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return 0.0

        # Mean loss inside the tail = expected shortfall
        cvar = tail_returns.mean()

        # Flip sign so the answer reads as a positive loss in %
        return float(-cvar * 100)
