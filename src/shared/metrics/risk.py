"""
Risk profile analysis module.

Provides risk-adjusted performance metrics including
beta, alpha, Sharpe ratio, and Sortino ratio.
"""

import warnings
import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.shared.data_provider import DataProvider


class RiskProfileAnalyzer:
    """
    Analyzes portfolio risk metrics.

    Calculates beta, alpha, Sharpe ratio, and other
    risk-adjusted performance measures.
    """

    def __init__(self, data_provider: DataProvider):
        """
        Initialize risk analyzer.

        Args:
            data_provider: DataProvider for market data
        """
        self.data_provider = data_provider

    def get_beta(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series,
        min_observations: int = 30
    ) -> Optional[float]:
        """
        Calculate beta coefficient.

        Beta measures systematic risk relative to market.
        Beta = Cov(asset, market) / Var(market)

        Args:
            asset_returns: Asset return series
            market_returns: Market return series
            min_observations: Minimum data points required

        Returns:
            Beta coefficient (None if insufficient data)

        Example:
            >>> beta = analyzer.get_beta(
            ...     portfolio_returns, spy_returns
            ... )
            >>> print(f"Portfolio beta: {beta:.2f}")
        """
        try:
            # Inner-join the two series on date so each row is
            # (asset, market) on the same trading day
            aligned_data = pd.DataFrame({
                'asset': asset_returns,
                'market': market_returns
            }).dropna()

            # Refuse to estimate beta on too-few observations
            if len(aligned_data) < min_observations:
                warnings.warn(
                    f"get_beta: only {len(aligned_data)} "
                    f"observations (need {min_observations}); "
                    f"returning None.")
                return None

            # Numerator of beta: Cov(asset, market)
            covariance = aligned_data['asset'].cov(aligned_data['market'])
            # Denominator: Var(market)
            market_variance = aligned_data['market'].var()

            # Guard against degenerate flat-market data
            if market_variance == 0:
                warnings.warn(
                    "get_beta: market variance is 0 "
                    "(flat-market data); returning None.")
                return None

            # Beta = Cov(asset, market) / Var(market)
            beta = covariance / market_variance
            # Return
            return float(beta)

        except (ValueError, TypeError, ZeroDivisionError) as e:
            warnings.warn(f"get_beta failed: {type(e).__name__}: {e}")
            return None

    def get_portfolio_beta(
        self,
        holdings: Dict[str, float],
        prices: Dict[str, pd.Series],
        benchmark: str = 'SPY',
        start_date = None,
        end_date = None
    ) -> Optional[float]:
        """
        Calculate portfolio-weighted beta.

        Calculates beta for each holding then takes 
        weighted average based on market values.

        Args:
            holdings: Dictionary {ticker: shares}
            prices: Dictionary {ticker: price_series}
            benchmark: Benchmark ticker for beta calc
            start_date: Start date for calculation
            end_date: End date for calculation

        Returns:
            Portfolio beta (None if calculation fails)
        """
        try:
            # Need the benchmark series to compute any beta
            if benchmark not in prices:
                warnings.warn(
                    f"benchmark {benchmark!r} not in price "
                    f"data; portfolio beta unavailable.")
                return None

            # Daily returns of the benchmark (close-to-close)
            market_returns = (prices[benchmark].pct_change().dropna())

            # Running aggregates for the weighted average
            total_value = 0.0
            weighted_beta_sum = 0.0

            for ticker, shares in holdings.items():
                if ticker not in prices or shares <= 0:
                    continue

                # Most recent close = current price
                current_price = prices[ticker].iloc[-1]
                # Position market value at that price
                market_value = shares * current_price

                # Daily returns of this name
                stock_returns = (prices[ticker].pct_change().dropna())

                # Per-name beta vs the benchmark
                beta = self.get_beta(stock_returns, market_returns)

                # Only positions whose beta computed cleanly
                # contribute to the average
                if beta is not None:
                    total_value += market_value
                    weighted_beta_sum += market_value * beta

            # Portfolio beta = sum(MV_i * beta_i) / sum(MV_i)
            if total_value > 0:
                return weighted_beta_sum / total_value

            return None

        except (ValueError, TypeError, ZeroDivisionError) as e:
            warnings.warn(f"get_portfolio_beta failed: "
                f"{type(e).__name__}: {e}")
            return None

    def get_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 2.0
    ) -> float:
        """
        Calculate Sharpe ratio.

        Measures risk-adjusted return.

        Sharpe = (Return - RFR) / Volatility, RFR = Risk-Free Rate

        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate (%)

        Returns:
            Sharpe ratio

        Example:
            >>> sharpe = analyzer.get_sharpe_ratio(
            ...     portfolio_returns, risk_free_rate=2.0
            ... )
            >>> print(f"Sharpe ratio: {sharpe:.3f}")
        """
        if len(returns) == 0:
            return 0.0

        # Compound the daily mean across 252 trading days,
        # subtract 1 to get a return, then convert to percent
        annual_return = ((1 + returns.mean()) ** 252 - 1) * 100

        # Scale daily stdev to annual via sqrt-of-time, in %
        annual_vol = returns.std() * np.sqrt(252) * 100

        # Guard against a flat return series
        if annual_vol == 0:
            return 0.0

        # Sharpe = excess annualised return / annualised vol
        sharpe = ((annual_return - risk_free_rate) / annual_vol)
        # Return
        return float(sharpe)

    def get_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 2.0
    ) -> float:
        """
        Calculate Sortino ratio.

        Like Sharpe but uses downside deviation instead of total volatility.

        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate (%)

        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0

        # Same compounding-then-percent conversion as Sharpe
        annual_return = ((1 + returns.mean()) ** 252 - 1) * 100

        # Sortino punishes only the downside; ignore up days
        downside_returns = returns[returns < 0]
        # No negative returns -> infinite Sortino by convention
        if len(downside_returns) == 0:
            return float('inf')

        # Annualised downside volatility (%)
        downside_vol = (downside_returns.std() * np.sqrt(252) * 100)

        if downside_vol == 0:
            return 0.0

        # Sortino = excess annualised return / downside vol
        sortino = ((annual_return - risk_free_rate) / downside_vol)
        # Return
        return float(sortino)

    def get_alpha(
        self,
        portfolio_returns: pd.Series,
        market_returns: pd.Series,
        risk_free_rate: float = 2.0
    ) -> Optional[float]:
        """
        Calculate Jensen's alpha.

        Alpha measures excess return vs expected return 
        from Capital Asset Pricing Model (CAPM).

        Alpha = Actual Return - Expected Return
        Expected Return = RFR + Beta * (Market Return - RFR),
        where RFR = Risk-Free Rate.

        See `docs/theory/capm.md` for a full description of
        the model, its components, and its limits.

        Args:
            portfolio_returns: Portfolio return series
            market_returns: Market return series
            risk_free_rate: Annual risk-free rate (%)

        Returns:
            Annualized alpha (%) or None if calc fails

        Example:
            >>> alpha = analyzer.get_alpha(
            ...     portfolio_returns,
            ...     spy_returns,
            ...     risk_free_rate=2.0
            ... )
            >>> print(f"Alpha: {alpha:.2f}%")
        """
        try:
            # CAPM expected return depends on beta vs market
            beta = self.get_beta( portfolio_returns, market_returns)

            if beta is None:
                return None

            # Compound daily means to annual, in percent
            portfolio_annual = ((1 + portfolio_returns.mean())**252 - 1) * 100
            market_annual = ((1 + market_returns.mean())**252 - 1) * 100

            # CAPM: E[r] = rf + beta * (rm - rf)
            expected_return = (
                risk_free_rate + beta * (market_annual - risk_free_rate))

            # Jensen's alpha = realised - CAPM-expected
            alpha = portfolio_annual - expected_return

            # Return
            return float(alpha)

        except (ValueError, TypeError, ZeroDivisionError) as e:
            warnings.warn(f"get_alpha failed: {type(e).__name__}: {e}")
            return None

    def get_calmar_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 2.0
    ) -> Optional[float]:
        """
        Calculate Calmar ratio.

        Calmar = (Annualised Return - Risk-Free Rate) / |Max
        Drawdown|. Same numerator convention as Sharpe; the
        denominator swaps total-volatility for the worst
        peak-to-trough loss, so the ratio rewards strategies
        that produced their return without deep drawdowns.

        Args:
            returns: Daily return series.
            risk_free_rate: Annual risk-free rate (%).

        Returns:
            Calmar ratio, or ``None`` when the return series
            has no observed drawdown (constant or rising
            wealth curve), where the ratio is undefined.

        Example:
            >>> calmar = analyzer.get_calmar_ratio(returns)
            >>> print(f"Calmar ratio: {calmar:.3f}")
        """
        if len(returns) == 0:
            return None

        # Compound the daily mean to an annual figure, percent
        annual_return = ((1 + returns.mean()) ** 252 - 1) * 100

        # Peak-to-trough drop, same idiom as VolatilityAnalyzer.
        # Kept inline to avoid a circular dependency between
        # risk and volatility modules.
        cum = (1 + returns).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        max_dd = float(drawdown.min() * 100)

        # No observed drawdown -> ratio undefined
        if max_dd == 0:
            return None

        # Calmar = excess annualised return / |max drawdown|
        return float(
            (annual_return - risk_free_rate) / abs(max_dd))

    def classify_risk_level(self, beta: Optional[float]) -> str:
        """
        Classify risk level based on beta.

        Args:
            beta: Beta coefficient

        Returns:
            Risk level classification

        Classification:
            < 0: Contrarian
            < 0.5: Very Low
            < 0.8: Low
            < 1.2: Moderate
            < 1.5: High
            >= 1.5: Very High
        """
        if beta is None:
            return 'Unknown'
        if beta < 0:
            return 'Contrarian'
        elif beta < 0.5:
            return 'Very Low'
        elif beta < 0.8:
            return 'Low'
        elif beta < 1.2:
            return 'Moderate'
        elif beta < 1.5:
            return 'High'
        else:
            return 'Very High'

    def get_information_ratio(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Optional[float]:
        """
        Calculate information ratio.

        Measures excess return per unit of tracking error.
        IR = (Portfolio Return - Benchmark Return) /
             Tracking Error

        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series

        Returns:
            Information ratio, or None when undefined (no
            overlapping dates, or tracking error is zero).
        """
        # Inner-join on date so we compare same-day returns
        aligned = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if len(aligned) == 0:
            warnings.warn(
                "get_information_ratio: no overlapping "
                "dates between portfolio and benchmark; "
                "returning None.")
            return None

        # Day-by-day outperformance over the benchmark
        excess_returns = (aligned['portfolio'] - aligned['benchmark'])

        # Tracking error = annualised stdev of excess returns
        tracking_error = (excess_returns.std() * np.sqrt(252) * 100)

        if tracking_error == 0:
            warnings.warn(
                "get_information_ratio: tracking error is 0 "
                "(returns are identical); returning None.")
            return None

        # Compound the daily excess to an annual figure (%)
        annual_excess = ((1 + excess_returns.mean()) ** 252 - 1) * 100

        # IR = annualised excess return / tracking error
        return float(annual_excess / tracking_error)
