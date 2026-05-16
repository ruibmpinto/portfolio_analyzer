"""
Options Greeks calculator.

Calculates option sensitivities (Greeks) using the
Black-Scholes model.
"""

import numpy as np
from scipy.stats import norm
from typing import Literal, Dict
from dataclasses import dataclass


@dataclass
class GreeksResult:
    """
    Container for all Greeks values.

    Attributes:
        delta: Sensitivity to underlying price
        gamma: Rate of change of delta
        theta: Time decay (per day)
        vega: Sensitivity to volatility (per 1%)
        rho: Sensitivity to interest rate (per 1%)
    """
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

    def __str__(self) -> str:
        """String representation of Greeks."""
        return (
            f"Greeks(delta={self.delta:.4f}, "
            f"gamma={self.gamma:.4f}, "
            f"theta={self.theta:.4f}, "
            f"vega={self.vega:.4f}, "
            f"rho={self.rho:.4f})"
        )


class GreeksCalculator:
    """
    Calculates option Greeks using Black-Scholes model.

    All Greeks are calculated for European-style options.

    Example:
        >>> calc = GreeksCalculator()
        >>> greeks = calc.calculate_all_greeks(
        ...     spot_price=100,
        ...     strike_price=105,
        ...     time_to_expiry=0.25,
        ...     volatility=0.20,
        ...     risk_free_rate=0.05,
        ...     option_type='call'
        ... )
        >>> print(f"Delta: {greeks.delta:.4f}")
    """

    def __init__(self):
        """Initialize Greeks calculator."""
        pass

    def calculate_all_greeks(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        option_type: Literal['call', 'put'] = 'call',
        dividend_yield: float = 0.0
    ) -> GreeksResult:
        """
        Calculate all Greeks at once.

        Args:
            spot_price: Current price of underlying
            strike_price: Option strike price
            time_to_expiry: Time to expiry in years
            volatility: Annual volatility (e.g., 0.20 for 20%)
            risk_free_rate: Risk-free rate (e.g., 0.05 for 5%)
            option_type: 'call' or 'put'
            dividend_yield: Annual dividend yield (default 0)

        Returns:
            GreeksResult with all Greeks values

        Example:
            >>> greeks = calc.calculate_all_greeks(
            ...     100, 105, 0.25, 0.20, 0.05, 'call'
            ... )
        """
        delta = self.calculate_delta(
            spot_price, strike_price, time_to_expiry,
            volatility, risk_free_rate, option_type,
            dividend_yield
        )

        gamma = self.calculate_gamma(
            spot_price, strike_price, time_to_expiry,
            volatility, risk_free_rate, dividend_yield
        )

        theta = self.calculate_theta(
            spot_price, strike_price, time_to_expiry,
            volatility, risk_free_rate, option_type,
            dividend_yield
        )

        vega = self.calculate_vega(
            spot_price, strike_price, time_to_expiry,
            volatility, risk_free_rate, dividend_yield
        )

        rho = self.calculate_rho(
            spot_price, strike_price, time_to_expiry,
            volatility, risk_free_rate, option_type,
            dividend_yield
        )

        return GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho
        )

    def calculate_delta(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        option_type: Literal['call', 'put'] = 'call',
        dividend_yield: float = 0.0
    ) -> float:
        """
        Calculate option delta.

        Delta measures sensitivity to $1 change in
        underlying price.

        Args:
            spot_price: Current price of underlying
            strike_price: Option strike price
            time_to_expiry: Time to expiry in years
            volatility: Annual volatility
            risk_free_rate: Risk-free rate
            option_type: 'call' or 'put'
            dividend_yield: Annual dividend yield

        Returns:
            Delta value (between -1 and 1)
        """
        d1 = self._calculate_d1(
            spot_price, strike_price, time_to_expiry,
            volatility, risk_free_rate, dividend_yield
        )

        if option_type == 'call':
            delta = np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1)
        else:  # put
            delta = -np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)

        return float(delta)

    def calculate_gamma(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0
    ) -> float:
        """
        Calculate option gamma.

        Gamma measures rate of change of delta.
        Same for calls and puts.

        Args:
            spot_price: Current price of underlying
            strike_price: Option strike price
            time_to_expiry: Time to expiry in years
            volatility: Annual volatility
            risk_free_rate: Risk-free rate
            dividend_yield: Annual dividend yield

        Returns:
            Gamma value
        """
        d1 = self._calculate_d1(
            spot_price, strike_price, time_to_expiry,
            volatility, risk_free_rate, dividend_yield
        )

        gamma = (
            np.exp(-dividend_yield * time_to_expiry) *
            norm.pdf(d1) /
            (spot_price * volatility * np.sqrt(time_to_expiry))
        )

        return float(gamma)

    def calculate_theta(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        option_type: Literal['call', 'put'] = 'call',
        dividend_yield: float = 0.0
    ) -> float:
        """
        Calculate option theta.

        Theta measures time decay (per day).
        Typically negative for long options.

        Args:
            spot_price: Current price of underlying
            strike_price: Option strike price
            time_to_expiry: Time to expiry in years
            volatility: Annual volatility
            risk_free_rate: Risk-free rate
            option_type: 'call' or 'put'
            dividend_yield: Annual dividend yield

        Returns:
            Theta value (per day)
        """
        d1 = self._calculate_d1(
            spot_price, strike_price, time_to_expiry,
            volatility, risk_free_rate, dividend_yield
        )
        d2 = self._calculate_d2(d1, volatility, time_to_expiry)

        sqrt_t = np.sqrt(time_to_expiry)

        # Common term
        term1 = -(
            spot_price * norm.pdf(d1) * volatility *
            np.exp(-dividend_yield * time_to_expiry)
        ) / (2 * sqrt_t)

        if option_type == 'call':
            term2 = (
                dividend_yield * spot_price * norm.cdf(d1) *
                np.exp(-dividend_yield * time_to_expiry)
            )
            term3 = (
                risk_free_rate * strike_price *
                np.exp(-risk_free_rate * time_to_expiry) *
                norm.cdf(d2)
            )
            theta = term1 - term2 - term3
        else:  # put
            term2 = (
                dividend_yield * spot_price * norm.cdf(-d1) *
                np.exp(-dividend_yield * time_to_expiry)
            )
            term3 = (
                risk_free_rate * strike_price *
                np.exp(-risk_free_rate * time_to_expiry) *
                norm.cdf(-d2)
            )
            theta = term1 + term2 + term3

        # Convert to per-day theta
        theta_per_day = theta / 365.0

        return float(theta_per_day)

    def calculate_vega(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0
    ) -> float:
        """
        Calculate option vega.

        Vega measures sensitivity to 1% change in
        volatility. Same for calls and puts.

        Args:
            spot_price: Current price of underlying
            strike_price: Option strike price
            time_to_expiry: Time to expiry in years
            volatility: Annual volatility
            risk_free_rate: Risk-free rate
            dividend_yield: Annual dividend yield

        Returns:
            Vega value (per 1% volatility change)
        """
        d1 = self._calculate_d1(
            spot_price, strike_price, time_to_expiry,
            volatility, risk_free_rate, dividend_yield
        )

        vega = (
            spot_price *
            np.exp(-dividend_yield * time_to_expiry) *
            norm.pdf(d1) *
            np.sqrt(time_to_expiry)
        )

        # Vega per 1% change in volatility
        vega_per_percent = vega / 100.0

        return float(vega_per_percent)

    def calculate_rho(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        option_type: Literal['call', 'put'] = 'call',
        dividend_yield: float = 0.0
    ) -> float:
        """
        Calculate option rho.

        Rho measures sensitivity to 1% change in
        interest rate.

        Args:
            spot_price: Current price of underlying
            strike_price: Option strike price
            time_to_expiry: Time to expiry in years
            volatility: Annual volatility
            risk_free_rate: Risk-free rate
            option_type: 'call' or 'put'
            dividend_yield: Annual dividend yield

        Returns:
            Rho value (per 1% rate change)
        """
        d1 = self._calculate_d1(
            spot_price, strike_price, time_to_expiry,
            volatility, risk_free_rate, dividend_yield
        )
        d2 = self._calculate_d2(d1, volatility, time_to_expiry)

        if option_type == 'call':
            rho = (
                strike_price * time_to_expiry *
                np.exp(-risk_free_rate * time_to_expiry) *
                norm.cdf(d2)
            )
        else:  # put
            rho = (
                -strike_price * time_to_expiry *
                np.exp(-risk_free_rate * time_to_expiry) *
                norm.cdf(-d2)
            )

        # Rho per 1% change in rate
        rho_per_percent = rho / 100.0

        return float(rho_per_percent)

    def _calculate_d1(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0
    ) -> float:
        """Calculate d1 for Black-Scholes formula."""
        d1 = (
            (
                np.log(spot_price / strike_price) +
                (risk_free_rate - dividend_yield +
                 0.5 * volatility ** 2) * time_to_expiry
            ) /
            (volatility * np.sqrt(time_to_expiry))
        )
        return d1

    def _calculate_d2(
        self,
        d1: float,
        volatility: float,
        time_to_expiry: float
    ) -> float:
        """Calculate d2 for Black-Scholes formula."""
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        return d2
