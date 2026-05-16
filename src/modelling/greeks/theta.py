"""
Theta calculation for options.

Theta measures the rate of time decay of an option's value.
"""

import numpy as np
from scipy.stats import norm
from typing import Literal


def calculate_theta(
    spot_price: float,
    strike_price: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    option_type: Literal['call', 'put'] = 'call',
    dividend_yield: float = 0.0
) -> float:
    """
    Calculate option theta (time decay).

    Theta measures how much the option's value decreases
    as time passes. Returned value is per day.

    Theta is typically negative for long options,
    indicating value loss over time.

    Args:
        spot_price: Current price of underlying asset
        strike_price: Option strike price
        time_to_expiry: Time to expiry in years
        volatility: Annual volatility (e.g., 0.20 for 20%)
        risk_free_rate: Risk-free rate (e.g., 0.05 for 5%)
        option_type: 'call' or 'put'
        dividend_yield: Annual dividend yield (default 0)

    Returns:
        Theta value (per day)

    Example:
        >>> theta = calculate_theta(
        ...     spot_price=100,
        ...     strike_price=105,
        ...     time_to_expiry=0.25,
        ...     volatility=0.20,
        ...     risk_free_rate=0.05,
        ...     option_type='call'
        ... )
        >>> print(f"Theta: ${theta:.4f} per day")
    """
    # Calculate d1 and d2
    d1 = (
        (
            np.log(spot_price / strike_price) +
            (risk_free_rate - dividend_yield +
             0.5 * volatility ** 2) * time_to_expiry
        ) /
        (volatility * np.sqrt(time_to_expiry))
    )
    d2 = d1 - volatility * np.sqrt(time_to_expiry)

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


def interpret_theta(
    theta: float, option_value: float
) -> str:
    """
    Interpret theta value.

    Args:
        theta: Theta value (per day)
        option_value: Current option value

    Returns:
        Interpretation string

    Example:
        >>> interpretation = interpret_theta(
        ...     -0.05, 3.50
        ... )
        >>> print(interpretation)
        "Moderate decay: 1.4% per day"
    """
    if option_value == 0:
        return "No time value remaining"

    decay_percent = abs(theta) / option_value * 100

    if decay_percent < 1:
        return f"Low decay: {decay_percent:.1f}% per day"
    elif decay_percent < 2:
        return f"Moderate decay: {decay_percent:.1f}% per day"
    else:
        return f"High decay: {decay_percent:.1f}% per day"
