"""
Delta calculation for options.

Delta measures the sensitivity of an option's price
to changes in the underlying asset price.
"""

import numpy as np
from scipy.stats import norm
from typing import Literal


def calculate_delta(
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

    Delta represents the expected change in option price
    for a $1 change in the underlying asset price.

    Range:
        - Call delta: 0 to 1
        - Put delta: -1 to 0

    Args:
        spot_price: Current price of underlying asset
        strike_price: Option strike price
        time_to_expiry: Time to expiry in years
        volatility: Annual volatility (e.g., 0.20 for 20%)
        risk_free_rate: Risk-free rate (e.g., 0.05 for 5%)
        option_type: 'call' or 'put'
        dividend_yield: Annual dividend yield (default 0)

    Returns:
        Delta value

    Example:
        >>> delta = calculate_delta(
        ...     spot_price=100,
        ...     strike_price=105,
        ...     time_to_expiry=0.25,
        ...     volatility=0.20,
        ...     risk_free_rate=0.05,
        ...     option_type='call'
        ... )
        >>> print(f"Call delta: {delta:.4f}")
    """
    # Calculate d1
    d1 = (
        (
            np.log(spot_price / strike_price) +
            (risk_free_rate - dividend_yield +
             0.5 * volatility ** 2) * time_to_expiry
        ) /
        (volatility * np.sqrt(time_to_expiry))
    )

    # Calculate delta
    if option_type == 'call':
        delta = (
            np.exp(-dividend_yield * time_to_expiry) *
            norm.cdf(d1)
        )
    else:  # put
        delta = (
            -np.exp(-dividend_yield * time_to_expiry) *
            norm.cdf(-d1)
        )

    return float(delta)


def interpret_delta(
    delta: float, option_type: Literal['call', 'put']
) -> str:
    """
    Interpret delta value.

    Args:
        delta: Delta value
        option_type: 'call' or 'put'

    Returns:
        Interpretation string

    Example:
        >>> interpretation = interpret_delta(0.7, 'call')
        >>> print(interpretation)
        "Significantly in-the-money"
    """
    abs_delta = abs(delta)

    if abs_delta < 0.25:
        return "Deep out-of-the-money"
    elif abs_delta < 0.45:
        return "Out-of-the-money"
    elif abs_delta < 0.55:
        return "At-the-money"
    elif abs_delta < 0.75:
        return "In-the-money"
    else:
        return "Deep in-the-money"
