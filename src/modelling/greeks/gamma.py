"""
Gamma calculation for options.

Gamma measures the rate of change of delta with respect
to changes in the underlying asset price.
"""

import numpy as np
from scipy.stats import norm


def calculate_gamma(
    spot_price: float,
    strike_price: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0
) -> float:
    """
    Calculate option gamma.

    Gamma measures the curvature of the option's value
    with respect to the underlying price. It represents
    the rate of change of delta.

    Gamma is same for calls and puts with same parameters.

    Args:
        spot_price: Current price of underlying asset
        strike_price: Option strike price
        time_to_expiry: Time to expiry in years
        volatility: Annual volatility (e.g., 0.20 for 20%)
        risk_free_rate: Risk-free rate (e.g., 0.05 for 5%)
        dividend_yield: Annual dividend yield (default 0)

    Returns:
        Gamma value

    Example:
        >>> gamma = calculate_gamma(
        ...     spot_price=100,
        ...     strike_price=105,
        ...     time_to_expiry=0.25,
        ...     volatility=0.20,
        ...     risk_free_rate=0.05
        ... )
        >>> print(f"Gamma: {gamma:.6f}")
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

    # Calculate gamma
    gamma = (
        np.exp(-dividend_yield * time_to_expiry) *
        norm.pdf(d1) /
        (spot_price * volatility * np.sqrt(time_to_expiry))
    )

    return float(gamma)


def interpret_gamma(gamma: float) -> str:
    """
    Interpret gamma value.

    Args:
        gamma: Gamma value

    Returns:
        Interpretation string

    Example:
        >>> interpretation = interpret_gamma(0.05)
        >>> print(interpretation)
        "High gamma: Delta changes rapidly"
    """
    if gamma < 0.01:
        return "Low gamma: Delta changes slowly"
    elif gamma < 0.03:
        return "Moderate gamma: Delta changes moderately"
    else:
        return "High gamma: Delta changes rapidly"
