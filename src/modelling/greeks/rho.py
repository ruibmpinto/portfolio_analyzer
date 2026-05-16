"""
Rho calculation for options.

Rho measures the sensitivity of an option's price
to changes in the risk-free interest rate.
"""

import numpy as np
from scipy.stats import norm
from typing import Literal


def calculate_rho(
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

    Rho measures how much the option's value changes
    for a 1% change in the risk-free interest rate.

    Call options have positive rho (benefit from
    higher rates). Put options have negative rho.

    Args:
        spot_price: Current price of underlying asset
        strike_price: Option strike price
        time_to_expiry: Time to expiry in years
        volatility: Annual volatility (e.g., 0.20 for 20%)
        risk_free_rate: Risk-free rate (e.g., 0.05 for 5%)
        option_type: 'call' or 'put'
        dividend_yield: Annual dividend yield (default 0)

    Returns:
        Rho value (per 1% rate change)

    Example:
        >>> rho = calculate_rho(
        ...     spot_price=100,
        ...     strike_price=105,
        ...     time_to_expiry=0.25,
        ...     volatility=0.20,
        ...     risk_free_rate=0.05,
        ...     option_type='call'
        ... )
        >>> print(f"Rho: ${rho:.4f} per 1% rate")
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

    # Calculate rho
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


def interpret_rho(
    rho: float,
    option_value: float,
    option_type: Literal['call', 'put']
) -> str:
    """
    Interpret rho value.

    Args:
        rho: Rho value (per 1% rate)
        option_value: Current option value
        option_type: 'call' or 'put'

    Returns:
        Interpretation string

    Example:
        >>> interpretation = interpret_rho(
        ...     0.08, 3.50, 'call'
        ... )
        >>> print(interpretation)
        "Moderate sensitivity: 2.3% per 1% rate"
    """
    if option_value == 0:
        return "No sensitivity to rates"

    sensitivity_percent = abs(rho) / option_value * 100

    direction = "benefits" if rho > 0 else "hurt by"

    if sensitivity_percent < 2:
        return (
            f"Low sensitivity: {sensitivity_percent:.1f}% "
            f"per 1% rate ({direction} rate increases)"
        )
    elif sensitivity_percent < 5:
        return (
            f"Moderate sensitivity: "
            f"{sensitivity_percent:.1f}% "
            f"per 1% rate ({direction} rate increases)"
        )
    else:
        return (
            f"High sensitivity: {sensitivity_percent:.1f}% "
            f"per 1% rate ({direction} rate increases)"
        )
