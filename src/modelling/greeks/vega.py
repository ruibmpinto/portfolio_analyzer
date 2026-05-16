"""
Vega calculation for options.

Vega measures the sensitivity of an option's price
to changes in volatility.
"""

import numpy as np
from scipy.stats import norm


def calculate_vega(
    spot_price: float,
    strike_price: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0
) -> float:
    """
    Calculate option vega.

    Vega measures how much the option's value changes
    for a 1% change in volatility.

    Vega is same for calls and puts with same parameters.

    Args:
        spot_price: Current price of underlying asset
        strike_price: Option strike price
        time_to_expiry: Time to expiry in years
        volatility: Annual volatility (e.g., 0.20 for 20%)
        risk_free_rate: Risk-free rate (e.g., 0.05 for 5%)
        dividend_yield: Annual dividend yield (default 0)

    Returns:
        Vega value (per 1% volatility change)

    Example:
        >>> vega = calculate_vega(
        ...     spot_price=100,
        ...     strike_price=105,
        ...     time_to_expiry=0.25,
        ...     volatility=0.20,
        ...     risk_free_rate=0.05
        ... )
        >>> print(f"Vega: ${vega:.4f} per 1% vol")
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

    # Calculate vega
    vega = (
        spot_price *
        np.exp(-dividend_yield * time_to_expiry) *
        norm.pdf(d1) *
        np.sqrt(time_to_expiry)
    )

    # Vega per 1% change in volatility
    vega_per_percent = vega / 100.0

    return float(vega_per_percent)


def interpret_vega(
    vega: float, option_value: float
) -> str:
    """
    Interpret vega value.

    Args:
        vega: Vega value (per 1% volatility)
        option_value: Current option value

    Returns:
        Interpretation string

    Example:
        >>> interpretation = interpret_vega(0.15, 3.50)
        >>> print(interpretation)
        "Moderate sensitivity: 4.3% per 1% vol"
    """
    if option_value == 0:
        return "No sensitivity to volatility"

    sensitivity_percent = vega / option_value * 100

    if sensitivity_percent < 3:
        return (
            f"Low sensitivity: "
            f"{sensitivity_percent:.1f}% per 1% vol"
        )
    elif sensitivity_percent < 7:
        return (
            f"Moderate sensitivity: "
            f"{sensitivity_percent:.1f}% per 1% vol"
        )
    else:
        return (
            f"High sensitivity: "
            f"{sensitivity_percent:.1f}% per 1% vol"
        )
