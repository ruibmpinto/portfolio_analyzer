"""
Options Greeks module.

Calculates option sensitivities (Greeks) using the
Black-Scholes model.

Available Greeks:
- Delta: Sensitivity to underlying price
- Gamma: Rate of change of delta
- Theta: Time decay
- Vega: Sensitivity to volatility
- Rho: Sensitivity to interest rates

Example:
    >>> from src.modelling.greeks import GreeksCalculator
    >>> calc = GreeksCalculator()
    >>> greeks = calc.calculate_all_greeks(
    ...     spot_price=100,
    ...     strike_price=105,
    ...     time_to_expiry=0.25,
    ...     volatility=0.20,
    ...     risk_free_rate=0.05,
    ...     option_type='call'
    ... )
    >>> print(greeks)
"""

from src.modelling.greeks.greeks_calculator import (
    GreeksCalculator, GreeksResult
)
from src.modelling.greeks.delta import calculate_delta
from src.modelling.greeks.gamma import calculate_gamma
from src.modelling.greeks.theta import calculate_theta
from src.modelling.greeks.vega import calculate_vega
from src.modelling.greeks.rho import calculate_rho

__all__ = [
    'GreeksCalculator',
    'GreeksResult',
    'calculate_delta',
    'calculate_gamma',
    'calculate_theta',
    'calculate_vega',
    'calculate_rho',
]
