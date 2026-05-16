"""
Strategy registry: name -> Strategy subclass.

Concrete strategies self-register via Strategy.__init_subclass__.
Importing this module triggers import of every concrete
strategy module so the registry is fully populated.
"""

from src.modelling.rebalancing.strategies.base import (
    Strategy, _registry as strategy_registry)
from src.modelling.rebalancing.strategies import equal_weight  # noqa: F401
from src.modelling.rebalancing.strategies import inverse_vol  # noqa: F401
from src.modelling.rebalancing.strategies import max_growth  # noqa: F401
from src.modelling.rebalancing.strategies import min_risk  # noqa: F401
from src.modelling.rebalancing.strategies import min_variance  # noqa: F401
from src.modelling.rebalancing.strategies import mvo  # noqa: F401
from src.modelling.rebalancing.strategies import risk_adjusted  # noqa: F401


def get_strategy(name: str) -> Strategy:
    """
    Instantiate a registered strategy by name.

    Args:
        name: Registry key (e.g. 'mvo', 'equal_weight').

    Returns:
        Fresh instance of the matching Strategy subclass.

    Raises:
        KeyError: When `name` is not registered. Message lists
            the known names.
    """
    if name not in strategy_registry:
        raise KeyError(
            f'Unknown strategy {name!r}. Known: '
            f'{sorted(strategy_registry)}')
    return strategy_registry[name]()
