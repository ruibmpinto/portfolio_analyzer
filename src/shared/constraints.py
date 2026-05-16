"""
Rebalance-policy constraints shared across strategies.

Defines `RebalanceConstraints`, a frozen dataclass capturing
all per-run rebalance policy: weight caps, excluded categories,
dust threshold, Swiss tax filter, new capital, and the HOLD
band. Consumed by every Strategy.propose() call and by
Rebalancer. The `default_constraints()` factory returns a
fresh instance with the canonical defaults from the design
spec (Contract 4).
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Tuple


@dataclass(frozen=True)
class RebalanceConstraints:
    """
    Per-run rebalance policy passed to every Strategy.

    Attributes:
        max_weight_default: Per-name maximum weight, as a fraction.
            Default 0.15.
        category_caps: Per-category overrides on `max_weight_default`. 
            Wrapped in MappingProxyType so the inner dict is read-only; 
            mutation raises TypeError. 
            Default MappingProxyType({'commodity': 0.05}).
        excluded_categories: Categories the rebalancer drops
            entirely before optimising. 
            Default ('speculative',).
        min_position_pct: Positions whose current weight (in
            PERCENT, not fraction) is below this value are
            treated as dust and dropped.
            Default 0.5 means weights below 0.5 percent.
        swiss_tax_filter: When True, exclude bond ETFs and
            high-dividend names per the Swiss tax-resident
            policy. 
            Default True.
        new_capital_chf: New capital to deploy on top of
            current NAV in rebalancing actions. 
            Default 0.0.
        rebalance_band_chf: (lower, upper) gap in CHF inside
            which a position is held; below the lower bound is
            REDUCE, above the upper bound is BUY. 
            Default (-200.0, 50.0).
    """

    max_weight_default: float = 0.15
    category_caps: Dict[str, float] = field(
        default_factory=lambda: MappingProxyType({'commodity': 0.05}))
    excluded_categories: Tuple[str, ...] = ('speculative',)
    min_position_pct: float = 0.5
    swiss_tax_filter: bool = True
    new_capital_chf: float = 0.0
    rebalance_band_chf: Tuple[float, float] = (-200.0, 50.0)


def default_constraints() -> RebalanceConstraints:
    """
    Return a fresh RebalanceConstraints with spec defaults.

    Returns:
        New instance; mutable members (`category_caps`) are
        per-instance, not shared.
    """
    return RebalanceConstraints()
