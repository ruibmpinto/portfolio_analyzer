"""
Strategy ABC for rebalancing.

Every concrete strategy subclasses Strategy and implements
propose(...) returning target weights {ticker: fraction}.
Action generation is shared logic on Rebalancer, never
duplicated per strategy.

Concrete strategies self-register in `_registry` via
__init_subclass__. The strategies package re-exports
`_registry` as `strategy_registry`.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type

import pandas as pd

from src.shared.constraints import RebalanceConstraints


_registry: Dict[str, Type['Strategy']] = {}


class Strategy(ABC):
    """
    Abstract rebalancing strategy.

    Attributes:
        name: Registry key. Must be set on every concrete
            subclass; used by `strategy_registry` and the
            CLI's `--strategy` flag. Subclasses with an empty
            `name` are NOT registered (treated as abstract
            intermediates).
    """

    name: str = ''

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            _registry[cls.name] = cls

    @abstractmethod
    def propose(
        self,
        holdings_df: pd.DataFrame,
        prices: pd.DataFrame,
        constraints: RebalanceConstraints,
        candidates: Optional[List] = None,
        risk_free_rate: Optional[float] = None) -> Dict[str, float]:
        """
        Return target portfolio weights.

        Args:
            holdings_df: Snapshot from
                analyzer.get_holdings_snapshot(); columns
                include ticker, value_chf, weight_pct,
                category.
            prices: Wide DataFrame from
                analyzer.get_price_panel(); one column per
                held ticker, daily close, native currency.
            constraints: Per-run RebalanceConstraints.
            candidates: Optional list of CandidateTicker (set
                by the screener). None for portfolio-only
                strategies; screener-fed strategies (Phase 4)
                require it.
            risk_free_rate: Annual rate as a decimal fraction
                (e.g. 0.005 = 0.5%). None means the strategy
                fetches via
                src.shared.risk_free_rate.get_live_risk_free_rate().

        Returns:
            Dict {ticker: fraction} summing to 1.0 within
            target_weights_sum_tol of RebalancePlan.
        """
        raise NotImplementedError
