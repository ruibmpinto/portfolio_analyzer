"""
Rebalancing action data model.

Implements Contract 5 of the refactor design spec. One
RebalancingAction = one row in the rebalance plan: what to do
(BUY/REDUCE/HOLD/CASH), for which ticker, and at what cost.

The CASH pseudo-ticker (see `cash_ticker`) carries the target
cash reserve as a single CASH action with shares=0 and an
``est_cost_chf`` equal to the target cash CHF magnitude.

Validation enforces the action vocabulary, non-negative cost
magnitude (sign is carried by the `action` field), and the
HOLD/CASH-must-be-zero-shares invariant.
"""

from dataclasses import dataclass


valid_actions = ('BUY', 'REDUCE', 'HOLD', 'CASH')
cash_ticker = 'CASH'


@dataclass(frozen=True)
class RebalancingAction:
    """
    Single row of a rebalance plan.

    Attributes:
        ticker: Stock or ETF ticker symbol. Equal to
            `cash_ticker` ('CASH') for the cash bucket row.
        action: One of 'BUY', 'REDUCE', 'HOLD', 'CASH'. 'CASH'
            is reserved for the cash bucket pseudo-ticker.
        shares: Number of shares affected. 0 for HOLD and CASH;
            positive for BUY (shares to add); positive for
            REDUCE (shares to drop). Always non-negative —
            direction is carried by `action`.
        est_cost_chf: Magnitude of CHF cash flow. 0 for HOLD;
            positive for BUY and REDUCE; for CASH it carries the
            target cash CHF magnitude. Always non-negative.
        current_wt_pct: Position's weight before the action,
            in percent.
        target_wt_pct: Strategy's aspirational target weight, in
            percent. What the strategy wants; not necessarily
            reachable when capital is limited.
        achievable_wt_pct: Post-trade weight this action actually
            reaches, in percent, against the post-trade NAV
            (holdings + new capital). Equals target_wt_pct only
            when the action is fully funded; falls short when the
            buy pool cannot cover the gap. The rebalancer always
            sets this explicitly; the 0.0 default exists only for
            ad-hoc/test constructions.
        note: Free-form annotation (e.g.
            'Only if held >6 months' for REDUCE actions,
            'Cash reserve' for CASH actions).
    """

    ticker: str
    action: str
    shares: int
    est_cost_chf: float
    current_wt_pct: float
    target_wt_pct: float
    note: str
    achievable_wt_pct: float = 0.0

    def __post_init__(self):
        if self.action not in valid_actions:
            raise ValueError(
                f'Unknown action {self.action!r}; must be one '
                f'of {valid_actions}.')
        if self.shares < 0:
            raise ValueError(
                f'shares must be non-negative '
                f'(got {self.shares}); direction is carried '
                f'by `action`.')
        if self.est_cost_chf < 0:
            raise ValueError(
                f'est_cost_chf must be non-negative '
                f'(got {self.est_cost_chf}); magnitude only.')
        if self.action in ('HOLD', 'CASH') and self.shares != 0:
            raise ValueError(
                f'{self.action} action must have shares=0, '
                f'got {self.shares}.')
