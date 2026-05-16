"""FIFO lot tracking for the 6-month-hold sell eligibility rule.

Walks ``portfolio.transactions`` in chronological order. Each
buy adds a lot (date, shares); each sell consumes the oldest
lots first (FIFO). A ticker is "sellable today" when its earliest
remaining open lot is at least ``min_hold_days`` old.

Functions
---------
held_lots
    Per-ticker list of remaining open lots after FIFO matching.
sellable_today
    Subset of tickers whose oldest open lot is >= 183 days old.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


# Conservative six-month definition: 183 calendar days (~6 mo).
min_hold_days_default = 183


def held_lots(portfolio):
    """Return remaining open buy lots per ticker after FIFO sells.

    Args:
        portfolio: ``analysis.core.portfolio.Portfolio`` instance.

    Returns:
        Dict ticker -> list of (acquired_date, shares_remaining)
        tuples ordered oldest-first. Only tickers with a non-zero
        remaining position are included.
    """
    lots: Dict[str, List[List]] = {}
    for t in portfolio.transactions:
        op = t.operation
        if op == 'buy':
            lots.setdefault(t.ticker, []).append(
                [t.date, float(t.amount)])
        elif op == 'sell':
            remaining = float(t.amount)
            queue = lots.get(t.ticker, [])
            i = 0
            while remaining > 0 and i < len(queue):
                lot_date, lot_shares = queue[i]
                take = min(lot_shares, remaining)
                queue[i][1] = lot_shares - take
                remaining -= take
                if queue[i][1] <= 0:
                    i += 1
            # Drop fully-consumed lots
            lots[t.ticker] = [
                pair for pair in queue if pair[1] > 0]
            # remaining > 0 means oversell vs known buy history;
            # we silently ignore so partial-history portfolios
            # don't blow up. The analyzer enforces share-count
            # invariants separately.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return only positions with shares remaining.
    return {
        ticker: [(d, s) for d, s in pairs]
        for ticker, pairs in lots.items()
        if pairs and sum(s for _, s in pairs) > 0
    }


def sellable_today(
        portfolio,
        as_of: Optional[datetime] = None,
        min_hold_days: int = min_hold_days_default):
    """Return per-ticker 6-month-hold eligibility info.

    Args:
        portfolio: ``analysis.core.portfolio.Portfolio`` instance.
        as_of: Reference date. Defaults to ``datetime.now()``.
        min_hold_days: Minimum days the oldest open lot must
            have aged before the position is sellable.

    Returns:
        Dict ticker -> dict with keys:
            shares : float
                Total shares currently held.
            earliest_lot_date : datetime
                Acquisition date of the oldest open lot.
            days_held : int
                Calendar days between earliest_lot_date and
                as_of.
            eligible : bool
                True when ``days_held >= min_hold_days``.
            unlock_date : datetime
                ``earliest_lot_date + min_hold_days``. Only
                meaningful when ``eligible`` is False.
    """
    if as_of is None:
        as_of = datetime.now()
    open_lots = held_lots(portfolio)
    result: Dict[str, dict] = {}
    for ticker, pairs in open_lots.items():
        # Oldest open lot drives the hold-period clock
        earliest_date, _ = pairs[0]
        days_held = (as_of - earliest_date).days
        shares = sum(s for _, s in pairs)
        result[ticker] = {
            'shares': shares,
            'earliest_lot_date': earliest_date,
            'days_held': int(days_held),
            'eligible': days_held >= min_hold_days,
            'unlock_date':
                earliest_date + timedelta(days=min_hold_days),
        }
    return result
