"""Sell-candidates endpoint.

Combines two sources:

  - The 6-month FIFO eligibility info per ticker (from
    ``services.lots``).
  - The REDUCE / sell actions proposed by the configured
    strategies, so each row carries a recommended action and
    reasoning when the underlying strategy wants to trim it.

Routes
------
GET /api/sells
"""

from datetime import datetime
from typing import Dict, List

from fastapi import APIRouter, Request

from src.dashboard_api import config
from src.dashboard_api.services import lots as lots_mod
from src.modelling.rebalancing import Rebalancer, get_strategy
from src.shared.constraints import RebalanceConstraints


router = APIRouter()


@router.get('/api/sells')
def get_sells(request: Request):
    """Return eligibility + recommended action per held ticker."""
    ctx = request.app.state.ctx
    return ctx.cache.get_or_compute(
        'sells', lambda: _build_payload(ctx))


def _build_payload(ctx) -> Dict:
    analyzer = ctx.analyzer
    eligibility = lots_mod.sellable_today(
        analyzer.portfolio, as_of=datetime.now())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Aggregate REDUCE recommendations across configured strategies
    reduce_reasons = _collect_reduce_reasons(ctx)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Combine
    eligible: List[Dict] = []
    locked: List[Dict] = []
    for ticker, info in eligibility.items():
        rec = {
            'ticker': ticker,
            'shares': info['shares'],
            'earliest_lot_date':
                info['earliest_lot_date'].strftime('%Y-%m-%d'),
            'days_held': info['days_held'],
            'unlock_date':
                info['unlock_date'].strftime('%Y-%m-%d'),
            'recommendations': reduce_reasons.get(ticker, []),
        }
        if info['eligible']:
            eligible.append(rec)
        else:
            locked.append(rec)
    # Sort eligible first by whether there's a recommendation,
    # then by days_held descending so the most aged-out lots
    # bubble up.
    eligible.sort(
        key=lambda r: (
            not bool(r['recommendations']),
            -r['days_held']))
    locked.sort(key=lambda r: r['unlock_date'])
    return {
        'eligible': eligible,
        'locked': locked,
        'min_hold_days': lots_mod.min_hold_days_default,
    }


def _collect_reduce_reasons(ctx) -> Dict[str, List[Dict]]:
    """Build {ticker: [{strategy, action, note}]} from REDUCE actions."""
    out: Dict[str, List[Dict]] = {}
    constraints = RebalanceConstraints(
        max_weight_default=config.mc_max_weight,
        new_capital_chf=config.mc_new_capital_chf)
    for name in config.strategies:
        try:
            rebalancer = Rebalancer(
                analyzer=ctx.analyzer,
                strategy=get_strategy(name),
                constraints=constraints,
                ticker_categories=ctx.categories or None)
            plan = rebalancer.propose()
        except Exception as exc:  # noqa: BLE001
            # Per-strategy failures must not block the others.
            out.setdefault('__errors__', []).append({
                'strategy': name, 'error': str(exc),
            })
            continue
        for action in plan.actions:
            if action.action != 'REDUCE':
                continue
            out.setdefault(action.ticker, []).append({
                'strategy': name,
                'action': action.action,
                'shares': action.shares,
                'note': action.note,
            })
    return out
