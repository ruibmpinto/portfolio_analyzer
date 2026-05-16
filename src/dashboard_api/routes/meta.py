"""Meta-info endpoints: screener freshness + rebalance log.

Surfaces the two pieces of system-state the dashboard needs to
nag the user about staleness:

  - When the screener cache was last refreshed.
  - When the portfolio was last rebalanced (and whether a new
    rebalance is overdue).

Routes
------
GET /api/meta
    Combined screener-refresh + last-rebalance payload.
POST /api/rebalance-done
    Append a new "rebalance executed" entry to the JSON log.
POST /api/refresh
    Invalidate every cached value so the next GET refetches.
"""

import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Body, Request

from src.dashboard_api import config
from src.dashboard_api.services import rebalance_log


router = APIRouter()


@router.get('/api/meta')
def get_meta(request: Request):
    """Return screener refresh + last-rebalance + due flag."""
    ctx = request.app.state.ctx
    screener = _read_screener_metadata()
    last_entry = rebalance_log.last_rebalance(
        path=config.rebalance_log_path)
    days = rebalance_log.days_since(last_entry)
    due = (
        last_entry is None
        or (days is not None
            and days >= rebalance_log.rebalance_due_days))
    return {
        'screener': screener,
        'last_rebalance': last_entry,
        'days_since_rebalance': days,
        'rebalance_due': due,
        'rebalance_due_threshold_days':
            rebalance_log.rebalance_due_days,
        'base_currency': ctx.base_currency,
    }


@router.post('/api/rebalance-done')
def post_rebalance_done(
        request: Request,
        payload: dict = Body(...)):
    """Append a new entry to the rebalance log."""
    strategy_name = payload.get('strategy_name')
    if not strategy_name:
        return {'error': 'strategy_name required'}, 400
    entry = rebalance_log.mark_done(
        strategy_name=strategy_name,
        path=config.rebalance_log_path)
    # Bust the meta cache so the new entry shows immediately
    request.app.state.ctx.cache.invalidate('meta')
    return {'entry': entry}


@router.post('/api/refresh')
def post_refresh(request: Request):
    """Drop every cached value and reload the analyzer."""
    request.app.state.ctx.invalidate()
    return {'status': 'invalidated',
            'ts_utc': datetime.now(timezone.utc).isoformat()}


def _read_screener_metadata() -> Optional[dict]:
    """Load screener metadata JSON; return None on failure."""
    path = config.screener_metadata_path
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    # Find the most recent updated timestamp across exchanges
    latest = None
    for _, info in data.items():
        if not isinstance(info, dict):
            continue
        stamp = info.get('updated')
        if stamp is None:
            continue
        if latest is None or stamp > latest:
            latest = stamp
    return {
        'exchanges': data,
        'latest_updated_utc': latest,
    }
