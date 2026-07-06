"""Safe-harbor endpoint: Swiss 5x-turnover + 6-month-hold tracker.

Thin wrapper over ``services.safe_harbor`` so the route stays trivial
and the composition logic (Jan-1 NAV + turnover + realized gains +
holding-period violations) lives in the service where it can be
tested without spinning up FastAPI.

Routes
------
GET /api/safe_harbor
"""

from fastapi import APIRouter, Request

from src.dashboard_api.services import safe_harbor as sh_mod


router = APIRouter()


@router.get('/api/safe_harbor')
def get_safe_harbor(request: Request):
    """Return the Swiss safe-harbor tracker payload."""
    ctx = request.app.state.ctx
    return ctx.cache.get_or_compute(
        'safe_harbor',
        lambda: sh_mod.build_safe_harbor_payload(ctx))
