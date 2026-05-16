"""FastAPI app + uvicorn entrypoint for the dashboard sidecar.

When the Tauri shell launches the bundled sidecar binary it
expects the **first line printed to stdout** to be the chosen
port, so the parent process can configure the WebView's
``API_BASE`` global. The chosen port is the one passed via
``DASHBOARD_PORT`` env var, or an OS-assigned ephemeral port
when ``DASHBOARD_PORT=0``.

Functions
---------
create_app
    Construct the FastAPI app, register every route, attach
    the shared ``DashboardContext``.
main
    Allocate a port, print it, run uvicorn.
"""

import socket
import sys
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.dashboard_api import config
from src.dashboard_api.context import DashboardContext
from src.dashboard_api.routes import (
    holdings, meta, overview, sells, strategies)


# Tauri WebViews load assets from tauri://localhost in production
# and http://localhost:1420 in dev. Allow both.
_default_origins = (
    'tauri://localhost',
    'http://localhost:1420',
    'http://127.0.0.1:1420',
    'http://localhost:5173',
)


def create_app(ctx: Optional[DashboardContext] = None) -> FastAPI:
    """Build the FastAPI app, register routes, attach context.

    Args:
        ctx: Optional pre-built DashboardContext (handy for
            tests). When None a fresh production context is
            constructed.

    Returns:
        Configured FastAPI app ready to hand to uvicorn or
        ``starlette.testclient.TestClient``.
    """
    app = FastAPI(
        title='stock_market dashboard sidecar',
        version='1.0.0')
    app.state.ctx = ctx or DashboardContext()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_default_origins),
        allow_credentials=False,
        allow_methods=['GET', 'POST'],
        allow_headers=['*'])
    app.include_router(meta.router)
    app.include_router(overview.router)
    app.include_router(holdings.router)
    app.include_router(sells.router)
    app.include_router(strategies.router)

    @app.get('/api/health')
    def _health():
        """Liveness probe used by the Tauri parent at startup."""
        return {'status': 'ok'}
    return app


def _pick_port(requested: int) -> int:
    """Return `requested` when set, else an OS-assigned free port."""
    if requested and requested > 0:
        return requested
    with socket.socket(
            socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def main():
    """Entry point used by both ``python -m`` and PyInstaller."""
    port = _pick_port(config.port)
    # Print first so the Tauri parent can read & route to it
    # before uvicorn finishes booting.
    print(port, flush=True)
    sys.stdout.flush()
    app = create_app()
    uvicorn.run(
        app, host=config.host, port=port, log_level='warning')


if __name__ == '__main__':
    main()
