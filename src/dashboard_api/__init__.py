"""FastAPI sidecar serving live portfolio analytics to the
Tauri-shelled dashboard frontend.

The sidecar is spawned as a child process by the Tauri shell at
app startup. It listens on a localhost port chosen at runtime
(printed to stdout as the first line so the Rust parent can pick
it up) and exposes the GET/POST routes consumed by the React UI.

Modules
-------
server
    FastAPI app, route registration, uvicorn entrypoint.
cache
    Singleton TTL-cache for analyzer-derived snapshots.
routes
    HTTP endpoint handlers (overview, holdings, strategies,
    sells, meta).
services
    Domain logic that is composed by the route handlers
    (fees, lot tracking, benchmarks, annual returns, rebalance
    log, weighted strategy CAGR).
"""
