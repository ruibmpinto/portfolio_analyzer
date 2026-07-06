"""Forward-looking per-trade decision helpers.

Sits alongside ``rebalancing/`` (portfolio-level reallocation)
and ``monte_carlo/`` (simulation). Contains helpers that answer
questions about a single hypothetical trade — cost, breakeven,
stop-loss thresholds — without touching the portfolio state.
"""
