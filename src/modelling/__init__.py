"""
Forward-looking quantitative models.

- `monte_carlo`: Clayton-copula simulation of rebalancing
  scenarios.
- `rebalancing`: scenario construction over a fixed
  monthly deployment.
- `greeks`: Black-Scholes option Greeks.

Imports from `src.analysis.transaction` and
`src.analysis.portfolio` are allowed (seed simulations with
the current holding state); imports from any other
`src.analysis` submodule are not.
"""
