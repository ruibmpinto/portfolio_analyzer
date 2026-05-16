"""
Cross-cutting infrastructure shared across all domains.

`shared` imports nothing from `analysis`, `modelling`, or
`screening` — it is the leaf of the dependency graph.

Re-exports the data-provider abstractions, the IBKR symbol
map, the live risk-free-rate fetcher, and the rebalance-
constraints dataclass + factory.
"""

from src.shared.data_provider import (
    DataProvider, YFinanceProvider, QFLibProvider,
    DefeatBetaProvider)
from src.shared.symbol_map import ibkr_symbol_map
from src.shared.risk_free_rate import (
    get_live_risk_free_rate,
    fred_series_id_default,
    fallback_rate_decimal)
from src.shared.constraints import (
    RebalanceConstraints, default_constraints)


__all__ = [
    'DataProvider',
    'YFinanceProvider',
    'QFLibProvider',
    'DefeatBetaProvider',
    'ibkr_symbol_map',
    'get_live_risk_free_rate',
    'fred_series_id_default',
    'fallback_rate_decimal',
    'RebalanceConstraints',
    'default_constraints',
]
