"""
Static portfolio analysis: load transaction history, track
holdings, value the portfolio, compute risk and ratio
metrics, render dashboards, write reports.

`PortfolioAnalyzer` is the main entry point. Data-provider
classes are re-exported from `src.shared` for convenience —
the same instances back both analysis and modelling.
"""

from src.shared.data_provider import (
    DataProvider, YFinanceProvider, QFLibProvider,
)
from src.analysis.core.transaction import Transaction
from src.analysis.core.portfolio import Portfolio
from src.analysis.core.analyzer import PortfolioAnalyzer

__all__ = [
    'PortfolioAnalyzer',
    'Transaction',
    'Portfolio',
    'DataProvider',
    'YFinanceProvider',
    'QFLibProvider',
]
