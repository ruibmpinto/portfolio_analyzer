"""
Central entities of the analysis domain.

- `Transaction`: data model for one buy/sell/dividend/tax row.
- `Portfolio`: holdings tracking over a transaction history.
- `PortfolioAnalyzer`: main analysis entry point, orchestrates
  loaders, metrics, and plots.
"""

from src.analysis.core.transaction import Transaction
from src.analysis.core.portfolio import Portfolio
from src.analysis.core.analyzer import PortfolioAnalyzer

__all__ = [
    'Transaction',
    'Portfolio',
    'PortfolioAnalyzer',
]
