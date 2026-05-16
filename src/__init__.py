"""
Portfolio analysis, modelling, and screening package.

Top-level re-exports of the most common public types. For
domain-specific entry points, import directly from a
sub-package: ``src.analysis``, ``src.modelling``,
``src.screening``, ``src.orchestration``, ``src.shared``.

Example:
    >>> from src import PortfolioAnalyzer
    >>> analyzer = PortfolioAnalyzer(
    ...     degiro_csv_file_path=(
    ...         'reports/degiro/statements/'
    ...         '20240101_20260514_degiro_statement.csv')
    ... )
    >>> sharpe = analyzer.get_sharpe()
"""

__version__ = '2.0.0'

from src.analysis import (
    PortfolioAnalyzer,
    Transaction,
    Portfolio,
    DataProvider,
    YFinanceProvider,
    QFLibProvider,
)

__all__ = [
    'PortfolioAnalyzer',
    'Transaction',
    'Portfolio',
    'DataProvider',
    'YFinanceProvider',
    'QFLibProvider',
]
