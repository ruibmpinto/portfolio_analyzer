"""Render the portfolio dashboard PNG.

Loads Degiro + IBKR transaction CSVs, runs the full analysis pipeline, 
and writes `results/dashboard/{today}/dashboard.png`.

Run from the project root:
    python -m src.user_scripts.run_analysis_dashboard
"""

import sys
import pathlib

# Ensure the project root is on sys.path so `from src...`
# resolves regardless of the caller's working directory
root_dir = str(pathlib.Path(__file__).parents[2])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.analysis.core.analyzer import PortfolioAnalyzer
from src.analysis.loaders.csv_loader import (
    default_degiro_path,
    default_ibkr_statement_path)


def main():
    """Render the dashboard from the latest broker statements."""
    degiro_path = default_degiro_path('transactions')
    ibkr_path = default_ibkr_statement_path()
    analyzer = PortfolioAnalyzer(
        degiro_csv_file_path=str(degiro_path) if degiro_path else None,
        ibkr_csv_file_path=str(ibkr_path) if ibkr_path else None)
    
    analyzer.plot_dashboard()
    print(
        f'dashboard written to '
        f'{analyzer.plotter.output_dir}/dashboard.png')


if __name__ == '__main__':
    main()
