#!/usr/bin/env python3
"""
Basic Portfolio Analysis Example.

Demonstrates basic usage of the PortfolioAnalyzer class
to calculate key metrics and save dashboard to results.
"""

import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(
    0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../..')
    )
)

from src.analysis import PortfolioAnalyzer, YFinanceProvider
from src.analysis.loaders.csv_loader import (
    default_degiro_path)


def main():
    """Run basic portfolio analysis."""
    # Initialize analyzer
    csv_path = str(default_degiro_path('transactions'))

    print("Initializing Portfolio Analyzer...")
    analyzer = PortfolioAnalyzer(csv_path)

    print("\nCalculating metrics...")

    # Get key metrics
    sharpe = analyzer.get_sharpe()
    beta = analyzer.get_beta('SPY')
    alpha = analyzer.get_alpha('SPY')
    vol = analyzer.get_volatility()
    pe = analyzer.get_pe_ratio()
    div_yield = analyzer.get_dividend_yield()

    # Print results
    print("\n" + "=" * 50)
    print("PORTFOLIO ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Sharpe Ratio:      {sharpe:.3f}")
    print(f"Beta vs SPY:       {beta:.3f}" if beta else
          "Beta vs SPY:       N/A")
    print(f"Alpha vs SPY:      {alpha:.3f}%" if alpha else
          "Alpha vs SPY:      N/A")
    print(f"Volatility:        {vol:.2f}%")
    print(f"P/E Ratio:         {pe:.2f}" if pe else
          "P/E Ratio:         N/A")

    if div_yield:
        print(f"Dividend Yield:    {div_yield*100:.2f}%")
    else:
        print("Dividend Yield:    N/A")

    print("=" * 50)

    # Create results directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(
        results_dir, f'portfolio_dashboard_{timestamp}.png')

    # Save dashboard
    print(f"\nSaving dashboard to {save_path}...")
    analyzer.plot_dashboard(save_path=save_path)
    print("Dashboard saved successfully!")


if __name__ == '__main__':
    main()
