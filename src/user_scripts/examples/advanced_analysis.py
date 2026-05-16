#!/usr/bin/env python3
"""
Advanced Portfolio Analysis Example.

Demonstrates advanced features including:
- Using different data providers
- Detailed metric calculations
- Custom visualizations
"""

import sys
import os
from datetime import datetime
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.analysis import (
    PortfolioAnalyzer,
    YFinanceProvider,
    QFLibProvider)
from src.analysis.loaders.csv_loader import (
    default_degiro_path)


def analyze_with_provider(csv_path: str, provider, provider_name: str):
    """
    Analyze portfolio with specific data provider.

    Args:
        csv_path: Path to CSV file
        provider: DataProvider instance
        provider_name: Name for display
    """
    print(f"\n{'='*60}")
    print(f"ANALYSIS USING {provider_name.upper()}")
    print(f"{'='*60}")

    # Initialize analyzer
    analyzer = PortfolioAnalyzer(csv_path, data_provider=provider)

    # Get comprehensive summary
    summary = analyzer.get_summary('SPY')

    print("\nRisk-Adjusted Performance:")
    print(f"  Sharpe Ratio:        {summary['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio:       {summary['sortino_ratio']:.3f}")
    print(f"  Information Ratio:   {summary['information_ratio']:.3f}")

    print("\nMarket Risk Metrics:")
    if summary['beta']:
        print(f"  Beta:                {summary['beta']:.3f}")
    else:
        print("  Beta:                N/A")

    if summary['alpha']:
        print(f"  Alpha:               {summary['alpha']:.3f}%")
    else:
        print("  Alpha:               N/A")

    print(f"  Volatility:          {summary['volatility']:.2f}%")

    print("\nValuation Metrics:")
    if summary['pe_ratio']:
        print(f"  P/E Ratio:           {summary['pe_ratio']:.2f}")
    else:
        print("  P/E Ratio:           N/A")

    if summary['dividend_yield']:
        print(
            f"  Dividend Yield:      "
            f"{summary['dividend_yield']*100:.2f}%"
        )
    else:
        print("  Dividend Yield:      N/A")

    return analyzer


def main():
    """Run advanced analysis."""
    csv_path = str(default_degiro_path('transactions'))

    # Analyze with yfinance
    print("\n\nUsing YFinance Provider")
    yf_provider = YFinanceProvider()
    yf_analyzer = analyze_with_provider(csv_path, yf_provider, "YFinance")

    # Analyze with qf-lib
    print("\n\nUsing QFLib Provider")
    qflib_provider = QFLibProvider()
    qflib_analyzer = analyze_with_provider(csv_path, qflib_provider, "QFLib")

    # Create results directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(
        results_dir, f'portfolio_dashboard_{timestamp}.png')

    # Save dashboard
    print(f"\n\nSaving dashboard to {save_path}...")
    yf_analyzer.plot_dashboard(save_path=save_path)
    print("Dashboard saved successfully!")

if __name__ == '__main__':
    main()
