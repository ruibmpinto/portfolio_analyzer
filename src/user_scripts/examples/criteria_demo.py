#!/usr/bin/env python3
"""
Screening Criteria Demonstration.

Shows how to define and use screening criteria.
"""

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.screening import (
    RangeCriterion,
    ComparisonCriterion,
    CompositeCriterion,
    print_metrics_guide)


def main():
    """Demonstrate screening criteria."""
    print("=" * 70)
    print("SCREENING CRITERIA DEMONSTRATION")
    print("=" * 70)

    # Show available metrics
    print("\n1. Available Metrics:")
    print("-" * 70)
    print_metrics_guide()

    # Example stock data
    stock_data = {
        'AAPL': {
            'price': 175.00,
            'market_cap': 2.8e12,
            'pe_ratio': 28.5,
            'forward_pe': 25.2,
            'dividend_yield': 0.52,
            'roe': 147.0,
            'profit_margin': 25.3,
            'debt_to_equity': 1.96,
        },
        'JNJ': {
            'price': 160.00,
            'market_cap': 400e9,
            'pe_ratio': 15.2,
            'forward_pe': 14.8,
            'dividend_yield': 3.1,
            'roe': 28.5,
            'profit_margin': 20.1,
            'debt_to_equity': 0.48,
        },
        'TSLA': {
            'price': 245.00,
            'market_cap': 780e9,
            'pe_ratio': 72.5,
            'forward_pe': 58.3,
            'dividend_yield': 0.0,
            'roe': 28.2,
            'profit_margin': 15.5,
            'debt_to_equity': 0.08,
        },
    }

    # Test Range Criterion
    print("\n2. Range Criterion Examples:")
    print("-" * 70)

    pe_criterion = RangeCriterion('pe_ratio', 10, 20)
    print(f"\nCriterion: {pe_criterion}")
    print("Matches:")
    for ticker, data in stock_data.items():
        if pe_criterion.matches(data):
            print(f"  {ticker}: P/E = {data['pe_ratio']}")

    # Test Comparison Criterion
    print("\n3. Comparison Criterion Examples:")
    print("-" * 70)

    div_criterion = ComparisonCriterion('dividend_yield', '>', 2.0)
    print(f"\nCriterion: {div_criterion}")
    print("Matches:")
    for ticker, data in stock_data.items():
        if div_criterion.matches(data):
            print(f"  {ticker}: Div Yield = {data['dividend_yield']}%")

    # Test Composite Criterion (AND)
    print("\n4. Composite Criterion (AND) Example:")
    print("-" * 70)

    value_criterion = CompositeCriterion(
        [
            RangeCriterion('pe_ratio', None, 20),
            ComparisonCriterion('dividend_yield', '>', 2.0),
            ComparisonCriterion('market_cap', '>', 100e9),
        ],
        logic='AND')
    
    print(f"\nCriterion: {value_criterion}")
    print("Matches:")
    for ticker, data in stock_data.items():
        if value_criterion.matches(data):
            print(
                f"  {ticker}: P/E={data['pe_ratio']}, "
                f"Div={data['dividend_yield']}%, "
                f"MCap=${data['market_cap']/1e9:.0f}B")

    # Test Composite Criterion (OR)
    print("\n5. Composite Criterion (OR) Example:")
    print("-" * 70)

    growth_or_value = CompositeCriterion(
        [
            RangeCriterion('pe_ratio', None, 15),
            ComparisonCriterion('roe', '>', 100),
        ],
        logic='OR')
    print(f"\nCriterion: {growth_or_value}")
    print("Matches:")
    for ticker, data in stock_data.items():
        if growth_or_value.matches(data):
            print(
                f"  {ticker}: P/E={data['pe_ratio']}, "
                f"ROE={data['roe']}%")

    # Complex nested example
    print("\n6. Complex Nested Criterion Example:")
    print("-" * 70)

    complex_criterion = CompositeCriterion(
        [
            CompositeCriterion(
                [
                    RangeCriterion('pe_ratio', 10, 25),
                    ComparisonCriterion('profit_margin', '>', 15),
                ],
                logic='AND'),
            ComparisonCriterion('debt_to_equity', '<', 0.5),
        ],
        logic='AND')
    print(f"\nCriterion: {complex_criterion}")
    print("Matches:")
    for ticker, data in stock_data.items():
        if complex_criterion.matches(data):
            print(
                f"  {ticker}: P/E={data['pe_ratio']}, "
                f"Margin={data['profit_margin']}%, "
                f"D/E={data['debt_to_equity']}")

    # Serialization example
    print("\n7. Criterion Serialization:")
    print("-" * 70)

    criterion = RangeCriterion('pe_ratio', 10, 20)
    print(f"Criterion: {criterion}")
    print(f"Serialized: {criterion.to_dict()}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
