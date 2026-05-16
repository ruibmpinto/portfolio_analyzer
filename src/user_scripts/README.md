# User Scripts

This directory contains example scripts demonstrating how to use the portfolio analysis package.

## Structure

```
user_scripts/
├── examples/
│   ├── basic_analysis.py      # Basic usage example
│   └── advanced_analysis.py   # Advanced features demo
└── README.md                   # This file
```

## Running Examples

### Basic Analysis

Demonstrates fundamental usage:

```bash
python user_scripts/examples/basic_analysis.py
```

Features shown:
- Loading portfolio from CSV
- Calculating Sharpe ratio, beta, alpha
- Getting volatility and P/E ratio
- Displaying comprehensive dashboard

### Advanced Analysis

Shows advanced features:

```bash
python user_scripts/examples/advanced_analysis.py
```

Features shown:
- Using different data providers (YFinance vs QFLib)
- Getting comprehensive metric summaries
- Comparing provider performance
- Advanced visualization

## Creating Your Own Scripts

### Template

```python
#!/usr/bin/env python3
import sys
import os

sys.path.insert(
    0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../..')
    )
)

from src.core import PortfolioAnalyzer
from src.analysis.loaders.csv_loader import (
    default_degiro_path,
    default_ibkr_statement_path)

def main():
    # Initialize analyzer (auto-pick the newest broker exports)
    analyzer = PortfolioAnalyzer(
        degiro_csv_file_path=str(default_degiro_path('transactions')),
        ibkr_csv_file_path=str(default_ibkr_statement_path()),
    )

    # Your analysis here
    sharpe = analyzer.get_sharpe()
    print(f"Sharpe Ratio: {sharpe:.3f}")

    # Show dashboard
    analyzer.plot_dashboard()

if __name__ == '__main__':
    main()
```

## Available Metrics

### Risk-Adjusted Performance
- `get_sharpe(risk_free_rate=2.0)` - Sharpe ratio
- `get_sortino_ratio(risk_free_rate=2.0)` - Sortino ratio
- `get_information_ratio(benchmark='SPY')` - Info ratio

### Market Risk
- `get_beta(benchmark='SPY')` - Beta coefficient
- `get_alpha(benchmark='SPY')` - Jensen's alpha
- `get_volatility(window=None)` - Volatility

### Valuation
- `get_pe_ratio()` - Portfolio P/E ratio
- `get_dividend_yield()` - Dividend yield

### Visualization
- `plot_dashboard(benchmark='SPY')` - Full dashboard

### Summary
- `get_summary(benchmark='SPY')` - All metrics dict

## Data Providers

### YFinance (Default)
```python
from src.core import YFinanceProvider

provider = YFinanceProvider()
analyzer = PortfolioAnalyzer(csv_path, provider)
```

Lightweight, suitable for most use cases.

### QFLib (Advanced)
```python
from src.core import QFLibProvider

provider = QFLibProvider()
analyzer = PortfolioAnalyzer(csv_path, provider)
```

Advanced quantitative finance library.
Requires: `pip install qf-lib`

## CSV Format

Expected transaction CSV format:

```csv
ticker,operation,date,price,amount,fee,auto_fx_fee
AAPL,buy,2023-01-15,150.00,100,9.95,0.00
MSFT,buy,2023-02-20,280.00,50,9.95,0.00
AAPL,sell,2023-06-15,180.00,50,9.95,0.00
```

Operations: `buy`, `sell`, `dividend`
