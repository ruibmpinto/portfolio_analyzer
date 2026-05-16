"""Portfolio rebalancing report - programmatic orchestrator.

Reads transaction CSVs, runs the chosen rebalance strategy
against current holdings, and writes a multi-sheet Excel
workbook with current holdings, target weights, and BUY /
REDUCE / HOLD actions.

Edit the values at the top of ``main()`` to change the run
inputs (strategy, capital, file paths, etc.), then either:
    python -m src.analysis.report
or call from a session / notebook:
    from src.analysis import report
    report.main()
"""

import json
import pathlib
from typing import Dict, Optional

import pandas as pd

from src.analysis.core.analyzer import PortfolioAnalyzer
from src.analysis.loaders.csv_loader import (
    default_degiro_path,
    default_ibkr_statement_path)
from src.analysis.loaders.data_exporter import DataExporter
from src.modelling.rebalancing import Rebalancer, get_strategy
from src.screening.candidate import load_candidates_from_json
from src.shared.constraints import RebalanceConstraints
from src.shared.data_provider import YFinanceProvider


def _load_categories(path: str) -> Dict[str, str]:
    """Load ticker -> category map from JSON."""
    with open(path) as f:
        return json.load(f)


def _summary_dataframe(
    plan_summary: Dict[str, float]) -> pd.DataFrame:
    """Single-row DataFrame for the Summary sheet."""
    return pd.DataFrame([plan_summary])


def main():
    """Generate the rebalancing Excel report.

    Args:
        strategy: Registered strategy name (e.g. 'mvo',
            'max_growth').
        new_capital_chf: Fresh capital to deploy this run.
        max_weight: Per-name cap fed into RebalanceConstraints.
        degiro_csv: Path to Degiro transaction CSV, or None.
        ibkr_csv: Path to IBKR transaction CSV, or None.
        ticker_categories_path: Path to the ticker -> category
            JSON map.
        candidates_from: Path to a screener JSON output (for
            screener-fed strategies); ``None`` for
            portfolio-only strategies.
        output_path: Excel destination.

    Raises:
        ValueError: When neither degiro_csv nor ibkr_csv is set.
        KeyError: When ``strategy`` is not registered.
    """
    strategy = 'mvo'
    new_capital_chf = 2000.0
    max_weight = 0.15
    # Auto-pick the newest broker statement; override these
    # two lines with a literal path when reproducing an older
    # run.
    degiro_path = default_degiro_path('transactions')
    ibkr_path = default_ibkr_statement_path()
    degiro_csv = str(degiro_path) if degiro_path else None
    ibkr_csv = str(ibkr_path) if ibkr_path else None
    ticker_categories_path = 'data/ticker_categories.json'
    candidates_from = None  # set to a JSON path for screener-fed
    output_path = 'results/portfolio_report.xlsx'


    if not degiro_csv and not ibkr_csv:
        raise ValueError(
            'report.main_with_inputs: must supply at least '
            'one of degiro_csv or ibkr_csv.')

    categories = _load_categories(ticker_categories_path)
    constraints = RebalanceConstraints(
        max_weight_default=max_weight,
        new_capital_chf=new_capital_chf)
    provider = YFinanceProvider()

    analyzer = PortfolioAnalyzer(
        degiro_csv_file_path=degiro_csv,
        ibkr_csv_file_path=ibkr_csv,
        data_provider=provider,
        base_currency='CHF')

    candidates = None
    if candidates_from is not None:
        candidates = load_candidates_from_json(candidates_from)

    rebalancer = Rebalancer(
        analyzer=analyzer,
        strategy=get_strategy(strategy),
        constraints=constraints,
        ticker_categories=categories)
    plan = rebalancer.propose(candidates=candidates)

    sheets = {
        'Holdings': plan.holdings_df,
        'Target Weights': plan.target_weights_dataframe(),
        'Rebalancing Actions': plan.to_dataframe(),
        'Summary': _summary_dataframe(plan.summary())}

    out_dir = pathlib.Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    DataExporter().export_to_excel(sheets, output_path)
    print(
        f'Report saved: {output_path} '
        f'(strategy={strategy}, '
        f'NAV CHF {plan.total_value_chf:,.2f}, '
        f'{len(plan.actions)} actions, '
        f'generated {plan.generated_at:%Y-%m-%d %H:%M})')

if __name__ == '__main__':
    main()
