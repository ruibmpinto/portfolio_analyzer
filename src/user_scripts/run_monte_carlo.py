"""Monte Carlo simulation - programmatic orchestrator.

Loads a portfolio via PortfolioAnalyzer, builds a RebalancePlan
via the registered strategy, then runs MonteCarloEngine.simulate
over the configured horizon and writes a multi-sheet Excel
report.

Edit the values at the top of ``main()`` to change the run
inputs (strategy, horizon, n_paths, etc.), then either:
    python -m src.user_scripts.run_monte_carlo
or call from a session / notebook:
    from src.user_scripts import run_monte_carlo
    run_monte_carlo.main()
"""

import json
import pathlib
from typing import Dict

import pandas as pd

from src.analysis.core.analyzer import PortfolioAnalyzer
from src.analysis.loaders.csv_loader import (
    default_degiro_path,
    default_ibkr_statement_path)
from src.analysis.loaders.data_exporter import DataExporter
from src.modelling.monte_carlo import (
    MonteCarloEngine, ibkr_default_cost_model)
from src.modelling.rebalancing import (
    Rebalancer, get_strategy)
from src.screening.candidate import load_candidates_from_json
from src.shared.constraints import RebalanceConstraints
from src.shared.data_provider import (
    DataProvider, YFinanceProvider)


def _build_data_provider() -> DataProvider:
    """Factory hook returning the live data provider."""
    return YFinanceProvider()


def _load_categories(path: str) -> Dict[str, str]:
    """Load ticker -> category map from JSON."""
    with open(path) as f:
        return json.load(f)


def main() -> None:
    """Generate the Monte Carlo Excel report.

    Inputs are local constants — edit them at the top of this
    function to change strategy, horizon, n_paths, cost model,
    rebalance frequency, etc.

    Raises:
        ValueError: When neither degiro_csv nor ibkr_csv is set.
        KeyError: When ``strategy`` is not a registered name.
    """
    strategy = 'mvo'
    horizon_days = 252
    n_paths = 5000
    rebalance_frequency = 'monthly'
    monthly_contribution_chf = 2000.0
    no_costs = False
    new_capital_chf = 2000.0
    max_weight = 0.15
    random_seed = 42
    candidates_from = None  # set to a JSON path for screener-fed
    ticker_categories_path = 'data/ticker_categories.json'
    # Auto-pick the newest broker statement; override with a
    # literal path when reproducing an older run.
    degiro_path = default_degiro_path('transactions')
    ibkr_path = default_ibkr_statement_path()
    degiro_csv = str(degiro_path) if degiro_path else None
    ibkr_csv = str(ibkr_path) if ibkr_path else None
    output_path = 'results/monte_carlo_report.xlsx'

    if not degiro_csv and not ibkr_csv:
        raise ValueError(
            'run_monte_carlo.main: must supply at least one '
            'of degiro_csv or ibkr_csv.')

    categories = _load_categories(ticker_categories_path)
    constraints = RebalanceConstraints(
        max_weight_default=max_weight,
        new_capital_chf=new_capital_chf)
    provider = _build_data_provider()

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

    ticker_currencies = {}
    if 'currency' in plan.holdings_df.columns:
        ticker_currencies = dict(zip(
            plan.holdings_df['ticker'],
            plan.holdings_df['currency']))
    if candidates:
        for c in candidates:
            ticker_currencies[c.ticker] = c.currency

    prices = analyzer.get_price_panel()
    cost_model = (
        None if no_costs else ibkr_default_cost_model())

    engine = MonteCarloEngine(
        n_paths=n_paths,
        cost_model=cost_model,
        rebalance_frequency=rebalance_frequency,
        monthly_contribution_chf=monthly_contribution_chf,
        random_seed=random_seed,
        ticker_currencies=ticker_currencies)

    dist = engine.simulate(
        plan, prices,
        horizon_days=horizon_days,
        initial_nav_chf=plan.total_value_chf)

    sheets = {
        'Plan': plan.to_dataframe(),
        'Summary': pd.DataFrame([dist.to_summary_dict()]),
        'Percentiles': dist.percentiles(),
        'TerminalDistribution': pd.DataFrame(
            {'terminal_nav': dist.terminal_distribution()})}

    out_path = pathlib.Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    DataExporter().export_to_excel(sheets, str(out_path))
    print(
        f'Monte Carlo report saved to {out_path} '
        f'(strategy={strategy}, n_paths={n_paths}, '
        f'horizon_days={horizon_days}).')


if __name__ == '__main__':
    main()
