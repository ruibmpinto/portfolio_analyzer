"""Validate ``TransactionCostModel.fees_breakdown`` against IBKR statements.

Loads the newest IBKR activity statement, samples 10 buys and 10 sells
at random (seeded, reproducible), and compares the fees the model
predicts against the fees IBKR actually charged.

Ground truth: ``Transaction.fee + Transaction.auto_fx_fee`` for each
sampled row (the IBKR loader collapses commission + regulatory +
exchange + tax into ``fee``; FX conversion sits in ``auto_fx_fee``).

Modeled fee: ``TransactionCostModel().fees_breakdown()`` called with
``fx_rate_to_chf=1.0`` so the ``*_chf`` output fields carry native-
currency values — a direct comparison to IBKR without FX drift.

Run from the project root:
    python -m src.user_scripts.validate_ibkr_fees
"""

import math
import random
import sys
import pathlib
from typing import Dict, List, Tuple


# Ensure the repo root is on sys.path when invoked as a plain script.
root_dir = str(pathlib.Path(__file__).resolve().parents[2])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


from src.analysis.core.transaction import Transaction
from src.analysis.loaders.csv_loader import (
    CSVLoader, default_ibkr_statement_path)
from src.modelling.monte_carlo.cost_model import (
    TransactionCostModel, ibkr_default_cost_model)
from src.shared.venue_taxes import lookup_venue


sample_size_per_side = 10
random_seed = 42
# Flag rows whose modelled fee misses reality by more than this many
# basis points of trade value; below this the model is judged OK.
alert_threshold_bps = 3.0


def main():
    """Load, sample, compute, and print the validation report."""
    statement_path = default_ibkr_statement_path()
    if statement_path is None or not statement_path.exists():
        raise SystemExit(
            'No IBKR statement found under reports/ibkr/statements/.')
    print(f'Loading IBKR statement: {statement_path.name}')

    loader = CSVLoader()
    transactions = loader.load_csv_ibkr(str(statement_path))
    print(f'Loaded {len(transactions)} transactions.')

    sampled = _sample_transactions(
        transactions, sample_size_per_side, random_seed)
    print(
        f'Sampled {len(sampled)} rows '
        f'(seed={random_seed}, {sample_size_per_side} per side).\n')

    cost_model = ibkr_default_cost_model()
    rows = [_row_diagnostics(t, cost_model) for t in sampled]
    _print_table(rows)
    print()
    _print_summary(rows)


def _sample_transactions(
        transactions: List[Transaction],
        k: int,
        seed: int) -> List[Transaction]:
    """Split by side, sample ``k`` per side, return the combined list.

    Skips zero-value trades so per-share fees have a denominator; a
    corporate action row that arrives as a 0-price sell shouldn't
    corrupt the sample.
    """
    rng = random.Random(seed)
    buys = [
        t for t in transactions
        if t.operation == 'buy' and t.amount * t.price > 0]
    sells = [
        t for t in transactions
        if t.operation == 'sell' and t.amount * t.price > 0]
    buy_sample = rng.sample(buys, min(k, len(buys)))
    sell_sample = rng.sample(sells, min(k, len(sells)))
    return buy_sample + sell_sample


def _actual_native_fee(t: Transaction) -> float:
    """Ground-truth fee reported by IBKR, in the trade's native currency."""
    return float(t.fee) + float(t.auto_fx_fee)


def _modeled_native_fee(
        t: Transaction,
        cost_model: TransactionCostModel) -> Tuple[float, Dict]:
    """Modelled fee in native currency plus the component breakdown.

    ``fx_rate_to_chf=1.0`` intentional: it makes the ``*_chf`` output
    fields carry native-currency amounts, so we compare like for like
    against ``Transaction.fee`` (also in native).
    """
    trade_value_native = float(t.amount) * float(t.price)
    breakdown = cost_model.fees_breakdown(
        shares=float(t.amount),
        trade_value_native=trade_value_native,
        currency=t.currency,
        ticker=t.ticker,
        side=t.operation,
        fx_rate_to_chf=1.0)
    return float(breakdown['total_chf']), breakdown


def _row_diagnostics(
        t: Transaction,
        cost_model: TransactionCostModel) -> Dict:
    """Bundle every field the table + summary need for one transaction."""
    trade_value = float(t.amount) * float(t.price)
    actual = _actual_native_fee(t)
    modeled, breakdown = _modeled_native_fee(t, cost_model)
    delta = modeled - actual
    delta_bps = (delta / trade_value * 1e4) if trade_value > 0 else 0.0
    return {
        'date': t.date.date().isoformat(),
        'ticker': t.ticker,
        'side': t.operation,
        'currency': t.currency,
        'shares': float(t.amount),
        'price': float(t.price),
        'trade_value': trade_value,
        'actual': actual,
        'modeled': modeled,
        'delta': delta,
        'delta_bps': delta_bps,
        'is_us': lookup_venue(t.ticker).is_us,
        'breakdown': breakdown,
    }


def _print_table(rows: List[Dict]) -> None:
    """Fixed-width, terminal-friendly per-row report."""
    header = (
        f'{"Date":<12} {"Ticker":<10} {"Side":<5} {"Ccy":<4} '
        f'{"Shares":>10} {"Value":>12} {"Actual":>10} '
        f'{"Modeled":>10} {"Delta":>10} {"Δ bps":>8} {"Flag":>5}')
    print(header)
    print('-' * len(header))
    for r in sorted(rows, key=lambda r: (r['side'], r['date'])):
        flag = '' if abs(r['delta_bps']) <= alert_threshold_bps else '!'
        print(
            f'{r["date"]:<12} {r["ticker"]:<10} {r["side"]:<5} '
            f'{r["currency"]:<4} {r["shares"]:>10.4f} '
            f'{r["trade_value"]:>12,.2f} {r["actual"]:>10.4f} '
            f'{r["modeled"]:>10.4f} {r["delta"]:>10.4f} '
            f'{r["delta_bps"]:>8.2f} {flag:>5}')


def _print_summary(rows: List[Dict]) -> None:
    """Aggregate diagnostics — mean / median / max delta, correlation."""
    if not rows:
        print('No rows sampled; nothing to summarise.')
        return
    deltas_bps = [r['delta_bps'] for r in rows]
    abs_deltas = [abs(d) for d in deltas_bps]
    mean_abs = sum(abs_deltas) / len(abs_deltas)
    median_abs = sorted(abs_deltas)[len(abs_deltas) // 2]
    max_abs = max(abs_deltas)
    flagged = sum(
        1 for d in abs_deltas if d > alert_threshold_bps)
    corr = _pearson(
        [r['actual'] for r in rows],
        [r['modeled'] for r in rows])
    print('Summary')
    print(f'  rows sampled           : {len(rows)}')
    print(f'  mean |delta_bps|       : {mean_abs:.2f}')
    print(f'  median |delta_bps|     : {median_abs:.2f}')
    print(f'  max |delta_bps|        : {max_abs:.2f}')
    print(
        f'  flagged (>{alert_threshold_bps:.0f} bps)         '
        f': {flagged} / {len(rows)}')
    print(f'  actual vs modelled r   : {corr:.4f}')
    _print_split_by_us(rows)


def _print_split_by_us(rows: List[Dict]) -> None:
    """Per-venue-class summary — US regulatory fees differ from RoW."""
    us_deltas = [r['delta_bps'] for r in rows if r['is_us']]
    other_deltas = [r['delta_bps'] for r in rows if not r['is_us']]
    print('  by venue class:')
    if us_deltas:
        print(
            f'    US       ({len(us_deltas):>2d} rows) '
            f'mean delta_bps = {_mean(us_deltas):+.2f}')
    if other_deltas:
        print(
            f'    non-US   ({len(other_deltas):>2d} rows) '
            f'mean delta_bps = {_mean(other_deltas):+.2f}')


def _mean(xs: List[float]) -> float:
    """Arithmetic mean; guards against empty input."""
    return sum(xs) / len(xs) if xs else 0.0


def _pearson(xs: List[float], ys: List[float]) -> float:
    """Sample Pearson correlation; returns nan on degenerate inputs."""
    n = len(xs)
    if n < 2:
        return float('nan')
    mx = _mean(xs)
    my = _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(dx2 * dy2)
    return num / denom if denom > 0 else float('nan')


if __name__ == '__main__':
    main()
