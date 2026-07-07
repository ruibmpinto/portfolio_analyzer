"""Validate ``TransactionCostModel.fees_breakdown`` against Degiro data.

Loads the newest Degiro transactions CSV, samples 10 buys and 10
sells at random (seeded, reproducible), and compares the fees the
model predicts against the fees Degiro actually charged.

Ground truth: the ``Transaction and/or third party fees CHF`` column
in the Degiro trades export (already CHF-denominated; sign is
negative for a debit — we take absolute value).

Modeled fee: ``TransactionCostModel().fees_breakdown()`` with:

- ``fx_rate_to_chf`` derived per-row as ``Value CHF / Local value``
  (i.e., 1 / Degiro's ``Exchange rate`` column) so the trade
  currency is correctly converted to CHF.
- ``commission_fx_rate_to_chf`` set to the EUR/CHF rate at trade
  date. For EUR-denominated trades this equals ``fx_rate_to_chf``;
  for USD/GBP trades EUR/CHF is fetched from yfinance
  (``EURCHF=X`` daily closes, cached in memory).

Run from the project root:
    python -m src.user_scripts.validate_degiro_fees
"""

import math
import random
import sys
import pathlib
from typing import Dict, List, Tuple

import pandas as pd


root_dir = str(pathlib.Path(__file__).resolve().parents[2])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


from src.analysis.loaders.csv_loader import (
    default_degiro_path, default_isin_map_path)


# Statement-side lines that Degiro charges alongside a trade
# (matched by Order Id) but reports OUTSIDE the transactions CSV.
# Signed sums of these rows are added to the actual fee so the
# validation compares like for like against the model.
statement_side_fee_descriptions = (
    'stamp duty', 'transaction tax', 'financial transaction tax',
    'french ftt', 'italian ftt', 'ftt')
from src.modelling.monte_carlo.cost_model import (
    TransactionCostModel, degiro_default_cost_model)


def main():
    """Load, sample, compute, and print the validation report."""
    csv_path = default_degiro_path('transactions')
    if csv_path is None or not csv_path.exists():
        raise SystemExit(
            'No Degiro transactions CSV found under '
            'reports/degiro/transactions/.')
    stmt_path = default_degiro_path('statement')
    if stmt_path is None or not stmt_path.exists():
        raise SystemExit(
            'No Degiro account statement found under '
            'reports/degiro/statements/.')
    print(f'Loading Degiro transactions: {csv_path.name}')
    print(f'Loading Degiro statement   : {stmt_path.name}')

    trades = _load_degiro_orders(str(csv_path))
    statement_fees = _load_statement_side_fees(str(stmt_path))
    for o in trades:
        o['actual_fee_chf'] += statement_fees.get(o['order_id'], 0.0)
    print(f'Aggregated to {len(trades)} orders.')

    sample_size_per_side = 10
    random_seed = 42
    alert_threshold_bps = 3.0

    sampled = _sample_orders(
        trades, sample_size_per_side, random_seed)
    print(
        f'Sampled {len(sampled)} rows '
        f'(seed={random_seed}, '
        f'{sample_size_per_side} per side).\n')

    dates_needed = sorted({r['date'] for r in sampled})
    eur_chf_by_date = _fetch_eur_chf_at(dates_needed)

    cost_model = degiro_default_cost_model()
    rows = [
        _row_diagnostics(o, cost_model, eur_chf_by_date)
        for o in sampled]
    _print_table(rows, alert_threshold_bps)
    print()
    _print_summary(rows, alert_threshold_bps)


def _load_degiro_orders(csv_path: str) -> List[Dict]:
    """Aggregate raw fills into one dict per Order ID.

    Mirrors the trade-side logic of ``CSVLoader._parse_degiro_trades``
    but keeps the CHF fee and the row's exchange rate — both of which
    the ``Transaction`` object drops.
    """
    with open(default_isin_map_path, 'r') as f:
        import json
        isin_to_ticker = json.load(f)
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Date', 'Quantity'])
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df['Order ID'] = df['Order ID'].fillna(df[df.columns[-1]])
    df = df.dropna(subset=['Order ID'])
    ccy_col = df.columns[df.columns.get_loc('Price') + 1]
    out: List[Dict] = []
    for oid, group in df.groupby('Order ID', sort=False):
        anchor = group.iloc[0]
        isin = str(anchor.get('ISIN') or '').strip()
        if not isin or isin not in isin_to_ticker:
            continue
        ticker = isin_to_ticker[isin]
        if ticker is None:
            continue
        # Aggregate fills within the order
        qtys = group['Quantity'].astype(float).tolist()
        prices = group['Price'].astype(float).tolist()
        currencies = {
            str(c).strip().upper()
            for c in group[ccy_col].dropna()}
        if len(currencies) != 1:
            continue
        currency = currencies.pop()
        # Degiro reports LSE small-price stocks in pence (GBX). The
        # CSV's ``Exchange rate`` column is also in pence-per-CHF,
        # so both prices AND the FX rate must be divided by 100 to
        # normalise everything into GBP.
        gbx_to_gbp = currency == 'GBX'
        if gbx_to_gbp:
            currency = 'GBP'
            prices = [p / 100.0 for p in prices]
        total_qty = sum(qtys)
        gross_native = sum(
            abs(q) * p for q, p in zip(qtys, prices))
        if total_qty == 0:
            continue
        avg_price = gross_native / abs(total_qty)
        side = 'buy' if total_qty > 0 else 'sell'
        # Fee and exchange rate: sum the fee column (CHF, may be
        # NaN on partial fills), grab the first nonzero xr.
        fee_col = 'Transaction and/or third party fees CHF'
        fees = group[fee_col].fillna(0.0).astype(float).tolist()
        fee_chf = abs(sum(fees))
        xrs = group['Exchange rate'].dropna().astype(float).tolist()
        if not xrs or xrs[0] <= 0:
            continue
        exchange_rate_native_per_chf = xrs[0]
        if gbx_to_gbp:
            exchange_rate_native_per_chf /= 100.0
        out.append({
            'order_id': oid,
            'date': anchor['Date'].to_pydatetime().date(),
            'ticker': ticker,
            'side': side,
            'currency': currency,
            'shares': abs(total_qty),
            'price': avg_price,
            'trade_value_native': abs(total_qty) * avg_price,
            'exchange_rate_native_per_chf': (
                exchange_rate_native_per_chf),
            'actual_fee_chf': fee_chf,
        })
    return out


def _load_statement_side_fees(stmt_path: str) -> Dict[str, float]:
    """Return ``{Order Id -> stamp/FTT in CHF}`` from the statement.

    Degiro reports UK Stamp Duty and FR/IT FTT as separate
    statement rows tied to the trade's Order Id; the transactions
    CSV excludes them. Collected here so the actual fee reflects
    the full trade cost.
    """
    df = pd.read_csv(stmt_path)
    desc = df['Description'].astype(str).str.lower()
    pattern = '|'.join(statement_side_fee_descriptions)
    hits = df[desc.str.contains(pattern, na=False, regex=True)]
    hits = hits.dropna(subset=[hits.columns[-1]])
    # In the Degiro statement schema the 'Change' column carries
    # the currency (e.g. 'CHF') and the *next* unnamed column
    # carries the signed amount. Order Id is the last column.
    amount_col = hits.columns[hits.columns.get_loc('Change') + 1]
    oid_col = hits.columns[-1]
    return {
        row[oid_col]: abs(float(row[amount_col]))
        for _, row in hits.iterrows()}


def _sample_orders(
        orders: List[Dict], k: int, seed: int) -> List[Dict]:
    """Split by side, sample ``k`` per side, return the combined list."""
    rng = random.Random(seed)
    buys = [o for o in orders if o['side'] == 'buy']
    sells = [o for o in orders if o['side'] == 'sell']
    return (rng.sample(buys, min(k, len(buys)))
            + rng.sample(sells, min(k, len(sells))))


def _fetch_eur_chf_at(dates) -> Dict:
    """Return EUR/CHF close on/before each date via yfinance.

    Downloads once, indexes on date, and forward-fills over weekends
    so a Saturday trade uses Friday's close.
    """
    if not dates:
        return {}
    import yfinance as yf
    start = min(dates) - pd.Timedelta(days=7)
    end = max(dates) + pd.Timedelta(days=1)
    hist = yf.download(
        'EURCHF=X', start=str(start), end=str(end),
        progress=False, auto_adjust=False)
    if hist is None or hist.empty:
        raise RuntimeError(
            'Failed to fetch EURCHF=X from yfinance.')
    closes = hist['Close']
    if hasattr(closes, 'columns'):
        closes = closes.iloc[:, 0]
    closes = closes.sort_index()
    # Reindex over every calendar day and forward-fill
    daily = closes.reindex(
        pd.date_range(closes.index.min(), closes.index.max(),
                      freq='D')).ffill()
    out = {}
    for d in dates:
        ts = pd.Timestamp(d)
        if ts in daily.index:
            out[d] = float(daily.loc[ts])
        else:
            available = daily[daily.index <= ts]
            if available.empty:
                raise RuntimeError(
                    f'No EURCHF=X quote on or before {d}.')
            out[d] = float(available.iloc[-1])
    return out


def _row_diagnostics(
        order: Dict,
        cost_model: TransactionCostModel,
        eur_chf_by_date: Dict) -> Dict:
    """Model this order and diff against the CSV's CHF fee."""
    native_per_chf = order['exchange_rate_native_per_chf']
    fx_rate_to_chf = 1.0 / native_per_chf
    if order['currency'] == 'EUR':
        commission_fx = fx_rate_to_chf
    elif order['currency'] == 'CHF':
        commission_fx = None
    else:
        commission_fx = eur_chf_by_date[order['date']]
    breakdown = cost_model.fees_breakdown(
        shares=order['shares'],
        trade_value_native=order['trade_value_native'],
        currency=order['currency'],
        ticker=order['ticker'],
        side=order['side'],
        fx_rate_to_chf=fx_rate_to_chf,
        commission_fx_rate_to_chf=commission_fx)
    modeled = float(breakdown['total_chf'])
    actual = order['actual_fee_chf']
    delta = modeled - actual
    value_chf = order['trade_value_native'] * fx_rate_to_chf
    delta_bps = (delta / value_chf * 1e4) if value_chf > 0 else 0.0
    return {
        'date': order['date'].isoformat(),
        'ticker': order['ticker'],
        'side': order['side'],
        'currency': order['currency'],
        'shares': order['shares'],
        'value_chf': value_chf,
        'actual': actual,
        'modeled': modeled,
        'delta': delta,
        'delta_bps': delta_bps,
        'breakdown': breakdown,
    }


def _print_table(rows: List[Dict], threshold: float) -> None:
    """Fixed-width, terminal-friendly per-row report."""
    header = (
        f'{"Date":<12} {"Ticker":<12} {"Side":<5} {"Ccy":<4} '
        f'{"Shares":>10} {"Value CHF":>12} {"Actual":>10} '
        f'{"Modeled":>10} {"Delta":>10} {"delta_bps":>10} '
        f'{"Flag":>5}')
    print(header)
    print('-' * len(header))
    for r in sorted(rows, key=lambda r: (r['side'], r['date'])):
        flag = '' if abs(r['delta_bps']) <= threshold else '!'
        print(
            f'{r["date"]:<12} {r["ticker"]:<12} '
            f'{r["side"]:<5} {r["currency"]:<4} '
            f'{r["shares"]:>10.4f} '
            f'{r["value_chf"]:>12,.2f} {r["actual"]:>10.4f} '
            f'{r["modeled"]:>10.4f} {r["delta"]:>10.4f} '
            f'{r["delta_bps"]:>10.2f} {flag:>5}')


def _print_summary(rows: List[Dict], threshold: float) -> None:
    """Aggregate diagnostics: mean/median/max delta, correlation."""
    if not rows:
        print('No rows sampled; nothing to summarise.')
        return
    deltas_bps = [r['delta_bps'] for r in rows]
    abs_deltas = [abs(d) for d in deltas_bps]
    mean_abs = sum(abs_deltas) / len(abs_deltas)
    median_abs = sorted(abs_deltas)[len(abs_deltas) // 2]
    max_abs = max(abs_deltas)
    flagged = sum(1 for d in abs_deltas if d > threshold)
    corr = _pearson(
        [r['actual'] for r in rows],
        [r['modeled'] for r in rows])
    print('Summary')
    print(f'  rows sampled           : {len(rows)}')
    print(f'  mean |delta_bps|       : {mean_abs:.2f}')
    print(f'  median |delta_bps|     : {median_abs:.2f}')
    print(f'  max |delta_bps|        : {max_abs:.2f}')
    print(
        f'  flagged (>{threshold:.0f} bps)         '
        f': {flagged} / {len(rows)}')
    print(f'  actual vs modelled r   : {corr:.4f}')
    _print_by_venue_class(rows)


def _print_by_venue_class(rows: List[Dict]) -> None:
    """Per-venue-class summary — Degiro fees differ sharply by row."""
    six_rows = [r for r in rows if r['currency'] == 'CHF']
    eur_rows = [r for r in rows if r['currency'] == 'EUR']
    usd_rows = [r for r in rows if r['currency'] == 'USD']
    print('  by venue class:')
    for label, subset in (
            ('SIX', six_rows), ('EUR', eur_rows),
            ('USD', usd_rows)):
        if subset:
            mean = _mean([r['delta_bps'] for r in subset])
            print(
                f'    {label:<8} ({len(subset):>2d} rows) '
                f'mean delta_bps = {mean:+.2f}')


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
