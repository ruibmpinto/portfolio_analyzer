"""FIFO realized-gains calculator for closed portfolio lots.

Walks the transaction stream in date order, opens a lot on every buy,
and consumes lots FIFO on every sell. Emits a closed-sub-lot table
with cost basis, proceeds, and realized P&L in both native and CHF.

Split-aware: share counts are normalised to post-today-equivalent units
via ``Portfolio._split_factor`` so FIFO matching lines up across pre-
and post-split rows in the same lot chain.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd

from src.analysis.core.portfolio import Portfolio


# Below broker precision, above float64 residues from proportional
# multi-lot FIFO consumption on fractional-share instruments.
_zero_share_epsilon = 1e-4


@dataclass
class _OpenLot:
    """Working state of an open buy lot during FIFO matching.

    Both share count and cost basis are drained together as sells consume
    the lot; when ``shares_remaining`` falls below ``_zero_share_epsilon``
    the lot is popped from its per-ticker queue.
    """

    ticker: str
    buy_date: datetime
    currency: str
    shares_remaining: float
    cost_basis_remaining: float


class RealizedGains:
    """FIFO matcher for closed lots and realized P&L.

    Instances are stateless from the caller's perspective — every public
    accessor returns computed data derived from ``portfolio.transactions``
    at construction time. The closed-lots frame is built lazily on first
    access and cached; downstream queries filter the cache.

    Attributes:
        portfolio: Source of ``transactions`` (chronologically sorted) and
            the split table used to normalise share counts to post-today-
            equivalent units.
        fx_to_chf: Callable ``(native_amount, currency, date) ->
            chf_amount``. Typically bound to
            ``PortfolioAnalyzer._native_to_base`` so realized P&L is
            converted at the sell-date FX rate.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        fx_to_chf: Callable[[float, str, datetime], float]):
        """Store dependencies; no work runs until an accessor is called."""
        self.portfolio = portfolio
        self.fx_to_chf = fx_to_chf
        self._closed_lots_df: Optional[pd.DataFrame] = None

    def closed_lots(self) -> pd.DataFrame:
        """All closed sub-lots produced by the FIFO walk.

        A single sell that consumes N buys yields N rows.

        Returns:
            DataFrame with columns ``ticker``, ``buy_date``, ``sell_date``,
            ``shares``, ``cost_basis_native``, ``proceeds_native``,
            ``realized_native``, ``realized_chf``, ``days_held``,
            ``currency``. Empty frame (with the same columns) when the
            portfolio has never closed a lot.
        """
        if self._closed_lots_df is None:
            self._closed_lots_df = self._match_fifo()
        return self._closed_lots_df

    def ytd_realized_chf(self, today: datetime) -> float:
        """Sum of ``realized_chf`` for sub-lots closed this year.

        Args:
            today: Cutoff date; the year is derived from it and the
                window is ``[Jan 1 of year, today]``.

        Returns:
            Signed CHF total (positive for net realized gain, negative
            for net realized loss). Zero when no lots have closed this
            year.
        """
        df = self.closed_lots()
        if df.empty:
            return 0.0
        year_start = datetime(today.year, 1, 1)
        # Inclusive on both ends: a sell exactly on year_start or today
        # counts toward YTD.
        in_year = df[
            (df['sell_date'] >= year_start)
            & (df['sell_date'] <= today)]
        return float(in_year['realized_chf'].sum())

    def holding_period_violations(
        self, min_days: int = 180) -> pd.DataFrame:
        """Closed sub-lots held for fewer than ``min_days`` calendar days.

        Args:
            min_days: Threshold in calendar days. Default 180 matches the
                Swiss safe-harbor 6-month holding period.

        Returns:
            Slice of ``closed_lots()`` where ``days_held < min_days``.
        """
        df = self.closed_lots()
        if df.empty:
            return df
        return df[df['days_held'] < min_days]

    # -------------------------------------------------------------------
    # Internal FIFO matching
    # -------------------------------------------------------------------

    def _match_fifo(self) -> pd.DataFrame:
        """Walk transactions once and build the closed-lots frame.

        Only ``buy`` and ``sell`` rows drive lot creation and matching.
        ``dividend`` and ``tax`` rows are explicitly ignored — they
        represent cash events, not share-lot activity. Any other
        operation string is a data error and raises ``RuntimeError``
        rather than being silently skipped.
        """
        # Per-ticker FIFO queue of currently-open lots
        open_lots: Dict[str, List[_OpenLot]] = {}
        # Accumulator: one row per matched (buy_lot, sell) pair
        closed_rows: List[Dict] = []

        for trans in self.portfolio.transactions:
            if trans.operation == 'buy':
                self._open_lot(open_lots, trans)
            elif trans.operation == 'sell':
                self._consume_lots(open_lots, trans, closed_rows)
            elif trans.operation in ('dividend', 'tax'):
                # Cash-only events; no share-lot side effect.
                continue
            else:
                raise RuntimeError(
                    f'Unknown transaction operation '
                    f'{trans.operation!r} for {trans.ticker} on '
                    f'{trans.date.date()}. Expected one of buy, '
                    f'sell, dividend, tax.')

        return pd.DataFrame(closed_rows, columns=[
            'ticker', 'buy_date', 'sell_date', 'shares',
            'cost_basis_native', 'proceeds_native',
            'realized_native', 'realized_chf',
            'days_held', 'currency'])

    def _open_lot(
        self,
        open_lots: Dict[str, List[_OpenLot]],
        trans) -> None:
        """Append a new open lot to the per-ticker FIFO queue.

        Called once per buy transaction. Two conversions happen
        before the lot is stored:

          - Shares: ``trans.amount`` is pre-split as recorded by
            the broker. We normalise it to post-today-equivalent
            shares via ``Portfolio._split_factor`` so later sells
            (also normalised the same way) can be matched against
            this lot even if splits occurred in between.
          - Cost basis: ``Transaction.total_cost`` is negative
            for buys (it represents a cash outflow); we store
            ``abs()`` so the running cost basis is a positive
            number that shrinks toward zero as the lot is
            consumed.

        A buy resolving to zero post-split shares is a malformed
        row — splits travel via ``Portfolio.splits``, not as
        zero-share buys, and synthetic split-buys are stripped by
        ``Portfolio.set_splits`` before reaching us. Raises
        ``RuntimeError`` so the data problem surfaces.

        Args:
            open_lots: Mutable per-ticker FIFO queue. This
                method appends to the list for ``trans.ticker``,
                creating an empty list first if the ticker is
                not yet known.
            trans: The buy transaction driving lot creation.

        Raises:
            RuntimeError: If the post-split share count is zero
                or below the floating-point epsilon (data error).
        """
        # Post-today-equivalent shares so FIFO consumption lines up with
        # later sells regardless of intervening splits.
        shares_post_today = (
            trans.amount
            * self.portfolio._split_factor(trans.ticker, trans.date))
        if shares_post_today <= _zero_share_epsilon:
            raise RuntimeError(
                f'Buy of {trans.ticker} on {trans.date.date()} '
                f'resolves to zero post-split shares '
                f'(amount={trans.amount}). Malformed row.')
        # Fully-loaded cost basis (positive), including fees.
        # Transaction.total_cost is negative for buys; abs() gives outflow.
        cost_basis = abs(trans.total_cost)
        lot = _OpenLot(
            ticker=trans.ticker,
            buy_date=trans.date,
            currency=trans.currency,
            shares_remaining=shares_post_today,
            cost_basis_remaining=cost_basis)
        open_lots.setdefault(trans.ticker, []).append(lot)

    def _consume_lots(
        self,
        open_lots: Dict[str, List[_OpenLot]],
        trans,
        closed_rows: List[Dict]) -> None:
        """Match a sell transaction against earliest-bought open lots.

        Walks the per-ticker FIFO queue head-first, taking shares from
        each open lot until ``trans``'s share count is fully covered.
        For every lot that gets touched (fully or partially), appends
        one row to ``closed_rows`` describing the closed sub-lot, and
        reduces that lot's remaining shares and cost basis in place.
        Lots whose remaining shares fall below ``_zero_share_epsilon`` 
        are popped from the queue.

        Attribution rules:
          - Cost basis is sliced in proportion to the LOT's own
            consumption: consuming half a lot's remaining shares
            removes half of that lot's remaining cost basis.
          - Proceeds are sliced in proportion to the SELL's total
            shares (not the running remainder), so shares across the
            appended rows sum to the sell's total shares and proceeds
            sum to the sell's total net proceeds exactly.

        Args:
            open_lots: Mutable per-ticker FIFO queue. Head of each
                list is the earliest buy. This method pops
                fully-consumed lots and reduces the head lot in place.
            trans: The sell transaction driving the consumption.
            closed_rows: Mutable list of dicts. One dict is appended
                per lot touched by ``trans``.

        Raises:
            RuntimeError: When the sell's total share count is zero
                or exceeds the sum of open lots for the ticker
                (transaction log is inconsistent — no silent
                zero-P&L default).
        """
        # Post-today-equivalent shares being sold and total net proceeds
        # (positive; Transaction.total_cost for a sell is already net
        # of fees).
        total_shares_to_close = (
            trans.amount
            * self.portfolio._split_factor(trans.ticker, trans.date))
        # A zero-share sell is a malformed row: it would silently
        # bypass the shortfall guard below and leave no trace.
        if total_shares_to_close <= _zero_share_epsilon:
            raise RuntimeError(
                f'Sell of {trans.ticker} on {trans.date.date()} '
                f'resolves to zero post-split shares '
                f'(amount={trans.amount}). Malformed row.')
        total_proceeds_native = abs(trans.total_cost)

        shares_remaining_to_close = total_shares_to_close
        lots = open_lots.get(trans.ticker, [])

        while shares_remaining_to_close > _zero_share_epsilon and lots:
            lot = lots[0]
            # Never consume more than the head lot has left
            consumed = min(lot.shares_remaining, shares_remaining_to_close)
            # Cost basis slice: proportional to the lot's consumption
            lot_fraction = consumed / lot.shares_remaining
            cost_slice = lot.cost_basis_remaining * lot_fraction
            # Proceeds slice: proportional to the sell's consumption; uses
            # the ORIGINAL total, not the running one, so shares across
            # rows sum to total_shares_to_close and proceeds sum to
            # total_proceeds_native exactly.
            sell_fraction = consumed / total_shares_to_close
            proceeds_slice = total_proceeds_native * sell_fraction
            realized_native = proceeds_slice - cost_slice
            realized_chf = self.fx_to_chf(
                realized_native, trans.currency, trans.date)
            closed_rows.append({
                'ticker': trans.ticker,
                'buy_date': lot.buy_date,
                'sell_date': trans.date,
                'shares': consumed,
                'cost_basis_native': cost_slice,
                'proceeds_native': proceeds_slice,
                'realized_native': realized_native,
                'realized_chf': realized_chf,
                'days_held': (trans.date - lot.buy_date).days,
                'currency': trans.currency,
            })
            # Drain the lot and the outer sell counter together
            lot.shares_remaining -= consumed
            lot.cost_basis_remaining -= cost_slice
            shares_remaining_to_close -= consumed
            if lot.shares_remaining <= _zero_share_epsilon:
                lots.pop(0)

        # Log-consistency guarantees a sell always has matching buys; if
        # it does not, the transaction data is malformed and we refuse
        # to silently absorb it (no default P&L).
        if shares_remaining_to_close > _zero_share_epsilon:
            # Full precision on the residue so a genuine mismatch is
            # distinguishable from a marginal float-noise miss.
            raise RuntimeError(
                f'Sell of {trans.ticker} on {trans.date.date()} exceeds '
                f'open lots by {shares_remaining_to_close!r} shares '
                f'(post-split; sold {total_shares_to_close!r}). '
                f'Missing buy in transaction log?')
