"""
Portfolio visualization module.

Provides comprehensive plotting functionality for
portfolio analysis and performance visualization.
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# Use a non-interactive backend so figures are written to
# disk without opening a window — this module is strictly
# file-output.
matplotlib.use('Agg')

plt.rcParams['figure.dpi'] = 180
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['lines.linewidth'] = 1.5


class PortfolioPlotter:
    """
    Handles all portfolio visualization.

    Every plot is *saved* to disk; no figure is ever shown
    interactively. Standalone calls write to
    ``self.output_dir / <plot_name>.png``.

    Attributes:
        colors: ordered colour palette used across plots.
        output_dir: directory where figures land. Defaults to
            ``results/dashboard/{YYYY-MM-DD}/`` resolved
            against the current working directory; override
            at construction time to redirect output.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None):
        """
        Initialise plotter and the output directory.

        Args:
            output_dir: Where to save figures. None (default)
                resolves to ``results/dashboard/{today}/``.
        """
        self.colors = [
            'blue', 'red', 'green', 'orange', 'purple',
            'brown', 'pink', 'gray', 'olive', 'cyan']
        # Default to results/dashboard/<today> when no override
        if output_dir is None:
            today = datetime.now().strftime('%Y-%m-%d')
            output_dir = Path('results') / 'dashboard' / today

        self.output_dir = Path(output_dir)

    def _save_and_close(self, fig, target: Path) -> Path:
        """
        Save `fig` to `target`, creating its parent
        directory if necessary, and close the figure.

        Args:
            fig: Matplotlib figure to save.
            target: Full output path (with extension).

        Returns:
            The resolved Path written.
        """
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(target)
        plt.close(fig)
        # Return
        return target

    def plot_portfolio_value(
        self,
        actual: pd.Series,
        hold: pd.Series,
        ax,
        benchmark: Optional[pd.Series] = None,
        base_currency: str = 'CHF') -> None:
        """
        Plot portfolio values over time into the given axis.

        Args:
            actual: Actual portfolio value series
            hold: Hold strategy portfolio value series
            ax: Matplotlib axis to draw into
            benchmark: Optional benchmark value series
            base_currency: Currency code used for the y-axis
                label (e.g. 'CHF', 'USD'). Default 'CHF'.
        """
        ax.plot(
            actual.index, actual.values,
            label='Actual Portfolio')

        ax.plot(
            hold.index, hold.values,
            label='Hold Strategy')

        if benchmark is not None and not benchmark.empty:
            ax.plot(
                benchmark.index, benchmark.values,
                label='Benchmark', alpha=0.8)

        ax.set_title('Portfolio Value Over Time')
        ax.set_ylabel(f'Portfolio Value ({base_currency})')
        ax.legend()

    def plot_returns(
        self,
        actual_cum: pd.Series,
        hold_cum: pd.Series,
        ax,
        benchmark_cum: Optional[pd.Series] = None) -> None:
        """Plot cumulative TWR (%) series.

        All three inputs are already in percent and start
        near zero; this method just draws them.
        """
        ax.plot(actual_cum.index, actual_cum.values,
                label='Actual Portfolio')
        ax.plot(hold_cum.index, hold_cum.values,
                label='Hold Strategy')
        
        if benchmark_cum is not None and not benchmark_cum.empty:
            ax.plot(benchmark_cum.index, benchmark_cum.values,
                    label='Benchmark', alpha=0.8)

        ax.set_title('Portfolio Returns vs Benchmark (%)')
        ax.set_ylabel('Return (%)')
        ax.legend()

    def plot_volatility(
        self,
        actual_returns: pd.Series,
        hold_returns: pd.Series,
        ax,
        benchmark_returns: Optional[pd.Series] = None,
        window: int = 30
    ) -> None:
        """Plot rolling annualised vol from daily returns."""

        actual_vol = self._calc_rolling_vol(actual_returns, window)
        hold_vol = self._calc_rolling_vol(hold_returns, window)

        ax.plot(actual_vol.index, actual_vol.values,
                label='Actual Portfolio')
        ax.plot(hold_vol.index, hold_vol.values,
                label='Hold Strategy')
        
        if (benchmark_returns is not None and not benchmark_returns.empty):
            bench_vol = self._calc_rolling_vol(benchmark_returns, window)

            ax.plot(bench_vol.index, bench_vol.values,
                    label='Benchmark', alpha=0.8)

        ax.set_title(f'{window}-Day Rolling Volatility (Annualized %)')
        ax.set_ylabel('Volatility (%)')
        ax.legend()

    def plot_risk_return_scatter(
        self,
        actual_returns: pd.Series,
        hold_returns: pd.Series,
        ax,
        benchmark_returns: Optional[pd.Series] = None) -> None:
        """Annualised return vs vol from daily return series."""
        
        actual_ret, actual_vol = self._calc_annual_metrics(
            actual_returns)
        hold_ret, hold_vol = self._calc_annual_metrics(
            hold_returns)

        ax.scatter(actual_vol, actual_ret, s=100,
                   label='Actual Portfolio', alpha=0.7)
        ax.scatter(hold_vol, hold_ret, s=100,
                   label='Hold Strategy', alpha=0.7)
        
        if (benchmark_returns is not None and not benchmark_returns.empty):
            bench_ret, bench_vol = self._calc_annual_metrics(benchmark_returns)

            ax.scatter(bench_vol, bench_ret, s=100,
                       label='Benchmark', alpha=0.7)

        ax.set_title('Risk-Return Analysis')
        ax.set_xlabel('Annualized Volatility (%)')
        ax.set_ylabel('Annualized Return (%)')
        ax.legend()

    def plot_pe_ratio(self, pe_series: pd.Series, ax) -> None:
        """
        Plot portfolio P/E ratio over time into the given axis.

        Args:
            pe_series: P/E ratio time series
            ax: Matplotlib axis to draw into
        """
        # Remove None/NaN values
        pe_clean = pe_series.dropna()

        if not pe_clean.empty:
            ax.plot(
                pe_clean.index, pe_clean.values,
                label='Portfolio P/E Ratio', color='green')
            ax.set_ylabel('P/E Ratio')
            ax.legend()
        else:
            ax.text(
                0.5, 0.5, 'No P/E Data Available',
                ha='center', va='center', transform=ax.transAxes)

        ax.set_title('Portfolio P/E Ratio Over Time')

    def plot_cashflows(
        self,
        dividends: Optional[pd.Series],
        taxes: Optional[pd.Series],
        fees: Optional[pd.Series],
        ax,
        base_currency: str = 'CHF'
    ) -> None:
        """
        Plot cumulative dividends, taxes, and fees as three
        positive-magnitude step lines in the base currency.

        Each series is independently cumsummed on its own
        event index. Taxes carry a non-positive natural sign
        in transaction data, so they are flipped to positive
        magnitude here — the plot displays "money lost to
        taxes", not the signed cash-flow.

        Args:
            dividends: Per-event gross dividend cash flows
                in `base_currency`, indexed by event date.
                Non-negative.
            taxes: Per-event withholding tax cash flows in
                `base_currency`, indexed by event date.
                Non-positive in natural sign; displayed as
                its absolute magnitude.
            fees: Per-event trading + FX fees in
                `base_currency`, indexed by event date.
                Non-negative.
            ax: Matplotlib axis to draw into.
            base_currency: Currency code used for the y-axis
                label (e.g. 'CHF', 'USD').
        """
        # Normalise inputs: None -> empty Series so the
        # downstream emptiness check sees a uniform shape
        if dividends is None:
            dividends = pd.Series(dtype=float)
        if taxes is None:
            taxes = pd.Series(dtype=float)
        if fees is None:
            fees = pd.Series(dtype=float)

        # All-empty short-circuit: leave the axis blank with
        # a placeholder so the dashboard grid stays aligned
        if dividends.empty and taxes.empty and fees.empty:
            ax.text(
                0.5, 0.5, 'No Cashflow Data Available',
                ha='center', va='center',
                transform=ax.transAxes)
            ax.set_title('Cumulative Dividends, Taxes, and Fees')
            return

        # Display semantics: taxes are non-positive in source
        # data; flip to a positive magnitude so all three
        # lines climb together on the same axis
        taxes_pos = taxes.abs()

        # Per-series cumulative sums (chronological)
        series_specs = (
            (dividends, 'Cumulative Dividends', 'green'),
            (taxes_pos, 'Cumulative Taxes Paid', 'red'),
            (fees, 'Cumulative Fees Paid', 'orange'),
        )

        for series, label, color in series_specs:
            # Skip empty series so the legend stays meaningful
            if series.empty:
                continue
            cumulative = series.sort_index().cumsum()
            # Step plot: each event lifts the line at its date
            ax.plot(
                cumulative.index, cumulative.values,
                label=label, color=color,
                drawstyle='steps-post')

        ax.set_title('Cumulative Dividends, Taxes, and Fees')
        ax.set_ylabel(f'Cumulative Cashflow ({base_currency})')
        ax.set_xlabel('Date')
        ax.legend()

    def create_comprehensive_dashboard(
        self,
        actual: pd.Series,
        hold: pd.Series,
        benchmark: Optional[pd.Series] = None,
        actual_returns: Optional[pd.Series] = None,
        hold_returns: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None,
        actual_cum_twr: Optional[pd.Series] = None,
        hold_cum_twr: Optional[pd.Series] = None,
        benchmark_cum_twr: Optional[pd.Series] = None,
        pe_series: Optional[pd.Series] = None,
        dividends: Optional[pd.Series] = None,
        taxes: Optional[pd.Series] = None,
        fees: Optional[pd.Series] = None,
        save_path: Optional[Union[str, Path]] = None,
        base_currency: str = 'CHF'
    ) -> Path:
        """Render and save the 2x3 portfolio dashboard.

        Subplots:
        1. Portfolio value over time (NAV)
        2. Cumulative TWR vs benchmark (%)
        3. P/E ratio over time
        4. Rolling annualised vol (from daily returns)
        5. Risk-return scatter (from daily returns)
        6. Cumulative dividends, taxes, and fees
           (base currency, all positive magnitudes)

        Plots 1 and 6 take NAV / per-event cashflow Series;
        plots 2, 4, 5 take pre-computed daily-return and
        cumulative TWR series, since correct TWR computation
        requires cash-flow data the plotter does not have
        access to.

        Returns:
            Path of the written PNG.
        """
        # 2 rows x 3 cols at (6, 6) per cell => (18, 12) total
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)
        ) = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1: Portfolio values (NAV)
        self.plot_portfolio_value(
            actual=actual, hold=hold, ax=ax1,
            benchmark=benchmark,
            base_currency=base_currency)

        # Plot 2: Cumulative TWR
        self.plot_returns(
            actual_cum=actual_cum_twr,
            hold_cum=hold_cum_twr, ax=ax2,
            benchmark_cum=benchmark_cum_twr)

        # Plot 3: P/E ratio
        if pe_series is not None:
            self.plot_pe_ratio(pe_series, ax=ax3)
        else:
            ax3.text(
                0.5, 0.5, 'No P/E Data',
                ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Portfolio P/E Ratio')

        # Plot 4: Rolling volatility (from daily returns)
        self.plot_volatility(
            actual_returns=actual_returns,
            hold_returns=hold_returns, ax=ax4,
            benchmark_returns=benchmark_returns)

        # Plot 5: Risk-return scatter (from daily returns)
        self.plot_risk_return_scatter(
            actual_returns=actual_returns,
            hold_returns=hold_returns, ax=ax5,
            benchmark_returns=benchmark_returns)

        # Plot 6: Cumulative dividends, taxes, and fees.
        # plot_cashflows internally handles the all-empty
        # placeholder, so no wrapping branch is needed.
        self.plot_cashflows(
            dividends=dividends, taxes=taxes, fees=fees,
            ax=ax6, base_currency=base_currency)

        # Force every subplot's plotting area to be square,
        # independent of title / label / legend overhead.
        for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
            ax.set_box_aspect(1)

        # Thin x-axis ticks on the time-series subplots
        # (ax5 is the risk-return scatter — no date axis).
        for ax in (ax1, ax2, ax3, ax4, ax6):
            locator = AutoDateLocator(maxticks=6)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))

        # Caller-provided path wins; otherwise default name
        # inside `self.output_dir`. 
        target = (Path(save_path) if save_path is not None
                  else self.output_dir / 'dashboard.png')
        
        return self._save_and_close(fig, target)

    def _calc_rolling_vol(self, returns: pd.Series, window: int) -> pd.Series:
        """Rolling annualised vol (%) from a daily return series."""

        return (returns.rolling(window=window).std() * np.sqrt(252) * 100)

    def _calc_annual_metrics(self, returns: pd.Series):
        """Annualised return and vol (%) from daily returns."""

        annual_return = ((1 + returns.mean()) ** 252 - 1) * 100
        annual_vol = returns.std() * np.sqrt(252) * 100

        return annual_return, annual_vol
