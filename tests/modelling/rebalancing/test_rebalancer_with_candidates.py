"""Tests for Rebalancer with screener-fed strategies + CASH."""

import pandas as pd
import pytest

from src.modelling.rebalancing.rebalancer import Rebalancer
from src.modelling.rebalancing.strategies.base import Strategy
from src.screening.candidate import CandidateTicker
from src.shared.constraints import RebalanceConstraints


class FakeAnalyzer:
    """Minimal PortfolioAnalyzer stub."""

    def __init__(self, holdings_df, prices, data_provider):
        self._h = holdings_df
        self._p = prices
        self.data_provider = data_provider

    def get_holdings_snapshot(self, categories=None):
        return self._h.copy()

    def get_price_panel(self):
        return self._p.copy()


class FakeProvider:
    def __init__(self, prices):
        self._prices = prices

    def get_current_price(self, ticker):
        if ticker not in self._prices:
            raise ValueError(f'no price for {ticker}')
        return self._prices[ticker]


class FakeStrategy(Strategy):
    """Returns fixed target weights for testing.

    Deliberately leaves ``name = ''`` to skip registry
    enrolment (otherwise the test asserting the registry
    contains exactly the four Phase 2 strategies breaks).
    """

    # Intentionally empty -> not registered in strategy_registry
    name = ''

    def __init__(self, target_weights):
        self._tw = target_weights
        # Set a non-registry display name on the instance so
        # RebalancePlan has a meaningful strategy_name.
        self.name = 'fake_phase4_test'

    def propose(
        self, holdings_df, prices, constraints,
        candidates=None, risk_free_rate=None):
        return dict(self._tw)


def _holdings(values_chf, prices_chf):
    rows = []
    total = sum(values_chf.values())
    for t, v in values_chf.items():
        rows.append({
            'ticker': t, 'shares': 1, 'value_chf': v,
            'weight_pct': v / total * 100.0,
            'currency': 'CHF', 'sector': 'X',
            'price_chf': prices_chf[t], 'category': 'core'})
    return pd.DataFrame(rows)


def _make_candidate(ticker, currency='USD'):
    return CandidateTicker(
        ticker=ticker, exchange='NASDAQ',
        currency=currency, sector='Tech',
        market_cap=1.0e11, composite_score=0.7,
        metrics={'sharpe': 1.2, 'max_drawdown': -0.2})


def test_rebalancer_generates_buy_for_new_candidate_ticker():
    """Target weight on a non-held ticker yields a BUY action."""
    holdings = _holdings(
        {'AMZN': 1000.0}, {'AMZN': 200.0})
    prices = pd.DataFrame({'AMZN': [200.0, 201.0, 199.0]})
    provider = FakeProvider(
        {'NEWCO': 100.0, 'USDCHF=X': 0.9})
    analyzer = FakeAnalyzer(holdings, prices, provider)
    strategy = FakeStrategy({'AMZN': 0.5, 'NEWCO': 0.5})
    candidates = [_make_candidate('NEWCO', 'USD')]
    constraints = RebalanceConstraints(
        new_capital_chf=2000.0,
        excluded_categories=(), category_caps={})

    rebalancer = Rebalancer(analyzer, strategy, constraints)
    plan = rebalancer.propose(candidates=candidates)

    actions_by_ticker = {a.ticker: a for a in plan.actions}
    assert 'NEWCO' in actions_by_ticker
    new = actions_by_ticker['NEWCO']
    assert new.action == 'BUY'
    assert new.shares > 0
    assert new.current_wt_pct == pytest.approx(0.0)
    assert new.target_wt_pct == pytest.approx(50.0)


def test_rebalancer_raises_when_candidate_price_missing():
    """Target weight on unknown ticker raises."""
    holdings = _holdings(
        {'AMZN': 1000.0}, {'AMZN': 200.0})
    prices = pd.DataFrame({'AMZN': [200.0, 201.0]})
    provider = FakeProvider({})
    analyzer = FakeAnalyzer(holdings, prices, provider)
    strategy = FakeStrategy({'AMZN': 0.5, 'GHOST': 0.5})
    constraints = RebalanceConstraints(
        new_capital_chf=2000.0,
        excluded_categories=(), category_caps={})

    rebalancer = Rebalancer(analyzer, strategy, constraints)
    with pytest.raises(RuntimeError, match='GHOST'):
        rebalancer.propose(candidates=[])


def test_rebalancer_uses_candidate_currency_for_fx():
    """A USD candidate uses USDCHF=X for the CHF conversion.

    NEWCO native price 100 USD, USDCHF=X = 0.5 -> 50 CHF/share.
    NAV=10000 in AMZN, new_capital=2000 -> target_total=12000.
    AMZN 50% / NEWCO 50% -> AMZN target 6000, NEWCO target
    6000. AMZN current 10000 -> REDUCE 20 shares (4000 CHF
    proceeds). BUY pool = 2000 + 4000 - 0 = 6000 CHF.
    NEWCO underweight gap = 6000 -> alloc 6000 / 50 = 120
    shares for 6000 CHF.
    """
    holdings = _holdings(
        {'AMZN': 10000.0}, {'AMZN': 200.0})
    prices = pd.DataFrame({'AMZN': [200.0, 201.0]})
    provider = FakeProvider(
        {'NEWCO': 100.0, 'USDCHF=X': 0.5})
    analyzer = FakeAnalyzer(holdings, prices, provider)
    strategy = FakeStrategy({'AMZN': 0.5, 'NEWCO': 0.5})
    candidates = [_make_candidate('NEWCO', 'USD')]
    constraints = RebalanceConstraints(
        new_capital_chf=2000.0,
        excluded_categories=(), category_caps={})

    rebalancer = Rebalancer(analyzer, strategy, constraints)
    plan = rebalancer.propose(candidates=candidates)

    new = next(a for a in plan.actions if a.ticker == 'NEWCO')
    assert new.shares == 120
    assert new.est_cost_chf == pytest.approx(6000.0)


def test_rebalancer_emits_cash_action_for_cash_ticker():
    """target_weights['CASH'] -> single CASH action with target CHF."""
    holdings = _holdings(
        {'AMZN': 1000.0}, {'AMZN': 200.0})
    prices = pd.DataFrame({'AMZN': [200.0, 201.0]})
    provider = FakeProvider({})
    analyzer = FakeAnalyzer(holdings, prices, provider)
    # NAV=1000, new_capital=2000 -> target_total=3000.
    # CASH 30% -> est_cost_chf = 900.
    strategy = FakeStrategy({'AMZN': 0.7, 'CASH': 0.3})
    constraints = RebalanceConstraints(
        new_capital_chf=2000.0,
        excluded_categories=(), category_caps={})

    rebalancer = Rebalancer(analyzer, strategy, constraints)
    plan = rebalancer.propose(candidates=None)

    cash_actions = [a for a in plan.actions if a.ticker == 'CASH']
    assert len(cash_actions) == 1
    cash = cash_actions[0]
    assert cash.action == 'CASH'
    assert cash.shares == 0
    assert cash.est_cost_chf == pytest.approx(900.0)
    assert cash.current_wt_pct == pytest.approx(0.0)
    assert cash.target_wt_pct == pytest.approx(30.0)
    assert plan.summary()['total_cash_chf'] == pytest.approx(900.0)


def test_rebalancer_does_not_request_price_for_cash():
    """CASH ticker is never sent to fetch_chf_price."""
    holdings = _holdings(
        {'AMZN': 1000.0}, {'AMZN': 200.0})
    prices = pd.DataFrame({'AMZN': [200.0, 201.0]})
    provider = FakeProvider({})
    analyzer = FakeAnalyzer(holdings, prices, provider)
    strategy = FakeStrategy({'AMZN': 0.5, 'CASH': 0.5})
    constraints = RebalanceConstraints(
        new_capital_chf=1000.0,
        excluded_categories=(), category_caps={})
    rebalancer = Rebalancer(analyzer, strategy, constraints)
    plan = rebalancer.propose(candidates=None)
    assert 'CASH' in {a.ticker for a in plan.actions}


def test_rebalancer_recycles_reduce_proceeds_into_buy_pool():
    """REDUCE proceeds + new_capital - target_cash fund BUYs.

    NAV=10000 in AMZN, new_capital=2000, target
    AMZN 50% / NEWCO 30% / CASH 20%.

    With Phase 2 logic NEWCO would BUY only 2000 (capped at
    new_capital). With Phase 4 recycling: AMZN REDUCEs 4000,
    target_cash = 0.20*12000 = 2400, BUY pool =
    2000 + 4000 - 2400 = 3600 -> NEWCO BUY hits its 3600
    target exactly.
    """
    holdings = _holdings(
        {'AMZN': 10000.0}, {'AMZN': 200.0})
    prices = pd.DataFrame({'AMZN': [200.0, 201.0]})
    # NEWCO priced in CHF at 100 (provider returns CHF directly
    # when candidate currency is CHF).
    provider = FakeProvider({'NEWCO': 100.0})
    analyzer = FakeAnalyzer(holdings, prices, provider)
    strategy = FakeStrategy({
        'AMZN': 0.5, 'NEWCO': 0.3, 'CASH': 0.2})
    candidates = [CandidateTicker(
        ticker='NEWCO', exchange='SIX', currency='CHF',
        sector='Tech', market_cap=1e10, composite_score=0.7,
        metrics={})]
    constraints = RebalanceConstraints(
        new_capital_chf=2000.0,
        excluded_categories=(), category_caps={})

    rebalancer = Rebalancer(analyzer, strategy, constraints)
    plan = rebalancer.propose(candidates=candidates)

    by_ticker = {a.ticker: a for a in plan.actions}
    assert by_ticker['NEWCO'].action == 'BUY'
    assert by_ticker['NEWCO'].est_cost_chf == pytest.approx(
        3600.0)
    assert by_ticker['NEWCO'].shares == 36
    assert by_ticker['AMZN'].action == 'REDUCE'
    assert by_ticker['CASH'].action == 'CASH'
    assert by_ticker['CASH'].est_cost_chf == pytest.approx(
        2400.0)


def test_rebalancer_cash_reservation_caps_buy_pool():
    """target_cash > funding -> BUY pool is 0, not negative.

    NAV=10000, new_capital=2000, target AMZN 30% / B 20% / CASH 50%.
    target_total=12000. AMZN target = 3600, B target = 2400.
    AMZN current=6000 delta=-2400 -> REDUCE.
    B current=4000 delta=-1600 -> REDUCE.
    CASH target = 6000. Funding = 2000 + (REDUCE proceeds)
    - 6000. Even if both REDUCEs land, funding stays below 0
    -> BUY pool clamped to 0 -> no BUYs even if any ticker were
    underweight.
    """
    holdings = _holdings(
        {'AMZN': 6000.0, 'B': 4000.0},
        {'AMZN': 200.0, 'B': 100.0})
    prices = pd.DataFrame({
        'AMZN': [200.0, 201.0],
        'B': [100.0, 101.0]})
    provider = FakeProvider({})
    analyzer = FakeAnalyzer(holdings, prices, provider)
    strategy = FakeStrategy({
        'AMZN': 0.3, 'B': 0.2, 'CASH': 0.5})
    constraints = RebalanceConstraints(
        new_capital_chf=2000.0,
        excluded_categories=(), category_caps={})

    rebalancer = Rebalancer(analyzer, strategy, constraints)
    plan = rebalancer.propose(candidates=None)

    total_buy = sum(
        a.est_cost_chf for a in plan.actions
        if a.action == 'BUY')
    assert total_buy == pytest.approx(0.0)
