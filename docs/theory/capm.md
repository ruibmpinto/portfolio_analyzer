# CAPM — Capital Asset Pricing Model

A theory that says: in equilibrium, an asset's expected return is determined entirely by its sensitivity to the overall market (β). The formula:

```
E[r_i] = RFR + β_i · (E[r_m] − RFR)
```

In words: take the risk-free rate as a baseline, then add a risk premium equal to β times the market's own premium over the risk-free rate.

## Components

- **RFR** — what you can earn for taking no risk
- **E[r_m] − RFR** — the **market risk premium**: how much extra return investors demand for holding the broad market instead of T-bills
- **β** — how much of *that* premium this asset should earn. β=1 means the asset moves 1-for-1 with the market and earns the full market premium. β=2 → twice the premium (and twice the volatility). β=0 → no equity risk, asset should just earn RFR. β<0 → contrarian, expected to earn *less* than RFR.

## How it is used in `get_alpha`

[`src/risk_profile/risk_analyzer.py`](../../src/risk_profile/risk_analyzer.py), function `get_alpha`:

```python
beta = self.get_beta(portfolio_returns, market_returns)
portfolio_annual = ((1 + portfolio_returns.mean()) ** 252 - 1) * 100
market_annual    = ((1 + market_returns.mean()) ** 252 - 1) * 100

# CAPM expected return
expected_return = risk_free_rate + beta * (market_annual - risk_free_rate)

# Jensen's alpha = realised − expected
alpha = portfolio_annual - expected_return
```

The function answers: *given how much market risk this portfolio actually took (β), what return would CAPM predict?* Then alpha is the gap between the portfolio's realised return and that prediction.

- **alpha > 0** → portfolio outperformed what β alone would justify (skill, or luck, or factor exposures CAPM doesn't model)
- **alpha = 0** → exactly what CAPM expects; you got paid for the market risk you took, nothing extra
- **alpha < 0** → underperformed; you took market risk and didn't even get the standard premium for it

## Limits worth knowing

- CAPM is a single-factor model — only market β. Real returns also depend on size, value, momentum, profitability (Fama-French + Carhart factors), so a positive CAPM alpha can just be a tilt to small-cap or value stocks rather than skill.
- Assumes investors are mean-variance optimizers with a single market portfolio, no taxes, free borrowing at RFR — none of which hold strictly.
- β is estimated from past returns and is not stable; estimates change with the window length and choice of benchmark (the `'SPY'` default in `get_portfolio_beta` anchors the whole calculation to the S&P 500).
- For Swiss CHF investors the chosen benchmark and RFR matter — CAPM-vs-SPY for a CHF portfolio mixes equity-risk alpha with FX exposure.

In this codebase CAPM only appears in `get_alpha`. The Sharpe/Sortino/Information ratios elsewhere don't use β; they're variance-based measures, not factor-model based.
