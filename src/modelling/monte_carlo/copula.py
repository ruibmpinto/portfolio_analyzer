"""Clayton copula sampler.

Pure function: given a Clayton parameter theta > 0 and a
random generator, draws joint uniforms whose marginals are
uniform[0, 1] but exhibit lower-tail dependence proportional
to theta. Used by MonteCarloEngine to drive co-crash structure
between equity tickers without inflating upper-tail dependence.

Algorithm: Marshall-Olkin. Shared frailty V ~ Gamma(1/theta, 1),
independent exponentials E_i ~ Exp(1) per asset, then
U_i = (1 + E_i / V) ^ (-1/theta).
"""

import numpy as np


u_clip_floor = 1e-7
u_clip_ceil = 1.0 - 1e-7


def sample_clayton(
    theta: float,
    n_samples: int,
    n_assets: int,
    rng: np.random.Generator) -> np.ndarray:
    """Draw joint uniforms with Clayton lower-tail dependence.

    Args:
        theta: Clayton parameter, > 0. Higher = stronger
            co-crash dependence.
        n_samples: Number of joint draws.
        n_assets: Number of asset dimensions per draw.
        rng: NumPy random generator (e.g. np.random.default_rng).

    Returns:
        Array (n_samples, n_assets) of values in
        [u_clip_floor, u_clip_ceil].

    Raises:
        ValueError: theta <= 0, n_samples < 1, or n_assets < 1.
    """
    if theta <= 0:
        raise ValueError(
            f'sample_clayton: theta must be positive, '
            f'got {theta}.')
    if n_samples < 1:
        raise ValueError(
            f'sample_clayton: n_samples must be >= 1, '
            f'got {n_samples}.')
    if n_assets < 1:
        raise ValueError(
            f'sample_clayton: n_assets must be >= 1, '
            f'got {n_assets}.')
    inv_theta = 1.0 / theta
    neg_inv_theta = -1.0 / theta
    v = rng.gamma(inv_theta, 1.0, size=(n_samples, 1))
    e = rng.exponential(1.0, size=(n_samples, n_assets))
    u = (1.0 + e / v) ** neg_inv_theta
    return np.clip(u, u_clip_floor, u_clip_ceil)
