"""Tests for sample_clayton."""

import numpy as np
import pytest

from src.modelling.monte_carlo.copula import sample_clayton


def test_sample_clayton_shape():
    """Output shape matches (n_samples, n_assets)."""
    rng = np.random.default_rng(0)
    u = sample_clayton(theta=2.0, n_samples=100, n_assets=4,
                       rng=rng)
    assert u.shape == (100, 4)


def test_sample_clayton_uniform_marginals():
    """Each column has approximately uniform[0,1] marginals."""
    rng = np.random.default_rng(0)
    u = sample_clayton(theta=2.0, n_samples=20000, n_assets=3,
                       rng=rng)
    for j in range(3):
        assert abs(u[:, j].mean() - 0.5) < 0.02
        assert abs(u[:, j].std() - 0.2887) < 0.02


def test_sample_clayton_lower_tail_dependence():
    """Higher theta -> stronger co-crash (lower-tail dependence)."""
    rng_low = np.random.default_rng(1)
    rng_high = np.random.default_rng(1)
    u_low = sample_clayton(
        theta=0.5, n_samples=20000, n_assets=2, rng=rng_low)
    u_high = sample_clayton(
        theta=8.0, n_samples=20000, n_assets=2, rng=rng_high)

    joint_low_low = (
        (u_low[:, 0] < 0.05) & (u_low[:, 1] < 0.05)).mean()
    joint_low_high = (
        (u_high[:, 0] < 0.05) & (u_high[:, 1] < 0.05)).mean()
    assert joint_low_high > joint_low_low * 2


def test_sample_clayton_values_strictly_inside_unit():
    """Clipping prevents 0 or 1 exactly."""
    rng = np.random.default_rng(0)
    u = sample_clayton(theta=2.0, n_samples=5000, n_assets=4,
                       rng=rng)
    assert u.min() > 0.0
    assert u.max() < 1.0


def test_sample_clayton_rejects_invalid_theta():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match='theta'):
        sample_clayton(theta=0.0, n_samples=10, n_assets=2,
                       rng=rng)
    with pytest.raises(ValueError, match='theta'):
        sample_clayton(theta=-1.0, n_samples=10, n_assets=2,
                       rng=rng)


def test_sample_clayton_rejects_invalid_sizes():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match='n_samples'):
        sample_clayton(theta=2.0, n_samples=0, n_assets=2,
                       rng=rng)
    with pytest.raises(ValueError, match='n_assets'):
        sample_clayton(theta=2.0, n_samples=10, n_assets=0,
                       rng=rng)


def test_sample_clayton_reproducible_with_same_seed():
    """Same rng seed yields identical output."""
    a = sample_clayton(
        theta=2.0, n_samples=200, n_assets=3,
        rng=np.random.default_rng(42))
    b = sample_clayton(
        theta=2.0, n_samples=200, n_assets=3,
        rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
