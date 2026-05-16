"""Tests for src.shared.constraints."""

import dataclasses

import pytest

from src.shared.constraints import (
    RebalanceConstraints, default_constraints)


def test_default_factory_returns_constraints():
    """default_constraints() returns a RebalanceConstraints."""
    c = default_constraints()
    assert isinstance(c, RebalanceConstraints)


def test_default_field_values():
    """Defaults match Contract 4 in the design spec."""
    c = default_constraints()
    assert c.max_weight_default == 0.15
    assert c.category_caps == {'commodity': 0.05}
    assert c.excluded_categories == ('speculative',)
    assert c.min_position_pct == 0.5
    assert c.swiss_tax_filter is True
    assert c.new_capital_chf == 0.0
    assert c.rebalance_band_chf == (-200.0, 50.0)


def test_constraints_are_frozen():
    """Mutating a field raises FrozenInstanceError."""
    c = default_constraints()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.max_weight_default = 0.25


def test_per_instance_category_caps_isolated():
    """Each default_constraints() call gets its own dict.

    Guards against the classic mutable-default bug.
    """
    a = default_constraints()
    b = default_constraints()
    assert a.category_caps is not b.category_caps


def test_category_caps_is_read_only():
    """Mutating category_caps must raise TypeError.

    Proves the MappingProxyType wrapper makes the dict
    read-only and that frozen=True alone is not enough to
    protect mutable members.
    """
    c = default_constraints()
    with pytest.raises(TypeError):
        c.category_caps['commodity'] = 0.99
    with pytest.raises(TypeError):
        c.category_caps['new_cat'] = 0.10
