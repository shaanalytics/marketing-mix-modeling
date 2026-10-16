import numpy as np
import pytest
from src.features.adstock import geometric_adstock, hill_saturation

def test_geometric_adstock_carries_previous_activity():
    np.testing.assert_allclose(geometric_adstock([10, 0, 0], decay=0.5), [10, 5, 2.5])

def test_geometric_adstock_zero_decay_is_identity():
    np.testing.assert_allclose(geometric_adstock([1, 3, 2], decay=0), [1, 3, 2])

def test_geometric_adstock_rejects_invalid_decay():
    with pytest.raises(ValueError):
        geometric_adstock([1, 2], decay=1)

def test_geometric_adstock_rejects_negative_values():
    with pytest.raises(ValueError):
        geometric_adstock([1, -1], decay=0.5)

def test_hill_saturation_is_bounded_and_half_at_half_saturation():
    result = hill_saturation([0, 10, 100], half_saturation=10, slope=1)
    np.testing.assert_allclose(result, [0, 0.5, 100 / 110])
    assert np.all((result >= 0) & (result <= 1))

def test_hill_saturation_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        hill_saturation([1, 2], half_saturation=0)
