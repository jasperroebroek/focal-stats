import numpy as np
import pandas as pd
import pytest

from spatial_stats.grouped_stats import grouped_min, grouped_min_pd

rs = np.random.RandomState(0)


@pytest.fixture
def ind():
    return rs.randint(0, 5, size=(10, 10), dtype=np.uintp)


@pytest.fixture
def v():
    return rs.rand(10, 10)


def test_grouped_min(ind, v):
    min_v = grouped_min(ind, v)

    for i in range(1, int(ind.max()) + 1):
        values_in_group = v[ind == i]
        expected_min = np.nanmin(values_in_group)
        assert np.isclose(min_v[i], expected_min, atol=1e-5)

    assert np.isnan(min_v[0])


def test_grouped_min_pd(ind, v):
    result_df = grouped_min_pd(ind, v)

    # result_df should be a Pandas DataFrame
    assert isinstance(result_df, pd.DataFrame)

    # Check that the mins are correct
    for i in result_df.index:
        values_in_group = v[ind == i]
        expected_min = np.nanmin(values_in_group)
        assert np.isclose(result_df.loc[i, "minimum"], expected_min, atol=1e-5)

    # Check that there are no entries for group 0 (if 0 is not a valid group)
    assert 0 not in result_df.index


# Test with empty arrays
def test_grouped_min_empty():
    ind = np.array([], dtype=np.uintp)
    v = np.array([], dtype=np.float32)
    min_v = grouped_min(ind, v)
    assert min_v.size == 1  # Expect only one group, for 0
    assert np.isnan(min_v[0])


# Test with all NaNs
def test_grouped_min_all_nans():
    ind = np.ones(10, dtype=np.uintp)
    v = np.full(10, np.nan, dtype=np.float32)
    min_v = grouped_min(ind, v)
    assert np.isnan(min_v[1])  # Expect infinity as the min for the group


# Test with a single group
def test_grouped_min_single_group():
    ind = np.ones(10, dtype=np.uintp)
    v = np.arange(10, dtype=np.float32)
    min_v = grouped_min(ind, v)
    assert min_v[1] == v.min()  # Expect the min to be the smallest value in v


# Test with non-contiguous groups
def test_grouped_min_non_contiguous_groups():
    ind = np.array([1, 3, 5], dtype=np.uintp)
    v = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    min_v = grouped_min(ind, v)
    assert min_v[1] == 10.0
    assert min_v[3] == 20.0
    assert min_v[5] == 30.0


# Test with large values
def test_grouped_min_large_values():
    ind = np.array([1, 1], dtype=np.uintp)
    v = np.array([1e38, 1e-38], dtype=np.float32)
    min_v = grouped_min(ind, v)
    assert min_v[1] == 1e-38  # Expect the min to be the smallest value in v
