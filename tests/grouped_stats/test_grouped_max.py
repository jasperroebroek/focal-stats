import numpy as np
import pandas as pd
import pytest

from focal_stats.grouped_stats import (
    grouped_max,
    grouped_max_pd,
)

rs = np.random.RandomState(0)


@pytest.fixture
def ind():
    return rs.randint(0, 5, size=(10, 10), dtype=np.uintp)


@pytest.fixture
def v():
    return rs.rand(10, 10)


def test_grouped_max(ind, v):
    max_v = grouped_max(ind, v)

    for i in range(1, int(ind.max()) + 1):
        values_in_group = v[ind == i]
        expected_max = np.nanmax(values_in_group)
        assert np.isclose(max_v[i], expected_max, atol=1e-5)

    assert np.isnan(max_v[0])


def test_grouped_max_pd(ind, v):
    result_df = grouped_max_pd(ind, v)

    assert isinstance(result_df, pd.DataFrame)

    for i in result_df.index:
        values_in_group = v[ind == i]
        expected_max = np.nanmax(values_in_group)
        assert np.isclose(result_df.loc[i, "maximum"], expected_max, atol=1e-5)

    assert 0 not in result_df.index


def test_grouped_max_empty():
    ind = np.array([], dtype=np.uintp)
    v = np.array([], dtype=np.float32)
    max_v = grouped_max(ind, v)
    assert max_v.size == 1
    assert np.isnan(max_v[0])  # Assuming that the function returns -inf for groups with no entries


def test_grouped_max_all_nans():
    ind = np.ones(10, dtype=np.uintp)
    v = np.full(10, np.nan, dtype=np.float32)
    max_v = grouped_max(ind, v)
    assert np.isnan(max_v[1])  # Assuming that the function returns -inf for groups with no valid (non-NaN) entries


def test_grouped_max_single_group():
    ind = np.ones(10, dtype=np.uintp)
    v = np.arange(10, dtype=np.float32)
    max_v = grouped_max(ind, v)
    assert max_v[1] == v.max()


def test_grouped_max_non_contiguous_groups():
    ind = np.array([1, 3, 5], dtype=np.uintp)
    v = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    max_v = grouped_max(ind, v)
    assert max_v[1] == 10.0
    assert max_v[3] == 20.0
    assert max_v[5] == 30.0


def test_grouped_max_large_values():
    ind = np.array([1, 1], dtype=np.uintp)
    v = np.array([1e38, 1e-38], dtype=np.float32)
    max_v = grouped_max(ind, v)
    assert max_v[1] == 1e38  # Expect the max to be the largest value in v
