import numpy as np
import pandas as pd
import pytest

from focal_stats.grouped_stats import grouped_mean_std_pd, grouped_std, grouped_std_pd

rs = np.random.RandomState(0)


@pytest.fixture
def ind():
    return rs.randint(0, 5, size=(10, 10), dtype=np.uintp)


@pytest.fixture
def v():
    return rs.rand(10, 10)


def test_grouped_std(ind, v):
    std_v = grouped_std(ind, v)

    for i in range(1, int(ind.max()) + 1):
        values_in_group = v[ind == i]
        expected_std = np.nanstd(values_in_group)
        assert np.isclose(std_v[i], expected_std, atol=1e-5)

    assert np.isnan(std_v[0])  # Assuming the function returns NaN for groups with no entries


def test_grouped_std_pd(ind, v):
    result_df = grouped_std_pd(ind, v)

    assert isinstance(result_df, pd.DataFrame)

    for i in result_df.index:
        values_in_group = v[ind == i]
        expected_std = np.nanstd(values_in_group)
        assert np.isclose(result_df.loc[i, "std"], expected_std, atol=1e-5)

    assert 0 not in result_df.index


def test_grouped_mean_std_pd(ind, v):
    result_df = grouped_mean_std_pd(ind, v)

    assert isinstance(result_df, pd.DataFrame)

    for i in result_df.index:
        values_in_group = v[ind == i]
        expected_std = np.nanstd(values_in_group)
        assert np.isclose(result_df.loc[i, "std"], expected_std, atol=1e-5)

    assert 0 not in result_df.index


def test_grouped_std_empty():
    ind = np.array([], dtype=np.uintp)
    v = np.array([], dtype=np.float32)
    std_v = grouped_std(ind, v)
    assert std_v.size == 1
    assert np.isnan(std_v[0])  # Assuming that the function returns NaN for groups with no entries


def test_grouped_std_all_nans():
    ind = np.ones(10, dtype=np.uintp)
    v = np.full(10, np.nan, dtype=np.float32)
    std_v = grouped_std(ind, v)
    assert np.isnan(std_v[1])  # Assuming the function returns NaN for groups with no valid (non-NaN) entries


def test_grouped_std_single_group():
    ind = np.ones(10, dtype=np.uintp)
    v = np.arange(10, dtype=np.float32)
    std_v = grouped_std(ind, v)
    assert np.isclose(std_v[1], v.std(), atol=1e-5)


def test_grouped_std_non_contiguous_groups():
    ind = np.array([1, 3, 5], dtype=np.uintp)
    v = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    std_v = grouped_std(ind, v)
    assert std_v[1] == 0
    assert std_v[3] == 0
    assert std_v[5] == 0


def test_grouped_std_large_values():
    ind = np.array([1, 1], dtype=np.uintp)
    v = np.array([1e38, 1e-38], dtype=np.float32)
    std_v = grouped_std(ind, v)
    assert np.isclose(std_v[1], np.std(v), atol=1e-5)
