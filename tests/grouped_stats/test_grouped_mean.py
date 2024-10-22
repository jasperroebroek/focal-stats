import numpy as np
import pandas as pd
import pytest

from spatial_stats.grouped_stats import (
    grouped_mean,
    grouped_mean_pd, grouped_mean_std_pd,
)

rs = np.random.RandomState(0)


@pytest.fixture
def ind():
    return rs.randint(0, 5, size=(10, 10), dtype=np.uintp)


@pytest.fixture
def v():
    return rs.rand(10, 10)


def test_grouped_mean(ind, v):
    mean_v = grouped_mean(ind, v)

    for i in range(1, int(ind.max()) + 1):
        values_in_group = v[ind == i]
        expected_mean = np.nanmean(values_in_group)
        assert np.isclose(mean_v[i], expected_mean, atol=1e-5)

    assert np.isnan(
        mean_v[0]
    )  # Assuming the function returns NaN for groups with no entries


def test_grouped_mean_pd(ind, v):
    result_df = grouped_mean_pd(ind, v)

    assert isinstance(result_df, pd.DataFrame)

    for i in result_df.index:
        values_in_group = v[ind == i]
        expected_mean = np.nanmean(values_in_group)
        assert np.isclose(result_df.loc[i, "mean"], expected_mean, atol=1e-5)

    assert 0 not in result_df.index



def test_grouped_mean_std_pd(ind, v):
    result_df = grouped_mean_std_pd(ind, v)

    assert isinstance(result_df, pd.DataFrame)

    for i in result_df.index:
        values_in_group = v[ind == i]
        expected_mean = np.nanmean(values_in_group)
        assert np.isclose(result_df.loc[i, "mean"], expected_mean, atol=1e-5)

    assert 0 not in result_df.index


def test_grouped_mean_empty():
    ind = np.array([], dtype=np.uintp)
    v = np.array([], dtype=np.float32)
    mean_v = grouped_mean(ind, v)
    assert mean_v.size == 1
    assert np.isnan(
        mean_v[0]
    )  # Assuming that the function returns NaN for groups with no entries


def test_grouped_mean_all_nans():
    ind = np.ones(10, dtype=np.uintp)
    v = np.full(10, np.nan, dtype=np.float32)
    mean_v = grouped_mean(ind, v)
    assert np.isnan(
        mean_v[1]
    )  # Assuming the function returns NaN for groups with no valid (non-NaN) entries


def test_grouped_mean_single_group():
    ind = np.ones(10, dtype=np.uintp)
    v = np.arange(10, dtype=np.float32)
    mean_v = grouped_mean(ind, v)
    assert np.isclose(mean_v[1], v.mean(), atol=1e-5)


def test_grouped_mean_non_contiguous_groups():
    ind = np.array([1, 3, 5], dtype=np.uintp)
    v = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    mean_v = grouped_mean(ind, v)
    assert mean_v[1] == 10.0
    assert mean_v[3] == 20.0
    assert mean_v[5] == 30.0


def test_grouped_mean_large_values():
    ind = np.array([1, 1], dtype=np.uintp)
    v = np.array([1e38, 1e-38], dtype=np.float32)
    mean_v = grouped_mean(ind, v)
    assert np.isclose(
        mean_v[1], (1e38 + 1e-38) / 2, atol=1e-5
    )  # Expect the mean to be the average of the values in v
