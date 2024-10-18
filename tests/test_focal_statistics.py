import numpy as np
import pytest
import scipy.stats
from numpy.ma.testutils import assert_array_almost_equal
from pydantic import ValidationError

import focal_stats


def test_focal_stats_values():
    a = np.random.rand(5, 5)

    # Values when not reducing
    assert np.allclose(focal_stats.focal_mean(a, window=5)[2, 2], a.mean())
    assert np.allclose(focal_stats.focal_sum(a, window=5)[2, 2], a.sum())
    assert np.allclose(focal_stats.focal_min(a, window=5)[2, 2], a.min())
    assert np.allclose(focal_stats.focal_max(a, window=5)[2, 2], a.max())
    assert np.allclose(focal_stats.focal_std(a, window=5)[2, 2], a.std())
    assert np.allclose(
        focal_stats.focal_std(a, window=5, std_df=1)[2, 2], a.std(ddof=1)
    )

    # Values when reducing
    assert np.allclose(focal_stats.focal_mean(a, window=5, reduce=True)[0, 0], a.mean())
    assert np.allclose(focal_stats.focal_sum(a, window=5, reduce=True)[0, 0], a.sum())
    assert np.allclose(focal_stats.focal_min(a, window=5, reduce=True)[0, 0], a.min())
    assert np.allclose(focal_stats.focal_max(a, window=5, reduce=True)[0, 0], a.max())
    assert np.allclose(focal_stats.focal_std(a, window=5, reduce=True)[0, 0], a.std())
    assert np.allclose(
        focal_stats.focal_std(a, window=5, std_df=1, reduce=True)[0, 0], a.std(ddof=1)
    )

    # compare for larger raster_shape
    a = np.random.rand(100, 100)

    assert np.allclose(
        focal_stats.focal_std(a, window=5, std_df=0, reduce=True),
        focal_stats.rolling_window(a, window=5, reduce=True, flatten=True).std(axis=-1),
    )
    assert np.allclose(
        focal_stats.focal_std(a, window=5, std_df=1, reduce=True),
        focal_stats.rolling_window(a, window=5, reduce=True, flatten=True).std(
            axis=-1, ddof=1
        ),
    )

    # majority modes
    rs = np.random.RandomState(0)
    a = rs.randint(0, 10, 25).reshape(5, 5)

    # Value when reducing
    mode = scipy.stats.mode(a.flatten()).mode
    if isinstance(mode, np.ndarray):
        mode = mode[0]

    # Values when reducing
    assert (
        focal_stats.focal_majority(a, window=5, majority_mode="ascending")[2, 2] == mode
    )
    # Values when not reducing
    assert (
        focal_stats.focal_majority(a, window=5, reduce=True, majority_mode="ascending")[
            0, 0
        ]
        == mode
    )

    # Same number of observations in several classes lead to NaN in majority_mode='nan'
    a = np.arange(100).reshape(10, 10)
    assert np.isnan(
        focal_stats.focal_majority(a, window=10, reduce=True, majority_mode="nan")
    )

    # Same number of observations in several classes lead to lowest number in majority_mode='ascending'
    assert (
        focal_stats.focal_majority(a, window=10, reduce=True, majority_mode="ascending")
        == 0
    )

    # Same number of observations in several classes lead to highest number in majority_mode='descending'
    assert (
        focal_stats.focal_majority(
            a, window=10, reduce=True, majority_mode="descending"
        )
        == 99
    )


@pytest.mark.parametrize(
    "fs,np_fs",
    [
        (focal_stats.focal_mean, np.nanmean),
        (focal_stats.focal_sum, np.nansum),
        (focal_stats.focal_min, np.nanmin),
        (focal_stats.focal_max, np.nanmax),
        (focal_stats.focal_std, np.nanstd),
    ],
)
def test_focal_stats_values_mask(fs, np_fs):
    a = np.random.rand(100, 100)
    assert_array_almost_equal(
        fs(a, window=5)[2:-2, 2:-2],
        np_fs(focal_stats.rolling_window(a, window=5, flatten=True), axis=-1),
    )


@pytest.mark.parametrize(
    "fs",
    [
        focal_stats.focal_mean,
        focal_stats.focal_sum,
        focal_stats.focal_min,
        focal_stats.focal_max,
        focal_stats.focal_std,
        focal_stats.focal_majority,
    ],
)
def test_focal_stats_shape(fs):
    a = np.random.rand(10, 10)
    assert a.shape == fs(a, window=3).shape
    assert fs(a, window=10, reduce=True).shape == (1, 1)


@pytest.mark.parametrize(
    "fs",
    [
        focal_stats.focal_mean,
        focal_stats.focal_sum,
        focal_stats.focal_min,
        focal_stats.focal_max,
        focal_stats.focal_std,
        focal_stats.focal_majority,
    ],
)
def test_focal_stats_errors(fs):
    # Not boolean
    with pytest.raises(ValidationError):
        fs(np.random.rand(10, 10), window=5, verbose=2)

    with pytest.raises(ValidationError):
        fs(np.random.rand(10, 10), window=5, reduce=2)

    # not 2D
    with pytest.raises(IndexError):
        a = np.random.rand(10, 10, 10)
        fs(a, window=5)

    with pytest.raises(ValidationError):
        a = np.random.rand(10, 10)
        fs(a, window="x")

    with pytest.raises(ValueError):
        a = np.random.rand(10, 10)
        fs(a, window=1)

    with pytest.raises(ValueError):
        a = np.random.rand(10, 10)
        fs(a, window=11)

    # uneven window_shape is not supported
    with pytest.raises(ValueError):
        a = np.random.rand(10, 10)
        fs(a, window=4)

    # Not exactly divided in reduce mode
    with pytest.raises(ValueError):
        a = np.random.rand(10, 10)
        fs(a, window=4, reduce=True)

    with pytest.raises(ValueError):
        a = np.random.rand(10, 10)
        fs(a, window=5, fraction_accepted=-0.1)

    with pytest.raises(ValueError):
        a = np.random.rand(10, 10)
        fs(a, window=5, fraction_accepted=1.1)


@pytest.mark.parametrize(
    "fs",
    [
        focal_stats.focal_mean,
        focal_stats.focal_sum,
        focal_stats.focal_min,
        focal_stats.focal_max,
        focal_stats.focal_std,
        focal_stats.focal_majority,
    ],
)
def test_focal_stats_nan_propagation(fs):
    a = np.random.rand(5, 5)
    a[2, 2] = np.nan
    assert np.isnan(fs(a, window=5)[2, 2])


@pytest.mark.parametrize(
    "fs,np_fs",
    [
        (focal_stats.focal_mean, np.nanmean),
        (focal_stats.focal_sum, np.nansum),
        (focal_stats.focal_min, np.nanmin),
        (focal_stats.focal_max, np.nanmax),
        (focal_stats.focal_std, np.nanstd),
    ],
)
def test_focal_stats_nan_behaviour_fraction_accepted(fs, np_fs):
    a = np.random.rand(5, 5)
    a[1, 1] = np.nan

    assert np.allclose(fs(a, window=5)[2, 2], np_fs(a))
    assert not np.isnan(fs(a, window=5, fraction_accepted=0)[2, 2])
    assert np.isnan(fs(a, window=5, fraction_accepted=1)[2, 2])


def test_focal_stats_nan_behaviour_majority():
    a = np.ones((5, 5)).astype(float)
    a[1, 1] = np.nan
    assert focal_stats.focal_majority(a, window=5)[2, 2] == 1
    assert not np.isnan(
        focal_stats.focal_majority(a, window=5, fraction_accepted=0)[2, 2]
    )
    assert np.isnan(focal_stats.focal_majority(a, window=5, fraction_accepted=1)[2, 2])


@pytest.mark.parametrize(
    "fs",
    [
        focal_stats.focal_mean,
        focal_stats.focal_sum,
        focal_stats.focal_min,
        focal_stats.focal_max,
        focal_stats.focal_std,
        focal_stats.focal_majority,
    ],
)
def test_focal_stats_dtype(fs):
    a = np.random.rand(5, 5).astype(np.int32)
    assert fs(a, window=5).dtype == np.float64

    a = np.random.rand(5, 5).astype(np.float64)
    assert fs(a, window=5).dtype == np.float64
