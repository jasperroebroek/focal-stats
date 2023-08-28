import numpy as np
from scipy.stats import pearsonr
import pytest
from focal_stats import focal_correlation
from focal_stats.core import focal_correlation_base
from focal_stats.rolling import _parse_window_and_mask


def overlapping_arrays(m, preserve_input=True):
    """
    Taking two maps of the same shape and returns them with  all the cells that don't exist in the other set to np.nan

    Parameters
    ----------
    m: iterable of :obj:`~numpy.ndarray`
        list of a minimum of 2 identically shaped numpy arrays.
    preserve_input : bool, optional
        if set to True the data is copied before applying the mask to preserve the input arrays. If set to False the
        memory space of the input arrays will be used.

    Returns
    -------
    m : list of :obj:`~numpy.ndarray`

    Raises
    ------
    TypeError
        if arrays are unreadable
    """
    if len(m) < 2:
        raise IndexError("list needs to contain a minimum of two arrays")

    for a in m:
        if not isinstance(a, np.ndarray):
            raise TypeError("all entries in the list need to be ndarrays")

    if not np.all([a.shape == m[0].shape for a in m]):
        raise ValueError("arrays are not of the same size")

    m_new = []
    for a in m:
        if a.dtype != np.float64:
            m_new.append(a.astype(np.float64))
        elif preserve_input:
            m_new.append(a.copy())
        else:
            m_new.append(a)
    m = m_new

    valid_cells = np.logical_and.reduce([~np.isnan(a) for a in m])

    for a in m:
        a[~valid_cells] = np.nan

    return m


def focal_correlation_simple(map1, map2, window_size=5, mask=None, fraction_accepted=0.7):
    """
    Takes two maps and returning the local correlation between them with the same dimensions as the input maps.
    Correlation calculated in a rolling window with the size `window_size`. If either of the input maps contains
    a NaN value on a location, the output map will also have a NaN on that location. This is a simplified version of
    correlate_maps() in raster_functions with the purpose of testing. It is super slow, so don't throw large maps
    at it.

    Parameters
    ----------
    map1, map2 : array-like
        Input arrays that will be correlated. If not present in dtype `np.float64` it will be converted internally.
    window_size : int, optional
        Size of the window used for the correlation calculations. It should be bigger than 1, the default is 5.
    mask : array_like, optional
        A boolean array. Its shape will be used as window_size, and its entries are used to mask every window.
    fraction_accepted : float, optional
        Fraction of the window that has to contain not-nans for the function to calculate the correlation. The default
        is 0.7.

    Returns
    -------
    corr : :obj:`~numpy.ndarray`
        numpy array of the same shape as a and b with the local correlation

    """
    map1, map2 = overlapping_arrays([map1, map2])
    fringe = window_size // 2
    corr = np.full(map1.shape, np.nan)
    _, mask = _parse_window_and_mask(map1, window_size, mask, reduce=False)
    if mask is None:
        mask = np.full(window_size, dtype=np.bool_, fill_value=1)

    for i in range(fringe, map1.shape[0] - fringe):
        for j in range(fringe, map1.shape[1] - fringe):
            ind = np.s_[i - fringe:i + fringe + 1, j - fringe:j + fringe + 1]

            if np.isnan(map1[i, j]) or np.isnan(map2[i, j]):
                continue

            d1 = map1[ind][mask]
            d2 = map2[ind][mask]

            d1 = d1[~np.isnan(d1)]
            d2 = d2[~np.isnan(d2)]

            if d1.size < fraction_accepted * mask.sum():
                continue

            if np.all(d1 == d1[0]) or np.all(d2 == d2[0]):
                corr[i, j] = 0
                continue

            corr[i, j] = pearsonr(d1, d2)[0]

    return corr


def test_correlation_values():
    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)

    # Cython implementation
    assert np.allclose(pearsonr(a.flatten(), b.flatten())[0],
                       focal_correlation(a, b, window_size=5, reduce=True))
    # Numpy implementation
    assert np.allclose(pearsonr(a.flatten(), b.flatten())[0],
                       focal_correlation_base(a, b, window_size=5)[2, 2])
    # Local implementation
    assert np.allclose(pearsonr(a.flatten(), b.flatten())[0],
                       focal_correlation_simple(a, b, window_size=5)[2, 2])

    # compare for larger shape
    a = np.random.rand(100, 100)
    b = np.random.rand(100, 100)

    assert np.allclose(focal_correlation(a, b), focal_correlation_simple(a, b), equal_nan=True)
    assert np.allclose(focal_correlation_base(a, b), focal_correlation_simple(a, b), equal_nan=True)


def test_correlation_values_mask():
    a = np.random.rand(100, 100)
    b = np.random.rand(100, 100)
    mask = np.random.RandomState().rand(5, 5)

    assert np.allclose(focal_correlation(a, b, mask=mask), focal_correlation_simple(a, b, window_size=5, mask=mask),
                       equal_nan=True)


def test_correlation_shape():
    a = np.random.rand(10, 10)
    b = np.random.rand(10, 10)

    assert focal_correlation(a, b, window_size=3).shape == a.shape
    assert focal_correlation(a, b, window_size=10, reduce=True).shape == (1, 1)


def test_correlation_errors():
    with pytest.raises(TypeError):
        focal_correlation(np.random.rand(10, 10), np.random.rand(10, 10), window_size=5, verbose=1)

    with pytest.raises(TypeError):
        focal_correlation(np.random.rand(10, 10), np.random.rand(10, 10), window_size=5, reduce=1)

    # not 2D
    with pytest.raises(IndexError):
        a = np.random.rand(10, 10, 10)
        b = np.random.rand(10, 10, 10)
        focal_correlation(a, b)

    # different shapes
    with pytest.raises(IndexError):
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 15)
        focal_correlation(a, b)

    with pytest.raises(TypeError):
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)
        focal_correlation(a, b, window_size="5")

    with pytest.raises(ValueError):
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)
        focal_correlation(a, b, window_size=1)

    with pytest.raises(ValueError):
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)
        focal_correlation(a, b, window_size=11)

    # uneven window_size is not supported
    with pytest.raises(ValueError):
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)
        focal_correlation(a, b, window_size=4)

    # Not exactly divided in reduce mode
    with pytest.raises((NotImplementedError, ValueError)):
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)
        focal_correlation(a, b, window_size=4, reduce=True)

    with pytest.raises(ValueError):
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)
        focal_correlation(a, b, fraction_accepted=-0.1)

    with pytest.raises(ValueError):
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)
        focal_correlation(a, b, fraction_accepted=1.1)


def test_nan_behaviour():
    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)
    a[2, 2] = np.nan
    assert np.allclose(focal_correlation(a, b), focal_correlation_simple(a, b), equal_nan=True)
    assert np.isnan(focal_correlation(a, b)[2, 2])

    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)
    a[1, 1] = np.nan
    assert np.allclose(focal_correlation(a, b), focal_correlation_simple(a, b), equal_nan=True)
    assert not np.isnan(focal_correlation(a, b)[2, 2])
    assert np.isnan(focal_correlation(a, b, fraction_accepted=1)[2, 2])
    assert not np.isnan(focal_correlation(a, b, fraction_accepted=0)[2, 2])


def test_correlation_dtype():
    a = np.random.rand(5, 5).astype(np.int32)
    b = np.random.rand(5, 5).astype(np.int32)
    assert focal_correlation(a, b).dtype == np.float64

    a = np.random.rand(5, 5).astype(np.float64)
    b = np.random.rand(5, 5).astype(np.float64)
    assert focal_correlation(a, b).dtype == np.float64


# def test_correlation_against_base():
#     import rasterio as rio
#     with rio.open("../data/tree_height.asc") as f:
#         m1 = f.read(indexes=1)
#     with rio.open("../data/wtd.tif") as f:
#         m2 = f.read(indexes=1)
#     assert np.allclose(focal_correlation_base(m1, m2),
#                        focal_correlation(m1, m2),
#                        equal_nan=True)
