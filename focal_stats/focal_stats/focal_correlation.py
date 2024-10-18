import time
from typing import Sequence

import numpy as np
from numpydantic import NDArray
from pydantic import validate_call

from focal_stats import rolling_sum, rolling_window
from focal_stats.focal_stats._focal_correlation_core import _correlate_rasters
from focal_stats.types import Fraction, Mask, PositiveInt, RasterFloat64
from focal_stats.utils import _parse_array, timeit
from focal_stats.window import Window, define_window, validate_window


@validate_call(config={"arbitrary_types_allowed": True})
@timeit
def focal_correlation(
    a: NDArray,
    b: NDArray,
    *,
    window: PositiveInt | Sequence[PositiveInt] | Mask | Window = 5,
    fraction_accepted: Fraction = 0.7,
    verbose: bool = False,
    reduce: bool = False,
) -> RasterFloat64:
    """
    Focal correlation

    Parameters
    ----------
    a, b : array-like
        Input arrays that will be correlated. If not present in dtype :obj:`~numpy.float64` it will be converted
        internally. They have exatly the same raster_shape and have two dimensions.
    window : int, array_like, Window
        Window that is applied over ``a``. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a mask or a Window object.
    fraction_accepted : float, optional
        Fraction of the window that has to contain not-nans for the function to calculate the correlation. The default
        is 0.7.
    reduce : bool, optional
        Reuse all cells exactly once by setting a stepsize of the same size as window_shape. The resulting raster will
        have the raster_shape: ``raster_shape/window_shape``
    verbose : bool, optional
        Times the correlation calculations

    Returns
    -------
    :obj:`~numpy.ndarray`
        numpy array of the local correlation. If ``reduce`` is set to False, the output has the same raster_shape as the input
        raster, while if ``reduce`` is True, the output is reduced by the window size: ``raster_shape // window_shape``.
    """
    a = _parse_array(a)
    b = _parse_array(b)

    raster_shape = np.asarray(a.shape)

    if a.shape != b.shape:
        raise ValueError(f"Input arrays have different shapes: {a.shape=}, {b.shape=}")

    window = define_window(window)
    validate_window(window, raster_shape, reduce, allow_even=False)

    mask = window.get_mask(2)
    window_shape = np.asarray(window.get_shape(2), dtype=np.int32)

    corr = _correlate_rasters(
        a,
        b,
        window_shape=window_shape,
        mask=mask,
        fraction_accepted=fraction_accepted,
        reduce=reduce,
    )

    return np.asarray(corr)


# @validate_call(config={'arbitrary_types_allowed': True})
def focal_correlation_base(
    a: NDArray,
    b: NDArray,
    *,
    window: PositiveInt = 5,
    fraction_accepted: Fraction = 0.7,
    verbose: bool = False,
    reduce: bool = False,
) -> RasterFloat64:
    """
    Focal correlation

    Parameters
    ----------
    a, b : array-like
        Input arrays that will be correlated. If not present in dtype :obj:`~numpy.float64` it will be converted
        internally. They have exactly the same raster_shape and have two dimensions.
    window : int
        Window that is applied over ``a``.
    fraction_accepted : float, optional
        Fraction of the window that has to contain not-nans for the function to calculate the correlation. The default
        is 0.7.
    reduce : bool, optional
        Reuse all cells exactly once by setting a stepsize of the same size as window_shape. The resulting raster will
        have the raster_shape: ``raster_shape/window_shape``
    verbose : bool, optional
        Times the correlation calculations

    Returns
    -------
    :obj:`~numpy.ndarray`
        numpy array of the local correlation. If ``reduce`` is set to False, the output has the same raster_shape as the input
        raster, while if ``reduce`` is True, the output is reduced by the window size: ``raster_shape // window_shape``.
    """
    if verbose:
        print("testing validity of request")

    a = _parse_array(a)
    b = _parse_array(b)

    raster_shape = np.asarray(a.shape)

    window_size = window
    window = define_window(window)
    validate_window(window, raster_shape, reduce, allow_even=False)

    if a.shape != b.shape:
        raise ValueError(f"Input arrays have different shapes: {a.shape=}, {b.shape=}")

    if reduce:
        raise NotImplementedError(
            "Reduction option is currently not implemented for the numpy function. Compile the "
            "cython version to obtain this functionality"
        )

    # overlapping the maps
    nans = np.logical_or(np.isnan(a), np.isnan(b))
    a[nans] = np.nan
    b[nans] = np.nan

    # parameters
    fringe = window_size // 2  # count of neighbouring cells in each window
    ind_inner = np.s_[fringe:-fringe, fringe:-fringe]

    # prepare storage for correlation maps
    corr = np.full(a.shape, np.nan)

    # Start main calculation
    start = time.perf_counter()

    # create the windowed view on the data. These are views, no copies
    a_view = rolling_window(a, window=window_size)
    b_view = rolling_window(b, window=window_size)

    # boolean mask if values are present
    values = ~np.isnan(a)

    # sum up the boolean mask in the windows to get the amount of values
    count_values = rolling_sum(values, window=window_size)

    # remove cases from count_values where the original cell was NaN and where there are too many NaNs present
    count_values[count_values < fraction_accepted * window_size**2] = 0
    count_values[~values[ind_inner]] = 0
    valid_cells = count_values > 0

    if valid_cells.sum() == 0:
        if verbose:
            print("- empty tile")
        return corr

    if verbose:
        print(f"- preparation: {time.perf_counter() - start}")

    # create a focal statistics mean map
    a[nans] = 0
    b[nans] = 0

    a_sum = rolling_sum(a, window=window_size)
    a_mean = np.divide(a_sum, count_values, where=valid_cells, out=a_sum)

    b_sum = rolling_sum(b, window=window_size)
    b_mean = np.divide(b_sum, count_values, where=valid_cells, out=b_sum)

    # add empty dimensions to make it possible to broadcast
    a_mean = a_mean[:, :, np.newaxis, np.newaxis]
    b_mean = b_mean[:, :, np.newaxis, np.newaxis]

    if verbose:
        print(f"- mean: {time.perf_counter() - start}")

    # subtract all values from the mean map, with a sampling mask to prevent nan operations. a/2_dist will therefore
    # not contain any NaNs but only zero because of np.full is 0 initialisation
    sampling_mask = np.logical_and(
        valid_cells[:, :, np.newaxis, np.newaxis],
        rolling_window(values, window=window_size),
    )
    shape = (*count_values.shape, window_size, window_size)

    a_dist = np.subtract(
        a_view, a_mean, where=sampling_mask, out=np.full(shape, 0, dtype=np.float64)
    )
    b_dist = np.subtract(
        b_view, b_mean, where=sampling_mask, out=np.full(shape, 0, dtype=np.float64)
    )

    # add empty dimensions (raster_shape=1) to make it possible to broadcast
    a_dist = a_dist.reshape(*valid_cells.shape, window_size**2)
    b_dist = b_dist.reshape(*valid_cells.shape, window_size**2)

    if verbose:
        print(f"- distances: {time.perf_counter() - start}")

    # calculate the numerator and denominator of the correlation formula
    r_num = np.sum(a_dist * b_dist, axis=2)
    r_den = np.sqrt(
        np.sum(a_dist**2, axis=2) * np.sum(b_dist**2, axis=2),
        where=valid_cells,
        out=corr[ind_inner],
    )

    # insert the correlation (r_num/r_den) into the predefined correlation map
    corr_inner = np.divide(
        r_num, r_den, where=np.logical_and(valid_cells, r_den != 0), out=corr[ind_inner]
    )

    corr_inner[np.where(r_den == 0)] = 0

    # End main calculation
    if verbose:
        print(f"- correlation: {time.perf_counter() - start}")

    return corr
