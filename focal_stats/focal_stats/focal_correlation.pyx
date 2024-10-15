# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
"""
Algorithm to correlate two arrays (2D) with each other
"""
import time
import numpy as np
from typing import Sequence

from numpydantic import NDArray
from pydantic import validate_call

from focal_stats.types import Fraction, Mask, PositiveInt, RasterFloat64
from focal_stats.utils import _parse_array, timeit
from focal_stats.rolling import rolling_sum, rolling_window
from focal_stats.focal_stats.iteration_params cimport _define_iter_params

cimport numpy as np
from libc.stdlib cimport free
from libc.math cimport isnan, sqrt
from focal_stats.window import Window, define_window, validate_window

# todo; accept xarray (or anything that supports numpy indexing and raster_shape property)
# todo; propagate nan?


cdef double[:, ::1] _correlate_rasters(double[:, ::1] a,
                                       double[:, ::1] b,
                                       int[:] window_shape,
                                       np.npy_uint8[:, ::1] mask,
                                       double fraction_accepted,
                                       bint reduce,
                                       ):
    cdef:
        size_t p, q, i, j, x, y
        double[:, ::1] corr
        double r_num, d1_mean, d2_mean, d1_sum, d2_sum, c1_dist, c2_dist, r_den_d1, r_den_d2
        double num_values, threshold, count_values, first_value1, first_value2
        bint all_equal_d1, all_equal_d2
        size_t shape[2]
        size_t ws[2]

    shape[0] = a.shape[0]
    shape[1] = a.shape[1]
    ws[0] = window_shape[0]
    ws[1] = window_shape[1]

    ip = _define_iter_params(shape, ws, fraction_accepted, reduce)
    corr = np.full(ip.shape, np.nan, dtype=np.float64)

    with nogil:
        for y in range(ip.iter[0]):
            for x in range(ip.iter[1]):
                i = y * ip.step[0]
                j = x * ip.step[1]

                if not reduce:
                    if isnan(a[i + ip.fringe[0], j + ip.fringe[1]]) or isnan(b[i + ip.fringe[0], j + ip.fringe[1]]):
                        continue

                d1_sum = 0
                d2_sum = 0
                count_values = 0
                all_equal_d1 = True
                all_equal_d2 = True

                for p in range(window_shape[0]):
                    for q in range(window_shape[1]):
                        if not isnan(a[i + p, j + q]) and not isnan(b[i + p, j + q]) and mask[p, q]:
                            if count_values == 0:
                                first_value1 = a[i + p, j + q]
                                first_value2 = b[i + p, j + q]
                            d1_sum = d1_sum + a[i + p, j + q]
                            d2_sum = d2_sum + b[i + p, j + q]

                            if a[i + p, j + q] != first_value1:
                                all_equal_d1 = False
                            if b[i + p, j + q] != first_value2:
                                all_equal_d2 = False

                            count_values = count_values + 1

                if count_values < ip.threshold:
                    pass

                elif all_equal_d1 or all_equal_d2:
                    corr[y + ip.fringe[0], x + ip.fringe[1]] = 0

                else:
                    d1_mean = d1_sum / count_values
                    d2_mean = d2_sum / count_values

                    r_num = 0
                    r_den_d1 = 0
                    r_den_d2 = 0

                    for p in range(window_shape[0]):
                        for q in range(window_shape[1]):
                            if not isnan(a[i + p, j + q]) and not isnan(b[i + p, j + q]) and mask[p, q]:
                                c1_dist = a[i + p, j + q] - d1_mean
                                c2_dist = b[i + p, j + q] - d2_mean

                                r_num = r_num + (c1_dist * c2_dist)
                                r_den_d1 = r_den_d1 + c1_dist ** 2
                                r_den_d2 = r_den_d2 + c2_dist ** 2

                    corr[y + ip.fringe[0], x + ip.fringe[1]] = r_num / sqrt(r_den_d1 * r_den_d2)

    free(ip)
    return corr


@validate_call(config={'arbitrary_types_allowed': True})
@timeit
def focal_correlation(a: NDArray,
                      b: NDArray,
                      *,
                      window: PositiveInt | Sequence[PositiveInt] | Mask | Window = 5,
                      fraction_accepted: Fraction = 0.7,
                      verbose: bool = False,
                      reduce: bool = False) -> RasterFloat64:
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

    corr = _correlate_rasters(a, b, window_shape=window_shape, mask=mask,
                           fraction_accepted=fraction_accepted, reduce=reduce)

    return np.asarray(corr)


# @validate_call(config={'arbitrary_types_allowed': True})
def focal_correlation_base(a: NDArray,
                           b: NDArray,
                           *,
                           window: PositiveInt = 5,
                           fraction_accepted: Fraction = 0.7,
                           verbose: bool = False,
                           reduce: bool = False) -> RasterFloat64:
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
        raise NotImplementedError("Reduction option is currently not implemented for the numpy function. Compile the "
                                  "cython version to obtain this functionality")

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
    count_values[count_values < fraction_accepted * window_size ** 2] = 0
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
    sampling_mask = np.logical_and(valid_cells[:, :, np.newaxis, np.newaxis],
                                   rolling_window(values, window=window_size))
    shape = (*count_values.shape, window_size, window_size)

    a_dist = np.subtract(a_view, a_mean, where=sampling_mask,
                            out=np.full(shape, 0, dtype=np.float64))
    b_dist = np.subtract(b_view, b_mean, where=sampling_mask,
                            out=np.full(shape, 0, dtype=np.float64))

    # add empty dimensions (raster_shape=1) to make it possible to broadcast
    a_dist = a_dist.reshape(*valid_cells.shape, window_size ** 2)
    b_dist = b_dist.reshape(*valid_cells.shape, window_size ** 2)

    if verbose:
        print(f"- distances: {time.perf_counter() - start}")

    # calculate the numerator and denominator of the correlation formula
    r_num = np.sum(a_dist * b_dist, axis=2)
    r_den = np.sqrt(np.sum(a_dist ** 2, axis=2) * np.sum(b_dist ** 2, axis=2),
                    where=valid_cells, out=corr[ind_inner])

    # insert the correlation (r_num/r_den) into the predefined correlation map
    corr_inner = np.divide(r_num, r_den, where=np.logical_and(valid_cells, r_den != 0), out=corr[ind_inner])

    corr_inner[np.where(r_den == 0)] = 0

    # End main calculation
    if verbose:
        print(f"- correlation: {time.perf_counter() - start}")

    return corr
