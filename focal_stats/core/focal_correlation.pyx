# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
"""
Algorithm to correlate two arrays (2D) with each other
"""
from focal_stats.core.utils import verify_keywords, timeit
from focal_stats.rolling import rolling_sum, rolling_window, _parse_window_and_mask

from focal_stats.core.iteration_params cimport _define_iter_params

import time
import numpy as np


cimport numpy as np
from libc.stdlib cimport free
from libc.math cimport isnan, sqrt

# todo; accept xarray (or anything that supports numpy indexing and shape property)
# todo; propagate nan?


cdef double[:, ::1] _correlate_maps(double[:, ::1] a,
                                    double[:, ::1] b,
                                    long[:] window_size,
                                    np.npy_uint8[:, ::1] mask,
                                    double fraction_accepted,
                                    bint reduce,
                                    ) nogil:
    cdef:
        long p, q, i, j, x, y
        double[:, ::1] corr
        double r_num, d1_mean, d2_mean, d1_sum, d2_sum, c1_dist, c2_dist, r_den_d1, r_den_d2
        double num_values, threshold, count_values, first_value1, first_value2
        bint all_equal_d1, all_equal_d2
        size_t shape[2]
        size_t ws[2]

    shape[0] = a.shape[0]
    shape[1] = a.shape[1]
    ws[0] = window_size[0]
    ws[1] = window_size[1]

    ip = _define_iter_params(shape, ws, fraction_accepted, reduce)

    with gil:
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

                for p in range(window_size[0]):
                    for q in range(window_size[1]):
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

                    for p in range(window_size[0]):
                        for q in range(window_size[1]):
                            if not isnan(a[i + p, j + q]) and not isnan(b[i + p, j + q]) and mask[p, q]:
                                c1_dist = a[i + p, j + q] - d1_mean
                                c2_dist = b[i + p, j + q] - d2_mean

                                r_num = r_num + (c1_dist * c2_dist)
                                r_den_d1 = r_den_d1 + c1_dist ** 2
                                r_den_d2 = r_den_d2 + c2_dist ** 2

                    corr[y + ip.fringe[0], x + ip.fringe[1]] = r_num / sqrt(r_den_d1 * r_den_d2)

    free(ip)
    return corr


def _correlate_maps_input_checks(a, b, window_size, fraction_accepted, reduce, verbose):
    """
    Input checks for correlate_maps. Check their docstring for input requirements.
    """
    if a.ndim != 2:
        raise IndexError("Only two dimensional arrays are supported")
    if b.ndim != 2:
        raise IndexError("Only two dimensional arrays are supported")
    if a.shape != b.shape:
        raise IndexError(f"Different shapes: {a.shape}, {b.shape}")

    elif window_size < 2:
        raise ValueError("window_size should be uneven and bigger than or equal to 2")

    if np.any(np.array(a.shape) < window_size):
        raise ValueError("window is bigger than the input array on at least one of the dimensions")

    if reduce:
        if ~np.all(np.array(a.shape) % window_size == 0):
            raise ValueError("The reduce parameter only works when providing a window_size that divedes the input "
                             "exactly.")
    else:
        if window_size % 2 == 0:
            raise ValueError("window_size should be uneven if reduce is set to False")


@timeit
@verify_keywords
def focal_correlation(a, b, *, window_size=5, mask=None, fraction_accepted=0.7, reduce=False, verbose=False):
    """
    Focal correlation

    Parameters
    ----------
    a, b : array-like
        Input arrays that will be correlated. If not present in dtype :obj:`~numpy.float64` it will be converted
        internally. They have exatly the same shape and have two dimensions.
    window_size : int, optional
        Size of the window used for the correlation calculations. It should be bigger than 1, the default is 5.
    mask : array_like, optional
        A boolean array. Its shape will be used as window_size, and its entries are used to mask every window.
    fraction_accepted : float, optional
        Fraction of the window that has to contain not-nans for the function to calculate the correlation. The default
        is 0.7.
    reduce : bool, optional
        Reuse all cells exactly once by setting a stepsize of the same size as window_size. The resulting raster will
        have the shape: ``shape/window_size``
    verbose : bool, optional
        Times the correlation calculations

    Returns
    -------
    :obj:`~numpy.ndarray`
        numpy array of the local correlation. If ``reduce`` is set to False, the output has the same shape as the input
        raster, while if ``reduce`` is True, the output is reduced by the window size: ``shape // window_size``.
    """
    start = time.perf_counter()
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)

    # Input checks
    _correlate_maps_input_checks(a, b, window_size, fraction_accepted, reduce, verbose)

    window_size, mask = _parse_window_and_mask(a, window_size, mask, reduce)
    if mask is None:
        mask = np.ones(window_size, dtype=np.bool_)

    corr = _correlate_maps(a, b, window_size=window_size, mask=mask,
                           fraction_accepted=fraction_accepted, reduce=reduce)

    return np.asarray(corr)


@verify_keywords
def focal_correlation_base(a, b, *, window_size=5, fraction_accepted=0.7, reduce=False, verbose=False):
    """
    Focal correlation

    Parameters
    ----------
    a, b : array-like
        Input arrays that will be correlated. If not present in dtype :obj:`~numpy.float64` it will be converted
        internally. They have exatly the same shape and have two dimensions.
    window_size : int, optional
        Size of the window used for the correlation calculations. It should be bigger than 1, the default is 5.
    fraction_accepted : float, optional
        Fraction of the window that has to contain not-nans for the function to calculate the correlation. The default
        is 0.7.
    reduce : bool, optional
        Reuse all cells exactly once by setting a stepsize of the same size as window_size. The resulting raster will
        have the shape: ``shape/window_size``
    verbose : bool, optional
        Times the correlation calculations

    Returns
    -------
    :obj:`~numpy.ndarray`
        numpy array of the local correlation. If ``reduce`` is set to False, the output has the same shape as the input
        raster, while if ``reduce`` is True, the output is reduced by the window size: ``shape // window_size``.
    """
    # Input checks
    _correlate_maps_input_checks(a, b, window_size, fraction_accepted, reduce, verbose)

    if verbose:
        print("testing validity of request")

    if reduce:
        raise NotImplementedError("Reduction option is currently not implemented for the numpy function. Compile the "
                                  "cython version to obtain this functionality")

    a = a.astype(np.float64)
    b = b.astype(np.float64)

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
    a_view = rolling_window(a, window_size=window_size)
    b_view = rolling_window(b, window_size=window_size)

    # boolean mask if values are present
    values = ~np.isnan(a)

    # sum up the boolean mask in the windows to get the amount of values
    count_values = rolling_sum(values, window_size=window_size)

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

    a_sum = rolling_sum(a, window_size=window_size)
    a_mean = np.divide(a_sum, count_values, where=valid_cells, out=a_sum)

    b_sum = rolling_sum(b, window_size=window_size)
    b_mean = np.divide(b_sum, count_values, where=valid_cells, out=b_sum)

    # add empty dimensions to make it possible to broadcast
    a_mean = a_mean[:, :, np.newaxis, np.newaxis]
    b_mean = b_mean[:, :, np.newaxis, np.newaxis]

    if verbose:
        print(f"- mean: {time.perf_counter() - start}")

    # subtract all values from the mean map, with a sampling mask to prevent nan operations. a/2_dist will therefore
    # not contain any NaNs but only zero because of np.full is 0 initialisation
    sampling_mask = np.logical_and(valid_cells[:, :, np.newaxis, np.newaxis],
                                   rolling_window(values, window_size=window_size))
    shape = (*count_values.shape, window_size, window_size)

    a_dist = np.subtract(a_view, a_mean, where=sampling_mask,
                            out=np.full(shape, 0, dtype=np.float64))
    b_dist = np.subtract(b_view, b_mean, where=sampling_mask,
                            out=np.full(shape, 0, dtype=np.float64))

    # add empty dimensions (shape=1) to make it possible to broadcast
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
