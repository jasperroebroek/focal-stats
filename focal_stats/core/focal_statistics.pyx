# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Optional, Union, Iterable

import numpy as np
import time

from focal_stats.core.utils import timeit, verify_keywords, _parse_array, _parse_nans, _parse_window, \
    _create_output_array, _calc_below_fraction_accepted_mask, _calc_count_values
from focal_stats.rolling import rolling_window, rolling_sum

from focal_stats.core.iteration_params cimport _define_iter_params

cimport numpy as np
from libc.math cimport isnan, sqrt, NAN
from libc.stdlib cimport free

# todo; propagate nans?

cdef enum MajorityMode:
    ascending, descending, nan


cdef double[:, ::1] _focal_std(double[:, ::1] a,
                               long[:] window_size,
                               np.npy_uint8[:, ::1] mask,
                               double fraction_accepted,
                               bint reduce,
                               long dof
                               ) nogil:
    cdef:
        long p, q, i, j, x, y
        double[:, ::1] r
        double a_sum, a_mean, x_sum
        double first_value1, first_value2
        long count_values
        bint all_equal_d1, all_equal_d2
        size_t shape[2]
        size_t ws[2]

    shape[0] = a.shape[0]
    shape[1] = a.shape[1]
    ws[0] = window_size[0]
    ws[1] = window_size[1]

    ip = _define_iter_params(shape, ws, fraction_accepted, reduce)

    with gil:
        r = np.full(ip.shape, np.nan, dtype=np.float64)

    with nogil:
        for y in range(ip.iter[0]):
            for x in range(ip.iter[1]):
                i = y * ip.step[0]
                j = x * ip.step[1]

                if not reduce:
                    if isnan(a[i + ip.fringe[0], j + ip.fringe[1]]):
                        continue

                a_sum = 0
                count_values = 0

                for p in range(window_size[0]):
                    for q in range(window_size[1]):
                        if not isnan(a[i + p, j + q]) and mask[p, q]:
                            a_sum = a_sum + a[i + p, j + q]
                            count_values = count_values + 1

                if count_values < ip.threshold:
                    pass

                else:
                    a_mean = a_sum / count_values
                    x_sum = 0

                    for p in range(window_size[0]):
                        for q in range(window_size[1]):
                            if not isnan(a[i + p, j + q]) and mask[p, q]:
                                x_sum = x_sum + (a[i + p, j + q] - a_mean) ** 2

                    r[y + ip.fringe[0], x + ip.fringe[1]] = sqrt(x_sum / (count_values - dof))

    free(ip)
    return r


cdef double[:, ::1] _focal_sum(double[:, ::1] a,
                               long[:] window_size,
                               np.npy_uint8[:, ::1] mask,
                               double fraction_accepted,
                               bint reduce,
                               ) nogil:
    cdef:
        long p, q, i, j, x, y
        double[:, ::1] r
        double a_sum
        long count_values
        size_t shape[2]
        size_t ws[2]

    shape[0] = a.shape[0]
    shape[1] = a.shape[1]
    ws[0] = window_size[0]
    ws[1] = window_size[1]

    ip = _define_iter_params(shape, ws, fraction_accepted, reduce)

    with gil:
        r = np.full(ip.shape, np.nan, dtype=np.float64)

    with nogil:
        for y in range(ip.iter[0]):
            for x in range(ip.iter[1]):
                i = y * ip.step[0]
                j = x * ip.step[1]

                if not reduce:
                    if isnan(a[i + ip.fringe[0], j + ip.fringe[1]]):
                        continue

                a_sum = 0
                count_values = 0

                for p in range(window_size[0]):
                    for q in range(window_size[1]):
                        if not isnan(a[i + p, j + q]) and mask[p, q]:
                            a_sum = a_sum + a[i + p, j + q]
                            count_values = count_values + 1

                if count_values < ip.threshold:
                    pass

                else:
                    r[y + ip.fringe[0], x + ip.fringe[1]] = a_sum

    free(ip)
    return r


cdef double[:, ::1] _focal_min(double[:, ::1] a,
                               long[:] window_size,
                               np.npy_uint8[:, ::1] mask,
                               double fraction_accepted,
                               bint reduce,
                               ) nogil:
    cdef:
        long p, q, i, j, x, y
        double[:, ::1] r
        double curr_min
        long count_values
        size_t shape[2]
        size_t ws[2]

    shape[0] = a.shape[0]
    shape[1] = a.shape[1]
    ws[0] = window_size[0]
    ws[1] = window_size[1]

    ip = _define_iter_params(shape, ws, fraction_accepted, reduce)

    with gil:
        r = np.full(ip.shape, np.nan, dtype=np.float64)

    with nogil:
        for y in range(ip.iter[0]):
            for x in range(ip.iter[1]):
                i = y * ip.step[0]
                j = x * ip.step[1]

                if not reduce:
                    if isnan(a[i + ip.fringe[0], j + ip.fringe[1]]):
                        continue

                curr_min = 0
                count_values = 0

                for p in range(window_size[0]):
                    for q in range(window_size[1]):
                        if not isnan(a[i + p, j + q]) and mask[p, q]:
                            if a[i + p, j + q] < curr_min or count_values == 0:
                                curr_min = a[i + p, j + q]
                            count_values = count_values + 1

                if count_values < ip.threshold:
                    pass

                else:
                    r[y + ip.fringe[0], x + ip.fringe[1]] = curr_min

    free(ip)
    return r


cdef double[:, ::1] _focal_max(double[:, ::1] a,
                               long[:] window_size,
                               np.npy_uint8[:, ::1] mask,
                               double fraction_accepted,
                               bint reduce,
                               ) nogil:
    cdef:
        long p, q, i, j, x, y
        double[:, ::1] r
        double curr_max
        long count_values
        size_t shape[2]
        size_t ws[2]

    shape[0] = a.shape[0]
    shape[1] = a.shape[1]
    ws[0] = window_size[0]
    ws[1] = window_size[1]

    ip = _define_iter_params(shape, ws, fraction_accepted, reduce)

    with gil:
        r = np.full(ip.shape, np.nan, dtype=np.float64)

    with nogil:
        for y in range(ip.iter[0]):
            for x in range(ip.iter[1]):
                i = y * ip.step[0]
                j = x * ip.step[1]

                if not reduce:
                    if isnan(a[i + ip.fringe[0], j + ip.fringe[1]]):
                        continue

                curr_max = 0
                count_values = 0

                for p in range(window_size[0]):
                    for q in range(window_size[1]):
                        if not isnan(a[i + p, j + q]) and mask[p, q]:
                            if a[i + p, j + q] > curr_max or count_values == 0:
                                curr_max = a[i + p, j + q]
                            count_values = count_values + 1

                if count_values < ip.threshold:
                    pass

                else:
                    r[y + ip.fringe[0], x + ip.fringe[1]] = curr_max

    free(ip)
    return r


cdef double[:, ::1] _focal_majority(double[:, ::1] a,
                                    long[:] window_size,
                                    np.npy_uint8[:, ::1] mask,
                                    double fraction_accepted,
                                    bint reduce,
                                    MajorityMode mode
                                    ) nogil:
    cdef:
        long p, q, i, j, x, y, v, c
        double[:, ::1] r
        double curr_value
        size_t curr_max_count
        long count_values
        size_t shape[2]
        size_t ws[2]
        bint in_store, is_double
        double[:] values
        long[:] counts

    shape[0] = a.shape[0]
    shape[1] = a.shape[1]
    ws[0] = window_size[0]
    ws[1] = window_size[1]

    ip = _define_iter_params(shape, ws, fraction_accepted, reduce)

    with gil:
        values = np.full(ip.num_values, dtype=np.float64, fill_value=np.nan)
        counts = np.zeros(ip.num_values, dtype=np.intp)
        r = np.full(ip.shape, np.nan, dtype=np.float64)

    with nogil:
        for y in range(ip.iter[0]):
            for x in range(ip.iter[1]):
                i = y * ip.step[0]
                j = x * ip.step[1]

                if not reduce:
                    if isnan(a[i + ip.fringe[0], j + ip.fringe[1]]):
                        continue

                values[0] = 0
                counts[0] = 0
                count_values = 0
                c = 1

                for p in range(window_size[0]):
                    for q in range(window_size[1]):
                        if not isnan(a[i + p, j + q]) and mask[p, q]:
                            in_store = False
                            if count_values == 0:
                                values[0] = a[i + p, j + q]
                            for v in range(c):
                                if a[i + p, j + q] == values[v]:
                                    counts[v] = counts[v] + 1
                                    in_store = True
                            if not in_store:
                                values[c] = a[i + p, j + q]
                                counts[c] = 1
                                c = c + 1

                            count_values = count_values + 1

                if count_values < ip.threshold:
                    pass

                else:
                    if mode == MajorityMode.ascending:
                        curr_max_count = 0
                        curr_value = NAN
                        for v in range(c):
                            if counts[v] > curr_max_count:
                                curr_max_count = counts[v]
                                curr_value = values[v]

                    if mode == MajorityMode.descending:
                        curr_max_count = 0
                        curr_value = NAN
                        for v in range(c):
                            if counts[v] >= curr_max_count:
                                curr_max_count = counts[v]
                                curr_value = values[v]

                    if mode == MajorityMode.nan:
                        curr_max_count = 0
                        curr_value = NAN
                        is_double = False
                        for v in range(c):
                            if counts[v] == curr_max_count:
                                is_double = True
                            if counts[v] > curr_max_count:
                                curr_max_count = counts[v]
                                curr_value = values[v]
                                is_double = False

                        if is_double:
                            curr_value = NAN

                    r[y + ip.fringe[0], x + ip.fringe[1]] = curr_value

    free(ip)
    return r


@timeit
@verify_keywords
def focal_min(a: Iterable,
              *,
              window_size: Optional[Union[int, Iterable[int]]] = None,
              mask: Optional[Iterable[bool]] = None,
              fraction_accepted: float = 0.7,
              verbose: bool = False,
              reduce: bool = False) -> np.ndarray:
    """
    Focal minimum

    Parameters
    ----------
    a : array_like
        Input array (2D)
    window_size : int, array_like, optional
        Size of the window that is applied over ``a``. Should be a positive integer. If it is an integer it will be
        interpreted as (window_size, window_size). If a list is provided it needs to have the same number of entries
        as the number of dimensions as a.
    mask : array_like, optional
        A boolean array (2D). If provided, its shape will be used as ``window_size``, and its entries are used to mask
        every window.
    fraction_accepted : float, optional
        Fraction of valid cells (not NaN) per window that is deemed acceptable
        ``0``: all window are calculated if at least 1 value is present
        ``1``: only windows completely filled with values are calculated
        ``0-1``: fraction of acceptability
    verbose : bool, optional
        Verbosity with timing. False by default
    reduce : bool, optional
        The way in which the windowed array is created. If true, all values are used exactly once. If False (which is
        the default), values are reduced and the output array has the same shape as the input array, albeit with a
        border of nans where there are not enough values to calculate the cells.

    Returns
    -------
    :obj:`~numpy.ndarray`
        if ``reduce`` is False:
            numpy ndarray of the function applied to input array ``a``. The shape will
            be the same as the input array. The border of the map will be filled with nan,
            because of the lack of data to calculate the border. In the future other
            behaviours might be implemented. To obtain only the useful cells the
            following might be done:

                >>> window_size = 5
                >>> fringe = window_size // 2
                >>> ind_inner = np.s_[fringe:-fringe, fringe:-fringe]
                >>> a = a[ind_inner]

            in which case a will only contain the cells for which all data was
            present
        if ``reduce`` is True:
            numpy ndarray of the function applied on input array ``a``. The shape
            will be the original shape divided by the ``window_size``. Dimensions
            remain equal. No border of NaN values is present.
    """
    a_parsed = _parse_array(a)
    empty_flag, nan_flag, nan_mask = _parse_nans(a_parsed)
    mask_flag, mask_parsed, window_size_parsed, fringes, ind_inner = _parse_window(a_parsed, window_size, mask, reduce)

    if empty_flag:
        if verbose:
            print("- Empty array")
        return _create_output_array(a_parsed, window_size_parsed, reduce)

    if not nan_flag and not mask_flag:
        r = _create_output_array(a_parsed, window_size_parsed, reduce)

        r[ind_inner] = (
            rolling_window(a_parsed, window_size=window_size_parsed, mask=None, reduce=reduce)
            .min(axis=(2, 3))
        )

        return r

    return np.asarray(
        _focal_min(a_parsed, window_size_parsed, mask_parsed, fraction_accepted, reduce)
    )


@timeit
@verify_keywords
def focal_max(a: Iterable,
              *,
              window_size: Optional[Union[int, Iterable[int]]] = None,
              mask: Optional[Iterable[bool]] = None,
              fraction_accepted: float = 0.7,
              verbose: bool = False,
              reduce: bool = False) -> np.ndarray:
    """
    Focal maximum

    Parameters
    ----------
    a : array_like
        Input array (2D)
    window_size : int, array_like, optional
        Size of the window that is applied over ``a``. Should be a positive integer. If it is an integer it will be
        interpreted as (window_size, window_size). If a list is provided it needs to have the same number of entries
        as the number of dimensions as a.
    mask : array_like, optional
        A boolean array (2D). If provided, its shape will be used as ``window_size``, and its entries are used to mask
        every window.
    fraction_accepted : float, optional
        Fraction of valid cells (not NaN) per window that is deemed acceptable
        ``0``: all window are calculated if at least 1 value is present
        ``1``: only windows completely filled with values are calculated
        ``0-1``: fraction of acceptability
    verbose : bool, optional
        Verbosity with timing. False by default
    reduce : bool, optional
        The way in which the windowed array is created. If true, all values are used exactly once. If False (which is
        the default), values are reduced and the output array has the same shape as the input array, albeit with a
        border of nans where there are not enough values to calculate the cells.

    Returns
    -------
    :obj:`~numpy.ndarray`
        if ``reduce`` is False:
            numpy ndarray of the function applied to input array ``a``. The shape will
            be the same as the input array. The border of the map will be filled with nan,
            because of the lack of data to calculate the border. In the future other
            behaviours might be implemented. To obtain only the useful cells the
            following might be done:

                >>> window_size = 5
                >>> fringe = window_size // 2
                >>> ind_inner = np.s_[fringe:-fringe, fringe:-fringe]
                >>> a = a[ind_inner]

            in which case a will only contain the cells for which all data was
            present
        if ``reduce`` is True:
            numpy ndarray of the function applied on input array ``a``. The shape
            will be the original shape divided by the ``window_size``. Dimensions
            remain equal. No border of NaN values is present.
    """
    a_parsed = _parse_array(a)
    empty_flag, nan_flag, nan_mask = _parse_nans(a_parsed)
    mask_flag, mask_parsed, window_size_parsed, fringes, ind_inner = _parse_window(a_parsed, window_size, mask, reduce)

    if empty_flag:
        if verbose:
            print("- Empty array")
        return _create_output_array(a_parsed, window_size_parsed, reduce)

    if not nan_flag and not mask_flag:
        r = _create_output_array(a_parsed, window_size_parsed, reduce)

        r[ind_inner] = (
            rolling_window(a_parsed, window_size=window_size_parsed, mask=None, reduce=reduce)
            .max(axis=(2, 3))
        )

        return r

    return np.asarray(
        _focal_max(a_parsed, window_size_parsed, mask_parsed, fraction_accepted, reduce)
    )


@timeit
@verify_keywords
def focal_mean(a: Iterable,
               *,
               window_size: Optional[Union[int, Iterable[int]]] = None,
               mask: Optional[Iterable[bool]] = None,
               fraction_accepted: float = 0.7,
               verbose: bool = False,
               reduce: bool = False) -> np.ndarray:
    """
    Focal mean

    Parameters
    ----------
    a : array_like
        Input array (2D)
    window_size : int, array_like, optional
        Size of the window that is applied over ``a``. Should be a positive integer. If it is an integer it will be
        interpreted as (window_size, window_size). If a list is provided it needs to have the same number of entries
        as the number of dimensions as a.
    mask : array_like, optional
        A boolean array (2D). If provided, its shape will be used as ``window_size``, and its entries are used to mask
        every window.
    fraction_accepted : float, optional
        Fraction of valid cells (not NaN) per window that is deemed acceptable
        ``0``: all window are calculated if at least 1 value is present
        ``1``: only windows completely filled with values are calculated
        ``0-1``: fraction of acceptability
    verbose : bool, optional
        Verbosity with timing. False by default
    reduce : bool, optional
        The way in which the windowed array is created. If true, all values are used exactly once. If False (which is
        the default), values are reduced and the output array has the same shape as the input array, albeit with a
        border of nans where there are not enough values to calculate the cells.

    Returns
    -------
    :obj:`~numpy.ndarray`
        if ``reduce`` is False:
            numpy ndarray of the function applied to input array ``a``. The shape will
            be the same as the input array. The border of the map will be filled with nan,
            because of the lack of data to calculate the border. In the future other
            behaviours might be implemented. To obtain only the useful cells the
            following might be done:

                >>> window_size = 5
                >>> fringe = window_size // 2
                >>> ind_inner = np.s_[fringe:-fringe, fringe:-fringe]
                >>> a = a[ind_inner]

            in which case a will only contain the cells for which all data was
            present
        if ``reduce`` is True:
            numpy ndarray of the function applied on input array ``a``. The shape
            will be the original shape divided by the ``window_size``. Dimensions
            remain equal. No border of NaN values is present.
    """
    r = focal_sum(a, window_size=window_size, mask=mask, fraction_accepted=fraction_accepted, verbose=verbose, reduce=reduce)

    a_parsed = _parse_array(a)
    empty_flag, nan_flag, nan_mask = _parse_nans(a_parsed)
    mask_flag, mask_parsed, window_size_parsed, fringes, ind_inner = _parse_window(a_parsed, window_size, mask, reduce)

    count_values = _calc_count_values(window_size_parsed, mask_parsed, nan_mask, reduce, ind_inner)

    r[ind_inner] = r[ind_inner] / count_values
    return r


@timeit
@verify_keywords
def focal_std(a: Iterable,
              *,
              window_size: Optional[Union[int, Iterable[int]]] = None,
              mask: Optional[Iterable[bool]] = None,
              fraction_accepted: float = 0.7,
              verbose: bool = False,
              reduce: bool = False,
              std_df: int = 0) -> np.ndarray:
    """
    Focal standard deviation

    Parameters
    ----------
    a : array_like
        Input array (2D)
    window_size : int, array_like, optional
        Size of the window that is applied over ``a``. Should be a positive integer. If it is an integer it will be
        interpreted as (window_size, window_size). If a list is provided it needs to have the same number of entries
        as the number of dimensions as a.
    mask : array_like, optional
        A boolean array (2D). If provided, its shape will be used as ``window_size``, and its entries are used to mask
        every window.
    fraction_accepted : float, optional
        Fraction of valid cells (not NaN) per window that is deemed acceptable
        ``0``: all window are calculated if at least 1 value is present
        ``1``: only windows completely filled with values are calculated
        ``0-1``: fraction of acceptability
    verbose : bool, optional
        Verbosity with timing. False by default
    reduce : bool, optional
        The way in which the windowed array is created. If true, all values are used exactly once. If False (which is
        the default), values are reduced and the output array has the same shape as the input array, albeit with a
        border of nans where there are not enough values to calculate the cells.
    std_df : {1,0}, optional
        Degrees of freedom; meaning if the function is divided by the count of observations or the count of observations
        minus one. Should be 0 or 1. See :func:`numpy.std` for documentation.

    Returns
    -------
    :obj:`~numpy.ndarray`
        if ``reduce`` is False:
            numpy ndarray of the function applied to input array ``a``. The shape will
            be the same as the input array. The border of the map will be filled with nan,
            because of the lack of data to calculate the border. In the future other
            behaviours might be implemented. To obtain only the useful cells the
            following might be done:

                >>> window_size = 5
                >>> fringe = window_size // 2
                >>> ind_inner = np.s_[fringe:-fringe, fringe:-fringe]
                >>> a = a[ind_inner]

            in which case a will only contain the cells for which all data was
            present
        if ``reduce`` is True:
            numpy ndarray of the function applied on input array ``a``. The shape
            will be the original shape divided by the ``window_size``. Dimensions
            remain equal. No border of NaN values is present.
    """
    a_parsed = _parse_array(a)
    empty_flag, nan_flag, nan_mask = _parse_nans(a_parsed)
    mask_flag, mask_parsed, window_size_parsed, fringes, ind_inner = _parse_window(a_parsed, window_size, mask, reduce)

    if empty_flag:
        if verbose:
            print("- Empty array")
        return _create_output_array(a_parsed, window_size_parsed, reduce)

    return np.asarray(
        _focal_std(a_parsed, window_size_parsed, mask_parsed, fraction_accepted, reduce, std_df)
    )


@timeit
@verify_keywords
def focal_sum(a: Iterable,
              *,
              window_size: Optional[Union[int, Iterable[int]]] = None,
              mask: Optional[Iterable[bool]] = None,
              fraction_accepted: float = 0.7,
              verbose: bool = False,
              reduce: bool = False) -> np.ndarray:
    """
    Focal summation

    Parameters
    ----------
    a : array_like
        Input array (2D)
    window_size : int, array_like, optional
        Size of the window that is applied over ``a``. Should be a positive integer. If it is an integer it will be
        interpreted as (window_size, window_size). If a list is provided it needs to have the same number of entries
        as the number of dimensions as a.
    mask : array_like, optional
        A boolean array (2D). If provided, its shape will be used as ``window_size``, and its entries are used to mask
        every window.
    fraction_accepted : float, optional
        Fraction of valid cells (not NaN) per window that is deemed acceptable
        ``0``: all window are calculated if at least 1 value is present
        ``1``: only windows completely filled with values are calculated
        ``0-1``: fraction of acceptability
    verbose : bool, optional
        Verbosity with timing. False by default
    reduce : bool, optional
        The way in which the windowed array is created. If true, all values are used exactly once. If False (which is
        the default), values are reduced and the output array has the same shape as the input array, albeit with a
        border of nans where there are not enough values to calculate the cells.

    Returns
    -------
    :obj:`~numpy.ndarray`
        if ``reduce`` is False:
            numpy ndarray of the function applied to input array ``a``. The shape will
            be the same as the input array. The border of the map will be filled with nan,
            because of the lack of data to calculate the border. In the future other
            behaviours might be implemented. To obtain only the useful cells the
            following might be done:

                >>> window_size = 5
                >>> fringe = window_size // 2
                >>> ind_inner = np.s_[fringe:-fringe, fringe:-fringe]
                >>> a = a[ind_inner]

            in which case a will only contain the cells for which all data was
            present
        if ``reduce`` is True:
            numpy ndarray of the function applied on input array ``a``. The shape
            will be the original shape divided by the ``window_size``. Dimensions
            remain equal. No border of NaN values is present.
    """
    a_parsed = _parse_array(a)
    empty_flag, nan_flag, nan_mask = _parse_nans(a_parsed)
    mask_flag, mask_parsed, window_size_parsed, fringes, ind_inner = _parse_window(a_parsed, window_size, mask, reduce)

    if empty_flag:
        if verbose:
            print("- Empty array")
        return _create_output_array(a_parsed, window_size_parsed, reduce)

    if not mask_flag:
        r = _create_output_array(a_parsed, window_size_parsed, reduce)
        a_parsed = a_parsed.copy()
        a_parsed[nan_mask] = 0

        below_fraction_accepted_mask = _calc_below_fraction_accepted_mask(
            window_size_parsed, mask_parsed, nan_mask, ind_inner, fraction_accepted, reduce)
        r[ind_inner] = rolling_sum(a_parsed, window_size=window_size_parsed, mask=None, reduce=reduce)
        r[ind_inner][below_fraction_accepted_mask] = np.nan
        return r

    return np.asarray(
        _focal_sum(a_parsed, window_size_parsed, mask_parsed, fraction_accepted, reduce)
    )


@timeit
@verify_keywords
def focal_majority(a: Iterable,
                   *,
                   window_size: Optional[Union[int, Iterable[int]]] = None,
                   mask: Optional[Iterable[bool]] = None,
                   fraction_accepted: float = 0.7,
                   verbose: bool = False,
                   reduce: bool = False,
                   majority_mode: str = 'nan') -> np.ndarray:
    """
    Focal majority

    Parameters
    ----------
    a : array_like
        Input array (2D)
    window_size : int, array_like, optional
        Size of the window that is applied over ``a``. Should be a positive integer. If it is an integer it will be
        interpreted as (window_size, window_size). If a list is provided it needs to have the same number of entries
        as the number of dimensions as a.
    mask : array_like, optional
        A boolean array (2D). If provided, its shape will be used as ``window_size``, and its entries are used to mask
        every window.
    fraction_accepted : float, optional
        Fraction of valid cells (not NaN) per window that is deemed acceptable
        ``0``: all window are calculated if at least 1 value is present
        ``1``: only windows completely filled with values are calculated
        ``0-1``: fraction of acceptability
    verbose : bool, optional
        Verbosity with timing. False by default
    reduce : bool, optional
        The way in which the windowed array is created. If true, all values are used exactly once. If False (which is
        the default), values are reduced and the output array has the same shape as the input array, albeit with a
        border of nans where there are not enough values to calculate the cells.
    majority_mode : {"nan", "ascending", "descending"}, optional
        Differt modes of dealing with more than one value occuring equally often:

            - ``nan``: when more than one class has the same score NaN will be assigned
            - ``ascending``: the first occurrence of the maximum count will be assigned
            - ``descending``: the last occurrence of the maximum count will be assigned

    Returns
    -------
    :obj:`~numpy.ndarray`
        if ``reduce`` is False:
            numpy ndarray of the function applied to input array ``a``. The shape will
            be the same as the input array. The border of the map will be filled with nan,
            because of the lack of data to calculate the border. In the future other
            behaviours might be implemented. To obtain only the useful cells the
            following might be done:

                >>> window_size = 5
                >>> fringe = window_size // 2
                >>> ind_inner = np.s_[fringe:-fringe, fringe:-fringe]
                >>> a = a[ind_inner]

            in which case a will only contain the cells for which all data was
            present
        if ``reduce`` is True:
            numpy ndarray of the function applied on input array ``a``. The shape
            will be the original shape divided by the ``window_size``. Dimensions
            remain equal. No border of NaN values is present.
    """
    modes = {'nan': MajorityMode.nan,
             'ascending': MajorityMode.ascending,
             'descending': MajorityMode.descending}

    c_mode = modes[majority_mode]

    a_parsed = _parse_array(a)
    empty_flag, nan_flag, nan_mask = _parse_nans(a_parsed)
    mask_flag, mask_parsed, window_size_parsed, fringes, ind_inner = _parse_window(a_parsed, window_size, mask, reduce)

    if empty_flag:
        if verbose:
            print("- Empty array")
        return _create_output_array(a_parsed, window_size_parsed, reduce)

    return np.asarray(
        _focal_majority(a_parsed, window_size_parsed, mask_parsed, fraction_accepted, reduce, c_mode)
    )
