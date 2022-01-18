# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from copy import copy

import numpy as np
import time
from focal_stats.rolling import rolling_window, rolling_sum, _parse_window_and_mask
from .utils cimport _define_iter_params

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


class _FocalStats:
    def __init__(self, a, *, window_size, mask, fraction_accepted, reduce):
        if not isinstance(reduce, bool):
            raise TypeError("reduce is a boolean variable")
        if fraction_accepted < 0 or fraction_accepted > 1:
            raise ValueError("fraction_accepted should be between 0 and 1")

        self.window_size, self.mask = _parse_window_and_mask(a, window_size, mask, reduce)
        if self.mask is None:
            self.mask_flag = False
            self.mask = np.ones(self.window_size, dtype=np.bool_)
        else:
            self.mask_flag = True

        if not reduce:
            if np.any(window_size % 2 == 0):
                raise ValueError("window_size should be uneven if reduce is set to False")

        if np.issubdtype(a.dtype, np.floating):
            self.nans = np.isnan(a)
            self.values = ~self.nans

            if self.values.sum() == 0:
                self.empty = True
            else:
                self.empty = False

            if self.nans.sum() == 0:
                self.nan_flag = False
            else:
                self.nan_flag = True

        else:
            self.empty = False
            self.nan_flag = False
            self.values = np.ones(a.shape, dtype=np.bool_)
            self.nans = ~self.values

        self.a = a
        self.fraction_accepted = fraction_accepted
        self.reduce = reduce

    def construct_r(self):
        if self.reduce:
            shape = list(np.array(self.a.shape) // self.window_size)
        else:
            shape = self.a.shape

        return np.full(shape, dtype=np.float64, fill_value=np.nan)

    @property
    def fringes(self):
        if self.reduce:
            return np.array((0, 0))
        else:
            return self.window_size // 2

    @property
    def ind_inner(self):
        if self.reduce:
            return np.s_[:, :]
        else:
            return np.s_[self.fringes[0]:-self.fringes[0], self.fringes[1]:-self.fringes[1]]

    @property
    def threshold(self):
        if self.mask is None:
            return self.fraction_accepted * self.window_size.prod()
        else:
            return self.fraction_accepted * self.mask.sum()

    @property
    def count_values(self):
        if self.mask_flag:
            count_values = np.asarray(
                _focal_sum(self.values.astype(np.float64), window_size=self.window_size, mask=self.mask,
                           reduce=self.reduce, fraction_accepted=1)
            )[self.ind_inner]
        else:
            count_values = rolling_sum(self.values, window_size=self.window_size, mask=None, reduce=self.reduce)

        if not self.reduce:
            count_values[self.nans[self.ind_inner]] = 0
        count_values[count_values < self.threshold] = 0

        return count_values

    @property
    def not_enough_values(self):
        return self.count_values < self.threshold

    def max(self):
        if not self.nan_flag and not self.mask_flag:
            r = self.construct_r()
            r[self.ind_inner] = (
                rolling_window(self.a, window_size=self.window_size, mask=None, reduce=self.reduce)
                    .max(axis=(2, 3))
            )
        else:
            r = np.asarray(
                _focal_max(self.a, self.window_size, self.mask, self.fraction_accepted, self.reduce)
            )
        return r

    def min(self):
        if not self.nan_flag and not self.mask_flag:
            r = self.construct_r()
            r[self.ind_inner] = (
                rolling_window(self.a, window_size=self.window_size, mask=None, reduce=self.reduce)
                    .min(axis=(2, 3))
            )
        else:
            r = np.asarray(
                _focal_min(self.a, self.window_size, self.mask, self.fraction_accepted, self.reduce)
            )

        return r

    def sum(self):
        if not self.mask_flag:
            r = self.construct_r()
            a = self.a.copy()
            a[self.nans] = 0
            r[self.ind_inner] = rolling_sum(a, window_size=self.window_size, mask=None, reduce=self.reduce)
            r[self.ind_inner][self.not_enough_values] = np.nan
        else:
            r = np.asarray(
                _focal_sum(self.a, self.window_size, self.mask, self.fraction_accepted, self.reduce)
            )

        return r

    def mean(self):
        r = self.sum()
        r[self.ind_inner] = r[self.ind_inner] / self.count_values
        return r

    def std(self, dof=0):
        return np.asarray(
            _focal_std(self.a, self.window_size, mask=self.mask, fraction_accepted=self.fraction_accepted,
                       reduce=self.reduce, dof=dof)
        )

    def majority(self, mode='nan'):
        if mode == 'ascending':
            c_mode = MajorityMode.ascending
        elif mode == 'descending':
            c_mode = MajorityMode.descending
        else:
            c_mode = MajorityMode.nan

        return np.asarray(
            _focal_majority(self.a, self.window_size, mask=self.mask, fraction_accepted=self.fraction_accepted,
                            reduce=self.reduce, mode=c_mode)
        )


def focal_statistics(a, *, window_size=5, mask=None, func=None, fraction_accepted=0.7, verbose=False, std_df=0,
                     reduce=False, majority_mode="nan"):
    """
    Focal statistics.

    Parameters
    ----------
    a : array_like
        Input array (2D)
    window_size : int, array_like, optional
        Size of the window that is applied over 'a'. Should be a positive integer. If it is an integer it will be
        interpreted as (window_size, window_size). If a list is provided it needs to have the same number of entries
        as the number of dimensions as a.
    mask : array_like, optional
        A boolean array (2D). If provided, its shape will be used as window_size, and its entries are used to mask
        every window.
    func : {"mean","min","max","std","majority", "sum"}
        Function to be applied on the windows
    fraction_accepted : float, optional
        Fraction of valid cells (not NaN) per window that is deemed acceptable
        0: all window are calculated if at least 1 value is present
        1: only windows completely filled with values are calculated
        0-1: fraction of acceptability
    verbose : bool, optional
        Verbosity with timing. False by default
    std_df : {1,0}, optional
        Only for std calculations. Degrees of freedom; meaning if the function is divided by the count of
        observations or the count of observations minus one. Should be 0 or 1. See `numpy.std` for documentation.
    reduce : bool, optional
        The way in which the windowed array is created. If true, all values are used exactly once. If False (which is
        the default), values are reduced and the output array has the same shape as the input array, albeit with a
        border of nans where there are not enough values to calculate the cells.
    majority_mode : {"nan", "ascending", "descending"}, optional
        nan: when more than one class has the same score NaN will be assigned
        ascending: the first occurrence of the maximum count will be assigned
        descending: the last occurrence of the maximum count will be assigned.
        Parameter only used when the `func` is "majority".
        
    Returns
    -------
    :obj:`~numpy.ndarray`
        if `reduce` is False:
            numpy ndarray of the function applied to input array `a`. The shape will
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
        if `reduce` is True:
            numpy ndarray of the function applied on input array `a`. The shape
            will be the original shape divided by the `window_size`. Dimensions
            remain equal. No border of NaN values is present.
    """
    func_list = ["mean", "majority", "min", "max", "std", "sum"]
    if func not in func_list:
        raise ValueError(f"Function needs to be one of {func_list}")

    if not isinstance(verbose, bool):
        raise TypeError("verbose is a boolean variable")

    start = time.perf_counter()

    a = np.array(a, dtype=np.float64)
    if a.ndim != 2:
        raise IndexError("Only 2D data is supported")

    fs = _FocalStats(a, window_size=window_size, mask=mask, fraction_accepted=fraction_accepted, reduce=reduce)

    if fs.empty:
        r = fs.construct_r()

        if verbose:
            print("- Empty array")

    else:
        if func == "majority":
            r = fs.majority(majority_mode)

        elif func == "mean":
            r = fs.mean()

        elif func == "max":
            r = fs.max()

        elif func == "min":
            r = fs.min()

        elif func == "std":
            r = fs.std(std_df)

        elif func == "sum":
            r = fs.sum()

        if verbose:
            print(f"- focal {func}: {time.perf_counter() - start}")

    return r


def focal_min(a, **kwargs):
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
    return focal_statistics(a, func="min", **kwargs)


def focal_max(a, **kwargs):
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
    return focal_statistics(a, func="max", **kwargs)


def focal_mean(a, **kwargs):
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
    return focal_statistics(a, func="mean", **kwargs)


def focal_std(a, **kwargs):
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
    return focal_statistics(a, func="std", **kwargs)


def focal_majority(a, **kwargs):
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
    return focal_statistics(a, func="majority", **kwargs)


def focal_sum(a, **kwargs):
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
    return focal_statistics(a, func="sum", **kwargs)
