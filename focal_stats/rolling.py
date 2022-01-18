#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to create a windowed view on the input array. Is similar to `np.lib.stride_tricks.sliding_window_view`,
but has some convenient options build in. Windows can be either defined by window size as an int (which casts it to all
dimensions) or a list, which is analogous to `sliding_window_view`. A mask can be provided, which explicitly defines
a window, which can have different shapes than rectangular. The `reduce` parameter results in non-overlapping windows,
stacking side by side.
"""
import numpy as np
from numpy.lib.stride_tricks import as_strided


def _parse_window_and_mask(a, window_size, mask, reduce):
    if mask is None:
        if window_size is None:
            raise ValueError("Neither window_size nor mask is set")
        window_size = np.asarray(window_size, dtype=np.intp).flatten()
        if window_size.size == 1:
            window_size = window_size.repeat(a.ndim)
        if np.all(window_size == 1):
            raise ValueError("Window_size can't only contain ones.")
    else:
        mask = np.asarray(mask, dtype=bool)
        window_size = np.asarray(mask.shape)

    if window_size.size != a.ndim:
        raise IndexError("Length of window_size (or the mask that defines it) should either be the same length as the "
                         "shape of `a`, or it needs to be 1.")

    if np.any(window_size < 1):
        raise ValueError("Window_size should be a positive integer")

    shape = np.asarray(a.shape)

    if np.any(np.array(a.shape) < window_size):
        raise ValueError(f"Window bigger than input array: {shape}, {window_size}")

    if reduce:
        if not np.array_equal(shape // window_size, shape / window_size):
            raise ValueError("not all dimensions are divisible by window_size")

    return window_size, mask


def rolling_window(a, *, window_size=None, mask=None, flatten=False, reduce=False, **kwargs):
    """
    Takes an array and returns a windowed version
    
    Parameters
    ----------
    a : array_like
        Input array
    window_size : int, array_like, optional
        Size of the window that is applied over ``a``. Should be a positive integer. If it is an integer it will be
        interpreted as ``(window_size, ) * a.ndim``. If a list is provided it needs to have the same length as the
        number of dimensions as a.
    mask : array_like, optional
        A boolean array. Its shape will be used as ``window_size``, and its entries are used to mask every window,
        resulting in dimensionality ``a.ndim + 1`` as the final result, just as in the case of `flatten` is True.
    flatten : bool, optional
        Flag to flatten the windowed view to 1 dimension. The shape of the returned array if set to True will be:

            *reduce* == False:
                shape : (s - window_size + 1) + (np.prod(window_size),)
            *reduce* == True:
                shape : (s // window_size) + (np.prod(window_size),)

        If set to False (which is the default) the shape of the window will not change and the data will be added in as
        many dimensions as the input array. The shape will be:

            *reduce* == False:
                shape : (s - window_size + 1) + (window_size)
            *reduce* == True:
                shape : (s // window_size) + (window_size)

        False has the nice property of returning a view, not copying the data while if True is passed, all the data will
        be copied. This can be very slow and memory intensive for large arrays.
    reduce : bool, optional
        Reuse data if set to False (which is the default) in which case an array will be returned with dimensions that
        are close to the original; see ``flatten``. If set to true, every entry is used exactly once. Creating much
        smaller dimensions.
    kwargs : dict, optional
        Arguments for :func:`~numpy.lib.stride_tricks.as_strided`, notably ``subok`` and ``writeable`` (see numpy
        documentation).

    Returns
    -------
    :obj:`~numpy.ndarray`
        windowed array of the data
        
    Raises
    ------
    ValueError
        - window_size too bigger than on of the dimensions of the input array
        - if reduce is True, the window_size needs to be an exact divisor for all dimensions of the input array
    """
    a = np.asarray(a)

    if not isinstance(flatten, bool):
        raise TypeError("flatten needs to be a boolean variable")
    if not isinstance(reduce, bool):
        raise TypeError("'reduce' needs to be a boolean variable")

    window_size, mask = _parse_window_and_mask(a, window_size, mask, reduce)

    shape = np.array(a.shape)
    strides = np.array(a.strides)

    if reduce:
        output_shape = np.r_[shape // window_size, window_size]
        output_strides = np.r_[strides * window_size, strides]

    else:
        output_shape = np.r_[shape - window_size + 1, window_size]
        output_strides = np.r_[strides, strides]

    # create view on the data with new shape and strides
    strided_a = as_strided(a, shape=output_shape, strides=output_strides, **kwargs)

    if mask is not None:
        strided_a = strided_a[..., mask]
    elif flatten:
        strided_a = strided_a.reshape(strided_a.shape[:a.ndim] + (-1,))

    return strided_a


def rolling_mean(a, *, window_size=None, mask=None, reduce=False):
    """
    Takes an array and returns the rolling mean. Not suitable for arrays with NaN values.
    
    Parameters
    ----------
    a : array_like
        Input array
    window_size : int, array_like, optional
        Size of the window that is applied over ``a``. Should be a positive integer. If it is an integer it will be
        interpreted as ``(window_size, ) * a.ndim``. If a list is provided it needs to have the same length as the
        number of dimensions as a.
    mask : array_like, optional
        A boolean array. Its shape will be used as ``window_size``, and its entries are used to mask every window,
        resulting in dimensionality ``a.ndim + 1`` as the final result, just as in the case of `flatten` is True.
    reduce : bool, optional
        Reuse data if set to False (which is the default) in which case an array will be returned with dimensions that
        are close to the original; see ``flatten``. If set to true, every entry is used exactly once. Creating much
        smaller dimensions.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Rolling mean over array ``a``. Resulting shape depends on reduce parameter. See :func:`rolling_window` for
        documentation. If a mask is provided, the last dimension has the length of the sum of ``mask``.
    """
    window_size, mask = _parse_window_and_mask(a, window_size, mask, reduce)
    if mask is None:
        div = np.prod(window_size)
    else:
        div = mask.sum()

    return rolling_sum(a, window_size=window_size, mask=mask, reduce=reduce) / div


def rolling_sum(a, *, window_size=None, mask=None, reduce=False):
    """
    Takes an array and returns the rolling sum. Not suitable for arrays with NaN values.

    Parameters
    ----------
    a : array_like
        Input array
    window_size : int, array_like, optional
        Size of the window that is applied over ``a``. Should be a positive integer. If it is an integer it will be
        interpreted as ``(window_size, ) * a.ndim``. If a list is provided it needs to have the same length as the
        number of dimensions as a.
    mask : array_like, optional
        A boolean array. Its shape will be used as ``window_size``, and its entries are used to mask every window,
        resulting in dimensionality ``a.ndim + 1`` as the final result, just as in the case of `flatten` is True.
    reduce : bool, optional
        Reuse data if set to False (which is the default) in which case an array will be returned with dimensions that
        are close to the original; see ``flatten``. If set to true, every entry is used exactly once. Creating much
        smaller dimensions.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Rolling sum over array ``a``. Resulting shape depends on reduce parameter. See :func:`rolling_window` for
        documentation. If a mask is provided, the last dimension has the length of the sum of ``mask``.
    """
    a = np.asarray(a)

    window_size, mask = _parse_window_and_mask(a, window_size, mask, reduce)

    if mask is not None or reduce is True:
        if mask is None:
            axis = tuple(range(a.ndim, 2 * a.ndim))
        else:
            axis = -1

        return rolling_window(a, window_size=window_size, reduce=reduce, mask=mask).sum(axis=axis)

    else:
        if np.issubdtype(a.dtype, np.bool_):
            dtype = np.int
        else:
            dtype = a.dtype

        shape = np.asarray(a.shape)
        r = np.zeros(shape + 1, dtype=dtype)
        r[(slice(1, None),) * a.ndim] = a

        for i in range(a.ndim):
            if window_size[i] == 1:
                continue
            else:
                ind1 = [slice(None)] * a.ndim
                ind1[i] = slice(window_size[i], None)
                ind1 = tuple(ind1)
                ind2 = [slice(None)] * a.ndim
                ind2[i] = slice(None, -window_size[i])
                ind2 = tuple(ind2)

                np.cumsum(r, axis=i, out=r)
                r[ind1] = r[ind1] - r[ind2]

        s = ()
        for i in range(a.ndim):
            s = s + (slice(window_size[i], None),)

        return r[s]
