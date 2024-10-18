from typing import Sequence

import numpy as np
from numpydantic import NDArray
from pydantic import validate_call

from focal_stats.rolling.rolling_window import rolling_window
from focal_stats.types import Mask, PositiveInt
from focal_stats.window import Window, define_window, validate_window


@validate_call(config={"arbitrary_types_allowed": True})
def rolling_sum(
    a: NDArray,
    *,
    window: PositiveInt | Sequence[PositiveInt] | Mask | Window,
    reduce: bool = False,
) -> NDArray:
    """
    Takes an array and returns the rolling sum. Not suitable for arrays with NaN values.

    Parameters
    ----------
    a : array_like
        Input array
    window : int, array_like, Window
        Window that is applied over ``a``. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a mask or a window object. If a mask is provided, its raster_shape will be used to flatten ``a``,
        resulting in dimensionality ``a.ndim + 1`` as the final result, just as in the case of `flatten` is True.
    reduce : bool, optional
        Reuse data if set to False (which is the default) in which case an array will be returned with dimensions that
        are close to the original; see ``flatten``. If set to true, every entry is used exactly once. Creating much
        smaller dimensions.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Rolling sum over array ``a``. Resulting raster_shape depends on reduce parameter. See :func:`rolling_window` for
        documentation. If a mask is provided, the last dimension has the length of the sum of ``mask``.
    """
    a = np.asarray(a)
    shape = np.asarray(a.shape)

    window = define_window(window)
    validate_window(window, shape, reduce)

    window_shape = np.asarray(window.get_shape(a.ndim))

    if window.masked or reduce:
        axis = -1 if window.masked else tuple(range(a.ndim, 2 * a.ndim))

        return rolling_window(a, window=window, reduce=reduce).sum(axis=axis)

    if np.issubdtype(a.dtype, np.bool_):
        dtype = np.intp
    else:
        dtype = a.dtype

    r = np.zeros(shape + 1, dtype=dtype)
    r[(slice(1, None),) * a.ndim] = a

    for i in range(a.ndim):
        if window_shape[i] == 1:
            continue
        else:
            ind1 = [slice(None)] * a.ndim
            ind1[i] = slice(window_shape[i], None)
            ind1 = tuple(ind1)
            ind2 = [slice(None)] * a.ndim
            ind2[i] = slice(None, -window_shape[i])
            ind2 = tuple(ind2)

            np.cumsum(r, axis=i, out=r)
            r[ind1] = r[ind1] - r[ind2]

    s = ()
    for i in range(a.ndim):
        s = s + (slice(window_shape[i], None),)

    return r[s]


@validate_call(config={"arbitrary_types_allowed": True})
def rolling_mean(
    a: NDArray,
    *,
    window: PositiveInt | Sequence[PositiveInt] | Mask | Window,
    reduce: bool = False,
) -> NDArray:
    """
    Takes an array and returns the rolling mean. Not suitable for arrays with NaN values.

    Parameters
    ----------
    a : array_like
        Input array
    window : int, array_like, Window
        Window that is applied over ``a``. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a mask or a window object. If a mask is provided, its raster_shape will be used to flatten ``a``,
        resulting in dimensionality ``a.ndim + 1`` as the final result, just as in the case of `flatten` is True.
    reduce : bool, optional
        Reuse data if set to False (which is the default) in which case an array will be returned with dimensions that
        are close to the original; see ``flatten``. If set to true, every entry is used exactly once. Creating much
        smaller dimensions.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Rolling mean over array ``a``. Resulting raster_shape depends on reduce parameter. See :func:`rolling_window` for
        documentation. If a mask is provided, the last dimension has the length of the sum of ``mask``.
    """
    a = np.asarray(a)
    shape = np.asarray(a.shape)

    window = define_window(window)
    validate_window(window, shape, reduce)

    window_shape = np.asarray(window.get_shape(a.ndim))

    div = window.get_mask(a.ndim).sum() if window.masked else np.prod(window_shape)

    return rolling_sum(a, window=window, reduce=reduce) / div
