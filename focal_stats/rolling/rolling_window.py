from typing import Sequence

import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpydantic.ndarray import NDArray
from pydantic import validate_call

from focal_stats.types import Mask, PositiveInt
from focal_stats.window import Window, define_window, validate_window


@validate_call(config={'arbitrary_types_allowed': True})
def rolling_window(a: NDArray, *,
                   window: PositiveInt | Sequence[PositiveInt] | Mask | Window,
                   flatten: bool = False,
                   reduce: bool = False,
                   **kwargs) -> NDArray:
    """
    Takes an array and returns a windowed version

    Parameters
    ----------
    a : array_like
        Input array
    window : int, array_like, Window
        Window that is applied over ``a``. It can be an integer or a sequence of integers, which will be interpreted as
        a rectangular window, a mask or a window object. If a mask is provided, its raster_shape will be used to flatten ``a``,
        resulting in dimensionality ``a.ndim + 1`` as the final result, just as in the case of `flatten` is True.
    flatten : bool, optional
        Flag to flatten the windowed view to 1 dimension. The raster_shape of the returned array if set to True will be:

            *reduce* == False:
                raster_shape : (s - window_shape + 1) + (np.prod(window_shape),)
            *reduce* == True:
                raster_shape : (s // window_shape) + (np.prod(window_shape),)

        If set to False (which is the default) the raster_shape of the window will not change and the data will be added in as
        many dimensions as the input array. The raster_shape will be:

            *reduce* == False:
                raster_shape : (s - window_shape + 1) + (window_shape)
            *reduce* == True:
                raster_shape : (s // window_shape) + (window_shape)

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
        - window_shape too bigger than on of the dimensions of the input array
        - if reduce is True, the window_shape needs to be an exact divisor for all dimensions of the input array
    """
    a = np.asarray(a)
    shape = np.asarray(a.shape)
    strides = np.asarray(a.strides)

    window = define_window(window)
    validate_window(window, shape, reduce)

    window_shape = np.asarray(window.get_shape(a.ndim))

    if reduce:
        output_shape = np.r_[shape // window_shape, window_shape]
        output_strides = np.r_[strides * window_shape, strides]

    else:
        output_shape = np.r_[shape - window_shape + 1, window_shape]
        output_strides = np.r_[strides, strides]

    # create view on the data with new raster_shape and strides
    strided_a = as_strided(a, shape=output_shape.astype(int), strides=output_strides.astype(int), **kwargs)

    if window.masked or flatten:
        return strided_a[..., window.get_mask(a.ndim)]

    return strided_a
