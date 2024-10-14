from collections import namedtuple
from typing import Any, Callable, Generator, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from numpydantic.ndarray import NDArray
from pydantic import validate_call
from rasterio.windows import Window

from focal_stats.types import PositiveInt

# WindowFunction arguments correspond to keys in the `input` dictionary, and should return another dict with the keys
# corresponding to the `output` dictionary of the `focal_function` function.
WindowFunction = Callable[[List[np.ndarray]], List[float]]
WindowPair = namedtuple("WindowPair", ['input', 'output'])


def construct_windows(shape: Tuple[int, int],
                      window_size: Tuple[int, int],
                      fringe: Tuple[int, int],
                      step: Tuple[int, int]) -> Generator[Window, None, None]:
    return (Window(x_idx, y_idx, window_size[1], window_size[0])
            for y_idx in range(fringe[0], shape[0] - window_size[0] - fringe[0] + 1, step[0])
            for x_idx in range(fringe[1], shape[1] - window_size[1] - fringe[1] + 1, step[1]))


def construct_tiles(shape: Tuple[int, int],
                    tile_size: Tuple[int, int],
                    fringe: Tuple[int, int]) -> Generator[Window, None, None]:
    return (Window(x_idx - fringe[1], y_idx - fringe[0], tile_size[1] + fringe[1] * 2, tile_size[0] + fringe[0] * 2)
            for y_idx in range(0, shape[0], tile_size[0])
            for x_idx in range(0, shape[1], tile_size[1]))


def define_windows(shape: Tuple[int, int],
                   window_size: Union[int, Tuple[int, int]],
                   reduce: bool = False) -> Generator[NamedTuple, None, None]:
    """define slices for input and output data for windowed calculations"""
    if isinstance(window_size, int):
        window_size = window_size, window_size

    if window_size[0] < 1 or window_size[1] < 1:
        raise ValueError("Window needs to have at least size 1")

    if not shape[1] >= window_size[1]:
        raise ValueError("Window needs to be smaller than the shape Y")
    if not shape[0] >= window_size[0]:
        raise ValueError("Window needs to be smaller than the shape X")

    if reduce:
        if shape[1] % window_size[1] > 0:
            raise ValueError("Shape Y not divisible by window_size")
        if shape[0] % window_size[0] > 0:
            raise ValueError("Shape X not divisible by window_size")

    if reduce:
        output_shape = (shape[0] // window_size[0], shape[1] // window_size[1])
        output_fringe = (0, 0)
        input_step = window_size
    else:
        output_shape = shape
        output_fringe = (window_size[0] // 2, window_size[1] // 2)
        input_step = (1, 1)

    input_windows = construct_windows(shape, window_size=window_size, fringe=(0, 0), step=input_step)
    output_windows = construct_windows(output_shape, window_size=(1, 1), fringe=output_fringe, step=(1, 1))

    return (WindowPair(iw, ow) for iw, ow in zip(input_windows, output_windows))


def define_tiles(shape: Tuple[int, int],
                 tile_size: Union[int, Tuple[int, int]],
                 window_size: Union[int, Tuple[int, int]],
                 reduce: bool = False) -> Generator[NamedTuple, None, None]:
    """define slices for input and output data for tiled and windowed calculations"""
    if isinstance(window_size, int):
        window_size = window_size, window_size
    if isinstance(tile_size, int):
        tile_size = tile_size, tile_size

    if window_size[0] < 1 or window_size[1] < 1:
        raise ValueError("Window needs to have at least size 1")
    if tile_size[0] < 1 or tile_size[1] < 1:
        raise ValueError("Tiles needs to have at least size 1")

    if not shape[1] >= tile_size[1]:
        raise ValueError("Tiles needs to be smaller than the shape Y")
    if not shape[0] >= tile_size[0]:
        raise ValueError("Tiles needs to be smaller than the shape X")

    if shape[1] % tile_size[1] > 0:
        raise ValueError("Shape Y not divisible by tile_size")
    if shape[0] % tile_size[0] > 0:
        raise ValueError("Shape X not divisible by tile_size")

    if reduce:
        if tile_size[1] % window_size[1] > 0:
            raise ValueError("Tile Y not divisible by window_size")
        if tile_size[0] % window_size[0] > 0:
            raise ValueError("Tile X not divisible by window_size")

    if reduce:
        fringe = (0, 0)
        output_shape = shape[0] // window_size[0], shape[1] // window_size[1]
        output_tile_size = tile_size[0] // window_size[0], tile_size[1] // window_size[1]
    else:
        fringe = (window_size[0] // 2, window_size[1] // 2)
        output_shape = shape
        output_tile_size = tile_size

    input_windows = construct_tiles(shape, tile_size=tile_size, fringe=fringe)
    output_windows = construct_tiles(output_shape, tile_size=output_tile_size, fringe=(0, 0))

    return (WindowPair(iw, ow) for iw, ow in zip(input_windows, output_windows))


@validate_call
def _parse_window_and_mask(a: NDArray,
                           window_size: Optional[PositiveInt | Sequence[PositiveInt]],
                           mask: Optional[NDArray[Any, bool]],
                           reduce: bool) -> Tuple[List[PositiveInt], Optional[NDArray[Any, bool]]]:
    """Parses window size and mask.

    Mask takes the upper hand when not None, in which case window_size is overridden by the shape of the mask.
    If mask is not provided, also the output of this function will be None.

    Returns
    -------
    window_size, mask
    """
    if mask is None:
        if window_size is None:
            raise ValueError("Neither window_size nor mask is set")
        window_size = np.asarray(window_size, dtype=np.int32).flatten()
        if window_size.size == 1:
            window_size = window_size.repeat(a.ndim)
    else:
        mask = np.asarray(mask, dtype=bool)
        window_size = np.asarray(mask.shape, dtype=np.int32)

    if window_size.size != a.ndim:
        raise IndexError("Length of window_size (or the mask that defines it) should either be the same length as the "
                         "shape of `a`, or it needs to be 1.")

    shape = np.asarray(a.shape)

    if np.any(np.array(a.shape) < window_size):
        raise ValueError(f"Window bigger than input array: {shape}, {window_size}")

    if reduce:
        if not np.array_equal(shape // window_size, shape / window_size):
            raise ValueError("not all dimensions are divisible by window_size")

    return window_size.tolist(), mask
