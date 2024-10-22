from typing import Callable, Generator, List, Tuple

import numpy as np
from pydantic import BaseModel, validate_call

from focal_stats.types import PositiveInt, Shape2D, UInt

# WindowFunction arguments correspond to keys in the `input` dictionary, and should return another dict with the keys
# corresponding to the `output` dictionary of the `focal_function` function.
WindowFunction = Callable[[List[np.ndarray]], List[float]]


class RasterWindow(BaseModel):
    col_off: UInt
    row_off: UInt
    width: PositiveInt
    height: PositiveInt

    @property
    def slices(self) -> tuple[slice, slice]:
        return (
            slice(self.row_off, self.row_off + self.height),
            slice(self.col_off, self.col_off + self.width),
        )


class RasterWindowPair(BaseModel):
    input: RasterWindow
    output: RasterWindow


def _construct_window_objects(
    raster_shape: Tuple[int, int],
    window_shape: Tuple[int, int],
    fringe: Tuple[int, int],
    step: Tuple[int, int],
) -> Generator[RasterWindow, None, None]:
    return (
        RasterWindow(
            col_off=x_idx, row_off=y_idx, width=window_shape[1], height=window_shape[0]
        )
        for y_idx in range(
            fringe[0], raster_shape[0] - window_shape[0] - fringe[0] + 1, step[0]
        )
        for x_idx in range(
            fringe[1], raster_shape[1] - window_shape[1] - fringe[1] + 1, step[1]
        )
    )


def _construct_tile_objects(
    raster_shape: Shape2D, tile_shape: Shape2D, fringe: Tuple[PositiveInt, PositiveInt]
) -> Generator[RasterWindow, None, None]:
    return (
        RasterWindow(
            col_off=x_idx - fringe[1],
            row_off=y_idx - fringe[0],
            width=tile_shape[1] + fringe[1] * 2,
            hight=tile_shape[0] + fringe[0] * 2,
        )
        for y_idx in range(0, raster_shape[0], tile_shape[0])
        for x_idx in range(0, raster_shape[1], tile_shape[1])
    )


@validate_call
def construct_windows(
    raster_shape: Shape2D, window_shape: Shape2D, reduce: bool = False
) -> Generator[RasterWindowPair, None, None]:
    """define slices for input and output data for windowed calculations"""
    if reduce:
        output_shape = (
            raster_shape[0] // window_shape[0],
            raster_shape[1] // window_shape[1],
        )
        output_fringe = (0, 0)
        input_step = window_shape
    else:
        output_shape = raster_shape
        output_fringe = (window_shape[0] // 2, window_shape[1] // 2)
        input_step = (1, 1)

    input_windows = _construct_window_objects(
        raster_shape, window_shape=window_shape, fringe=(0, 0), step=input_step
    )
    output_windows = _construct_window_objects(
        output_shape, window_shape=(1, 1), fringe=output_fringe, step=(1, 1)
    )

    return (
        RasterWindowPair(input=iw, output=ow)
        for iw, ow in zip(input_windows, output_windows)
    )


@validate_call
def construct_tiles(
    raster_shape: Shape2D,
    tile_shape: Shape2D,
    window_shape: Shape2D,
    reduce: bool = False,
) -> Generator[RasterWindowPair, None, None]:
    """define slices for input and output data for tiled and windowed calculations"""
    if reduce:
        fringe = (0, 0)
        output_shape = (
            raster_shape[0] // window_shape[0],
            raster_shape[1] // window_shape[1],
        )
        output_tile_shape = (
            tile_shape[0] // window_shape[0],
            tile_shape[1] // window_shape[1],
        )
    else:
        fringe = (window_shape[0] // 2, window_shape[1] // 2)
        output_shape = raster_shape
        output_tile_shape = tile_shape

    input_windows = _construct_tile_objects(
        raster_shape, tile_shape=tile_shape, fringe=fringe
    )
    output_windows = _construct_tile_objects(
        output_shape, tile_shape=output_tile_shape, fringe=(0, 0)
    )

    return (
        RasterWindowPair(input=iw, output=ow)
        for iw, ow in zip(input_windows, output_windows)
    )
