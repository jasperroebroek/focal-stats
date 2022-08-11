from collections import namedtuple
from typing import Tuple, List, Callable, NamedTuple, Generator, Dict

import numpy as np
from joblib import Parallel, delayed
from rasterio.windows import Window

WindowPair = namedtuple("WindowPair", ['input', 'output'])
WindowFunction = Callable[[List[np.ndarray]], List[float]]


def construct_windows(x: int, y: int, window_size: int, fringe: int, step: int) -> Generator[Window, None, None]:
    return (Window(x_idx, y_idx, window_size, window_size)
            for y_idx in range(fringe, y - window_size - fringe + 1, step)
            for x_idx in range(fringe, x - window_size - fringe + 1, step))


def define_windows(shape: Tuple[int, int], window_size: int, reduce: bool = False) -> Generator[NamedTuple, None, None]:
    """define slices for input and output data for windowed calculations"""
    if window_size < 1:
        raise ValueError("Window needs to have at least size 1")

    y, x = shape

    if not y >= window_size:
        raise ValueError("Window needs to be smaller than the shape Y")
    if not x >= window_size:
        raise ValueError("Window needs to be smaller than the shape X")

    if reduce:
        if y % window_size > 0:
            raise ValueError("Shape Y not divisible by window_size")
        if x % window_size > 0:
            raise ValueError("Shape X not divisible by window_size")

    if reduce:
        input_windows = construct_windows(x, y, window_size=window_size, fringe=0, step=window_size)
        output_windows = construct_windows(x // window_size, y // window_size, window_size=1, fringe=0, step=1)
    else:
        input_windows = construct_windows(x, y, window_size=window_size, fringe=0, step=1)
        output_windows = construct_windows(x, y, window_size=1, fringe=window_size // 2, step=1)

    return (WindowPair(iw, ow) for iw, ow in zip(input_windows, output_windows))


def process_window(fn: Callable,
                   inputs: Dict[str, np.ndarray],
                   outputs: Dict[str, np.ndarray],
                   windows: WindowPair,
                   **kwargs) -> None:
    input_slices = windows.input.toslices()
    result = fn(**{key: inputs[key][..., input_slices[0], input_slices[1]] for key in inputs}, **kwargs)

    for key in outputs:
        output_slices = windows.output.toslices()
        outputs[key][..., output_slices[0], output_slices[1]] = result[key]


def focal_function(fn: Callable,
                   inputs: Dict[str, np.ndarray],
                   outputs: Dict[str, np.ndarray],
                   window_size: int,
                   reduce: bool = False,
                   # joblib Parallel arg
                   n_jobs: int = 1,
                   verbose: bool = False,
                   prefer: str = 'threads',
                   **kwargs) -> None:
    """Focal statistics with an arbitrary function. prefer 'threads' always works, 'processes' only works with memmaps,
    but provides potentially large speed-ups"""

    shapes = []
    for key in inputs:
        s = inputs[key].shape[-2:]
        if len(s) != 2:
            raise IndexError("All inputs need to be at least 2D")
        shapes.append(s)

    for s in shapes:
        if not s == shapes[0]:
            raise IndexError(f'Not all input rasters have the same shape: {shapes}')

    for key in outputs:
        shape = outputs[key].shape[-2:]
        if (
                reduce and (shapes[0][0] // window_size, shapes[0][1] // window_size) != shape or
                not reduce and shape != shapes[0]
        ):
            raise IndexError(f"Output shapes not matching input shapes: {shapes[0]} {shape}")

    window_pairs = define_windows(shapes[0], window_size, reduce)

    Parallel(n_jobs=n_jobs, verbose=verbose, prefer=prefer, mmap_mode='r+')(
        delayed(process_window)(
            fn, inputs, outputs, wp, **kwargs) for wp in window_pairs
    )
