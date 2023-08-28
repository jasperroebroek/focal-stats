from typing import Callable, Dict

import numpy as np
from joblib import Parallel, delayed

from focal_stats.core.utils import timeit
from focal_stats.windows import define_windows, WindowPair


def process_window(fn: Callable,
                   inputs: Dict[str, np.ndarray],
                   outputs: Dict[str, np.ndarray],
                   windows: WindowPair,
                   **kwargs) -> None:
    input_slices = windows.input.toslices()
    print({key: inputs[key][..., input_slices[0], input_slices[1]] for key in inputs})
    result = fn(**{key: inputs[key][..., input_slices[0], input_slices[1]] for key in inputs}, **kwargs)

    for key in outputs:
        output_slices = windows.output.toslices()
        outputs[key][..., output_slices[0], output_slices[1]] = result[key]


@timeit
def focal_function(fn: Callable,
                   inputs: Dict[str, np.ndarray],
                   outputs: Dict[str, np.ndarray],
                   window_size: int,
                   reduce: bool = False,
                   # joblib Parallel arg
                   n_jobs: int = 1,
                   verbose: bool = False,
                   prefer: str = 'threads',
                   # kwargs go to fn
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
