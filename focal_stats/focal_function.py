import tempfile
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from focal_stats.raster_windows import WindowPair, define_windows
from joblib import Parallel, delayed

from focal_stats.utils import timeit


class MemmapContext:
    def __init__(self, shape: Tuple[int, int], window_size: int, reduce: bool, dtype: np.dtype = np.float64):
        if len(shape) != 2:
            raise IndexError("Only 2D")

        if reduce:
            self.memmap_shape = shape[0] // window_size, shape[1] // window_size
        else:
            self.memmap_shape = shape

        self.dtype = dtype
        self.open: bool = False
        self.temp_file = None
        self.memmap: Optional[np.memmap] = None

    def create(self) -> np.memmap:
        if not self.open:
            self.open = True
            self.temp_file = tempfile.NamedTemporaryFile(mode='w+')
            self.memmap = np.memmap(filename=self.temp_file.name, dtype=self.dtype, mode='w+', shape=self.memmap_shape)

        return self.memmap

    def close(self):
        if not self.open:
            raise FileNotFoundError("File is not open")
        else:
            self.open = False
            self.temp_file.close()

    def __enter__(self) -> np.memmap:
        return self.create()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()


class OutputDict:
    def __init__(self, keys: List[str], **kwargs):
        self.keys = keys
        self.kw = kwargs
        self.contexts = {}
        self.memmaps = {}

    def __enter__(self):
        for key in self.keys:
            self.contexts[key] = MemmapContext(**self.kw)
            self.memmaps[key] = self.contexts[key].create()
        return self.memmaps

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key in self.keys:
            self.contexts[key].close()


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
