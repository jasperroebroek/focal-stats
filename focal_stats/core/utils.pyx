import tempfile
from typing import List, Optional, Tuple

import numpy as np
from libc.stdlib cimport malloc

cdef IterParams* _define_iter_params(size_t[2] shape,
                                     size_t[2] window_size,
                                     double fraction_accepted,
                                     bint reduce) nogil:
    cdef IterParams* ip = <IterParams*> malloc(sizeof(IterParams))

    if reduce:
        ip.shape[0] = shape[0] // window_size[0]
        ip.shape[1] = shape[1] // window_size[1]
        ip.fringe[0] = 0
        ip.fringe[1] = 0
        ip.stop[0] = shape[0]
        ip.stop[1] = shape[1]
        ip.step[0] = window_size[0]
        ip.step[1] = window_size[1]

    else:
        ip.shape[0] = shape[0]
        ip.shape[1] = shape[1]
        ip.fringe[0] = window_size[0] // 2
        ip.fringe[1] = window_size[1] // 2
        ip.stop[0] = shape[0] - window_size[0] + 1
        ip.stop[1] = shape[1] - window_size[1] + 1
        ip.step[0] = 1
        ip.step[1] = 1

    ip.iter[0] = ip.stop[0] / ip.step[0]
    ip.iter[1] = ip.stop[1] / ip.step[1]
    ip.num_values = window_size[0] * window_size[1]
    ip.threshold = fraction_accepted * ip.num_values

    return ip


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
