import tempfile
import time
from functools import wraps
from typing import Tuple, Optional, Union, Iterable, List

import numpy as np

from focal_stats.rolling import _parse_window_and_mask


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        if kwargs is None:
            verbose = False
        else:
            verbose = kwargs.get('verbose', False)

        if verbose:
            print(f'{func.__name__}{args}{kwargs} Took {total_time:.4f} seconds')

        return result

    return timeit_wrapper


def verify_keywords(func):
    @wraps(func)
    def verifier(*args, **kwargs):
        if kwargs is not None:
            if not isinstance(kwargs.get('verbose', True), bool):
                raise TypeError("'verbose' is a boolean variable")
            if not isinstance(kwargs.get('reduce', True), bool):
                raise TypeError("'reduce' is a boolean variable")
            if not isinstance(kwargs.get('flatten', True), bool):
                raise TypeError("'flatten' is a boolean variable")
            if not isinstance(kwargs.get('window_size', 5), (int, tuple, list, np.ndarray)):
                raise TypeError("'window_size' needs to be an integer or iterable of integers")
            if not 0 <= kwargs.get('fraction_accepted', True) <= 1:
                raise ValueError("fraction_accepted should be between 0 and 1")
            if not kwargs.get("std_df", 0) in (0, 1):
                raise ValueError("std_df needs to be either 0 or 1")
            if not kwargs.get("majority_mode", 'nan') in ("nan", "ascending", "descending"):
                raise ValueError("majority_mode needs to be one of {'nan', 'ascending', 'descending'}")

        return func(*args, **kwargs)

    return verifier


def _parse_array(a: Iterable) -> np.ndarray:
    a_array = np.array(a, dtype=np.float64)
    if a_array.ndim != 2:
        raise IndexError("Only 2D data is supported")
    return a_array


def _parse_nans(a: np.ndarray) -> Tuple[bool, bool, np.ndarray]:
    if not np.issubdtype(a.dtype, np.floating):
        empty_flag = False
        nan_flag = False
        nan_mask = np.zeros(a.shape, dtype=np.bool_)

    else:
        nan_mask = np.isnan(a)

        if ~nan_mask.sum() == 0:
            empty_flag = True
        else:
            empty_flag = False

        if nan_mask.sum() == 0:
            nan_flag = False
        else:
            nan_flag = True

    return empty_flag, nan_flag, nan_mask


def _parse_window(a: np.ndarray,
                  window_size: Optional[Union[int, Iterable[int]]],
                  mask: Optional[Iterable[bool]],
                  reduce: bool) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, Tuple[slice, slice]]:
    window_size_parsed, mask_parsed = _parse_window_and_mask(a, window_size, mask, reduce)

    if mask_parsed is None:
        mask_flag = False
        mask_parsed = np.ones(window_size_parsed, dtype=np.bool_)
    else:
        mask_flag = True

    if reduce:
        fringes = np.array((0, 0))
        ind_inner = np.s_[:, :]
    else:
        if not np.all(window_size_parsed % 2):
            raise ValueError("Uneven window size is not allowed when not reducing")

        fringes = window_size_parsed // 2
        ind_inner = np.s_[fringes[0]:-fringes[0], fringes[1]:-fringes[1]]

    return mask_flag, mask_parsed, window_size_parsed, fringes, ind_inner


def _create_output_array(a: np.ndarray, window_size: np.ndarray, reduce: bool):
    if reduce:
        shape = list(np.array(a.shape) // window_size)
    else:
        shape = a.shape

    return np.full(shape, dtype=np.float64, fill_value=np.nan)


def _calc_count_values(window_size: np.ndarray,
                       mask: np.ndarray,
                       nan_mask: np.ndarray,
                       reduce: bool,
                       ind_inner: Tuple[slice, slice]) -> np.ndarray:
    from focal_stats.core.focal_statistics import focal_sum
    from focal_stats.rolling import rolling_sum

    if ~mask.sum() > 0:
        count_values = np.asarray(
            focal_sum(~nan_mask, window_size=window_size, mask=mask,
                      reduce=reduce, fraction_accepted=1)
        )[ind_inner]
    else:
        count_values = rolling_sum(~nan_mask, window_size=window_size, mask=None, reduce=reduce)

    if not reduce:
        count_values[nan_mask[ind_inner]] = 0

    return count_values


def _calc_below_fraction_accepted_mask(window_size: np.ndarray,
                                       mask: np.ndarray,
                                       nan_mask: np.ndarray,
                                       ind_inner: Tuple[slice, slice],
                                       fraction_accepted: float,
                                       reduce: bool) -> np.ndarray:
    threshold = fraction_accepted * mask.sum()
    count_values = _calc_count_values(window_size, mask, nan_mask, reduce, ind_inner)

    return count_values < threshold


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