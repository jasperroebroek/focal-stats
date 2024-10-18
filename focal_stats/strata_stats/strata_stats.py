from typing import Callable

import numpy as np
from numpydantic import NDArray

from focal_stats.grouped_stats.utils import parse_data
from focal_stats.strata_stats._strata_stats import (
    _strata_correlation,
    _strata_linear_regression,
    _strata_max,
    _strata_mean,
    _strata_mean_std,
    _strata_min,
    _strata_std, StrataLinearRegressionResult, StrataCorrelationResult,
)
from focal_stats.types import RasterFloat32


def strata_fun(fun: Callable, ind: NDArray, **data) -> RasterFloat32:
    ind = np.asarray(ind)
    if ind.ndim != 2:
        raise IndexError("Only 2D data is supported")
    rows, cols = ind.shape
    parsed_data = parse_data(ind, **data)
    print([parsed_data[d].shape for d in parsed_data])
    return fun(rows=rows, cols=cols, **parsed_data)


def strata_min(ind: NDArray, v: NDArray) -> RasterFloat32:
    return strata_fun(_strata_min, ind=ind, v=v)


def strata_max(ind: NDArray, v: NDArray) -> RasterFloat32:
    return strata_fun(_strata_max, ind=ind, v=v)


def strata_mean(ind: NDArray, v: NDArray) -> RasterFloat32:
    return strata_fun(_strata_mean, ind=ind, v=v)


def strata_std(ind: NDArray, v: NDArray) -> RasterFloat32:
    return strata_fun(_strata_std, ind=ind, v=v)


def strata_mean_std(ind: NDArray, v: NDArray) -> RasterFloat32:
    return strata_fun(_strata_mean_std, ind=ind, v=v)


def strata_correlation(ind: NDArray, v1: NDArray, v2: NDArray) -> StrataCorrelationResult:
    return strata_fun(_strata_correlation, ind=ind, v1=v1, v2=v2)


def strata_linear_regression(ind: NDArray, v1: NDArray, v2: NDArray) -> StrataLinearRegressionResult:
    return strata_fun(_strata_linear_regression, ind=ind, v1=v1, v2=v2)
