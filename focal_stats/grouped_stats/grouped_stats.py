import pandas as pd
from numpydantic import Shape
from numpydantic.dtype import Float32, Int64
from numpydantic.ndarray import NDArray

from focal_stats.grouped_stats._grouped_correlation import (
    grouped_correlation_npy,
    grouped_correlation_npy_filtered,
)
from focal_stats.grouped_stats._grouped_linear_regression import (
    grouped_linear_regression_npy,
    grouped_linear_regression_npy_filtered,
)
from focal_stats.grouped_stats._grouped_std import (
    grouped_std_npy,
    grouped_std_npy_filtered,
)
from focal_stats.grouped_stats.utils import generate_index, grouped_fun, grouped_fun_pd, parse_array
from focal_stats.grouped_stats._grouped_count import (
    grouped_count_npy,
    grouped_count_npy_filtered,
)
from focal_stats.grouped_stats._grouped_max import (
    grouped_max_npy,
    grouped_max_npy_filtered,
)
from focal_stats.grouped_stats._grouped_mean import (
    grouped_mean_npy,
    grouped_mean_npy_filtered,
)
from focal_stats.grouped_stats._grouped_min import (
    grouped_min_npy,
    grouped_min_npy_filtered,
)


def grouped_max(ind: NDArray, v: NDArray) -> NDArray[Shape["*"], Float32]:
    return grouped_fun(grouped_max_npy, ind=ind, v=v)


def grouped_max_pd(ind: NDArray, v: NDArray) -> pd.DataFrame:
    return grouped_fun_pd(grouped_max_npy_filtered, name="maximum", ind=ind, v=v)


def grouped_min(ind: NDArray, v: NDArray) -> NDArray[Shape["*"], Float32]:
    return grouped_fun(grouped_min_npy, ind=ind, v=v)


def grouped_min_pd(ind: NDArray, v: NDArray) -> pd.DataFrame:
    return grouped_fun_pd(grouped_min_npy_filtered, name="minimum", ind=ind, v=v)


def grouped_count(ind: NDArray, v: NDArray) -> NDArray[Shape["*"], Int64]:
    return grouped_fun(grouped_count_npy, ind=ind, v=v)


def grouped_count_pd(ind: NDArray, v: NDArray) -> pd.DataFrame:
    return grouped_fun_pd(grouped_count_npy_filtered, name="count", ind=ind, v=v)


def grouped_mean(ind: NDArray, v: NDArray) -> NDArray[Shape["*"], Float32]:
    return grouped_fun(grouped_mean_npy, ind=ind, v=v)


def grouped_mean_pd(ind: NDArray, v: NDArray) -> pd.DataFrame:
    return grouped_fun_pd(grouped_mean_npy_filtered, name="mean", ind=ind, v=v)


def grouped_std(ind: NDArray, v: NDArray) -> NDArray[Shape["*"], Float32]:
    return grouped_fun(grouped_std_npy, ind=ind, v=v)


def grouped_std_pd(ind: NDArray, v: NDArray) -> pd.DataFrame:
    return grouped_fun_pd(grouped_std_npy_filtered, name="std", ind=ind, v=v)


def grouped_mean_std_pd(ind: NDArray, v: NDArray) -> pd.DataFrame:
    ind = parse_array("ind", ind).ravel()
    v = parse_array("v", v).ravel()

    if ind.size != v.size:
        raise IndexError(f"Arrays are not all of the same size: {ind.size=}, {v.size=}")

    index = generate_index(ind, v)
    mean_v = grouped_mean_npy_filtered(ind, v)
    std_v = grouped_std_npy_filtered(ind, v)

    return pd.DataFrame(data={"mean": mean_v, "std": std_v}, index=index)


def grouped_correlation(ind: NDArray, v1: NDArray, v2: NDArray) -> NDArray[Shape["*"], Float32]:
    return grouped_fun(grouped_correlation_npy, ind=ind, v1=v1, v2=v2)


def grouped_correlation_pd(ind: NDArray, v1: NDArray, v2: NDArray) -> pd.DataFrame:
    return grouped_fun_pd(grouped_correlation_npy_filtered, name="correlation", ind=ind, v1=v1, v2=v2)


def grouped_linear_regression(ind: NDArray, v1: NDArray, v2: NDArray) -> NDArray[Shape["*"], Float32]:
    return grouped_fun(grouped_linear_regression_npy, ind=ind, v1=v1, v2=v2)


def grouped_linear_regression_pd(ind: NDArray, v1: NDArray, v2: NDArray) -> pd.DataFrame:
    return grouped_fun_pd(grouped_linear_regression_npy_filtered, name="lr", ind=ind, v1=v1, v2=v2)
