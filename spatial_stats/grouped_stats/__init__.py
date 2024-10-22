"""
ind     Contains group index, with 0 being no group allocation ind=0 is skipped
v*      Variable(s). If one, 'v', otherwise v1, v2 etc.
"""
from focal_stats.grouped_stats.utils import define_max_ind, generate_index
from focal_stats.grouped_stats.grouped_stats import (
    grouped_correlation,
    grouped_correlation_pd,
    grouped_count,
    grouped_count_pd,
    grouped_linear_regression,
    grouped_linear_regression_pd,
    grouped_max,
    grouped_max_pd,
    grouped_mean,
    grouped_mean_pd,
    grouped_mean_std_pd,
    grouped_min,
    grouped_min_pd,
    grouped_std,
    grouped_std_pd,
)

__all__ = [
    "define_max_ind",
    "generate_index",
    "grouped_count",
    "grouped_count_pd",
    "grouped_min",
    "grouped_min_pd",
    "grouped_max",
    "grouped_max_pd",
    "grouped_mean",
    "grouped_mean_pd",
    "grouped_std",
    "grouped_std_pd",
    "grouped_mean_std_pd",
    "grouped_correlation",
    "grouped_correlation_pd",
    "grouped_linear_regression",
    "grouped_linear_regression_pd",
]
