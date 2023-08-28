import warnings
from typing import Dict

import numpy as np

from focal_stats import focal_mean
from focal_stats.core.focal_function import focal_function


def mean_fun_t(x: np.ndarray, key: str) -> Dict[str, float]:
    """Mean of first output is fed to the first output"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {key: np.nanmean(x).item()}


def test_focal_function():
    inputs = {'x': np.random.rand(10, 10)}
    outputs = {'y': np.full((2, 2), np.nan)}

    focal_function(mean_fun_t, inputs, outputs, window_size=5, reduce=True, key='y')

    # Check for equality
    m = focal_mean(inputs['x'], fraction_accepted=0, window_size=5, reduce=True)
    np.allclose(m, outputs['y'], equal_nan=True)


def test_focal_function_keyword_only():
    def _mean_fun_t_keyword_only(*, x: np.ndarray, key: str) -> Dict[str, float]:
        """Mean of first output is fed to the first output"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return {key: np.nanmean(x).item()}

    inputs = {'x': np.random.rand(10, 10)}
    outputs = {'y': np.full((2, 2), np.nan)}

    focal_function(_mean_fun_t_keyword_only, inputs, outputs, window_size=5, reduce=True, key='y')
