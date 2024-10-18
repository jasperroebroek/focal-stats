import numpy
from Cython.Build import cythonize
from distutils.extension import Extension
from setuptools import find_packages, setup

extensions = [
    # FOCAL STATS
    Extension(
        "focal_stats.focal_stats.iteration_params",
        ["focal_stats/focal_stats/iteration_params.pyx"],
    ),
    Extension(
        "focal_stats.focal_stats._focal_correlation_core",
        ["focal_stats/focal_stats/_focal_correlation_core.pyx"],
    ),
    Extension(
        "focal_stats.focal_stats._focal_statistics_core",
        ["focal_stats/focal_stats/_focal_statistics_core.pyx"],
    ),
    # GROUPED STATS
    Extension(
        "focal_stats.grouped_stats._grouped_count",
        ["focal_stats/grouped_stats/_grouped_count.pyx"],
    ),
    Extension(
        "focal_stats.grouped_stats._grouped_min",
        ["focal_stats/grouped_stats/_grouped_min.pyx"],
    ),
    Extension(
        "focal_stats.grouped_stats._grouped_max",
        ["focal_stats/grouped_stats/_grouped_max.pyx"],
    ),
    Extension(
        "focal_stats.grouped_stats._grouped_mean",
        ["focal_stats/grouped_stats/_grouped_mean.pyx"],
    ),
    Extension(
        "focal_stats.grouped_stats._grouped_std",
        ["focal_stats/grouped_stats/_grouped_std.pyx"],
    ),
    Extension(
        "focal_stats.grouped_stats._grouped_correlation",
        ["focal_stats/grouped_stats/_grouped_correlation.pyx"],
    ),
    Extension(
        "focal_stats.grouped_stats._grouped_linear_regression",
        ["focal_stats/grouped_stats/_grouped_linear_regression.pyx"],
    ),
    # Srata stats
    Extension(
        "focal_stats.strata_stats._strata_stats",
        ["focal_stats/strata_stats/_strata_stats.pyx"],
    )
]

setup(
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)
