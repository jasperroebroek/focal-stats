import numpy
from Cython.Build import cythonize
from distutils.extension import Extension
from setuptools import find_packages, setup

extensions = [
    # FOCAL STATS
    Extension(
        "spatial_stats.focal_stats._iteration_params",
        ["spatial_stats/focal_stats/_iteration_params.pyx"],
    ),
    Extension(
        "spatial_stats.focal_stats._focal_correlation_core",
        ["spatial_stats/focal_stats/_focal_correlation_core.pyx"],
    ),
    Extension(
        "spatial_stats.focal_stats._focal_statistics_core",
        ["spatial_stats/focal_stats/_focal_statistics_core.pyx"],
    ),
    # GROUPED STATS
    Extension(
        "spatial_stats.grouped_stats._grouped_count",
        ["spatial_stats/grouped_stats/_grouped_count.pyx"],
    ),
    Extension(
        "spatial_stats.grouped_stats._grouped_min",
        ["spatial_stats/grouped_stats/_grouped_min.pyx"],
    ),
    Extension(
        "spatial_stats.grouped_stats._grouped_max",
        ["spatial_stats/grouped_stats/_grouped_max.pyx"],
    ),
    Extension(
        "spatial_stats.grouped_stats._grouped_mean",
        ["spatial_stats/grouped_stats/_grouped_mean.pyx"],
    ),
    Extension(
        "spatial_stats.grouped_stats._grouped_std",
        ["spatial_stats/grouped_stats/_grouped_std.pyx"],
    ),
    Extension(
        "spatial_stats.grouped_stats._grouped_correlation",
        ["spatial_stats/grouped_stats/_grouped_correlation.pyx"],
    ),
    Extension(
        "spatial_stats.grouped_stats._grouped_linear_regression",
        ["spatial_stats/grouped_stats/_grouped_linear_regression.pyx"],
    ),
    # Srata stats
    Extension(
        "spatial_stats.strata_stats._strata_stats",
        ["spatial_stats/strata_stats/_strata_stats.pyx"],
    )
]

setup(
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)
