import numpy
from Cython.Build import cythonize
from distutils.extension import Extension
from setuptools import find_packages, setup

numpy_include_dir = numpy.get_include()

extensions = [
    # FOCAL STATS
    Extension(
        "pyspatialstats.focal_stats._iteration_params",
        ["pyspatialstats/focal_stats/_iteration_params.pyx"],
        include_dirs=[numpy_include_dir]
    ),
    Extension(
        "pyspatialstats.focal_stats._focal_correlation_core",
        ["pyspatialstats/focal_stats/_focal_correlation_core.pyx"],
        include_dirs=[numpy_include_dir]
    ),
    Extension(
        "pyspatialstats.focal_stats._focal_statistics_core",
        ["pyspatialstats/focal_stats/_focal_statistics_core.pyx"],
        include_dirs=[numpy_include_dir]
    ),
    # GROUPED STATS
    Extension(
        "pyspatialstats.grouped_stats._grouped_count",
        ["pyspatialstats/grouped_stats/_grouped_count.pyx"],
        include_dirs=[numpy_include_dir]
    ),
    Extension(
        "pyspatialstats.grouped_stats._grouped_min",
        ["pyspatialstats/grouped_stats/_grouped_min.pyx"],
        include_dirs=[numpy_include_dir]
    ),
    Extension(
        "pyspatialstats.grouped_stats._grouped_max",
        ["pyspatialstats/grouped_stats/_grouped_max.pyx"],
        include_dirs=[numpy_include_dir]
    ),
    Extension(
        "pyspatialstats.grouped_stats._grouped_mean",
        ["pyspatialstats/grouped_stats/_grouped_mean.pyx"],
        include_dirs=[numpy_include_dir]
    ),
    Extension(
        "pyspatialstats.grouped_stats._grouped_std",
        ["pyspatialstats/grouped_stats/_grouped_std.pyx"],
        include_dirs=[numpy_include_dir]
    ),
    Extension(
        "pyspatialstats.grouped_stats._grouped_correlation",
        ["pyspatialstats/grouped_stats/_grouped_correlation.pyx"],
        include_dirs=[numpy_include_dir]
    ),
    Extension(
        "pyspatialstats.grouped_stats._grouped_linear_regression",
        ["pyspatialstats/grouped_stats/_grouped_linear_regression.pyx"],
        include_dirs=[numpy_include_dir]
    ),
    # STRATA STATS
    Extension(
        "pyspatialstats.strata_stats._strata_stats",
        ["pyspatialstats/strata_stats/_strata_stats.pyx"],
        include_dirs=[numpy_include_dir]
    )
]

setup(
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    include_dirs=[],
)
