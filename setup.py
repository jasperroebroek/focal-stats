import numpy
from Cython.Build import cythonize
from distutils.extension import Extension
from setuptools import setup

extensions = [
    Extension(
        "focal_stats.focal_stats.iteration_params",
        ["focal_stats/focal_stats/iteration_params.pyx"]),
    Extension(
        "focal_stats.focal_stats.focal_correlation",
        ["focal_stats/focal_stats/focal_correlation.pyx"]),
    Extension(
        "focal_stats.focal_stats.focal_statistics",
        ["focal_stats/focal_stats/focal_statistics.pyx"])
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)
