import numpy
from Cython.Build import cythonize
from distutils.extension import Extension
from setuptools import find_packages, setup

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
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)
