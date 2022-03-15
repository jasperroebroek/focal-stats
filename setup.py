from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension("focal_stats.core.utils",
                        ["focal_stats/core/utils.pyx"]),
              Extension("focal_stats.core.focal_correlation",
                        ["focal_stats/core/focal_correlation.pyx"]),
              Extension("focal_stats.core.focal_statistics",
                        ["focal_stats/core/focal_statistics.pyx"])
              ]

setup(
    name='focal_stats',
    version='0.0.1',
    packages=find_packages(),
    url='',
    license='MIT',
    author='Jasper Roebroek',
    author_email='roebroek.jasper@gmail.com',
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
    install_requires=['numpy'],
    extras_require={
        'develop': ['cython', 'sphinx', 'sphinx_rtd_theme', 'numpydoc', 'scipy', 'jupyter', 'rasterio', 'matplotlib', 'pytest'],
        'test': ['scipy', 'pytest']
    }
)
