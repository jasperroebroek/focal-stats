import os

import numpy
from Cython.Build import cythonize
from distutils.extension import Extension
from setuptools import find_packages, setup


def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as file:
        return file.read()


extensions = [Extension("focal_stats.focal_stats.iteration_params",
                        ["focal_stats/focal_stats/iteration_params.pyx"]),
              Extension("focal_stats.focal_stats.focal_correlation",
                        ["focal_stats/focal_stats/focal_correlation.pyx"]),
              Extension("focal_stats.focal_stats.focal_statistics",
                        ["focal_stats/focal_stats/focal_statistics.pyx"])
              ]

setup(
    name='focal_stats',
    version='0.1.0',
    packages=find_packages(),
    url='',
    license='MIT',
    author='Jasper Roebroek',
    author_email='roebroek.jasper@gmail.com',
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
    setup_requires=['cython', 'numpy', 'setuptools', 'rasterio'],
    install_requires=['numpy'],
    extras_require={
        'develop': ['cython', 'sphinx', 'sphinx_rtd_theme', 'numpydoc', 'scipy', 'jupyter', 'matplotlib',
                    'pytest', 'joblib'],
        'test': ['scipy', 'pytest']},
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
)
