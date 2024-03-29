from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os


def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as file:
        return file.read()


extensions = [Extension("focal_stats.core.iteration_params",
                        ["focal_stats/core/iteration_params.pyx"]),
              Extension("focal_stats.core.focal_correlation",
                        ["focal_stats/core/focal_correlation.pyx"]),
              Extension("focal_stats.core.focal_statistics",
                        ["focal_stats/core/focal_statistics.pyx"])
              ]

setup(
    name='focal_stats',
    version='0.0.6',
    packages=find_packages(),
    url='',
    license='MIT',
    author='Jasper Roebroek',
    author_email='roebroek.jasper@gmail.com',
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
    setup_requires=['cython', 'numpy', 'setuptools'],
    install_requires=['numpy'],
    extras_require={
        'develop': ['cython', 'sphinx', 'sphinx_rtd_theme', 'numpydoc', 'scipy', 'jupyter', 'matplotlib',
                    'pytest', 'joblib'],
        'test': ['scipy', 'pytest', 'rasterio']},
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
)
