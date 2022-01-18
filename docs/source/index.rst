.. focal_stats documentation master file, created by
   sphinx-quickstart on Wed Dec  1 14:19:25 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

***************************
Focal statistics for python
***************************

This module aims to provide focal statistics for python, that runs without the installation of extensive GIS packages. 
The only dependency is numpy. Additionally, the package provides sliding window functionality to implement your own
focal statistics functions (see Tutorials). This is implemented in line with
:func:`numpy.lib.stride_tricks.sliding_window_view`, but extends its functionality.

*************
Documentation
*************

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation

.. toctree::
   :maxdepth: 2
   :caption: Methods

   methods

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/tutorial
   notebooks/custom_focal_stats

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   whats_new
   api


License
=======

focal_stats is published under a MIT license.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
