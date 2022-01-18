.. currentmodule:: focal_stats

#############
API reference
#############

Focal statistics
================

Focal statistics functions operating on array-like input data. They share functionality of ``window_size``
(determining the size of the sliding window), ``mask`` (superseding ``window_size`` and allowing for non-rectangular
windows), ``fraction_accepted`` (nan-behaviour based on the fraction of available data in the input window) and
``reduce`` (switch between returning same shape as input data and patching the sliding window without overlapping,
leading to a much smaller output array). :func:`focal_correlation` calculates the correlation between two arrays in
contrast to the other functions that operate on a single array.

.. autosummary::
    :toctree: generated/focal_statistics

    focal_min
    focal_max
    focal_mean
    focal_std
    focal_sum
    focal_majority
    focal_correlation

Rolling functions
=================

This module additionally implements standalone rolling functions, accepting ND arrays and accepting ``window_size``,
``mask`` and ``reduce`` similarly to the focal statistics functionality. It does however not guard against NaN
occurrences specifically, staying with the raw numpy behaviour. The functions are intended to be used to define custom
focal statistics functionality, potentially in higher than 2D dimensionality.

.. todo; implement example of how this might work (e.g. based on rolling_sum)

.. autosummary::
    :toctree: generated/rolling

    rolling_window
    rolling_sum
    rolling_mean
