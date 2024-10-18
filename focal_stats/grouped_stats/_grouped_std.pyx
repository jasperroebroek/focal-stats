# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np

from libc.math cimport isnan, sqrt
from libc.stdlib cimport calloc, free, malloc
from numpy.math cimport NAN
cimport numpy as cnp

from ._grouped_count cimport _define_max_ind, _grouped_count
from ._grouped_mean cimport _grouped_mean


cdef float* _grouped_std(size_t[:] ind, float[:] v, float *mean_v, size_t max_ind) except * nogil:
    cdef:
        size_t i, k, n = ind.shape[0]
        long *num_v = <long *> calloc(max_ind + 1, sizeof(long))
        float *dev_v = <float *> calloc(max_ind + 1, sizeof(float))
        float *std_v = dev_v

    if num_v == NULL or dev_v == NULL:
        free(num_v)
        free(dev_v)
        with gil:
            raise MemoryError("num_v or dev_v memory error")

    for i in range(n):
        if ind[i] == 0:
            continue
        if isnan(v[i]):
            continue
        dev_v[ind[i]] += (v[i] - mean_v[ind[i]]) ** 2
        num_v[ind[i]] += 1

    for k in range(max_ind + 1):
        if num_v[k] == 0:
            std_v[k] = NAN
        else:
            std_v[k] = sqrt(dev_v[k] / num_v[k])

    free(num_v)
    return dev_v


def grouped_std_npy(size_t[:] ind, float[:] v) -> np.ndarray:
    cdef:
        float *mean_v, *r
        size_t max_ind

    try:
        with nogil:
            max_ind = _define_max_ind(ind)
            mean_v = _grouped_mean(ind, v, max_ind)
            r = _grouped_std(ind, v, mean_v, max_ind)

        result_array = cnp.PyArray_SimpleNewFromData(1, [max_ind + 1], cnp.NPY_FLOAT, r)
        cnp.PyArray_ENABLEFLAGS(result_array, cnp.NPY_ARRAY_OWNDATA)

    finally:
        free(mean_v)

    return result_array


def grouped_std_npy_filtered(size_t[:] ind, float[:] v) -> np.ndarray:
    cdef:
        size_t i, max_ind, c = 0, num_inds = 0
        long* count_v
        float *r_v, *mean_v, *rf_v

    try:
        with nogil:
            max_ind = _define_max_ind(ind)
            mean_v = _grouped_mean(ind, v, max_ind)
            r_v = _grouped_std(ind, v, mean_v, max_ind)
            count_v = _grouped_count(ind, v, max_ind)

            for i in range(max_ind + 1):
                if count_v[i] > 0:
                    num_inds += 1

            rf_v = <float *> malloc(num_inds * sizeof(float))

            if rf_v == NULL:
                with gil:
                    raise MemoryError("rf_v memory error")

            for i in range(max_ind + 1):
                if count_v[i] > 0:
                    rf_v[c] = r_v[i]
                    c += 1

        result_array = cnp.PyArray_SimpleNewFromData(1, [num_inds], cnp.NPY_FLOAT, rf_v)
        cnp.PyArray_ENABLEFLAGS(result_array, cnp.NPY_ARRAY_OWNDATA)

    finally:
        free(r_v)
        free(count_v)
        free(mean_v)

    return result_array
