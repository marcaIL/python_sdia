# cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt


# Define numpy dtype aliases for clarity
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t ITYPE_t


cdef euclidean_distance(
    DTYPE_t[:] x,
    DTYPE_t[:] y
    ):

    cdef float distance
    cdef int n_features, idx

    n_features = x.shape[0]
    distance = 0.0
    for idx in range(n_features):
        distance += (x[idx] - y[idx])**2

    return sqrt(distance)


def knn_optimized(
    cnp.ndarray[DTYPE_t, ndim=2] x_train,
    cnp.ndarray[ITYPE_t, ndim=1] class_train,
    cnp.ndarray[DTYPE_t, ndim=2] x_test,
    int N_test,
    int K
):
    cdef int i, j, n_train, n_features, label, max_count
    n_train = x_train.shape[0]
    n_features = x_train.shape[1]

    #For label counts
    cdef dict counts

    # Create arrays for predicted classes
    # Remark : we used np.empty instead of np.zeros to save memory
    cdef cnp.ndarray[ITYPE_t, ndim=1] class_pred = np.empty(N_test, dtype=np.int64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] distance_xi = np.empty(n_train, dtype=np.float64)
    cdef cnp.ndarray[ITYPE_t, ndim=1] labels = np.empty(K, dtype=np.int64)

    for i in range(N_test):
        # Compute Euclidean distances
        for j in range(n_train):
            distance_xi[j] = euclidean_distance(x_test[i], x_train[j])

        # Sort and take top K
        # We used argpartition to speed up (log(n) instead of nlog(n) with argsort)
        id_sorted = np.argpartition(distance_xi, K)[:K]
        labels = class_train[id_sorted]

    class_pred[i] = np.bincount(labels).argmax()

    return class_pred
