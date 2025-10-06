# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt


# Define numpy dtype aliases for clarity
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t ITYPE_t


def knn_optimized_v1(
    cnp.ndarray[DTYPE_t, ndim=2] x_train,
    cnp.ndarray[ITYPE_t, ndim=1] class_train,
    cnp.ndarray[DTYPE_t, ndim=2] x_test,
    int N_test,
    int K
):
    """
    KNN classification implemented in Cython, following the same approach as in ex1, but optimized by computing Euclidean distances manually in nested loops rather than using np.linalg.norm.
    """"

    cdef int n_train = x_train.shape[0]
    cdef int n_features = x_train.shape[1]
    cdef int i, j, k, best_label

    # Output array
    cdef cnp.ndarray[ITYPE_t, ndim=1] class_pred = np.empty(N_test, dtype=np.int64)

    # Temporary arrays
    cdef cnp.ndarray[DTYPE_t, ndim=1] distances = np.empty(n_train, dtype=np.float64)
    cdef cnp.ndarray[ITYPE_t, ndim=1] indices
    cdef cnp.ndarray[ITYPE_t, ndim=1] labels

    cdef double diff, dist

    for i in range(N_test):
        # Compute distances manually
        for j in range(n_train):
            dist = 0.0
            for k in range(n_features):
                diff = x_test[i, k] - x_train[j, k]
                dist += diff * diff
            distances[j] = sqrt(dist)

        # Get indices of K nearest neighbors
        indices = np.argpartition(distances, K)[:K]

        # Get corresponding class labels
        labels = class_train[indices]

        # Majority vote
        best_label = np.bincount(labels).argmax()
        class_pred[i] = best_label

    return class_pred


def knn_optimized_v2(
    double[:, :] x_train,
    long[:] class_train,
    double[:, :] x_test,
    int N_test,
    int K
):

    """
    KNN classification in Cython as in knn_optimized_v1, adding typed memoryviews to speed up computation.
    """

    cdef int n_train = x_train.shape[0]
    cdef int n_features = x_train.shape[1]
    cdef int i, j, k, kk, best_label

    # Arrays definitions
    cdef cnp.ndarray[ITYPE_t, ndim=1] class_pred = np.empty(N_test, dtype=np.int64)

    cdef cnp.ndarray[DTYPE_t, ndim=1] distances = np.empty(n_train, dtype=np.float64)
    cdef cnp.ndarray[ITYPE_t, ndim=1] indices
    cdef cnp.ndarray[ITYPE_t, ndim=1] labels = np.empty(K, dtype=np.int64)

    cdef double diff, dist

    for i in range(N_test):
        # Compute distances manually
        for j in range(n_train):
            dist = 0.0
            for k in range(n_features):
                diff = x_test[i, k] - x_train[j, k]
                dist += diff * diff
            distances[j] = sqrt(dist)

        # Get indices of K nearest neighbors
        indices = np.argpartition(distances, K)[:K]

        # Get corresponding class labels
        for kk in range(K):
            labels[kk] = class_train[indices[kk]]

        # Get final label
        best_label = np.bincount(labels).argmax()
        class_pred[i] = best_label

    return class_pred
