"""
We put back here the functions used for TV computation in the lab 2
"""


import numpy as np
from numba import njit,prange

@njit(parallel=True, fastmath=True)
def TV_optimised(X):
    """
    Computes the discrete isotropic total variation of a 2D matrix.

    Args:
    X(np.ndarray): the input matrix

    Return:
    float: the value of the total variation, calculated using the gradient2D function
    """
    shape=X.shape
    grad_h=XDh(X)
    grad_v = DvX(X)

    row_sums=np.zeros(shape[0], dtype=np.float64)

    for i in prange(shape[0]):
        local_sum=0.0
        for j in range(shape[1]):
            g_1,g_2=grad_h[i,j],grad_v[i,j]
            #We do not use any function for sqrt
            local_sum+=np.sqrt(np.abs(g_1)**2+np.abs(g_2)**2)
        row_sums[i]=local_sum

    result=0.0
    for i in range(shape[0]):
        result+=row_sums[i]

    return result

#We deleted the usage of gradient2D, useless from an optimisation point of view

#Gradient functions

@njit(parallel=True)
def XDh(X):
    """
    Returns the horizontal gradient of the matrix X

    Parameters:
    X(np.ndarray): the input 2D matrix

    Returns:
    np.ndarray: 2D horizontal gradient matrix of X
    """
    if (X.ndim!=2):
        raise Exception("The input array as more than two dimensions")
    shape=X.shape
    #To avoid using python object for better numba optimisation, we change
    #the usage of np.c_, to a classic loop for calculation
    result=np.zeros(shape)#, dtype=np.complex128)
    for i in prange(shape[0]):
        #We keep a range here to avoid double parallelisation
        for j in range(shape[1]-1):
            #def of first gradient
            result[i,j]=X[i,j+1]-X[i,j]
        result[i,shape[1]-1]=0.0
    return result


@njit(parallel=True)
def DvX(X):
    """
    Returns the vertical gradient of the matrix X

    Parameters:
    X(np.ndarray): the input 2D matrix

    Returns:
    np.ndarray: 2D vertical gradient matrix of X
    """
    if (X.ndim!=2):
        raise Exception("The input array as more than two dimensions")
    shape=X.shape

    #As same as for XDh, we compute a classical parallel loop with numba
    #for better optimisation
    result=np.zeros(shape)#,  dtype=np.complex128)
    for j in prange(shape[1]):
        #We keep a range here to avoid double parallelisation
        for i in range(shape[0]-1):
            result[i,j]=X[i+1,j]-X[i,j]
        result[shape[0]-1,j]=0.0
    return result
