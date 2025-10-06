"""
We put back here the functions used for TV computation in the lab 2
"""


import numpy as np
from numba import njit,prange
import math

@njit(fastmath=True, parallel=True)
def TV_compact(X):
    n,m=X.shape
    result=0.
    for i in prange(n-1):
        for j in range(m-1):
            Dx=X[i,j+1]-X[i,j]
            dV =X[i+1,j] - X[i,j]
            result +=np.sqrt(Dx*Dx+dV*dV)
    return result


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
        g_1,g_2=grad_h[i],grad_v[i]
        for j in range(shape[1]):
            g_11=g_1[j]
            g_22=g_2[j]
            local_sum+=np.sqrt(np.abs(g_11)**2+np.abs(g_22)**2)
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
    result=np.empty(shape)
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
    result=np.empty(shape)
    for j in prange(shape[1]):
        #We keep a range here to avoid double parallelisation
        for i in range(shape[0]-1):
            result[i,j]=X[i+1,j]-X[i,j]
        result[shape[0]-1,j]=0.0
    return result
