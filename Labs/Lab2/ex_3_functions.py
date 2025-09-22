import numpy as np

"""
In this file you will find some complementary functions for exercise 3
"""

#Gradient functions

def XDh(X):
    if (X.ndim!=2):
        raise Exception("The input array as more than two dimensions")
    shape=X.shape
    first=X[:,1:]
    second=X[:,:-1]
    #We compute the difference of the columns
    difference=first-second
    #We generate the 0 vector
    zero_col=np.zeros((shape[0],1))
    result = np.c_[difference,zero_col]
    return result

def DvX(X):
    if (X.ndim!=2):
        raise Exception("The input array as more than two dimensions")
    shape=X.shape
    first=X[1:,:].T
    second=X[:-1,:].T
    #We compute the difference of the columns
    difference=first-second
    #We generate the 0 vector
    zero_col=np.zeros((shape[-1],1))
    result = np.c_[difference,zero_col].T
    return result


#Adjoint gradient functions

def YDh(Y1):
    first_col=(-Y1[:,0])[:, np.newaxis]
    last_col=(Y1[:,-2])[:, np.newaxis]
    #We extract the matrix used with the difference
    first=Y1[:,1:-1]
    second=Y1[:,:-2]
    difference=-(first-second)
    result=np.c_[first_col,difference,last_col]
    return result

def DvY(Y2):
    first_line=-Y2[0].T
    last_line=Y2[-2].T
    #Matrix
    first=Y2[1:-1,:].T
    second=Y2[:-2,:].T
    difference=-(first-second)
    result=np.c_[first_line,difference,last_line].T
    return result


#Complementary functions to test D operator and adjoint D*

def generate_random_matrix():
    #We fix the seed for reproducibility
    seed =42
    rng =np.random.default_rng(seed)
    m = rng.integers(2,11)
    n = rng.integers(2,11)
    matrix = rng.random((m,n))
    return matrix

def generate_random_matrix_m_n(m,n):
    seed =42
    rng =np.random.default_rng(seed)
    matrix = rng.random((m,n))
    return matrix

def scalar_product(U,V):
    if (U.ndim !=3 or V.ndim !=3):
        raise Exception("Error on matrix dimensions")
    return scalar_product_C(U[0],V[0])+scalar_product_C(U[1],V[1])

def scalar_product_C(U,V):
    return np.dot(U.T,V).trace()
