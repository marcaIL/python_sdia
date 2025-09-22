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
