"""
We put back here the functions used for TV computation in the lab 2
"""
import numpy as np
def TV(X):
    """
    Computes the discrete isotropic total variation of a 2D matrix.

    Args:
    X(np.ndarray): the input matrix

    Return:
    float: the value of the total variation, calculated using the gradient2D function
    """
    gradient=gradient2D(X)
    norm1,norm2= np.abs(gradient[0])**2, np.abs(gradient[1])**2
    sum_matrix = norm1+norm2
    square=np.sqrt(sum_matrix)
    result =np.sum(square)
    return result

def gradient2D(X):
    """
    Computes the horizontal and vertical gradient of a 2D matrix.

    Args:
    X(np.ndarray): the input matrix

    Return:
    np.ndarray a 3D Volume containing the two gradient matrix, horizontal and vertical
    """
    if (X.ndim!=2):
        raise Exception("The input array as more than two dimensions")
    return np.stack([XDh(X),DvX(X)],axis=0)

#Gradient functions

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
    first=X[:,1:]
    second=X[:,:-1]
    #We compute the difference of the columns
    difference=first-second
    #We generate the 0 vector
    zero_col=np.zeros((shape[0],1))
    result = np.c_[difference,zero_col]
    return result

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
    first=X[1:,:].T
    second=X[:-1,:].T
    #We compute the difference of the columns
    difference=first-second
    #We generate the 0 vector
    zero_col=np.zeros((shape[-1],1))
    result = np.c_[difference,zero_col].T
    return result



#Complementary functions to test D operator and adjoint D*

def generate_random_matrix():
    """
    Returns a random matrix of random size
    """
    #We fix the seed for reproducibility
    seed =42
    rng =np.random.default_rng(seed)
    m = rng.integers(2,11)
    n = rng.integers(2,11)
    matrix = rng.random((m,n))
    return matrix

def generate_random_matrix_m_n(m,n):
    """
    Returns a random matrix of size (m,n)
    """
    if m<=1 and n<=1:
        return None
    #We fix the seed for reproducibility
    seed =42
    rng =np.random.default_rng(seed)
    matrix = rng.random((m,n))
    return matrix
