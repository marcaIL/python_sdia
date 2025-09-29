import numpy as np
from scipy.signal.windows import gaussian


def generate_gaussian_kernel(sigma_X,sigma_Y,M,N):
    """
    Generates a gaussian window of given size with given standard deviations.

    Args:
    sigma_X(double): x standard deviation
    sigma_Y(double): y standard deviation
    M(int): first dimension of the window
    N(int): second dimension of the window

    Return:
    np.ndarray: gaussian kernel of size MxN
    """
    w_x=gaussian(M,std=sigma_X)
    w_y=gaussian(N,std=sigma_Y)
    kernel=w_y[:,np.newaxis]*w_x[np.newaxis,:]
    kernel/=kernel.sum()
    return kernel

def padding(X,M,N):
    """
    Computes the padding of a given matrix X, with the required dimensions MxN
    """
    shp=X.shape
    if (shp[0]>M or shp[1]>N):
        raise Exception("Dimensions problem")
    result=np.zeros((M,N))
    result[:shp[0],:shp[1]]=X[:shp[0],:shp[1]]
    return result
