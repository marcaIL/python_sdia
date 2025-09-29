import numpy as np
from scipy.signal.windows import gaussian


def generate_gaussian_kernel(sigma_X,sigma_Y,M,N):
    w_x=gaussian(M,std=sigma_X)
    w_y=gaussian(N,std=sigma_Y)
    kernel=w_y[:,np.newaxis]*w_x[np.newaxis,:]
    kernel/=kernel.sum()
    return kernel

def padding(X,M,N):
    shp=X.shape
    if (shp[0]>M or shp[1]>N):
        raise Exception("Dimensions problem")
    result=np.zeros((M,N))
    result[:shp[0],:shp[1]]=X[:shp[0],:shp[1]]
    return result
