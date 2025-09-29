import numpy as np
from scipy.signal import gaussian
from functions_ex_2 import *

def test_generate_gaussian_kernel_shape():
    kernel=generate_gaussian_kernel(1,1,5,7)
    assert kernel.shape==(7,5)

def test_generate_gaussian_kernel_sum():
    kernel=generate_gaussian_kernel(1,1,3,3)
    assert np.isclose(kernel.sum(), 1.0)

def test_padding_shape():
    X =np.ones((2,3))
    padded=padding(X,4,5)
    assert padded.shape==(4,5)

def test_padding_values():
    X = np.array([[1,2],[3,4]])
    padded = padding(X,3,3)
    expected = np.array([[1,2,0],
                         [3,4,0],
                         [0,0,0]])
    assert np.array_equal(padded,expected)

def test_padding_too_small_raises():
    X = np.ones((5,5))
    try:
        padding(X,3,3)
    except Exception as e:
        assert str(e)=="Dimensions problem"
    else:
        assert False,"Expected exception not raised"
