import pytest
import numpy as np
from ex_3_functions import XDh,DvX

def test_XDh_basic():
    X=np.array([[1, 2, 3],
                  [4, 5, 6]])
    expected=np.array([[1, 1, 0],
                         [1, 1, 0]])
    result=XDh(X)
    np.testing.assert_array_equal(result, expected)

def test_XDh_invalid_input():
    X = np.zeros((2, 2, 2))
    with pytest.raises(Exception, match="The input array as more than two dimensions"):
        XDh(X)

def test_DvX_basic():
    X=np.array([[1, 2],
                  [4, 5],
                  [7, 8]])
    expected=np.array([[3, 3],
                         [3, 3],
                         [0, 0]])
    result=DvX(X)
    np.testing.assert_array_equal(result,expected)

def test_DvX_invalid_input():
    X = np.zeros((3,2,1))
    with pytest.raises(Exception, match="The input array as more than two dimensions"):
        DvX(X)

pytest.main(["-v"])
