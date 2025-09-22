import pytest
import numpy as np
from ex_3_functions import XDh,DvX, generate_random_matrix,generate_random_matrix_m_n,scalar_product,scalar_product_C

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


######
def test_generate_random_matrix():
    matrix =generate_random_matrix()
    assert isinstance(matrix, np.ndarray)
    assert matrix.ndim==2
    assert 2<=matrix.shape[0] <=10
    assert 2<= matrix.shape[1] <=10

def test_generate_random_matrix_m_n_valid():
    m,n = 4,5
    matrix = generate_random_matrix_m_n(m,n)
    assert isinstance(matrix,np.ndarray)
    assert matrix.shape==(m, n)

def test_generate_random_matrix_m_n_invalid():
    assert generate_random_matrix_m_n(1,1) is None
    assert generate_random_matrix_m_n(0,0) is None

def test_scalar_product_C_basic():
    A = np.array([[1,2], [3,4]])
    B = np.array([[5,6], [7,8]])
    result =scalar_product_C(A,B)
    expected=np.trace(np.dot(A.T, B))
    assert result==expected

def test_scalar_product_C_invalid():
    with pytest.raises(Exception, match="Error on matrix dimensions"):
        scalar_product_C(np.ones((2,2,2)), np.ones((2,2)))

def test_scalar_product_basic():
    U0 = np.array([[1,2], [3,4]])
    U1 = np.array([[5,6], [7,8]])
    V0 = np.array([[2,0], [1,3]])
    V1 = np.array([[4,1], [0,2]])
    U = np.stack([U0, U1])
    V = np.stack([V0, V1])
    result=scalar_product(U, V)
    expected = scalar_product_C(U0,V0) + scalar_product_C(U1,V1)
    assert result==expected

def test_scalar_product_invalid():
    with pytest.raises(Exception,match="Error on matrix dimensions"):
        scalar_product(np.ones((2,2)), np.ones((2,2,2)))
