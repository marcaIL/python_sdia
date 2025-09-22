import pytest
from ex_2_functions import mean,variance,line

def test_mean_basic():
    X=[1, 2, 3]
    assert mean(X)==2

def test_mean_single_value():
    X=[10]
    assert mean(X)==10

def test_mean_empty():
    X=[]
    assert mean(X) is None

def test_variance_basic():
    X =[1,2,3]
    m=mean(X)
    assert variance(X,m)==pytest.approx(0.66666666,rel=1e-6)

def test_variance_zero():
    X=[10, 10, 10]
    m=mean(X)
    assert variance(X,m)==0

def test_variance_empty():
    X =[]
    m= 0
    assert variance(X,m) is None

def test_variance_none_mean():
    X = [1,2]
    m =None
    assert variance(X, m) is None

def test_line_basic():
    x=2
    f=(3, 1)
    assert line(x, f) == 7

def test_line_zero():
    x=0
    f=(5, -2)
    assert line(x,f) == -2

def test_line_negative():
    x = -1
    f = (2,4)
    assert line(x,f) == 2

pytest.main(["-v"])
