import numpy as np
import pytest
from functions_ex_1 import *

def test_simple_check_both_inside():
    A=np.array([0.5,0.5])
    B=np.array([0.6,0.6])
    result= simple_check(A, B, np.linalg.norm(A), np.linalg.norm(B))
    assert result is None

def test_simple_check_on_boundary():
    A=np.array([1.0,0.0])
    B=np.array([0.5,0.0])
    result=simple_check(A,B,np.linalg.norm(A), np.linalg.norm(B))
    assert np.allclose(result,A)

def test_interpolate_intersects():
    A=np.array([2.0,0.0])
    B=np.array([0.0,0.0])
    t=interpolate(A, B)
    point=(1-t)*A + t*B
    assert np.isclose(np.linalg.norm(point),1.0)

def test_interpolate_no_intersection_returns_none():
    A=np.array([0.5,0.0])
    B=np.array([0.4,0.0])
    t=interpolate(A, B)
    assert t is None
