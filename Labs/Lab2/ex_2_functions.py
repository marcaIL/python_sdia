from numpy import sqrt

"""
In this file you will find some complementary functions used for the exercise 2
"""

def mean(X):
    """
    Returns the mean of a list X of elements
    """
    if (len(X)==0):
        return None
    return sum(X)/len(X)

def variance(X,mean):
    """
    Returns the variance of a list X of elements
    """
    if (len(X)==0 or mean==None):
        return None
    n=len(X)
    values = [(x-mean)**2 for x in X]
    return sum(values)/n

def line(x,f):
    """
    Returns the linear value for a double x with f line coeffecients

    Parameters:
    x(double): point of evaluation,
    f(tuple): regression line coefficients
    """
    return f[0]*x+f[-1]
