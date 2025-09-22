from numpy import sqrt

"""
In this file you will find some complementary functions used for the exercise 2
"""

def mean(X):
    if (len(X)==0):
        return None
    return sum(X)/len(X)

def variance(X,mean):
    if (len(X)==0 or mean==None):
        return None
    n=len(X)
    values = [(x-mean)**2 for x in X]
    return sum(values)/n

def line(x,f):
    return f[0]*x+f[-1]
