from scipy.special import gamma
import scipy
import numpy as np

"""
In this file you will find some complementary functions used for exercise 1
"""

#First definition of the density by hand
def p(x,alpha,beta):
    if (x<=0):
        return 0
    first=(beta**alpha)/gamma(alpha)
    second=x**(alpha-1)*np.exp(-beta*x)
    return first*second

#Second definition using scipy.stats.gamma
def p2(x,alpha, beta):
    return scipy.stats.gamma.pdf(x,a=alpha, scale=1/beta)
