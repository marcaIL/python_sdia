import numpy as np
import matplotlib.pyplot as plt

def interpolate(A,B):
    """
    Compute the interpolation intersection between the two vectors and unit ball
    """
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    #We check the existence of an intersection with unit ball
    simple_check(A,B,norm_A,norm_B)
    #After calculation, we want to find the root of this polynom
    #between 0 and 1 (should exist now)
    coeffs = np.array([
       np.linalg.norm(A-B)**2,
       -2*norm_B**2 + 2*np.dot(A,B),
       norm_B**2-1
    ])
    roots = np.roots(coeffs)
    positive_and_compatible = roots[(roots>=0) & (roots<=1)]
    if (positive_and_compatible.size==0):
        return None
    return positive_and_compatible[0]


def simple_check(A,B,norm_A,norm_B):
    """
    Check quickly the existence of an intersection with the unit ball
    and return the smallest vector

    Args:
    A(np.ndarray): first vector
    B(np.ndarray): second vector
    norm_A(double): norm of first vector
    norm_B(double): norm of second vector

    Return:
    np.ndarray: None or the smallest vector
    """
    if (norm_A<1 and norm_B<1):
        return None
    elif (norm_A==1  and norm_B<1):
        return A
    elif (norm_A)==1  and (norm_B<1):
        return B
    elif (norm_A)==1  and (norm_B==1):
        return A


def plot_brownian_motion(nb,N,x,step,rng,brownian_f):

    """
    Plot a fixed number of brownian trajectories and crossing point with the unit ball

    Args:
    nb(int): number of trajectories between 0 and 5
    brownian_f(function): brownian movement function
    """
    if (nb<0 or nb>5):
        return
    colors=["red","pink","yellow","green","orange","blue"]
    circle = plt.Circle((0,0), 1)
    fig, ax = plt.subplots()
    plt.xlim(-1.25,1.25)
    plt.ylim(-1.25,1.25)
    plt.grid(linestyle = "--", zorder = 1)

    for i in range(nb):
        result=brownian_f(N,x,step,rng)
        points=np.array(result[0])
        boundary_point=result[-1]
        X=points[:,0]
        Y=points[:,1]
        plt.plot(X,Y,linestyle="-",marker="x",color=colors[i])
        if boundary_point is not None:
            plt.scatter(boundary_point[0],boundary_point[1],color=colors[i],marker="o",s=100,label=f"boundary point {i}",zorder=10)

    plt.title("Brownian motion within B(0,1)")
    ax.set_aspect(1)
    ax.add_artist(circle)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
