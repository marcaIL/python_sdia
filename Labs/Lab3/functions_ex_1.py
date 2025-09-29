import numpy as np
import matplotlib.pyplot as plt

def interpolate(A,B):
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    simple_check(A,B,norm_A,norm_B)
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
    if (norm_A<1 and norm_B<1):
        return None
    elif (norm_A==1  and norm_B<1):
        return A
    elif (norm_A)==1  and (norm_B<1):
        return B
    elif (norm_A)==1  and (norm_B==1):
        return A


def plot_brownian_motion(nb,N,x,step,rng,brownian_f):
    colors=["red","pink","yellow","green","orange"]
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
