import numpy as np
from math import *

def f(x):
    return (x[0]**2+1)**(1/2)

def g(x):
    a = 1
    b = 5
    c = 1
    k = 5
    #bad practice, but parameters stored here for convenience in syntax
    return np.exp(a*x[0]**2+b*(x[1]-c*np.sin(k*x[0])))

def h(x, a):
    sum = 0
    n = len(x)
    for i in range(n-1):
        sum += (x[i+1]-x[i])**2+a*(x[i+1]-x[i])**4
    return x[1]**2+a*x[1]**4+sum+x[n]**2+a*x[n]**4

def gradient(f, coord, dx):
    """
    f is an np array containg sample values around point x
    coord is a tuple of the coord of x in f
    dim of coord must match dim of f
    """
    dim = f.ndim
    grad = []
    retval = np.gradient(f, dx)
    if dim == 1:
            grad.append(retval[coord])
    else:
        for i in range(dim):
            grad.append(retval[i][coord])
    return np.array(grad)

def hessian(f, coord, dx):

    """
    f is an np array containg sample values around point x
    coord is a tuple of the coord of x in f
    dim of coord must match dim of f
    """

    dim = len(coord)
    H = []
    for i in range(dim):
        H.append([])
        for j in range(dim):
            H[i].append(0)
    grad = np.gradient(f, dx)
    for j in range(len(H)):
        if (f.ndim > 1):
            partial = gradient(grad[j], coord, dx)
        else:
            partial = gradient(grad, coord, dx)
        for i in range(len(H)):
            H[i][j] = partial[i]
    return np.array(H)

def grid(xk, n, dx):
    dim = len(xk)
    axis = []
    for i in range(dim):
        axis.append(np.linspace(xk[i]-n*dx, xk[i]+n*dx, 2*n+1))
    axis = np.array(axis)
    return np.meshgrid(*axis)
