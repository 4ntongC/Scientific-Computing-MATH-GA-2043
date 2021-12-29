from Functions import gradient
from Functions import hessian
from Functions import grid
import Functions
from scipy import linalg as la
import time
import numpy as np

def two_safe(xk, f, n, dx, max_t):
    """
    Pertubation with 2 safeguards, Labs(D)L and line search
    xk is an ndarray where the pertubation currently stands
    f is the function f, g, or h
    n is an int on how large the grid should be
    dx is a float on how fine the grid should be
    max_t is a float for stopping the line search in case
    """

    dim = len(xk)
    gridpts = grid(xk, n, dx)
    coord = tuple(np.zeros(dim).astype(int)+n)
    grad = gradient(f(gridpts), coord, dx)
    H = hessian(f(gridpts), coord, dx)
    L, D, perm = la.ldl(H)
    L = L[perm, :]
    D = np.abs(D)

    step = 1
    H_abs = L@D@np.transpose(L)
    if (la.det(H_abs) == 0):
        return xk, False
    dir = -1*la.inv(H_abs)@grad
    start = time.time()
    while f(xk) <= f(xk+step*dir):
        step /= 2
        if (time.time()-start) > max_t:
            return xk, False
    return xk+step*dir, True

def no_safe(xk, f, n, dx):
    """
    Pertubation with no safeguard
    xk is an ndarray where the pertubation currently stands
    f is the function f, g, or h
    n is an int on how large the grid should be
    dx is a float on how fine the grid should be
    """

    dim = len(xk)
    gridpts = grid(xk, n, dx)
    coord = tuple(np.zeros(dim).astype(int)+n)
    grad = gradient(f(gridpts), coord, dx)
    H = hessian(f(gridpts), coord, dx)
    if (la.det(H) == 0):
        return xk, False
    dir = -1*la.inv(H)@grad
    return xk+dir, True

def test_f():
    xk = np.zeros(1)+1000
    no_xk = xk
    two_xk = xk
    n = 5
    dx = 0.1
    max_t = 10
    step = 3
    no_cont = True
    two_cont = True
    for i in range(step):
        if (no_cont):
            no_xk, no_cont = no_safe(no_xk, Functions.f, n, dx)
        if (two_cont):
            two_xk, two_cont = two_safe(two_xk, Functions.f, n, dx, max_t)
    print("Starting point is: " + str(xk[0]))
    print("Newton's method ran for " + str(step) + " steps")
    print("The pertubation without safeguard stands at: {:+.2e}".format(no_xk[0]))
    print("The pertubation with safeguard stands at:    {:+.2e}".format(two_xk[0]))

def test_g():
    xk = np.zeros(2)+5
    no_xk = xk
    two_xk = xk
    n = 5
    dx = 0.1
    max_t = 10
    step = 10
    no_cont = True
    two_cont = True
    for i in range(step):
        if (no_cont):
            no_xk, no_cont = no_safe(no_xk, Functions.g, n, dx)
        if (two_cont):
            two_xk, two_cont = two_safe(two_xk, Functions.g, n, dx, max_t)
    print("Starting point is: " + str(xk))
    print("Newton's method ran for " + str(step) + " steps")
    print("The pertubation without safeguard stands at: {:.2e}".format(no_xk[0]))
    print("The pertubation with safeguard stands at:    {:.2e}".format(two_xk[0]))
def testh():
    return

test_f()
test_g()
