import numpy as np
from Numeric_Integration import m
from Numeric_Integration import RR1f
from Numeric_Integration import MR2f
from Numeric_Integration import RR1a
from Numeric_Integration import SR4a
import matplotlib.pyplot as plt

def linear(x):
    return x

def RR1f_MR2f_Test():
    accu = 1

    sine = m(np.sin)
    skip = 1000
    step = 10
    interval = 10
    err = np.zeros(step)
    a = 0
    b = np.pi/2

    for i in range(step):
        n = skip+(i + 1)*interval
        err[i] = abs(accu - RR1f(a, b, n, sine)) * n

    plt.title("RR1f vs MR2f Test")
    plt.plot(np.arange(step)+1, err, 'r')

    for i in range(step):
        n = skip+(i + 1)*interval
        err[i] = abs(accu - MR2f(a, b, n, sine)) * n**2

    plt.plot(np.arange(step)+1, err, 'y')
    plt.show()

def Part_D():
    skip = 1000
    step = 10
    interval = 10
    a = 0
    b = 1
    err = np.zeros(step)
    weight_2 = 1000 #weight to make the graph more visible for small constant
    weight_3 = 100

    accu_1 = 2

    def f_1(x):
        return 1/np.sqrt(x)

    integ_1 = m(f_1)

    for i in range(step):
        n = skip+(i + 1)*interval
        err[i] = abs(accu_1 - MR2f(a, b, n, integ_1)) * n
    plt.title("Part D Function Comparison")
    plt.plot(np.arange(step)+1, err, 'r')


    accu_2 = 2/3

    def f_2(x):
        return np.sqrt(x)

    integ_2 = m(f_2)

    for i in range(step):
        n = skip+(i + 1)*interval
        err[i] = abs(accu_2 - MR2f(a, b, n, integ_2)) * n * weight_2
    plt.plot(np.arange(step)+1, err, 'y')


    accu_3 = 2/5

    def f_3(x):
        return x**(3/2)

    integ_3 = m(f_3)

    for i in range(step):
        n = skip+(i + 1)*interval
        err[i] = abs(accu_3 - MR2f(a, b, n, integ_3)) * n**2 * weight_3
    plt.plot(np.arange(step)+1, err, 'g')
    plt.show()

def Part_E(lam):
    def exponential(x):
        return lam*np.exp(-lam*x)

    approx, n0, status = RR1a(0, 1, 1, m(exponential), 0.0000001)
    print("approx: ", approx, " n0: ", n0, " status: ", status)
    #Does not work: tol too small

    approx, n0, status = RR1a(0, 1, 1, m(exponential), 0.00001)
    print("approx: ", approx, " n0: ", n0, " status: ", status)
    #Converges at n0 = 131072

def Part_F(t):
    def trig(x):
        return np.cos(t*x**2)

    approx, n0, status = RR1a(0, 1, 1, m(trig), 0.0001)
    print("approx: ", approx, " n0: ", n0, " status: ", status)

def Part_G(t):
    def trig(x):
        return np.cos(t*x**2)

    approx, n0, status = SR4a(0, 1, 1, m(trig), 0.0001)
    print("approx: ", approx, " n0: ", n0, " status: ", status)

def main():
    """
    For the order graphs, we want the coefficient to stay relatively constant
    such that we know it is within constant multiples of the order. The
    reciprocal of the order has been multiplied at the end of the error.
    e.g. for O(h^2) we divide it by 1/h^2, which is n^2
    """

    RR1f_MR2f_Test()

    Part_D()

    print("Part E: ")
    Part_E(2)

    print("Part F: ")
    for t in [1, 10, 100, 200, 300, 400, 500]:
        #too large t results in scipy (for accurate answer) breaking
        Part_F(t)

    print("Part G: ")
    for t in [1, 10, 100, 200, 300, 400, 500]:
        Part_G(t)

if __name__ == "__main__":
    main()
