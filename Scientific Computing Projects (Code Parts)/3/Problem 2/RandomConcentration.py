import numpy as np
import random as rn

def gen_rand(n, seed, mu, sig):
    rn.seed(seed)
    arr = np.zeros(n)
    for i in range(n):
        arr[i] = rn.normalvariate(mu, sig)
    return arr

def fake_data(m, n, A_seed, A_mu, A_sig, r_seed, r_mu, r_sig, xi_seed, xi_mu, \
    xi_sig):

    A = gen_rand(m, A_seed, A_mu, A_sig)
    r = gen_rand(m, r_seed, r_mu, r_sig)
    def F(j, t):
        ret = 0
        for i in range(m):
            ret += A[i]*np.exp(-r[i]*t)
        return ret

    t = (np.arange(n)+1)/10

    xi = gen_rand(n, xi_seed, xi_mu, xi_sig)

    F_ret = np.zeros(n)

    for j in range(n):
        F_ret[j] = F(j, t[j])+xi[j]

    return r, t, F_ret
