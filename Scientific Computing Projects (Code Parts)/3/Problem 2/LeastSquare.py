from RandomConcentration import fake_data
import numpy as np
import numpy.linalg as la

def generate_A(r, t):
    m = len(r)
    n = len(t)
    A = np.zeros((n, m))
    for j in range(n):
        for i in range(m):
            A[j][i] = np.exp(-1*r[i]*t[j])
    return A

def Cholesky(A, b):
    b = np.transpose(A)@b
    A = np.transpose(A)@A
    L = la.cholesky(A)
    y = la.lstsq(L, b, rcond=None)[0]
    x = la.lstsq(np.transpose(L), y, rcond=None)[0]
    return x

def QR(A, b):
    Q, R = la.qr(A)
    y = la.lstsq(Q, b, rcond=None)[0]
    x = la.lstsq(R, y, rcond=None)[0]
    return x

def SVD(A, b):
    U, S, Vh = la.svd(A)
    y = la.lstsq(U, b, rcond=None)[0]
    x = la.lstsq(np.diag(S)@Vh, y, rcond=None)[0]
    cond = max(S)/min(S)
    return x, cond

#easy problem
m = 5
n = 5
A_seed = 3
A_mu = 20.
A_sig = 1
r_seed = 19
r_mu = 20.
r_sig = 1
xi_seed = 2
xi_mu = 0
xi_sig = 0
r, t, F = fake_data(m, n, A_seed, A_mu, A_sig, r_seed, r_mu, r_sig, xi_seed, \
    xi_mu, xi_sig)
A = generate_A(r, t)
x_cho = Cholesky(A, F)
x_qr = QR(A, F)
x_svd, cond = SVD(A, F)

print("\nRMS differences between resulting solution vectors\n")
dim_info = "Dimension: m = {m:3d}, n = {n:3d}\n"
dim_info = dim_info.format(m = m, n = n)
print(dim_info)
A_info = "A:  seed = {A_seed:3d}, entry mean = {A_mu:8.1f}, sd = {A_sig:8.2f}"
A_info = A_info.format( A_seed = A_seed, A_mu = A_mu, A_sig = A_sig)
print(A_info)
r_info = "r:  seed = {r_seed:3d}, entry mean = {r_mu:8.1f}, sd = {r_sig:8.2f}"
r_info = r_info.format( r_seed = r_seed, r_mu = r_mu, r_sig = r_sig)
print(r_info)
xi_info = "xi: seed = {xi_seed:3d}, entry mean = {xi_mu:8.1f}, sd = {xi_sig:8.2f}\n"
xi_info = xi_info.format( xi_seed = xi_seed, xi_mu = xi_mu, xi_sig = xi_sig)
print(xi_info)
cq_info = "Cholesky vs. QR: {cq:14.6e}"
cs_info = "Cholesky vs SVD: {cs:14.6e}, condition number = {cond:14.6e}"
qs_info = "QR vs SVD:       {qs:14.6e}"
cq_info = cq_info.format( cq = la.norm(x_cho-x_qr))
cs_info = cs_info.format( cs = la.norm(x_cho-x_svd), cond = cond)
qs_info = qs_info.format( qs = la.norm(x_qr-x_svd))
print(cq_info)
print(cs_info)
print(qs_info)
