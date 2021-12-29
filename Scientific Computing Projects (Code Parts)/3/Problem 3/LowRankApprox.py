import numpy as np
import numpy.linalg as la
import time
from ReadParticleData import read

def generate_low_rank(epsilon, max_t):
    x, y = read()
    ns = len(x)
    nt = len(y)
    A = np.zeros((nt, ns))
    print("Data Read.")

    start = time.time()
    for j in range(nt):
        for i in range(ns):
            A[j, i] = 1/la.norm(x[i]-y[j])**2
    end = time.time()
    print("A completed,   elapsed time is " + str(int(end-start)) + " seconds")

    start = time.time()
    SVD = la.svd(A, full_matrices=False)
    u, s, v = SVD
    for k in (np.arange(len(s))+1)*10:
        Ak = np.zeros((nt, ns))
        for i in range(k):
            Ak += s[i] * np.outer(u.T[i], v[i])
        if la.norm(A-Ak) <= epsilon:
            break
        end = time.time()
        if (end-start >= max_t):
            print("Failed to converge within " + str(max_t) + " seconds")
            break

    print("SVD completed, elapsed time is " + str(int(end-start)) + " seconds.")
    return x, A, Ak

def test(x, A, Ak):
    w = np.zeros(len(x))
    r = np.zeros(3)
    for j in range(len(x)):
        w[j] = 1/la.norm(x[j]-r)**2

    bk = Ak@w
    b = A@w

    pred_accu = epsilon*la.norm(w)
    real_accu = la.norm(bk-b)

    pred_info = "Predicted Error: {pa:8.4f}"
    real_info = "Actual Error:    {ra:8.4f}"
    pred_info = pred_info.format(pa = pred_accu)
    real_info = real_info.format(ra = real_accu)
    print(pred_info)
    print(real_info)

def total_illum(r, x, A):
    b = 0
    w = np.zeros(len(x))
    for i in range(len(r)):
        for j in range(len(x)):
            w[j] = 1/la.norm(x[j]-r[i])**2
        b += A@w

epsilon = 0.1
max_t = 300
num_test = 1000
x, A, Ak = generate_low_rank(epsilon, max_t)
test(x, A, Ak)
r = np.random.rand(num_test, 3)*10
start = time.time()
total_illum(r, x, A)
end = time.time()
print("Raw calculation takes          " + str(int(end-start)) + " seconds.")
start = time.time()
total_illum(r, x, Ak)
end = time.time()
print("Reduced rank calculation takes " + str(int(end-start)) + " seconds.")
