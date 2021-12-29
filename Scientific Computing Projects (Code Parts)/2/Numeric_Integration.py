from scipy import integrate

class m:
    def __init__(this, f):
        this.f = f

def RR1f(a, b, n, m):
    ans = 0;
    delta = (b - a) / n #length of each interval
    x = a - delta

    for i in range(n):
        x += delta
        ans += m.f(x)*delta

    return ans

def MR2f(a, b, n, m):
    ans = 0
    delta = (b - a) / n
    x = a - delta / 2

    for i in range(n):
        x += delta
        ans += m.f(x)*delta

    return ans

def RR1a(a, b, n0, m, tol):
    accu = integrate.quad(m.f, a, b)[0]
    approx = RR1f(a, b, n0, m)
    err = abs(accu - approx)
    n_max = 1000000
    while err / accu > tol:
        n0 *= 2
        approx = RR1f(a, b, n0, m)
        err = abs(accu - approx)
        if n0 > n_max:
            break
    return (approx, n0, n0 <= n_max)

def SR4f(a, b, n, m):
    ans = 0
    delta = (b - a) / n
    x = a - delta

    for i in range(n):
        x += delta
        ans += (m.f(x)+4*m.f(x+delta/2)+m.f(x+delta))*delta/6

    return ans

def SR4a(a, b, n0, m, tol):
    accu = integrate.quad(m.f, a, b)[0]
    approx = SR4f(a, b, n0, m)
    err = abs(accu - approx)
    n_max = 1000000
    while err / accu > tol:
        n0 *= 2
        approx = SR4f(a, b, n0, m)
        err = abs(accu - approx)
        if n0 > n_max:
            break
    return (approx, n0, n0 <= n_max)
