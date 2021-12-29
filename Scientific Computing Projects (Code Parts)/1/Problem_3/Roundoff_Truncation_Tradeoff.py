import numpy as np
from tabulate import tabulate

steps = 300
interval_length = 10**(-10)
delta_x = np.arange(1.0, 1.0 + steps) * interval_length
result = np.zeros((steps, 10))
"""
The indices are:
    0. delta x
    2. derivative of f (accurate)
    4. derivative of f (two point forward approximate)
    6. absolute error
    8. relative error
The odd indices are the previous ones in single precision instead of double
"""
def f64(x):
    return np.float64(np.exp(x))
def f32(x):
    return np.float32(f64(x))
x = 1
for i in range(steps):
    result[i, 0] = np.float64(delta_x[i])
    result[i, 2] = np.float64(np.exp(x)) #alter manually according to f(x)
    result[i, 4] = np.float64((f64(x + result[i, 0]) - f64(x)) / result[i, 0])
    result[i, 6] = np.float64(abs(result[i, 4] - result[i, 2]))
    result[i, 8] = np.float64(result[i, 6] / x)

    result[i, 1] = np.float32(delta_x[i])
    result[i, 3] = np.float32(np.exp(x)) #alter manually according to f(x)
    result[i, 5] = np.float32((f32(x + result[i, 0]) - f32(x)) / result[i, 0])
    result[i, 7] = np.float32(abs(result[i, 4] - result[i, 2]))
    result[i, 9] = np.float32(result[i, 6] / x)
print(tabulate(result, floatfmt=".10f"))

"""
Output too large to include unfortunately (counting linebreaks that's hundreds,
perhaps over a thousand lines). The specs declared at the beginning of the code
are handpicked to demonstrate this phenomenon the best.
"""
