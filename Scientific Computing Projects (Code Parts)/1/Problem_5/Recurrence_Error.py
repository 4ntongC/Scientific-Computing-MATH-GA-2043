import numpy as np

def fibonacci_a(a):
    steps = 100
    sequence = np.zeros((2, steps)) #back and forth
    sequence[0, 0] = 1
    sequence[0, 1] = 1
    for i in range(2, steps):
        sequence[0, i] = a*sequence[0, i-2] + sequence[0, i-1]
    sequence[1, steps-1] = sequence[0, steps-1]
    sequence[1, steps-2] = sequence[0, steps-2]
    for i in range(3, steps+1):
        sequence[1, steps-i] = (sequence[1, steps-i+2] - sequence[1, steps-i+1]) / a
    print(sequence)

print("The case for a = 1")
fibonacci_a(1)
print("The case for a = 1 + .1*sqrt(2)")
fibonacci_a(1+0.1*np.sqrt(2))

"""
The operation of addition (i.e. the values on the first row) was generally a
success. The case for a=1 achieved the exact answer for the 100th term of the
Fibonacci sequence, and although the exact answer for the second case isn't
available I would assume the same since it uses the same algorithm.

For both of these cases, the answer eventually blows up due to prematurely reaching
a negative value. This could be attributed to the fact that huge values exceeding
the limit of what a float64 value could store are rounded by the computer,
whose effect became quite noticable by the time the order of the terms got small again
during the subtraction operations.
"""
