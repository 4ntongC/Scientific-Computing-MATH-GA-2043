import numpy as np
from scipy import special

x = 1.1
n = 20
expansion = 0

for i in range(1, n+2):
    term = special.comb(n, i-1)*(x**(n+1-i))*((-1)**(i-1))
    print(term)
    expansion += term
print("Final polynomial result: ", expansion)
print("Direct computation: ", (x-1)**n)
print("Accurate answer: 1e-20")
"""
Command Output:

6.727499949325611
-122.31818089682929
1056.3842895635255
-5762.0961248919575
22262.644118900742
-64764.05561862033
147191.03549686438
-267620.0645397534
395347.8226155448
-479209.481958236
479209.481958236
-396040.8941803603
270027.8823957002
-151064.54959200008
68665.70436000003
-24969.347040000008
7093.564500000002
-1517.3400000000004
229.90000000000003
-22.0
1.0
Final polynomial result:  4.0188297134591267e-11
Direct computation:  1.0000000000000177e-20
Accurate answer: 1e-20
"""
