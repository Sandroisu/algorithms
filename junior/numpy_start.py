import numpy as np

a = np.array([1, 2, 3, 4])
print(a)
print(type(a))


a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

print(a + b)     # [11 22 33]
print(a * b)     # [10 40 90]
print(a ** 2)    # [1 4 9]
print(np.sqrt(a))  # [1.         1.41421356 1.73205081]