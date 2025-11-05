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


X = np.array([[2, 3]])  # два признака
W = np.array([[0.4], [0.6]])  # два веса

y = X @ W
print(y)  # [[2.8]]


A = np.array([[2, 0],
              [1, 3]])
B = np.array([[1, 4],
              [2, 5]])