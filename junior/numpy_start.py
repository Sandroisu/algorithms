import time

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

import numpy as np

# Матрица поворота на угол θ
def rotate(theta_deg):
    theta = np.radians(theta_deg)
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

v = np.array([1, 0])
R = rotate(90)
print(R @ v)


# Исходные данные (рост → вес)
X = np.array([[150], [160], [170], [180], [190]])  # рост
y = np.array([[50], [55], [60], [65], [70]])       # вес

# Добавляем столбец единиц (bias)
X_b = np.hstack([X, np.ones((X.shape[0], 1))])

# Решаем формулой нормальных уравнений
w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print("Параметры модели (w, b):", w.ravel())

# Признаки: площадь, количество комнат
X = np.array([
    [50, 1],
    [60, 2],
    [80, 3],
    [100, 4]
])
# Целевая переменная: цена (тыс. $)
y = np.array([[150], [180], [240], [300]])

# Добавляем bias
X_b = np.hstack([X, np.ones((X.shape[0], 1))])

# Решаем
w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print("w:", w.ravel())


y_pred = X_b @ w
mse = np.mean((y - y_pred) ** 2)
print("MSE:", mse)

def new_age():
    current = time.perf_counter()
    print(f"{current}")

new_age()

def min_max_scale(x):
    x = np.array(x, dtype=float)
    mn = x.min(axis=0)
    mx = x.max(axis=0)
    return (x - mn) / (mx - mn)

def softmax(x):
    x = np.array(x, dtype=float)
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)