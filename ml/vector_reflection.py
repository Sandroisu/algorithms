import numpy as np
import matplotlib.pyplot as plt

# Берём вектор v = (2, 1)
v = np.array([2, 1])

# -----------------------------
# 1. Матрицы отражения
# -----------------------------
# Отражение относительно оси X:
# Rx = [[1,  0],
#       [0, -1]]

# Отражение относительно оси Y:
# Ry = [[-1, 0],
#       [ 0, 1]]

# Выберем отражение относительно оси X
Rx = np.array([[1, 0],
               [0, -1]])

# Умножаем матрицу на вектор
v_reflected = Rx @ v

print("Исходный вектор:", v)
print("После отражения относительно X:", v_reflected)

# -----------------------------
# 2. Визуализация
# -----------------------------
plt.figure()

# Исходный вектор (синий)
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
           color='blue', label='v')

# Отражённый вектор (красный)
plt.quiver(0, 0, v_reflected[0], v_reflected[1], angles='xy', scale_units='xy', scale=1,
           color='red', label='Rx*v')

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.gca().set_aspect('equal', adjustable='box')
plt.axhline(0, color='black', linewidth=0.5)  # рисуем ось X
plt.axvline(0, color='black', linewidth=0.5)  # рисуем ось Y
plt.legend()
plt.show()
