import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Исходный вектор
# -----------------------------
v = np.array([2, 1])  # исходный вектор
print("Исходный вектор:", v)

# -----------------------------
# 2. Матрица поворота
# -----------------------------
# Формула матрицы поворота:
# R(θ) = [[cosθ, -sinθ],
#          [sinθ,  cosθ]]

# Выберем угол 45 градусов (π/4 радиан)
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

# Применяем поворот
v_rotated = R @ v
print("После поворота:", v_rotated)

# -----------------------------
# 3. Матрица масштабирования
# -----------------------------
# Увеличим по X в 1.5 раза, по Y — в 0.7 раза
S = np.array([[1.5, 0],
              [0, 0.7]])

v_scaled = S @ v_rotated
print("После масштабирования:", v_scaled)

# -----------------------------
# 4. Матрица отражения относительно оси X
# -----------------------------
Rx = np.array([[1, 0],
               [0, -1]])

v_reflected = Rx @ v_scaled
print("После отражения:", v_reflected)

# -----------------------------
# 5. Визуализация пошагового результата
# -----------------------------
plt.figure(figsize=(6, 6))

# Исходный вектор — синий
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
           color='blue', label='Исходный вектор')

# После поворота — оранжевый
plt.quiver(0, 0, v_rotated[0], v_rotated[1], angles='xy', scale_units='xy', scale=1,
           color='orange', label='После поворота')

# После масштабирования — зелёный
plt.quiver(0, 0, v_scaled[0], v_scaled[1], angles='xy', scale_units='xy', scale=1,
           color='green', label='После масштабирования')

# После отражения — красный
plt.quiver(0, 0, v_reflected[0], v_reflected[1], angles='xy', scale_units='xy', scale=1,
           color='red', label='После отражения')

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.title("Композиция линейных преобразований: поворот → масштаб → отражение")
plt.show()
