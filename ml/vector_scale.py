import numpy as np
import matplotlib.pyplot as plt

# Берём вектор v = (2, 1)
v = np.array([2, 1])

# -----------------------------
# 1. Матрица масштабирования
# -----------------------------
# Общая форма:
# S = [[sx, 0 ],
#      [0 , sy]]
# sx — коэффициент растяжения по оси X
# sy — коэффициент растяжения по оси Y

# Например, растянем в 2 раза по X и в 0.5 раза по Y
S = np.array([[2, 0],
              [0, 0.5]])

# Умножаем матрицу на вектор
v_scaled = S @ v

print("Исходный вектор:", v)
print("После масштабирования:", v_scaled)

# -----------------------------
# 2. Визуализация
# -----------------------------
plt.figure()

# Исходный вектор (синий)
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
           color='blue', label='v')

# Масштабированный вектор (зелёный)
plt.quiver(0, 0, v_scaled[0], v_scaled[1], angles='xy', scale_units='xy', scale=1,
           color='green', label='S*v')

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.show()
