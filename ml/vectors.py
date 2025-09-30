import numpy as np
import matplotlib.pyplot as plt
# 1. Векторы
v1 = np.array([2, 3])
v2 = np.array([1, -1])

# Длина (норма)
norm_v1 = np.linalg.norm(v1)
norm_v2 = np.linalg.norm(v2)

# Скалярное произведение
dot = np.dot(v1, v2)

# Угол между векторами
cos_theta = dot / (norm_v1 * norm_v2)
theta = np.arccos(cos_theta)  # в радианах

print("||v1|| =", norm_v1)
print("||v2|| =", norm_v2)
print("dot(v1, v2) =", dot)
print("theta (в градусах) =", np.degrees(theta))

# 2. Матрица вращения
R = np.array([[0, -1],
              [1,  0]])

v1_rotated = R @ v1

# Визуализация
plt.figure()
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v1')
plt.quiver(0, 0, v1_rotated[0], v1_rotated[1], angles='xy', scale_units='xy', scale=1, color='red', label='R*v1')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.show()

# 3. Собственные значения и векторы
eigvals, eigvecs = np.linalg.eig(R)
print("Eigenvalues:", eigvals)
print("Eigenvectors:\n", eigvecs)