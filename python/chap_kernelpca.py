import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate circular non-linearly separable data
np.random.seed(0)
n = 100
theta = 2 * np.pi * np.random.rand(n)
r = 0.3 + 0.1 * np.random.randn(n)
x1 = r * np.cos(theta)
x2 = r * np.sin(theta)

# Stack into 2D data
X = np.vstack((x1, x2)).T

# Define nonlinear mapping (e.g., to simulate kernel trick)
def phi(x):
    # Feature map: x1, x2, and x1^2 + x2^2 (like RBF)
    return np.array([x[0], x[1], x[0]**2 + x[1]**2])

# Apply mapping
X_mapped = np.array([phi(x) for x in X])

# Plot original 2D data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='blue', s=30, alpha=0.6)
plt.title('Espace d\'entrée (non linéaire)')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')

# Plot mapped 3D data
ax = plt.subplot(1, 2, 2, projection='3d')
ax.scatter(X_mapped[:, 0], X_mapped[:, 1], X_mapped[:, 2], c='red', s=30, alpha=0.6)
ax.set_title('Espace de caractéristiques $\phi(x)$')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_1^2 + x_2^2$')

plt.tight_layout()
plt.savefig("kernel_mapping.png")
plt.show()
