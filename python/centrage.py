import numpy as np
import matplotlib.pyplot as plt

# Données 2D non centrées
np.random.seed(0)
X = np.random.multivariate_normal(mean=[5, 10], cov=[[2, 1], [1, 2]], size=200)

# Calcul de la moyenne
mean = np.mean(X, axis=0)

# Centrage des données : soustraction de la moyenne
X_centered = X - mean

# Tracer les données avant et après centrage
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Avant centrage
axes[0].scatter(X[:, 0], X[:, 1], alpha=0.5, label='Données originales')
axes[0].scatter(mean[0], mean[1], color='red', label='Centre des données', zorder=5)
axes[0].set_title("Avant centrage")
axes[0].axhline(0, color='gray', linestyle='--')
axes[0].axvline(0, color='gray', linestyle='--')
axes[0].legend()
axes[0].axis('equal')
axes[0].grid(True)

# Après centrage
axes[1].scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.5, label='Données centrées')
axes[1].scatter(0, 0, color='green', label='Nouvelle moyenne (0,0)', zorder=5)
axes[1].set_title("Après centrage")
axes[1].axhline(0, color='gray', linestyle='--')
axes[1].axvline(0, color='gray', linestyle='--')
axes[1].legend()
axes[1].axis('equal')
axes[1].grid(True)

plt.suptitle("Illustration du centrage des données")
plt.tight_layout()
plt.show()
