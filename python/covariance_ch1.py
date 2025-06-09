import numpy as np
import matplotlib.pyplot as plt

# Générer des données corrélées (relation linéaire)
np.random.seed(0)
X_corr = np.random.rand(100)
Y_corr = X_corr + np.random.normal(0, 0.1, 100)  # Corrélées avec un peu de bruit

# Générer des données non corrélées
X_non_corr = np.random.rand(100)
Y_non_corr = np.random.rand(100)  # Non corrélées, générées indépendamment

# Créer la figure
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Données corrélées
ax[0].scatter(X_corr, Y_corr, color='blue')
ax[0].set_title("Données Corrélées")
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")

# Données non corrélées
ax[1].scatter(X_non_corr, Y_non_corr, color='red')
ax[1].set_title("Données Non Corrélées")
ax[1].set_xlabel("X")
ax[1].set_ylabel("Y")

# Affichage de la figure
plt.tight_layout()
plt.show()
