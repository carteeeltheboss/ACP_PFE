import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Création de données 2D synthétiques
np.random.seed(42)
mean = [0, 0]
cov = [[3, 2], [2, 2]]  # matrice de covariance
X = np.random.multivariate_normal(mean, cov, 200)

# Appliquer l'ACP
pca = PCA(n_components=2)
pca.fit(X)

# Moyenne des données (centre)
origin = pca.mean_

# Vecteurs propres (directions principales)
eigenvectors = pca.components_

# Valeurs propres (grandeur de la variance dans chaque direction)
eigenvalues = pca.explained_variance_

# Affichage des données
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.2, label='Données')

# Tracer les vecteurs propres (échelle par la racine des valeurs propres)
for i in range(2):
    vec = eigenvectors[i] * np.sqrt(eigenvalues[i]) * 3  # échelle arbitraire
    plt.quiver(*origin, *vec, angles='xy', scale_units='xy', scale=1, color=['r', 'g'][i],
               label=f'PC{i+1}')

plt.axis('equal')
plt.grid(True)
plt.title("Vecteurs propres (composantes principales) et direction de la variance maximale")
plt.xlabel("Axe X")
plt.ylabel("Axe Y")
plt.legend()
plt.show()
