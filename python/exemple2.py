# Réexécution après réinitialisation de l'environnement
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Génération des données simulées
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                           n_redundant=5, n_classes=3, n_clusters_per_class=1, random_state=42)

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Réduction à 3 composantes principales
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# Tracé 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y, cmap='tab10', alpha=0.7)

ax.set_title("Projection 3D des données simulées via l'ACP")
ax.set_xlabel("Composante principale 1")
ax.set_ylabel("Composante principale 2")
ax.set_zlabel("Composante principale 3")
plt.show()
