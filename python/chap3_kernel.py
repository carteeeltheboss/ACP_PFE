import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

# -------------------------
# 1. Génération de données
# -------------------------

np.random.seed(42)

# Données normales
X_normal = np.random.multivariate_normal(
    mean=[100, 12, 300],
    cov=[[100, 5, 20], [5, 10, 0], [20, 0, 10000]],
    size=200
)

# Données frauduleuses (hors distribution normale)
X_fraud = np.array([
    [500, 3, 20000],
    [700, 22, 5000],
    [50, 0, 30000],
])

# Regrouper
X = np.vstack([X_normal, X_fraud])
y = np.array([0]*len(X_normal) + [1]*len(X_fraud))  # 0 = normal, 1 = fraude

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# 2. Application du Kernel PCA
# -------------------------

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_kpca = kpca.fit_transform(X_scaled)

# -------------------------
# 3. Détection d'anomalies
# -------------------------

# On suppose que les points normaux sont proches du centre
center = X_kpca[y == 0].mean(axis=0)
distances = np.linalg.norm(X_kpca - center, axis=1)

# Seuil heuristique : 95e percentile des distances normales
threshold = np.percentile(distances[y == 0], 95)

# Marquer comme anomalie si distance > seuil
y_pred = (distances > threshold).astype(int)

# -------------------------
# 4. Affichage des résultats
# -------------------------

plt.figure(figsize=(8, 6))
plt.scatter(X_kpca[y_pred == 0][:, 0], X_kpca[y_pred == 0][:, 1], label='Normal', alpha=0.5)
plt.scatter(X_kpca[y_pred == 1][:, 0], X_kpca[y_pred == 1][:, 1], label='Fraude détectée', color='red')
plt.title("Détection de fraudes avec Kernel PCA")
plt.legend()
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.grid(True)
plt.show()
