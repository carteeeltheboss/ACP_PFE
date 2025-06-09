import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Simulation de comportements passagers
# -----------------------------

# 1000 passagers "normaux"
np.random.seed(42)
normaux = np.random.multivariate_normal(
    mean=[10, 5, 2, 0.5],  # distance, arrêts, virages, proximité
    cov=[[2, 0.5, 0.3, 0],
         [0.5, 1, 0.2, 0],
         [0.3, 0.2, 1, 0],
         [0, 0, 0, 0.2]],
    size=1000
)

# 20 passagers "anormaux"
anormaux = np.random.multivariate_normal(
    mean=[3, 10, 8, 2],
    cov=[[1, 0.2, 0.1, 0],
         [0.2, 1, 0.2, 0],
         [0.1, 0.2, 1, 0],
         [0, 0, 0, 0.1]],
    size=20
)

X = np.vstack((normaux, anormaux))
labels = np.array([0]*1000 + [1]*20)  # 0 = normal, 1 = anormal

# -----------------------------
# 2. Standardisation et ACP
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# -----------------------------
# 3. Détection d’anomalies via IsolationForest
# -----------------------------
clf = IsolationForest(contamination=0.02, random_state=42)
outliers = clf.fit_predict(X_pca)
# -1 = anomalie, 1 = normal
predicted_anomalies = (outliers == -1).astype(int)

# -----------------------------
# 4. Visualisation
# -----------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Visualisation selon les vraies étiquettes
ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="coolwarm", edgecolor='k', alpha=0.7)
ax[0].set_title("Projection ACP - Vérité terrain")
ax[0].set_xlabel("Composante principale 1")
ax[0].set_ylabel("Composante principale 2")
ax[0].legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Normaux', markerfacecolor='blue', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Anormaux', markerfacecolor='red', markersize=8)
])

# Visualisation selon la détection automatique
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=predicted_anomalies, cmap="coolwarm", edgecolor='k', alpha=0.7)
ax[1].set_title("Projection ACP - Détection automatique")
ax[1].set_xlabel("Composante principale 1")
ax[1].set_ylabel("Composante principale 2")
ax[1].legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Détecté normal', markerfacecolor='blue', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Détecté anormal', markerfacecolor='red', markersize=8)
])

plt.tight_layout()
plt.show()
