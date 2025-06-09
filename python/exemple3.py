import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Données simulées pour classification (1000 échantillons, 20 features, 3 classes)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                           n_redundant=5, n_classes=3, n_clusters_per_class=1, random_state=42)

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ACP
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ACP complète pour Scree plot
pca_full = PCA().fit(X_scaled)
explained_variance = pca_full.explained_variance_ratio_

# Générer un signal 1D bruité pour la démonstration de débruitage
np.random.seed(42)
signal = np.sin(np.linspace(0, 6 * np.pi, 100))
noisy_signal = signal + np.random.normal(0, 0.3, size=signal.shape)
noisy_matrix = np.tile(noisy_signal, (20, 1))  # Simuler plusieurs observations similaires
noisy_matrix += np.random.normal(0, 0.1, noisy_matrix.shape)

# ACP pour débruitage
pca_denoise = PCA(n_components=2)
compressed = pca_denoise.fit_transform(noisy_matrix)
reconstructed = pca_denoise.inverse_transform(compressed)

# Création des figures
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 1. Projection ACP 2D des classes
scatter = axs[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
axs[0, 0].set_title("Projection 2D des données simulées via l'ACP")
axs[0, 0].set_xlabel("Composante principale 1")
axs[0, 0].set_ylabel("Composante principale 2")

# 2. Scree plot
axs[0, 1].plot(np.cumsum(explained_variance)*100, marker='o')
axs[0, 1].set_title("Variance expliquée cumulée")
axs[0, 1].set_xlabel("Nombre de composantes")
axs[0, 1].set_ylabel("Variance expliquée (%)")
axs[0, 1].grid(True)

# 3. Signal bruité
axs[1, 0].plot(noisy_signal, label="Signal bruité", alpha=0.7)
axs[1, 0].plot(signal, label="Signal original", linestyle="--")
axs[1, 0].set_title("Signal 1D bruité vs original")
axs[1, 0].legend()

# 4. Signal reconstruit après ACP
axs[1, 1].plot(reconstructed[0], label="Reconstruit par ACP", color='green')
axs[1, 1].plot(signal, label="Signal original", linestyle="--", color='black')
axs[1, 1].set_title("Débruitage via ACP (reconstruction)")
axs[1, 1].legend()

plt.tight_layout()
plt.show()
